//! Port of PyMusicLooper's `find_best_loop_points` (analysis.py), wired to the
//! Rust DSP in `crate::dsp`.

use crate::audio::AudioData;
use crate::dsp::beat_track;
use crate::dsp::chroma::{chroma_filterbank, chroma_stft};
use crate::dsp::mel::{mel_filterbank, melspectrogram};
use crate::dsp::onset::onset_strength;
use crate::dsp::plp;
use crate::dsp::stft::{power_from_stft, stft_centered};
use crate::dsp::tempo;
use crate::dsp::power_db::power_to_db_ref_median;
use crate::dsp::weighting::{fft_frequencies, perceptual_weighting_peak};
use crate::dsp::window::hann_periodic;
use crate::dsp::zero_cross::nearest_zero_crossing;
use crate::model::LoopPairResponse;

pub const N_FFT: usize = 2048;
pub const HOP_LENGTH: usize = 512;
pub const N_MELS: usize = 128;
pub const N_CHROMA: usize = 12;

const ACCEPTABLE_NOTE_DEVIATION: f32 = 0.15;
const ACCEPTABLE_LOUDNESS_DIFFERENCE: f32 = 0.5;
const NUM_TEST_BEATS: usize = 12;
const TUNING: f32 = 0.0; // simplified: skip librosa's estimate_tuning

#[derive(Debug, Clone)]
pub struct CandidatePair {
    start_frame: usize,
    end_frame: usize,
    note_distance: f32,
    loudness_difference: f32,
    score: f32,
}

pub struct Params {
    pub min_duration_multiplier: f32,
    pub min_loop_duration: Option<f32>,
    pub max_loop_duration: Option<f32>,
    pub approx_loop_start: Option<f32>,
    pub approx_loop_end: Option<f32>,
    pub brute_force: bool,
    pub disable_pruning: bool,
    pub debug: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            min_duration_multiplier: 0.35,
            min_loop_duration: None,
            max_loop_duration: None,
            approx_loop_start: None,
            approx_loop_end: None,
            brute_force: false,
            disable_pruning: false,
            debug: false,
        }
    }
}

/// Run the full loop-point detection pipeline.
pub fn find_loop_pairs(audio: &AudioData, params: &Params) -> Vec<LoopPairResponse> {
    let n_bins = N_FFT / 2 + 1;
    let rate = audio.rate;

    // ---- STFT + power ----
    let window = hann_periodic(N_FFT);
    let stft = stft_centered(&audio.audio, N_FFT, HOP_LENGTH, &window);
    let n_frames = stft.n_frames;
    let s_power = power_from_stft(&stft);

    // ---- chroma ----
    let chroma_fb = chroma_filterbank(rate as f64, N_FFT, TUNING, N_CHROMA);
    let chroma = chroma_stft(&s_power, n_bins, n_frames, &chroma_fb, N_CHROMA);

    // ---- perceptual-weighted power (S_weighed): ref=peak + A-weighting ----
    // PyMusicLooper: S_weighed = perceptual_weighting(S=S_power, ...) (no ref -> peak)
    let freqs = fft_frequencies(rate as f64, N_FFT);
    let mut s_weighted = s_power.clone();
    perceptual_weighting_peak(&mut s_weighted, n_bins, &freqs);

    // ---- mel spectrogram (for onset): input = S_weighed (dB domain) ----
    // PyMusicLooper: melspectrogram(S=S_weighed, sr, n_mels=128, fmax=8000)
    let mel_fb = mel_filterbank(rate as f64, N_FFT, N_MELS, 0.0, 8000.0);
    let mel_spec = melspectrogram(&s_weighted, n_bins, n_frames, &mel_fb, N_MELS);

    // ---- power_db: S_weighed re-converted to dB with ref=median ----
    // PyMusicLooper: power_db = power_to_db(S_weighed, ref=np.median)
    let mut power_db = s_weighted.clone();
    power_to_db_ref_median(&mut power_db);

    let has_approx = params.approx_loop_start.is_some()
        && params.approx_loop_end.is_some();

    // ---- beats + bpm ----
    let bpm: f32;
    let beats: Vec<usize> = if has_approx {
        bpm = 120.0;
        // Approximate loop position: search a +/-2s window around each of the
        // approximate start/end (converted to frames, accounting for trim).
        let seconds_to_frames = |sec: f64| {
            let t = (sec * rate as f64 / HOP_LENGTH as f64) as i64;
            t.max(0).min(n_frames as i64) as usize
        };
        let trim_secs = audio.trim_offset as f32 / rate as f32;
        let approx_start = params
            .approx_loop_start
            .map(|s| seconds_to_frames((s - trim_secs) as f64))
            .unwrap();
        let approx_end = params
            .approx_loop_end
            .map(|s| seconds_to_frames((s - trim_secs) as f64))
            .unwrap();
        let window = (2.0 * rate as f64 / HOP_LENGTH as f64) as usize;
        let lo = |f: usize| f.saturating_sub(window);
        let hi = |f: usize| (f + window).min(n_frames);
        let mut v: Vec<usize> = (lo(approx_start)..hi(approx_start))
            .chain(lo(approx_end)..hi(approx_end))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    } else if params.brute_force {
        bpm = 120.0;
        (0..n_frames).collect()
    } else {
        let env = onset_strength(&mel_spec, N_MELS, n_frames);
        bpm = tempo::estimate_tempo(&env, rate as f32, HOP_LENGTH);
        let beats_dp = beat_track::beat_track(&env, rate as f32, HOP_LENGTH, bpm);
        let n_dp = beats_dp.len();
        // PLP: local maxima of the predominant local pulse (librosa.beat.plp).
        let pulse = plp::plp(&env, rate as f32, HOP_LENGTH);
        let beats_plp = plp::localmax(&pulse);
        // Union of beat_track and PLP beats (matching PyMusicLooper's
        // np.union1d(beats, beats_plp)), sorted and deduplicated.
        let mut beats: Vec<usize> = beats_dp;
        beats.extend_from_slice(&beats_plp);
        beats.sort_unstable();
        beats.dedup();
        if params.debug {
            let n = env.len() as f64;
            let mean = env.iter().map(|&v| v as f64).sum::<f64>() / n;
            let std = (env.iter().map(|&v| {
                let d = v as f64 - mean;
                d * d
            }).sum::<f64>() / n).sqrt();
            let max = env.iter().cloned().fold(0.0f32, f32::max);
            eprintln!(
                "debug: bpm={:.1} beats_dp={} beats_plp={} beats_union={} onset: mean={:.4} std={:.4} max={:.4}",
                bpm, n_dp, beats_plp.len(), beats.len(), mean, std, max
            );
        }
        beats
    };

    // ---- duration constraints (in frames) ----
    let total_duration_frames = (audio.total_duration * rate as f64 / HOP_LENGTH as f64) as usize;
    let (min_loop_duration, max_loop_duration) = if has_approx {
        // Adjust the duration window to the +/-2s search range.
        let window = (2.0 * rate as f64 / HOP_LENGTH as f64) as usize;
        let trim_secs = audio.trim_offset as f32 / rate as f32;
        let approx_start = params
            .approx_loop_start
            .map(|s| (((s - trim_secs) as f64) * rate as f64 / HOP_LENGTH as f64) as i64)
            .unwrap()
            .max(0) as usize;
        let approx_end = params
            .approx_loop_end
            .map(|s| (((s - trim_secs) as f64) * rate as f64 / HOP_LENGTH as f64) as i64)
            .unwrap()
            .max(0) as usize;
        let min = (approx_end as i64 - window as i64) - (approx_start as i64 + window as i64) - 1;
        let max = (approx_end as i64 + window as i64) - (approx_start as i64 - window as i64) + 1;
        (min.max(1) as usize, max.max(1) as usize)
    } else {
        let min_frames = params
            .min_loop_duration
            .map(|d| (d * rate as f32 / HOP_LENGTH as f32) as usize)
            .unwrap_or_else(|| {
                let total_frames_f32 = audio.total_duration as f32 * rate as f32 / HOP_LENGTH as f32;
                (params.min_duration_multiplier * total_frames_f32) as usize
            });
        let max_frames = params
            .max_loop_duration
            .map(|d| (d * rate as f32 / HOP_LENGTH as f32) as usize)
            .unwrap_or(total_duration_frames);
        (min_frames.max(1), max_frames)
    };

    // ---- candidate pairs ----
    let candidates = find_candidate_pairs(&chroma, &power_db, n_frames, n_bins, &beats, min_loop_duration, max_loop_duration, params.debug);
    if params.debug {
        eprintln!("debug: candidates (pre-prune) = {}", candidates.len());
    }

    if candidates.is_empty() {
        return vec![];
    }

    // ---- score + prune ----
    let beats_per_second = bpm / 60.0;
    let seconds_to_test = NUM_TEST_BEATS as f32 / beats_per_second;
    let mut test_offset = (seconds_to_test * rate as f32 / HOP_LENGTH as f32) as usize;
    if test_offset > n_frames {
        test_offset = n_frames / 4;
    }
    let weights = geomspace(test_offset, 2.max(test_offset / NUM_TEST_BEATS), 1);

    let candidates = if candidates.len() >= 100 && !params.disable_pruning {
        let pruned = prune_candidates(&candidates);
        if params.debug {
            eprintln!("debug: candidates (post-prune) = {}", pruned.len());
        }
        pruned
    } else {
        candidates
    };

    let mut scored: Vec<CandidatePair> = candidates
        .iter()
        .map(|c| {
            let score = calculate_loop_score(c.start_frame, c.end_frame, &chroma, n_frames, test_offset, &weights);
            CandidatePair {
                start_frame: c.start_frame,
                end_frame: c.end_frame,
                note_distance: c.note_distance,
                loudness_difference: c.loudness_difference,
                score,
            }
        })
        .collect();

    scored.sort_by(|a, b| b.score.total_cmp(&a.score));

    // ---- duration prioritization ----
    if scored.len() > 1 {
        prioritize_duration(&mut scored);
    }

    // ---- zero crossing + seconds ----
    let mut out: Vec<LoopPairResponse> = Vec::with_capacity(scored.len());
    for c in &scored {
        let mut sf = c.start_frame;
        let mut ef = c.end_frame;
        if audio.trim_offset > 0 {
            // apply_trim_offset: samples_to_frames(frames_to_samples(frame)+offset)
            sf = ((sf as u64 * HOP_LENGTH as u64 + audio.trim_offset as u64) / HOP_LENGTH as u64) as usize;
            ef = ((ef as u64 * HOP_LENGTH as u64 + audio.trim_offset as u64) / HOP_LENGTH as u64) as usize;
        }
        let start_sample = sf.saturating_mul(HOP_LENGTH);
        let end_sample = ef.saturating_mul(HOP_LENGTH);
        let zc_start = nearest_zero_crossing(&audio.playback, audio.n_channels, rate, start_sample);
        let zc_end = nearest_zero_crossing(&audio.playback, audio.n_channels, rate, end_sample);
        out.push(LoopPairResponse {
            loop_start_sample: zc_start as i64,
            loop_end_sample: zc_end as i64,
            loop_start_seconds: zc_start as f64 / rate as f64,
            loop_end_seconds: zc_end as f64 / rate as f64,
            score: c.score as f64,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// candidate generation
// ---------------------------------------------------------------------------

fn norm12(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn find_candidate_pairs(
    chroma: &[f32],
    power_db: &[f32],
    n_frames: usize,
    n_bins: usize,
    beats: &[usize],
    min_loop_duration: usize,
    max_loop_duration: usize,
    debug: bool,
) -> Vec<CandidatePair> {
    // deviation[idx] = norm(chroma_frame(beats[idx]) * ACCEPTABLE_NOTE_DEVIATION)
    let mut deviation = vec![0.0f32; beats.len()];
    for (idx, &b) in beats.iter().enumerate() {
        let col: Vec<f32> = (0..N_CHROMA).map(|c| chroma[c * n_frames + b] * ACCEPTABLE_NOTE_DEVIATION).collect();
        deviation[idx] = norm12(&col);
    }

    let mut out = Vec::new();
    let mut note_fail = 0usize;
    let mut loud_fail = 0usize;
    let mut max_note_dist = 0.0f32;
    let mut min_deviation = f32::MAX;
    for (idx, &loop_end) in beats.iter().enumerate() {
        min_deviation = min_deviation.min(deviation[idx]);
        for &loop_start in beats {
            let loop_length = loop_end as i64 - loop_start as i64;
            if loop_length < min_loop_duration as i64 {
                break;
            }
            if loop_length > max_loop_duration as i64 {
                continue;
            }
            let mut diff = vec![0.0f32; N_CHROMA];
            for c in 0..N_CHROMA {
                diff[c] = chroma[c * n_frames + loop_end] - chroma[c * n_frames + loop_start];
            }
            let note_distance = norm12(&diff);
            max_note_dist = max_note_dist.max(note_distance);
            if note_distance <= deviation[idx] {
                // loudness difference = |max(power_db[:,end]) - max(power_db[:,start])|
                let mut mx_end = f32::MIN;
                let mut mx_start = f32::MIN;
                for f in 0..n_bins {
                    let ve = power_db[loop_end * n_bins + f];
                    let vs = power_db[loop_start * n_bins + f];
                    if ve > mx_end { mx_end = ve; }
                    if vs > mx_start { mx_start = vs; }
                }
                let loudness_difference = (mx_end - mx_start).abs();
                if loudness_difference <= ACCEPTABLE_LOUDNESS_DIFFERENCE {
                    out.push(CandidatePair {
                        start_frame: loop_start,
                        end_frame: loop_end,
                        note_distance,
                        loudness_difference,
                        score: 0.0,
                    });
                } else {
                    loud_fail += 1;
                }
            } else {
                note_fail += 1;
            }
        }
    }
    if debug {
        eprintln!(
            "debug: find_candidate_pairs: beats={} min_dur={} max_dur={} note_fail={} loud_fail={} candidates={} max_note_dist={:.4} min_deviation={:.4}",
            beats.len(), min_loop_duration, max_loop_duration,
            note_fail, loud_fail, out.len(), max_note_dist, min_deviation
        );
    }
    out
}

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

fn calculate_subseq_beat_similarity(
    b1_start: usize,
    b2_start: usize,
    chroma: &[f32],
    n_frames: usize,
    test_end_offset: i64,
    weights: &[f32],
) -> f32 {
    let test_length = test_end_offset.unsigned_abs() as usize;

    let (s1, e1, s2, _e2, max_offset) = if test_end_offset < 0 {
        // lookbehind
        let off = (-test_end_offset).min(b1_start as i64).min(b2_start as i64) as i64;
        let s1 = b1_start - off as usize;
        let e1 = b1_start;
        let s2 = b2_start - off as usize;
        let e2 = b2_start;
        (s1, e1, s2, e2, off as usize)
    } else {
        let e1 = (b1_start + test_length as usize).min(n_frames);
        let e2 = (b2_start + test_length as usize).min(n_frames);
        let o = (e1 - b1_start).min(e2 - b2_start);
        (b1_start, b1_start + o, b2_start, b2_start + o, o)
    };

    let len = e1 - s1;
    if len == 0 {
        return 0.0;
    }

    // Per-frame cosine similarity, then weighted average.
    let mut sims = vec![0.0f32; len];
    for k in 0..len {
        let mut d = 0.0f32;
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for c in 0..N_CHROMA {
            let x = chroma[c * n_frames + s1 + k];
            let y = chroma[c * n_frames + s2 + k];
            d += x * y;
            a += x * x;
            b += y * y;
        }
        sims[k] = d / (a.sqrt() * b.sqrt()).max(1e-10);
    }

    if max_offset < test_length {
        // pad with zeros up to test_length
        let mut padded = vec![0.0f32; test_length as usize];
        padded[..max_offset].copy_from_slice(&sims);
        weighted_average(&padded, weights)
    } else {
        weighted_average(&sims, weights)
    }
}

fn weighted_average(values: &[f32], weights: &[f32]) -> f32 {
    let n = values.len().min(weights.len());
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..n {
        num += values[i] * weights[i];
        den += weights[i];
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

fn calculate_loop_score(
    b1: usize,
    b2: usize,
    chroma: &[f32],
    n_frames: usize,
    test_duration: usize,
    weights: &[f32],
) -> f32 {
    let lookahead = calculate_subseq_beat_similarity(b1, b2, chroma, n_frames, test_duration as i64, weights);
    let lookbehind = calculate_subseq_beat_similarity(b1, b2, chroma, n_frames, -(test_duration as i64), &weights.iter().rev().cloned().collect::<Vec<_>>());
    lookahead.max(lookbehind)
}

// ---------------------------------------------------------------------------
// pruning & prioritization
// ---------------------------------------------------------------------------

fn percentile(sorted: &mut [f32], p: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = (p as f64 / 100.0 * (n as f64 - 1.0)) as usize;
    sorted[idx.min(n - 1)]
}

fn prune_candidates(pairs: &[CandidatePair]) -> Vec<CandidatePair> {
    const KEEP_TOP_NOTES: f32 = 75.0;
    const KEEP_TOP_LOUDNESS: f32 = 50.0;
    const ACCEPTABLE_LOUDNESS: f32 = 0.25;
    const EPSILON: f32 = 1e-3;

    let db = pairs.iter().map(|p| p.loudness_difference).collect::<Vec<_>>();
    let nd = pairs.iter().map(|p| p.note_distance).collect::<Vec<_>>();

    let db_pos: Vec<f32> = db.iter().filter(|&&v| v > EPSILON).copied().collect();
    let nd_pos: Vec<f32> = nd.iter().filter(|&&v| v > EPSILON).copied().collect();

    let mut db_sorted = db_pos.clone();
    db_sorted.sort_by(|a, b| a.total_cmp(b));
    let db_threshold = if db_sorted.len() > 3 {
        percentile(&mut db_sorted, KEEP_TOP_LOUDNESS)
    } else {
        db.iter().cloned().fold(f32::MIN, f32::max)
    };

    let mut nd_sorted = nd_pos.clone();
    nd_sorted.sort_by(|a, b| a.total_cmp(b));
    let note_threshold = if nd_sorted.len() > 3 {
        percentile(&mut nd_sorted, KEEP_TOP_NOTES)
    } else {
        nd.iter().cloned().fold(f32::MIN, f32::max)
    };

    let db_thr = (ACCEPTABLE_LOUDNESS).max(db_threshold);
    pairs
        .iter()
        .filter(|p| p.loudness_difference <= db_thr && p.note_distance <= note_threshold)
        .cloned()
        .collect()
}

fn prioritize_duration(list: &mut Vec<CandidatePair>) {
    if list.len() < 2 {
        return;
    }
    let db = list.iter().map(|p| p.loudness_difference).collect::<Vec<_>>();
    let mut db_sorted = db.clone();
    db_sorted.sort_by(|a, b| a.total_cmp(b));
    let db_threshold = percentile(&mut db_sorted, 50.0);

    let scores = list.iter().map(|p| p.score).collect::<Vec<_>>();
    let mut sc_sorted = scores.clone();
    sc_sorted.sort_by(|a, b| a.total_cmp(b));
    let mut score_threshold = percentile(&mut sc_sorted, 90.0);
    score_threshold = score_threshold.max(list[0].score - 1e-4);

    let mut duration_max = 0usize;
    let mut duration_argmax = 0usize;
    for (idx, pair) in list.iter().enumerate() {
        if pair.score < score_threshold {
            break;
        }
        let duration = pair.end_frame - pair.start_frame;
        if duration > duration_max && pair.loudness_difference <= db_threshold {
            duration_max = duration;
            duration_argmax = idx;
        }
    }
    if duration_argmax > 0 {
        let item = list.remove(duration_argmax);
        list.insert(0, item);
    }
}

/// np.geomspace(start, stop, num)
fn geomspace(num: usize, start: usize, stop: usize) -> Vec<f32> {
    if num == 0 {
        return vec![];
    }
    if num == 1 {
        return vec![start as f32];
    }
    let start_f = start as f32;
    let stop_f = stop as f32;
    let ratio = if start_f > 0.0 && stop_f > 0.0 {
        (stop_f / start_f).log(1.0 / (num - 1) as f32)
    } else {
        1.0
    };
    (0..num).map(|i| start_f * ratio.powi(i as i32)).collect()
}
