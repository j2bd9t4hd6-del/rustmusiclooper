//! Onset strength + beat detection matching librosa's spectral flux.
//!
//! `onset_strength` computes the spectral-flux onset envelope:
//!     env[t] = mean_f max(0, S[f,t] - S[f,t-1])
//! matching librosa's `onset_strength(S=mel_spectrogram)` with default
//! `aggregate=np.mean` and `max_size=1` (no local-max reference filter).
//!
//! Beat detection is done via `beat_track` (dynamic programming, Ellis 2007)
//! in `crate::dsp::beat_track`. The legacy `detect_beats` below is retained
//! for reference but is no longer used by the analysis pipeline.

/// Spectral-flux onset envelope: `env[t] = mean_f max(0, S[f,t]-S[f,t-1])`.
pub fn onset_strength(mel: &[f32], n_mels: usize, n_frames: usize) -> Vec<f32> {
    let mut env = vec![0.0f32; n_frames];
    for t in 0..n_frames {
        if t == 0 {
            env[0] = 0.0;
            continue;
        }
        let mut sum = 0.0f32;
        for m in 0..n_mels {
            let cur = mel[m * n_frames + t];
            let prev = mel[m * n_frames + t - 1];
            sum += (cur - prev).max(0.0);
        }
        env[t] = sum / n_mels as f32;
    }
    env
}

fn mean_std(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let m = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| {
        let d = x as f64 - m;
        d * d
    }).sum::<f64>()
        / n;
    (m as f32, var.sqrt() as f32)
}

/// Detect beats (frame indices, ascending) from the onset envelope.
///
/// This is a simplified stand-in for librosa's `beat_track` (DTW around
/// `start_bpm=120`) plus `plp` local-maxima union. We:
///   1. pick local maxima of the envelope above a median+0.5*std threshold,
///   2. estimate the dominant beat period from onset-peak autocorrelation
///      (120..180 bpm band, i.e. the upper half of librosa's 60..180 range),
///   3. keep peaks spaced at least `0.7 * period` apart (greedy, strongest first).
pub fn detect_beats(env: &[f32], sr: f32, hop_length: usize) -> Vec<usize> {
    let n = env.len();
    if n < 4 {
        return vec![];
    }

    let (med, std) = mean_std(env);
    let thresh = med + 0.5 * std;

    // Candidate local maxima (greater than or equal to left, strictly greater
    // than right, above threshold).
    let mut peaks: Vec<(usize, f32)> = Vec::new();
    for t in 1..n - 1 {
        if env[t] >= env[t - 1] && env[t] > env[t + 1] && env[t] >= thresh {
            peaks.push((t, env[t]));
        }
    }

    if peaks.is_empty() {
        return vec![];
    }

    // Dominant beat period via autocorrelation over 120..180 bpm.
    let fps = sr / hop_length as f32;
    let min_lag = (fps * 60.0 / 180.0).ceil() as usize; // 180 bpm
    let max_lag = (fps * 60.0 / 120.0).floor() as usize; // 120 bpm
    let max_lag = max_lag.min(n / 2).max(1);

    let mut best_lag = 0usize;
    let mut best_corr = f64::NEG_INFINITY;
    let thr_masked: Vec<f64> = env.iter().map(|&e| if e >= thresh { e as f64 } else { 0.0 }).collect();
    for lag in min_lag.max(1)..=max_lag {
        let mut corr = 0.0f64;
        for t in 0..n - lag {
            corr += thr_masked[t] * thr_masked[t + lag];
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }
    let period = if best_lag > 0 {
        best_lag
    } else {
        1.max((fps * 60.0 / 120.0) as usize)
    };
    let min_gap = ((period as f32) * 0.7).max(1.0) as usize;

    // Greedy thinning by strength.
    let mut order: Vec<usize> = (0..peaks.len()).collect();
    order.sort_by(|&a, &b| peaks[b].1.partial_cmp(&peaks[a].1).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = vec![false; peaks.len()];
    let mut result: Vec<usize> = Vec::new();
    if let Some(&first) = order.first() {
        result.push(peaks[first].0);
        keep[first] = true;
    }
    for &p in order.iter().skip(1) {
        if keep[p] {
            continue;
        }
        let pos = peaks[p].0 as i64;
        let mut too_close = false;
        for &r in &result {
            if (r as i64 - pos).abs() < min_gap as i64 {
                too_close = true;
                break;
            }
        }
        if !too_close {
            result.push(pos as usize);
            keep[p] = true;
        }
    }
    result.sort_unstable();
    result
}
