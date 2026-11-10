//! Dynamic programming beat tracker matching librosa's `beat.beat_track`.
//!
//! Implements the method of Ellis (2007):
//!   1. Normalize the onset envelope by its standard deviation.
//!   2. Smooth with a Gaussian window (local score) sized to the estimated
//!      beat period.
//!   3. Run a dynamic program that picks the highest-scoring path of beats
//!      spaced approximately one beat period apart.
//!   4. Backtrack from the best terminal beat to recover the beat sequence.
//!   5. Trim weak leading/trailing beats.

const TIGHTNESS: f32 = 100.0;
const TINY: f32 = 1e-10;

/// Detect beats (frame indices, ascending) from an onset strength envelope
/// and an estimated tempo.
///
/// Matches `librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop,
/// start_bpm=bpm)` with default `tightness=100`, `trim=True`.
pub fn beat_track(env: &[f32], sr: f32, hop_length: usize, bpm: f32) -> Vec<usize> {
    let n = env.len();
    if n < 4 || bpm <= 0.0 {
        return vec![];
    }

    let frames_per_beat = (sr / hop_length as f32 * 60.0 / bpm).round() as usize;
    if frames_per_beat < 2 {
        return vec![];
    }

    // --- 1. Normalize onsets by std ---
    let std = env_std(env);
    let norm = std + TINY;
    let env_norm: Vec<f32> = env.iter().map(|&v| v / norm).collect();

    // --- 2. Local score (Gaussian same-convolution) ---
    let localscore = beat_local_score(&env_norm, frames_per_beat);

    // --- 3. Dynamic programming ---
    let (backlink, cumscore) = beat_track_dp(&localscore, frames_per_beat);

    // --- 4. Backtrack ---
    let tail = last_beat(&cumscore);
    let mut beats = vec![false; n];
    dp_backtrack(&backlink, tail, &mut beats);

    // --- 5. Trim weak leading/trailing beats ---
    trim_beats(&localscore, &mut beats, true);

    (0..n).filter(|&i| beats[i]).collect()
}

/// `librosa.beat.__normalize_onsets`: `onsets / (std + tiny)`.
fn env_std(env: &[f32]) -> f32 {
    let n = env.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = env.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = env.iter().map(|&v| {
        let d = v as f64 - mean;
        d * d
    }).sum::<f64>()
        / (n - 1.0);
    var.sqrt() as f32
}

/// `librosa.beat.__beat_local_score`: Gaussian same-mode convolution.
///
/// The window is `exp(-0.5 * ((k - fpb) * 32 / fpb)^2)` for `k` in `0..2*fpb+1`,
/// which is a Gaussian centered at `fpb` with `sigma = fpb / sqrt(32)`.
fn beat_local_score(env: &[f32], frames_per_beat: usize) -> Vec<f32> {
    let n = env.len();
    let fpb = frames_per_beat;
    let k_len = 2 * fpb + 1;
    let center = fpb;

    // Build the Gaussian window.
    let window: Vec<f32> = (0..k_len)
        .map(|k| {
            let x = (k as f32 - center as f32) * 32.0 / fpb as f32;
            (-0.5 * x * x).exp()
        })
        .collect();

    // Same-mode convolution.
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0f32;
        for (k, &w) in window.iter().enumerate() {
            let j = i as i64 + center as i64 - k as i64;
            if j < 0 || j >= n as i64 {
                continue;
            }
            acc += w * env[j as usize];
        }
        out[i] = acc;
    }
    out
}

/// `librosa.beat.__beat_track_dp`: the core dynamic program.
///
/// Returns `(backlink, cumscore)`.
fn beat_track_dp(
    localscore: &[f32],
    frames_per_beat: usize,
) -> (Vec<i64>, Vec<f32>) {
    let n = localscore.len();
    let fpb = frames_per_beat as f32;
    let score_thresh = 0.01 * localscore.iter().cloned().fold(0.0f32, f32::max);

    let mut backlink = vec![-1i64; n];
    let mut cumscore = vec![0.0f32; n];

    backlink[0] = -1;
    cumscore[0] = localscore[0];

    let mut first_beat = true;

    for i in 1..n {
        let score_i = localscore[i];
        let mut best_score = f32::NEG_INFINITY;
        let mut best_loc: i64 = -1;

        // Search predecessors: loc in (i - fpb/2) down to (i - 2*fpb - 1)
        let start = (i as i64 - (fpb / 2.0).round() as i64).max(0);
        let end = (i as i64 - 2 * frames_per_beat as i64 - 1).max(-1);

        let mut loc = start;
        while loc > end {
            if loc < 0 {
                break;
            }
            let dist = i as i64 - loc;
            let penalty = (dist as f32).ln() - fpb.ln();
            let score = cumscore[loc as usize] - TIGHTNESS * penalty * penalty;
            if score > best_score {
                best_score = score;
                best_loc = loc;
            }
            loc -= 1;
        }

        if best_loc >= 0 {
            cumscore[i] = score_i + best_score;
        } else {
            cumscore[i] = score_i;
        }

        if first_beat && score_i < score_thresh {
            backlink[i] = -1;
        } else {
            backlink[i] = best_loc;
            first_beat = false;
        }
    }

    (backlink, cumscore)
}

/// `librosa.beat.__last_beat`: find the terminal beat position.
///
/// The tail is the last local maximum of `cumscore` that exceeds half the
/// median of local-maximum values.
fn last_beat(cumscore: &[f32]) -> usize {
    let n = cumscore.len();
    if n == 0 {
        return 0;
    }

    // Find local maxima of cumscore.
    let mut lmax_values: Vec<f32> = Vec::new();
    for i in 1..n.saturating_sub(1) {
        if cumscore[i] > cumscore[i - 1] && cumscore[i] >= cumscore[i + 1] {
            lmax_values.push(cumscore[i]);
        }
    }
    // Handle last element.
    if n >= 2 && cumscore[n - 1] > cumscore[n - 2] {
        lmax_values.push(cumscore[n - 1]);
    }

    if lmax_values.is_empty() {
        return n - 1;
    }

    // Median of local-max values.
    let mut sorted = lmax_values.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    };
    let threshold = 0.5 * median;

    // Find the last position that is a local max and exceeds threshold.
    let mut tail = n - 1;
    for i in (0..n).rev() {
        let is_lmax = if i == 0 {
            false
        } else if i == n - 1 {
            cumscore[i] > cumscore[i - 1]
        } else {
            cumscore[i] > cumscore[i - 1] && cumscore[i] >= cumscore[i + 1]
        };
        if is_lmax && cumscore[i] >= threshold {
            tail = i;
            break;
        }
    }

    tail
}

/// `librosa.beat.__dp_backtrack`: follow backlinks from `tail` to mark beats.
fn dp_backtrack(backlink: &[i64], tail: usize, beats: &mut [bool]) {
    let mut n = tail as i64;
    while n >= 0 {
        beats[n as usize] = true;
        n = backlink[n as usize];
    }
}

/// `librosa.beat.__trim_beats`: remove weak leading/trailing beats.
fn trim_beats(localscore: &[f32], beats: &mut [bool], trim: bool) {
    if !trim {
        return;
    }

    let n = localscore.len();
    if n == 0 {
        return;
    }

    // Collect beat scores.
    let beat_scores: Vec<f32> = (0..n).filter(|&i| beats[i]).map(|i| localscore[i]).collect();
    if beat_scores.is_empty() {
        return;
    }

    // Hanning window of size 5 for smoothing.
    let w = [0.05, 0.35, 0.5, 0.35, 0.05]; // hanning(5) normalized
    // smooth_boe = convolve(beat_scores, w, mode='same')
    let mut smooth = vec![0.0f32; beat_scores.len()];
    for i in 0..beat_scores.len() {
        let mut acc = 0.0f32;
        for (k, &wk) in w.iter().enumerate() {
            let j = i as i32 + k as i32 - 2;
            if j >= 0 && j < beat_scores.len() as i32 {
                acc += wk * beat_scores[j as usize];
            }
        }
        smooth[i] = acc;
    }

    // RMS of smoothed beat scores.
    let rms = (smooth.iter().map(|v| v * v).sum::<f32>() / smooth.len() as f32).sqrt();
    let threshold = 0.5 * rms;

    // Trim leading beats.
    let mut i = 0usize;
    while i < n && beats[i] {
        if localscore[i] <= threshold {
            beats[i] = false;
        } else {
            break;
        }
        i += 1;
    }

    // Trim trailing beats.
    let mut i = n - 1;
    loop {
        if !beats[i] {
            break;
        }
        if localscore[i] <= threshold {
            beats[i] = false;
        } else {
            break;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
}
