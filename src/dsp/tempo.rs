//! Tempo (BPM) estimation matching librosa's `feature.tempo`.
//!
//! Pipeline:
//!   1. Compute a tempogram (windowed autocorrelation of the onset envelope)
//!      with a Hann window and l-infinity normalization.
//!   2. Map autocorrelation lags to BPM via `tempo_frequencies`.
//!   3. Weight the autocorrelation with a pseudo-log-normal prior centered on
//!      `start_bpm` (default 120) and select the argmax as the estimated tempo.

const AC_SIZE: f32 = 8.0;
const START_BPM: f32 = 120.0;
const STD_BPM: f32 = 1.0;
const MAX_TEMPO: f32 = 320.0;

/// Estimate the global tempo (BPM) from an onset strength envelope.
///
/// Matches `librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=hop)`
/// with default parameters (`start_bpm=120`, `std_bpm=1.0`, `ac_size=8.0`,
/// `max_tempo=320.0`).
pub fn estimate_tempo(env: &[f32], sr: f32, hop_length: usize) -> f32 {
    let n = env.len();
    if n < 4 {
        return START_BPM;
    }

    // --- 1. tempogram (windowed autocorrelation) ---
    let win_length = ((AC_SIZE * sr) / hop_length as f32).round() as usize;
    let win_length = win_length.min(n).max(2);

    // Hann window for the autocorrelation frames.
    let window = hann(win_length);

    // Number of autocorrelation lags to compute.
    let n_lags = win_length;

    // Accumulate windowed autocorrelation over all frames.
    let mut tg = vec![0.0f32; n_lags];
    let half = win_length / 2;

    for t in 0..n {
        // Centered frame: [t - half, t + half)
        let start = t.saturating_sub(half);
        let end = (t + half).min(n);
        let len = end - start;
        if len == 0 {
            continue;
        }

        // Apply window and accumulate autocorrelation.
        for lag in 0..n_lags {
            if lag >= len {
                break;
            }
            let mut acc = 0.0f32;
            for i in 0..(len - lag) {
                let w_i = window[i.min(win_length - 1)];
                let w_j = window[(i + lag).min(win_length - 1)];
                acc += w_i * w_j * env[start + i] * env[start + i + lag];
            }
            tg[lag] += acc;
        }
    }

    // Normalize: l-infinity (divide by max).
    let tg_max = tg.iter().cloned().fold(0.0f32, f32::max);
    if tg_max > 0.0 {
        for v in tg.iter_mut() {
            *v /= tg_max;
        }
    }

    // --- 2. tempo_frequencies ---
    // bpms[k] = 60 * sr / (hop_length * k), k = 1..n_lags
    // bpms[0] = inf (0-lag), skipped.
    let bpms: Vec<f32> = (1..=n_lags)
        .map(|k| 60.0 * sr / (hop_length as f32 * k as f32))
        .collect();

    // --- 3. log-normal prior ---
    let log_start = (START_BPM).log2();
    let logprior: Vec<f32> = bpms
        .iter()
        .map(|&bpm| {
            if bpm > MAX_TEMPO {
                f32::NEG_INFINITY
            } else {
                let d = (bpm.log2() - log_start) / STD_BPM;
                -0.5 * d * d
            }
        })
        .collect();

    // Mask out 0-lag (index 0 in tg, which is the DC / max component).
    // The autocorrelation at lag 0 is always the maximum, so we skip it.
    // bpms[0] corresponds to lag 1 (first non-DC lag).
    // Actually: tg[0] = autocorr at lag 0 (max), tg[1] = lag 1, ...
    // bpms[0] = 60*sr/(hop*1) corresponds to lag 1 = tg[1].
    // So we use tg[1..] and bpms[0..].

    // Build the score array: log1p(1e6 * tg[k]) + logprior[k]
    // for k = 1..n_lags (skipping lag 0).
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for k in 0..n_lags.saturating_sub(1) {
        let tg_val = tg[k + 1]; // skip lag 0
        let score = (1e6 * tg_val + 1.0).ln() + logprior[k];
        if score > best_score {
            best_score = score;
            best_idx = k;
        }
    }

    bpms[best_idx]
}

/// Periodic Hann window: `w[n] = 0.5 - 0.5 * cos(2*pi*n / Nx)`.
fn hann(n: usize) -> Vec<f32> {
    let two_pi_over_n = std::f32::consts::TAU / n as f32;
    (0..n)
        .map(|k| {
            let x = two_pi_over_n * k as f32;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}
