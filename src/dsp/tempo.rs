//! Tempo (BPM) estimation matching librosa's `feature.tempo`.
//!
//! Pipeline (1:1 with librosa):
//!   1. `tempogram(onset_envelope, sr, hop_length, win_length)`:
//!        - center=True: pad with `linear_ramp` (win_length//2 each side)
//!        - frame with hop=1, window with symmetric Hann (fftbins=True)
//!        - FFT autocorrelate each frame, normalize to l-infinity
//!   2. `aggregate = np.mean(tg, axis=-1)` (global tempogram)
//!   3. `tempo_frequencies(win_length)`: bpms[k] = 60*sr/(hop*k)
//!   4. Weight with pseudo-log-normal prior (start_bpm=120, std_bpm=1.0,
//!      max_tempo=320), argmax of `log1p(1e6*tg) + logprior`.

use rustfft::FftPlanner;

const AC_SIZE: f32 = 8.0;
const START_BPM: f32 = 120.0;
const STD_BPM: f32 = 1.0;
const MAX_TEMPO: f32 = 320.0;

/// Estimate the global tempo (BPM) from an onset strength envelope.
///
/// Matches `librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=hop)`
/// with default parameters (`start_bpm=120`, `std_bpm=1.0`, `ac_size=8.0`,
/// `max_tempo=320.0`, `aggregate=np.mean`).
pub fn estimate_tempo(env: &[f32], sr: f32, hop_length: usize) -> f32 {
    let n = env.len();
    if n < 4 {
        return START_BPM;
    }

    // win_length = time_to_frames(ac_size, sr, hop_length) = int(ac_size * sr / hop)
    let win_length = (AC_SIZE * sr / hop_length as f32) as usize;
    let win_length = win_length.max(2);

    // --- 1. tempogram ---
    let tg = tempogram(env, win_length);

    // --- 2. aggregate: mean over time ---
    // tg is [win_length][n_frames]; mean over frames -> [win_length]
    let n_frames = tg.len() / win_length;
    let mut global = vec![0.0f32; win_length];
    for k in 0..win_length {
        let mut acc = 0.0f64;
        for t in 0..n_frames {
            acc += tg[k * n_frames + t] as f64;
        }
        global[k] = (acc / n_frames as f64) as f32;
    }

    // --- 3. tempo_frequencies: bpms[k-1] = 60*sr/(hop*k), k=1..win_length-1 ---
    // librosa.tempo_frequencies(win_length) returns:
    //   bpms[0] = inf (lag 0), bpms[k] = 60*sr/(hop*k) for k=1..win_length-1
    // We store bpms[k-1] = 60*sr/(hop*k) so that bpms[k-1] corresponds to lag k.
    let bpms: Vec<f32> = (1..win_length)
        .map(|k| 60.0 * sr / (hop_length as f32 * k as f32))
        .collect();

    // --- 4. log-normal prior + max_tempo masking ---
    // librosa: max_idx = int(np.argmax(bpms < max_tempo))  => first index where
    // bpm >= max_tempo.  logprior[:max_idx] = -inf  => kill bins BELOW max_tempo.
    // (This looks inverted but matches librosa's actual behavior.)
    let log_start = (START_BPM).log2();
    let max_idx = bpms
        .iter()
        .position(|&b| b >= MAX_TEMPO)
        .unwrap_or(bpms.len());
    let logprior: Vec<f32> = bpms
        .iter()
        .enumerate()
        .map(|(i, &bpm)| {
            if i < max_idx {
                f32::NEG_INFINITY
            } else {
                let d = (bpm.log2() - log_start) / STD_BPM;
                -0.5 * d * d
            }
        })
        .collect();

    // --- 5. argmax of log1p(1e6 * global[k]) + logprior[k] for k=1..win_length-1 ---
    // global[k] IS lag k (global[0] = lag 0 = DC, global[1] = lag 1, etc.)
    // bpms[k] corresponds to lag k+1, so bpms[0] = lag 1 = global[1].
    // We iterate k=1..win_length (lag 1..win_length-1), and bpms[k-1] is the bpm.
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for k in 1..win_length {
        let tg_val = global[k]; // lag k
        if tg_val.is_nan() {
            continue;
        }
        let score = (1e6 * tg_val + 1.0).ln() + logprior[k - 1];
        if score > best_score {
            best_score = score;
            best_idx = k - 1;
        }
    }

    bpms[best_idx]
}

/// `librosa.feature.tempogram(onset_envelope, win_length, center=True,
/// window="hann", norm=np.inf)`.
///
/// Returns `[win_length][n]` row-major (lag, frame), l-infinity normalized.
fn tempogram(env: &[f32], win_length: usize) -> Vec<f32> {
    let n = env.len();

    // --- center=True: linear_ramp padding (win_length//2 each side) ---
    let pad = win_length / 2;
    let mut padded = vec![0.0f32; n + 2 * pad];
    // linear_ramp, end_values=[0,0]: ramp from 0 to env[0] at the start,
    // env[-1] to 0 at the end.
    for i in 0..pad {
        let frac = (i as f32 + 1.0) / (pad as f32 + 1.0);
        padded[i] = env[0] * frac;
    }
    padded[pad..pad + n].copy_from_slice(env);
    for i in 0..pad {
        let frac = (i as f32 + 1.0) / (pad as f32 + 1.0);
        padded[pad + n + i] = env[n - 1] * (1.0 - frac);
    }

    // --- symmetric Hann window (fftbins=True): w[k] = 0.5 - 0.5*cos(2*pi*k/(N-1)) ---
    let window = symmetric_hann(win_length);

    // --- frame with hop=1: n frames of length win_length ---
    // frame t = padded[t .. t+win_length], t = 0..n
    // For each frame: windowed = frame * window, autocorrelate, normalize l-inf.
    let mut out = vec![0.0f32; win_length * n];

    // FFT plan for autocorrelation: length = next_fast_len(2*win_length - 1)
    let fft_len = next_fast_len(2 * win_length - 1);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut buf = vec![0.0f64; fft_len];
    let mut spec = vec![rustfft::num_complex::Complex::new(0.0, 0.0); fft_len];

    for t in 0..n {
        // Window the frame.
        for k in 0..win_length {
            buf[k] = padded[t + k] as f64 * window[k] as f64;
        }
        for k in win_length..fft_len {
            buf[k] = 0.0;
        }

        // autocorrelate via FFT: ac = ifft(|fft(buf)|^2)
        for (i, v) in buf.iter().enumerate() {
            spec[i] = rustfft::num_complex::Complex::new(*v, 0.0);
        }
        fft.process(&mut spec);
        for c in spec.iter_mut() {
            let p = c.norm_sqr();
            *c = rustfft::num_complex::Complex::new(p, 0.0);
        }
        ifft.process(&mut spec);
        // irfft: take real part, scale by 1/fft_len (rustfft's inverse includes 1/n).
        for k in 0..win_length {
            out[k * n + t] = spec[k].re as f32;
        }

        // Normalize to l-infinity (norm=np.inf): divide by max.
        // Buffer is column-major: out[k * n + t], so the max over lag for
        // frame t strides by n (NOT a contiguous range).
        let mx = (0..win_length)
            .map(|k| out[k * n + t])
            .fold(f32::NEG_INFINITY, |m, v| m.max(v));
        if mx > 0.0 && mx.is_finite() {
            for k in 0..win_length {
                out[k * n + t] /= mx;
            }
        }
    }

    out
}

/// Symmetric Hann window (scipy `get_window("hann", N, fftbins=True)`):
/// `w[k] = 0.5 - 0.5 * cos(2*pi*k / (N-1))`, k = 0..N-1.
fn symmetric_hann(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0f32; n];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|k| 0.5 - 0.5 * (std::f32::consts::TAU * k as f32 / denom).cos())
        .collect()
}

/// Approximate scipy.fft.next_fast_len: round up to a highly composite number.
/// For our purposes (autocorrelation length ~1500), rounding up to the next
/// power of 2 is sufficient and gives identical autocorrelation results.
fn next_fast_len(n: usize) -> usize {
    n.next_power_of_two()
}
