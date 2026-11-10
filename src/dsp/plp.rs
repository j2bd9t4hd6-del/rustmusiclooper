//! Predominant Local Pulse (PLP) estimation, matching librosa's `beat.plp`.
//!
//! Pipeline (Grosche & Muller 2011):
//!   1. Compute the Fourier tempogram (STFT of the onset envelope with
//!      `n_fft=win_length`, `hop_length=1`, Hann window, center=True).
//!   2. Mask out tempo bins below `tempo_min` and above `tempo_max`.
//!   3. Keep only the peak bin per frame (zero out everything below the max).
//!   4. Normalize to retain phase information only.
//!   5. Invert with ISTFT to get the pulse curve.
//!   6. Clip negatives and normalize.
//!
//! The local maxima of the returned pulse are additional beat candidates that
//! PyMusicLooper unions with the DP `beat_track` results.

use crate::dsp::stft::istft;
use crate::dsp::window::hann_periodic;
use rustfft::num_complex::Complex;
use rustfft::num_complex::ComplexFloat;
use rustfft::FftPlanner;

const TINY: f32 = 1e-10;

/// Default PLP parameters (librosa defaults).
pub const WIN_LENGTH: usize = 384;
pub const TEMPO_MIN: f32 = 30.0;
pub const TEMPO_MAX: f32 = 300.0;

/// Compute the predominant local pulse from an onset strength envelope.
///
/// Matches `librosa.beat.plp(onset_envelope=env, sr=sr, hop_length=hop)`.
pub fn plp(env: &[f32], sr: f32, hop_length: usize) -> Vec<f32> {
    let n = env.len();
    if n < 4 {
        return vec![0.0f32; n];
    }

    // --- 1. Fourier tempogram (STFT of onset envelope) ---
    let win = hann_periodic(WIN_LENGTH);
    let n_bins = WIN_LENGTH / 2 + 1;
    // center=True: n_frames = (n + n_fft) / hop + 1  (hop=1)
    let n_frames = (n + WIN_LENGTH) / 1 + 1;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(WIN_LENGTH);

    // Build the Fourier tempogram: `[n_bins][n_frames]`.
    let mut ftgram = vec![Complex::new(0.0f32, 0.0f32); n_bins * n_frames];
    for t in 0..n_frames {
        let frame_start = t * 1; // hop_length = 1
        let mut buf = vec![Complex::new(0.0f32, 0.0f32); WIN_LENGTH];
        for i in 0..WIN_LENGTH {
            let idx = frame_start + i;
            let sample = if idx < n { env[idx] } else { 0.0 };
            buf[i] = Complex::new(sample * win[i], 0.0);
        }
        fft.process(&mut buf);
        let base = t * n_bins;
        for f in 0..n_bins {
            ftgram[base + f] = buf[f];
        }
    }

    // --- 2. Tempo masking ---
    // tempo_frequencies = fft_frequencies(sr * 60 / hop_length, n_fft = WIN_LENGTH)
    let tempo_freqs: Vec<f32> = (0..n_bins)
        .map(|k| (k as f32) * (sr * 60.0 / hop_length as f32) / WIN_LENGTH as f32)
        .collect();

    for t in 0..n_frames {
        for f in 0..n_bins {
            let bpm = tempo_freqs[f];
            if bpm < TEMPO_MIN || bpm > TEMPO_MAX {
                ftgram[t * n_bins + f] = Complex::new(0.0, 0.0);
            }
        }
    }

    // --- 3. Peak-only filter (discard below peak) ---
    let mut ftmag = vec![0.0f32; n_bins * n_frames];
    for i in 0..ftgram.len() {
        let c = &ftgram[i];
        ftmag[i] = (1e6f32 * c.abs()).ln_1p(); // log1p(1e6 * |ftgram|)
    }

    for t in 0..n_frames {
        let mut peak = 0.0f32;
        for f in 0..n_bins {
            peak = peak.max(ftmag[t * n_bins + f]);
        }
        for f in 0..n_bins {
            if ftmag[t * n_bins + f] < peak {
                ftgram[t * n_bins + f] = Complex::new(0.0, 0.0);
            }
        }
    }

    // --- 4. Phase normalize ---
    // ftgram /= tiny + |ftgram.max(axis=freq)|
    for t in 0..n_frames {
        let mut mx = 0.0f32;
        for f in 0..n_bins {
            mx = mx.max(ftgram[t * n_bins + f].abs());
        }
        let denom = (TINY * TINY).sqrt() + mx; // tiny^0.5 + max
        for f in 0..n_bins {
            ftgram[t * n_bins + f] /= denom;
        }
    }

    // --- 5. ISTFT ---
    // librosa: istft(ftgram, hop_length=1, n_fft=win_length, length=env.len())
    let pulse_raw = istft(&ftgram, WIN_LENGTH, 1, &win, Some(n));

    // --- 6. Clip negatives + normalize ---
    let mut pulse = vec![0.0f32; n];
    for i in 0..n {
        pulse[i] = pulse_raw.get(i).copied().unwrap_or(0.0).max(0.0);
    }
    let mx = pulse.iter().cloned().fold(0.0f32, f32::max);
    if mx > TINY {
        for v in pulse.iter_mut() {
            *v /= mx;
        }
    }

    pulse
}

/// Find local maxima of a signal (librosa.util.localmax).
///
/// An element `x[i]` is a local maximum if `x[i] > x[i-1]` and
/// `x[i] >= x[i+1]`. The first element is never a local maximum.
/// Returns the indices (ascending).
pub fn localmax(x: &[f32]) -> Vec<usize> {
    let n = x.len();
    if n < 2 {
        return vec![];
    }
    let mut out = Vec::new();
    for i in 1..n {
        let left = x[i] > x[i - 1];
        let right = if i == n - 1 { true } else { x[i] >= x[i + 1] };
        if left && right {
            out.push(i);
        }
    }
    out
}
