//! Mel filterbank matching `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax,
//! htk=False, norm='slaney')` and `librosa.feature.melspectrogram(S, sr, ...)`.
//!
//! librosa uses the Slaney (Auditory Toolbox) mel scale by default (htk=False):
//!     linear  below 1 kHz:  mel = f / (200/3)
//!     log     above 1 kHz:  mel = min_log_mel + log(f/1000)/logstep
//! with `mel_frequencies` doing a linear spacing in the *mel* domain between
//! `hz_to_mel(fmin)` and `hz_to_mel(fmax)`, then `mel_to_hz` back to Hz.

use crate::dsp::weighting::fft_frequencies;

const F_SP: f32 = 200.0 / 3.0;
const MIN_LOG_HZ: f32 = 1000.0;
const MIN_LOG_MEL: f32 = 1000.0 / (200.0 / 3.0);
// ln(6.4)/27
const LOGSTEP_V: f32 = 0.08443733930837494;

/// Slaney hz->mel (htk=False).
fn hz_to_mel(f: f32) -> f32 {
    if f >= MIN_LOG_HZ {
        MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP_V
    } else {
        f / F_SP
    }
}

/// Slaney mel->hz (htk=False).
fn mel_to_hz(m: f32) -> f32 {
    if m >= MIN_LOG_MEL {
        MIN_LOG_HZ * (LOGSTEP_V * (m - MIN_LOG_MEL)).exp()
    } else {
        F_SP * m
    }
}

/// `librosa.mel_frequencies(n_mels, fmin, fmax, htk=False)`: linearly spaced
/// points in the mel domain between the two endpoint mels, mapped back to Hz.
fn mel_frequencies(n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    if n_mels == 1 {
        return vec![mel_to_hz(mel_min)];
    }
    let step = (mel_max - mel_min) / (n_mels as f32 - 1.0);
    (0..n_mels)
        .map(|i| mel_to_hz(mel_min + step * i as f32))
        .collect()
}

/// Build the mel filterbank matrix `[n_mels][n_fft/2+1]` (row-major), matching
/// `librosa.filters.mel` with `norm='slaney'`.
pub fn mel_filterbank(sr: f64, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let fftfreqs: Vec<f32> = fft_frequencies(sr, n_fft).into_iter().map(|f| f as f32).collect();

    // mel_f of length n_mels + 2 (extra endpoints for the ramps).
    let mel_f = mel_frequencies(n_mels + 2, fmin, fmax);
    // fdiff = np.diff(mel_f), length n_mels+1; fdiff[k] = mel_f[k+1]-mel_f[k]
    let fdiff: Vec<f32> = (0..(n_mels + 1))
        .map(|k| mel_f[k + 1] - mel_f[k])
        .collect();

    let mut weights = vec![0.0f32; n_mels * n_bins];
    for i in 0..n_mels {
        // librosa: lower = -ramps[i]/fdiff[i] = -(mel_f[i]-fr)/fdiff[i]
        //           upper = ramps[i+2]/fdiff[i+1] = (mel_f[i+2]-fr)/fdiff[i+1]
        // fdiff[i+1] = mel_f[i+2]-mel_f[i+1]
        for (j, &fr) in fftfreqs.iter().enumerate() {
            let lower = -(mel_f[i] - fr) / fdiff[i];
            let upper = (mel_f[i + 2] - fr) / fdiff[i + 1];
            weights[i * n_bins + j] = lower.min(upper).max(0.0);
        }
    }

    // Slaney area normalization.
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for j in 0..n_bins {
            weights[i * n_bins + j] *= enorm;
        }
    }

    weights
}

/// Map a power matrix `[n_fft/2+1][n_frames]` through the mel filterbank,
/// producing `[n_mels][n_frames]` (row-major). Mirrors
/// `np.einsum('...ft,mf->...mt', S, mel_basis)`.
pub fn melspectrogram(
    power: &[f32],
    n_bins: usize,
    n_frames: usize,
    melfb: &[f32],
    n_mels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_mels * n_frames];
    for m in 0..n_mels {
        for t in 0..n_frames {
            let mut acc = 0.0f32;
            for f in 0..n_bins {
                acc += melfb[m * n_bins + f] * power[t * n_bins + f];
            }
            out[m * n_frames + t] = acc;
        }
    }
    out
}
