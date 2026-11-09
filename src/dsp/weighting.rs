//! `perceptual_weighting(S_power, frequencies, kind='A', ref=np.median)`.
//!
//! librosa:
//!     offset = frequency_weighting(frequencies, kind='A').reshape((-1, 1))
//!     result = offset + power_to_db(S, ref=np.median)
//!
//! `frequency_weighting` with kind 'A' is `A_weighting(frequencies, min_db=-80.0)`:
//!     f_sq = f**2
//!     const = [12194.217, 20.598997, 107.65265, 737.86223]**2
//!     w = 2 + 20*( log10(const0) + 2*log10(f_sq)
//!                  - log10(f_sq+const0) - log10(f_sq+const1)
//!                  - 0.5*log10(f_sq+const2) - 0.5*log10(f_sq+const3) )
//!     w = max(min_db, w)   # min_db = -80
//!
//! Note librosa evaluates in float64; we replicate with f64 for the offset,
//! then cast to f32 when adding to the (f32) power_to_db result.
//! The DC bin (f=0) makes log10(f_sq) -> -inf; the offset saturates to min_db.

use crate::dsp::power_db::power_to_db_ref_median;

/// FFT bin center frequencies: `np.fft.rfftfreq(n_fft, d=1/sr)`.
pub fn fft_frequencies(sr: f64, n_fft: usize) -> Vec<f64> {
    (0..(n_fft / 2 + 1))
        .map(|f| (f as f64) * (sr / n_fft as f64))
        .collect()
}

/// A-weighting per IEC 61672, matching librosa `A_weighting(min_db=-80.0)`.
pub fn a_weighting(freqs: &[f64]) -> Vec<f64> {
    const C0: f64 = 12194.217;
    const C1: f64 = 20.598997;
    const C2: f64 = 107.65265;
    const C3: f64 = 737.86223;
    const MIN_DB: f64 = -80.0;

    let c0 = C0 * C0;
    let c1 = C1 * C1;
    let c2 = C2 * C2;
    let c3 = C3 * C3;
    let log_c0 = c0.log10();

    freqs
        .iter()
        .map(|&f| {
            let f_sq = f * f;
            let mut w = if f_sq > 0.0 {
                2.0
                    + 20.0
                        * (log_c0
                            + 2.0 * f_sq.log10()
                            - (f_sq + c0).log10()
                            - (f_sq + c1).log10()
                            - 0.5 * (f_sq + c2).log10()
                            - 0.5 * (f_sq + c3).log10())
            } else {
                f64::NEG_INFINITY
            };
            if w.is_finite() {
                w = w.max(MIN_DB);
            } else {
                w = MIN_DB;
            }
            w
        })
        .collect()
}

/// Apply perceptual A-weighting + power_to_db(ref=median) in place.
/// `power` must be `[n_fft/2+1][n_frames]` row-major.
pub fn perceptual_weighting_a(power: &mut [f32], n_bins: usize, freqs: &[f64]) {
    let offset = a_weighting(freqs);
    // First compute power_to_db with the unweighted matrix's median.
    power_to_db_ref_median(power);
    // Add the per-frequency offset.
    for t in 0..power.len() / n_bins {
        for f in 0..n_bins {
            power[t * n_bins + f] += offset[f] as f32;
        }
    }
}
