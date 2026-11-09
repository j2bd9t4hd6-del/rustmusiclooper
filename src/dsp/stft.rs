//! STFT matching `librosa.stft(y, n_fft=2048, hop_length=512, win_length=n_fft,
//! window='hann', center=True, pad_mode='constant')`.
//!
//! Returns the complex spectrum in column-major-ish layout: `S[f][t]` is the
//! FFT coefficient for frequency bin `f` at frame `t`. Only the real-input
//! `1 + n_fft/2` bins are kept (equivalent to `np.fft.rfft` / `scipy.fft.rfft`).
//!
//! Frame count for center=True with zero padding is:
//!     n_frames = (n + n_fft) // hop_length + 1
//! where `n` is the (unpadded) signal length, matching librosa's
//! `start_k = ceil(n_fft // 2 / hop_length)` head + `tail_k` tail logic.

use rustfft::FftPlanner;

/// Result of an STFT: `spectrum[f][t]`, complex64, `n_fft/2+1` frequency bins.
pub struct StftOutput {
    pub n_fft: usize,
    pub hop_length: usize,
    /// `[n_fft/2 + 1][n_frames]`
    pub data: Vec<rustfft::num_complex::Complex<f32>>,
    pub n_frames: usize,
}

pub fn stft_centered(y: &[f32], n_fft: usize, hop_length: usize, window: &[f32]) -> StftOutput {
    debug_assert_eq!(window.len(), n_fft);
    let n = y.len();
    let n_bins = n_fft / 2 + 1;
    let n_frames = (n + n_fft) / hop_length + 1;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut spectrum = vec![rustfft::num_complex::Complex::new(0.0f32, 0.0f32); n_bins * n_frames];

    for (t, frame_start) in (0..n_frames).map(|t| t * hop_length).enumerate() {
        // Gather the frame with zero padding, then apply the window.
        let mut buf: Vec<rustfft::num_complex::Complex<f32>> = vec![
            rustfft::num_complex::Complex::new(0.0f32, 0.0f32);
            n_fft
        ];
        for i in 0..n_fft {
            let idx = frame_start + i;
            let sample = if idx < n { y[idx] } else { 0.0 };
            buf[i] = rustfft::num_complex::Complex::new(sample * window[i], 0.0);
        }
        fft.process(&mut buf);

        let base = t * n_bins;
        for f in 0..n_bins {
            spectrum[base + f] = buf[f];
        }
    }

    StftOutput {
        n_fft,
        hop_length,
        data: spectrum,
        n_frames,
    }
}

/// Real-valued power spectrogram `|S|^2` (magnitude squared), matching
/// `S_power = np.abs(S) ** 2`. Layout `[n_fft/2+1][n_frames]` as `Vec<f32>`.
pub fn power_from_stft(stft: &StftOutput) -> Vec<f32> {
    let n_bins = stft.n_fft / 2 + 1;
    let mut power = vec![0.0f32; n_bins * stft.n_frames];
    for (i, c) in stft.data.iter().enumerate() {
        power[i] = c.re * c.re + c.im * c.im;
    }
    power
}
