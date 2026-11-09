/// Hann window as produced by `scipy.signal.get_window('hann', Nx, fftbins=True)`.
///
/// librosa's `filters.get_window` delegates to scipy, and with `fftbins=True`
/// it yields the *periodic* (FFT-friendly) Hann:
///     w[n] = 0.5 - 0.5 * cos(2*pi*n / Nx),   n in 0..Nx
///
/// Note the denominator is `Nx` (not `Nx-1`), matching the periodic form.
pub fn hann_periodic(n: usize) -> Vec<f32> {
    let two_pi_over_n = std::f32::consts::TAU / n as f32;
    (0..n)
        .map(|k| {
            let x = two_pi_over_n * k as f32;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}
