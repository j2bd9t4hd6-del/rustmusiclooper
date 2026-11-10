//! `power_to_db(S, ref=np.median, amin=1e-10, top_db=None)`.
//!
//! librosa computes:
//!     log_spec = 10 * log10(max(amin, S)) - 10 * log10(max(amin, ref))
//! with `ref = np.median(S)` (over the whole matrix) when a callable is given.
//!
//! We mirror this element-wise in f32.

const AMIN: f32 = 1e-10;

fn median(values: &[f32]) -> f32 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    }
}

/// In-place convert a power matrix to dB relative to its median.
pub fn power_to_db_ref_median(db: &mut [f32]) {
    let ref_value = median(db);
    let log_ref = 10.0 * (ref_value.max(AMIN)).log10();
    for v in db.iter_mut() {
        *v = 10.0 * (*v).max(AMIN).log10() - log_ref;
    }
}

/// In-place convert a power matrix to dB with `ref=1.0` (peak reference).
///
/// Mirrors librosa `power_to_db(S)` with no `ref` argument (default `ref=1.0`):
///     10 * log10(max(amin, S)) - 10 * log10(max(amin, 1.0))
/// `max(amin, 1.0) == 1.0`, so `log_ref` is the peak power (>= 1.0 after clamping).
pub fn power_to_db_ref_peak(db: &mut [f32]) {
    let ref_value = db.iter().cloned().fold(0.0f32, f32::max).max(1.0);
    let log_ref = 10.0 * ref_value.max(AMIN).log10();
    for v in db.iter_mut() {
        *v = 10.0 * (*v).max(AMIN).log10() - log_ref;
    }
}
