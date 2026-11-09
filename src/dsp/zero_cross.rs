//! Port of PyMusicLooper's `nearest_zero_crossing` (Audacity's "At Zero
//! Crossings" behaviour).
//!
//! `audio` is `[samples][n_channels]` (row-major), `rate` the sample rate,
//! `sample_idx` the target sample. Returns the best rising zero crossing near
//! `sample_idx`, or `sample_idx` unchanged if nothing suitable is found.

pub fn nearest_zero_crossing(audio: &[f32], n_channels: usize, rate: usize, sample_idx: usize) -> usize {
    let n_samples = audio.len() / n_channels;
    if n_channels == 0 || n_samples == 0 {
        return sample_idx.min(n_samples.saturating_sub(1));
    }

    let window_size = (rate / 100).max(1);
    let offset = window_size / 2;

    let neg_offset = sample_idx.saturating_sub(offset);
    let pos_offset = (sample_idx + offset).min(n_samples);
    if neg_offset >= pos_offset {
        return sample_idx.min(n_samples.saturating_sub(1));
    }
    let win_len = pos_offset - neg_offset;
    let offset_correction = if sample_idx < offset {
        offset - sample_idx
    } else {
        0
    };

    let mut dist = vec![0.0f32; win_len];

    for ch in 0..n_channels {
        let mut one_dist = vec![0.0f32; win_len];
        for i in 0..win_len {
            one_dist[i] = audio[(neg_offset + i) * n_channels + ch];
        }
        let mut prev = 2.0f32;
        for i in 0..win_len {
            let mut fdist = one_dist[i].abs();
            if prev * one_dist[i] > 0.0 {
                fdist += 0.4;
            } else if prev > 0.0 {
                fdist += 0.1;
            }
            prev = one_dist[i];
            one_dist[i] = fdist;
        }
        for i in 0..win_len {
            dist[i] += one_dist[i];
            dist[i] += 0.1 * ((i as i32 - offset as i32 + offset_correction as i32).unsigned_abs() as f32) / (window_size / 2) as f32;
        }
    }

    let mut argmin = 0usize;
    for i in 1..win_len {
        if dist[i] < dist[argmin] {
            argmin = i;
        }
    }
    let minimum_dist = dist[argmin];

    if n_channels == 1 && minimum_dist > 0.2 * n_channels as f32 {
        return sample_idx.min(n_samples.saturating_sub(1));
    }
    if n_channels > 1 && minimum_dist > 0.6 * n_channels as f32 {
        return sample_idx.min(n_samples.saturating_sub(1));
    }

    let idx = (sample_idx as i64 + argmin as i64 - offset as i64 + offset_correction as i64).max(0);
    (idx as usize).min(n_samples.saturating_sub(1))
}
