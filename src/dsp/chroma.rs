//! Chroma matching `librosa.feature.chroma_stft(S=S_power, sr, n_chroma=12,
//! norm=np.inf)` with the *simplified* tuning = 0.0 (A440).
//!
//! `librosa.filters.chroma(sr, n_fft, tuning, n_chroma=12, ctroct=5.0,
//! octwidth=2, norm=2, base_c=True)`:
//!   frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]
//!   frqbins = n_chroma * hz_to_octs(frequencies, tuning, bins_per_octave=n_chroma)
//!   frqbins = concat([frqbins[0] - 1.5*n_chroma, frqbins])
//!   binwidthbins = concat(max(frqbins[1:]-frqbins[:-1], 1.0), [1])
//!   D = (frqbins - arange(n_chroma)).T ; D = (D + n_chroma/2 + 10*n_chroma) % n_chroma - n_chroma/2
//!   wts = exp(-0.5 * (2*D / binwidthbins)**2)
//!   wts = normalize(wts, norm=2, axis=0)            # l2 per column
//!   wts *= tile(exp(-0.5*(((frqbins/n_chroma - ctroct)/octwidth)**2)))
//!   wts = roll(wts, -3*(n_chroma//12), axis=0)      # base_c
//!   return wts[:, :n_fft/2+1]
//!
//! Then `raw_chroma = einsum('cf,ft->ct', chromafb, S)` and the result is
//! l-infinity normalized per frame (column): `raw[:, t] /= max(|raw[:, t]|)`.

/// hz -> octaves, `log2(f / (A440/16))` with `A440 = 440 * 2**(tuning/bins_per_octave)`.
fn hz_to_octs(f: f32, tuning: f32, bins_per_octave: f32) -> f32 {
    let a440 = 440.0 * 2.0_f32.powf(tuning / bins_per_octave);
    (f / (a440 / 16.0)).log2()
}

/// Build the chroma filterbank `[n_chroma][n_fft/2+1]` (row-major).
pub fn chroma_filterbank(sr: f64, n_fft: usize, tuning: f32, n_chroma: usize) -> Vec<f32> {
    let n_fft = n_fft;
    // frequencies = linspace(0, sr, n_fft, endpoint=False)[1:]
    let step = sr / n_fft as f64;
    let freqs: Vec<f32> = (1..n_fft).map(|k| (k as f64 * step) as f32).collect();

    let bpo = n_chroma as f32;
    let mut frqbins: Vec<f32> = freqs.iter().map(|&f| bpo * hz_to_octs(f, tuning, bpo)).collect();
    // prepend frqbins[0] - 1.5*n_chroma
    let first = frqbins[0] - 1.5 * bpo;
    frqbins.insert(0, first);
    // length is now n_fft (n_fft-1 freqs + 1 prepended)

    // binwidthbins = concat(max(frqbins[1:]-frqbins[:-1],1.0), [1])
    let mut binwidth: Vec<f32> = (1..frqbins.len())
        .map(|k| (frqbins[k] - frqbins[k - 1]).max(1.0))
        .collect();
    binwidth.push(1.0);

    let half = (n_chroma as f32 / 2.0).round();
    let nc = n_chroma as f32;

    let mut wts = vec![0.0f32; n_chroma * n_fft];
    for j in 0..n_fft {
        let bw = binwidth[j];
        for c in 0..n_chroma {
            let d = frqbins[j] - c as f32;
            let d2 = ((d + half + 10.0 * nc).rem_euclid(nc)) - half;
            wts[c * n_fft + j] = (-0.5 * (2.0 * d2 / bw).powi(2)).exp();
        }
    }

    // l2 normalize each column
    for j in 0..n_fft {
        let mut ss = 0.0f32;
        for c in 0..n_chroma {
            ss += wts[c * n_fft + j] * wts[c * n_fft + j];
        }
        let norm = ss.sqrt();
        if norm > 0.0 {
            for c in 0..n_chroma {
                wts[c * n_fft + j] /= norm;
            }
        }
    }

    // apply octwidth (ctroct=5.0, octwidth=2)
    const CTROCT: f32 = 5.0;
    const OCTWIDTH: f32 = 2.0;
    for j in 0..n_fft {
        let octs = frqbins[j] / bpo;
        let g = (-0.5 * (((octs - CTROCT) / OCTWIDTH).powi(2))).exp();
        for c in 0..n_chroma {
            wts[c * n_fft + j] *= g;
        }
    }

    // base_c: roll -3*(n_chroma//12) along chroma axis
    let shift = 3 * (n_chroma / 12);
    let mut rolled = vec![0.0f32; n_chroma * n_fft];
    for j in 0..n_fft {
        for c in 0..n_chroma {
            let src = (c + shift) % n_chroma;
            rolled[c * n_fft + j] = wts[src * n_fft + j];
        }
    }

    // keep only the first n_fft/2+1 bins (drop aliasing columns)
    let keep = n_fft / 2 + 1;
    let mut out = vec![0.0f32; n_chroma * keep];
    for c in 0..n_chroma {
        for j in 0..keep {
            out[c * keep + j] = rolled[c * n_fft + j];
        }
    }
    out
}

/// Compute chroma `[n_chroma][n_frames]` from a power matrix
/// `[n_fft/2+1][n_frames]`, l-infinity normalized per frame.
pub fn chroma_stft(
    power: &[f32],
    n_bins: usize,
    n_frames: usize,
    fb: &[f32],
    n_chroma: usize,
) -> Vec<f32> {
    let mut raw = vec![0.0f32; n_chroma * n_frames];
    for t in 0..n_frames {
        for c in 0..n_chroma {
            let mut acc = 0.0f32;
            for f in 0..n_bins {
                acc += fb[c * n_bins + f] * power[t * n_bins + f];
            }
            raw[c * n_frames + t] = acc;
        }
    }
    // l-inf normalize per frame (column)
    for t in 0..n_frames {
        let mut mx = 0.0f32;
        for c in 0..n_chroma {
            mx = mx.max(raw[c * n_frames + t].abs());
        }
        if mx > f32::EPSILON {
            for c in 0..n_chroma {
                raw[c * n_frames + t] /= mx;
            }
        }
    }
    raw
}
