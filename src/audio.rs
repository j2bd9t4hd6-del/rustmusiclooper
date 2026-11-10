//! Audio loading via symphonia, mirroring PyMusicLooper's `MLAudio`:
//!
//!   raw_audio, sr = librosa.load(path, sr=None, mono=False)
//!   mono = to_mono(raw_audio)                 # channel mean
//!   mono /= max(|mono|)                        # normalize to [-1, 1]
//!   audio, trim_offset = librosa.effects.trim(mono, top_db=40)
//!
//! We keep both:
//!   * `mono`      - the normalized mono signal used for analysis (pre-trim).
//!   * `playback`  - the *original* multi-channel signal (unnormalized) used
//!                   for zero-crossing detection, laid out `[samples][channels]`.

use anyhow::{anyhow, Result};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use std::io::Cursor;

#[derive(Debug, Clone)]
pub struct AudioData {
    /// Normalized mono signal used for analysis (length `mono_samples`).
    pub mono: Vec<f32>,
    pub rate: usize,
    pub n_channels: usize,
    /// Total duration in seconds of the original signal.
    pub total_duration: f64,
    /// Original multi-channel signal `[samples][channels]`, unnormalized.
    pub playback: Vec<f32>,
    /// `trim_offset` in samples (leading trim of the mono signal).
    pub trim_offset: usize,
    /// Trimmed mono signal used for STFT/chroma analysis.
    pub audio: Vec<f32>,
    /// Total number of samples of the original signal (`playback` rows).
    pub total_samples: usize,
}

/// Decode an audio file (m4a/mp4/aac, wav, etc.) via symphonia.
pub fn load(path: &str) -> Result<AudioData> {
    let bytes = std::fs::read(path).map_err(|e| anyhow!("cannot read {}: {}", path, e))?;
    decode(&bytes)
}

/// Decode audio from an in-memory byte buffer (for iOS/Flutter).
pub fn load_from_bytes(bytes: &[u8]) -> Result<AudioData> {
    decode(bytes)
}

fn decode(bytes: &[u8]) -> Result<AudioData> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let format: Box<dyn FormatReader> = symphonia::default::get_probe()
        .probe(&Hint::default(), mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| anyhow!("probe failed: {}", e))?;

    let mut format = format;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("no audio track"))?;
    let cp = track
        .codec_params
        .as_ref()
        .ok_or_else(|| anyhow!("track has no codec"))?;
    let audio_cp = match cp {
        symphonia::core::codecs::CodecParameters::Audio(a) => a,
        _ => return Err(anyhow!("non-audio default track")),
    };
    let rate = audio_cp.sample_rate.unwrap_or(48000) as usize;
    let n_channels = audio_cp
        .channels
        .as_ref()
        .map_or(1, |c| c.count() as usize)
        .max(1);
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_cp, &AudioDecoderOptions::default())
        .map_err(|e| anyhow!("no decoder for codec {:?}: {}", audio_cp.codec, e))?;

    // Read all packets into the decoder, collect interleaved samples.
    let mut raw: Vec<f32> = Vec::new();
    loop {
        match format.next_packet() {
            Ok(Some(packet)) => {
                if packet.track_id != track_id {
                    continue;
                }
                match decoder.decode(&packet) {
                    Ok(buf) => {
                        // copy_to_vec_interleaved resizes the Vec to the buffer size,
                        // so copy into a temporary buffer and append instead.
                        let nf = buf.frames();
                        let mut tmp = vec![0.0f32; nf * n_channels];
                        buf.copy_to_vec_interleaved::<f32>(&mut tmp);
                        raw.extend_from_slice(&tmp);
                    }
                    Err(e) => {
                        eprintln!("decode warning: {}", e);
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("read warning: {}", e);
                break;
            }
        }
    }

    let total_samples = raw.len() / n_channels;
    if total_samples == 0 {
        return Err(anyhow!("no audio samples decoded"));
    }

    // Channel mean -> mono.
    let mut mono = vec![0.0f32; total_samples];
    for s in 0..total_samples {
        let mut acc = 0.0f32;
        for ch in 0..n_channels {
            acc += raw[s * n_channels + ch];
        }
        mono[s] = acc / n_channels as f32;
    }

    // Normalize to [-1, 1] on the mono signal.
    let max_abs = mono.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if max_abs > 0.0 {
        for v in mono.iter_mut() {
            *v /= max_abs;
        }
    }

    // Playback: original multi-channel, [samples][channels], unnormalized.
    let playback = raw;

    // Trim leading/trailing silence: 1:1 port of librosa.effects.trim(mono, top_db=40).
    //   rms = feature.rms(y, frame_length=2048, hop_length=512, center=True)
    //   db  = amplitude_to_db(rms, ref=np.max, top_db=None)   // 20*log10(rms/max(rms))
    //   non_silent = db > -top_db
    //   start = frames_to_samples(first_nonzero, hop=512) = first * 512
    //   end   = min(n, frames_to_samples(last_nonzero + 1, hop=512))
    const FRAME_LEN: usize = 2048;
    const HOP: usize = 512;
    const TOP_DB: f32 = 40.0;
    const AMIN: f32 = 1e-5; // librosa amplitude_to_db default amin

    // 1) center=True: zero-pad frame_length//2 = 1024 on each side (librosa rms padding).
    let pad = FRAME_LEN / 2;
    let mut padded = vec![0.0f32; total_samples + 2 * pad];
    padded[pad..pad + total_samples].copy_from_slice(&mono);
    let padded_len = padded.len();

    // 2) util.frame(frame_length=2048, hop_length=512): n_frames = (L - 2048)/512 + 1.
    //    Every frame is exactly 2048 samples (zero-padded at the edges).
    let n_frames = (padded_len - FRAME_LEN) / HOP + 1;
    let mut rms = vec![0.0f32; n_frames];
    for t in 0..n_frames {
        let start = t * HOP;
        let mut acc = 0.0f64;
        for i in 0..FRAME_LEN {
            let v = padded[start + i] as f64;
            acc += v * v;
        }
        rms[t] = (acc / FRAME_LEN as f64).sqrt() as f32; // mean(|x|^2) -> sqrt
    }

    // 3) ref = max(rms) (amplitude_to_db ref=np.max).
    let ref_val = rms.iter().cloned().fold(0.0f32, f32::max);

    // 4) non_silent[t] = 20*log10(max(amin, rms)/max(amin, ref)) > -top_db.
    let mut non_silent = vec![false; n_frames];
    for t in 0..n_frames {
        let db = if ref_val > 0.0 {
            20.0f32 * (rms[t].max(AMIN) / ref_val.max(AMIN)).log10()
        } else {
            f32::NEG_INFINITY
        };
        non_silent[t] = db > -TOP_DB;
    }

    // 5) start/end via frames_to_samples(frame, hop=512) = frame * 512.
    let nonzero: Vec<usize> = (0..n_frames).filter(|&t| non_silent[t]).collect();
    let (start, end) = if nonzero.is_empty() {
        (0, 0)
    } else {
        (
            nonzero[0] * HOP,
            ((nonzero[nonzero.len() - 1] + 1) * HOP).min(total_samples),
        )
    };
    let trim_offset = start;
    let audio = mono[start..end].to_vec();

    let total_duration = total_samples as f64 / rate as f64;

    Ok(AudioData {
        mono,
        rate,
        n_channels,
        total_duration,
        playback,
        trim_offset,
        audio,
        total_samples,
    })
}
