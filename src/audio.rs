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
use std::fs::File;

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
    let file = File::open(path).map_err(|e| anyhow!("cannot open {}: {}", path, e))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let format: Box<dyn FormatReader> = symphonia::default::get_probe()
        .probe(&Hint::default(), mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| anyhow!("probe failed for {}: {}", path, e))?;

    let mut format = format;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("no audio track in {}", path))?;
    let cp = track
        .codec_params
        .as_ref()
        .ok_or_else(|| anyhow!("track has no codec in {}", path))?;
    let audio_cp = match cp {
        symphonia::core::codecs::CodecParameters::Audio(a) => a,
        _ => return Err(anyhow!("non-audio default track in {}", path)),
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
        return Err(anyhow!("no audio samples decoded from {}", path));
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

    // Trim leading/trailing silence below -40 dB (librosa.effects.trim top_db=40).
    let threshold = 10f32.powf(-40.0 / 20.0);
    let mut start = 0usize;
    while start < total_samples && mono[start].abs() < threshold {
        start += 1;
    }
    let mut end = total_samples;
    while end > start && mono[end - 1].abs() < threshold {
        end -= 1;
    }
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
