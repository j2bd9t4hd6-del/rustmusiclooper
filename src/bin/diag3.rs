use rustmusiclooper::audio;
use rustmusiclooper::dsp::mel::{mel_filterbank, melspectrogram};
use rustmusiclooper::dsp::onset::onset_strength;
use rustmusiclooper::dsp::stft::{power_from_stft, stft_centered};
use rustmusiclooper::dsp::weighting::{fft_frequencies, perceptual_weighting_peak};
use rustmusiclooper::dsp::window::hann_periodic;
use rustmusiclooper::dsp::beat_track::beat_track;
use rustmusiclooper::dsp::plp;
use rustmusiclooper::dsp::tempo;
use rustmusiclooper::analysis::{N_FFT, HOP_LENGTH, N_MELS};

fn main() {
    let path = r"C:\Users\atsug\GlintedAlchemyStarsSoundCollection.wav";
    let audio = audio::load(path).expect("load");
    let n_bins = N_FFT / 2 + 1;
    let rate = audio.rate as f32;
    let window = hann_periodic(N_FFT);
    let stft = stft_centered(&audio.audio, N_FFT, HOP_LENGTH, &window);
    let n_frames = stft.n_frames;
    let s_power = power_from_stft(&stft);

    let freqs = fft_frequencies(audio.rate as f64, N_FFT);
    let mut s_weighted = s_power.clone();
    perceptual_weighting_peak(&mut s_weighted, n_bins, &freqs);

    let mel_fb = mel_filterbank(audio.rate as f64, N_FFT, N_MELS, 0.0, 8000.0);
    let mel_spec = melspectrogram(&s_weighted, n_bins, n_frames, &mel_fb, N_MELS);

    let env = onset_strength(&mel_spec, N_MELS, n_frames, N_FFT, HOP_LENGTH);
    let mean = env.iter().map(|&v| v as f64).sum::<f64>() / env.len() as f64;
    let std = (env.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / env.len() as f64).sqrt();
    let max = env.iter().cloned().fold(0.0f32, f32::max);
    println!("rust n_frames {}", n_frames);
    println!("rust onset mean {:.6} std {:.6} max {:.6}", mean, std, max);

    let bpm = tempo::estimate_tempo(&env, rate, HOP_LENGTH);
    println!("rust bpm {:.4}", bpm);

    let beats_dp = beat_track(&env, rate, HOP_LENGTH, bpm);
    let pulse = plp::plp(&env, rate, HOP_LENGTH);
    let beats_plp = plp::localmax(&pulse);
    let mut beats: Vec<usize> = beats_dp.clone();
    beats.extend_from_slice(&beats_plp);
    beats.sort_unstable();
    beats.dedup();
    println!("rust n_beats_dp {} n_beats_plp {} n_beats_union {}", beats_dp.len(), beats_plp.len(), beats.len());
    println!("rust first 10 beats {:?}", &beats[..beats.len().min(10)]);
    println!("rust last 5 beats {:?}", &beats[beats.len().saturating_sub(5)..]);
}
