pub mod analysis;
pub mod audio;
pub mod dsp;
pub mod model;

use anyhow::Result;
use analysis::Params;
use model::{AnalysisMeta, AnalysisOutput, LoopPairResponse};

/// Load audio from a file path and detect loop points. Returns pairs sorted by score (desc).
pub fn detect_loop_points(path: &str, params: &Params) -> Result<AnalysisOutput> {
    let audio = audio::load(path)?;
    analyze_audio(&audio, params)
}

/// Decode audio from an in-memory byte buffer and detect loop points.
/// For use from iOS/Flutter where file access is restricted.
pub fn detect_loop_points_from_bytes(bytes: &[u8], params: &Params) -> Result<AnalysisOutput> {
    let audio = audio::load_from_bytes(bytes)?;
    analyze_audio(&audio, params)
}

fn analyze_audio(audio: &audio::AudioData, params: &Params) -> Result<AnalysisOutput> {
    let pairs = analysis::find_loop_pairs(audio, params);
    let meta = AnalysisMeta {
        rate: audio.rate as i64,
        n_channels: audio.n_channels as i64,
        total_samples: audio.total_samples as i64,
        total_duration: audio.total_duration,
        trim_offset: audio.trim_offset as i64,
        n_pairs: pairs.len() as i64,
    };
    Ok(AnalysisOutput { meta, pairs })
}

pub type LoopPair = LoopPairResponse;

// ---------------------------------------------------------------------------
// FFI (for Flutter / iOS)
// ---------------------------------------------------------------------------

use std::ffi::c_char;

/// FFI entry point for Flutter/iOS.
///
/// Decodes the audio buffer and detects loop points. Returns a null-terminated
/// UTF-8 JSON string. The caller **must** free the returned pointer with
/// [`rustmusiclooper_free_string`].
///
/// JSON schema:
/// ```json
/// {
///   "meta": { "rate": 48000, "n_channels": 2, "total_samples": 0,
///             "total_duration": 0.0, "trim_offset": 0, "n_pairs": 0 },
///   "pairs": [ { "loop_start_sample": 0, "loop_end_sample": 0,
///                "loop_start_seconds": 0.0, "loop_end_seconds": 0.0,
///                "score": 0.0 } ],
///   "error": null
/// }
/// ```
/// On failure, `error` contains the message and `pairs` is an empty array.
///
/// Parameters:
/// - `data` / `data_len`: raw audio file bytes (wav, m4a, aac, ...)
/// - `min_duration_multiplier`: minimum loop duration as multiplier of track length
/// - `min_loop_duration`: minimum loop duration in seconds (-1.0 = not set)
/// - `max_loop_duration`: maximum loop duration in seconds (-1.0 = not set)
/// - `approx_loop_start`: approximate loop start in seconds (-1.0 = not set)
/// - `approx_loop_end`: approximate loop end in seconds (-1.0 = not set)
/// - `brute_force`: 1 = check entire track, 0 = use detected beats
/// - `disable_pruning`: 1 = return all candidates, 0 = prune
#[unsafe(no_mangle)]
pub extern "C" fn rustmusiclooper_analyze(
    data: *const u8,
    data_len: usize,
    min_duration_multiplier: f32,
    min_loop_duration: f32,
    max_loop_duration: f32,
    approx_loop_start: f32,
    approx_loop_end: f32,
    brute_force: u8,
    disable_pruning: u8,
) -> *mut c_char {
    let result = unsafe {
        if data.is_null() || data_len == 0 {
            Err(anyhow::anyhow!("empty audio buffer"))
        } else {
            let bytes = std::slice::from_raw_parts(data, data_len);
            let params = Params {
                min_duration_multiplier,
                min_loop_duration: (min_loop_duration > 0.0).then_some(min_loop_duration),
                max_loop_duration: (max_loop_duration > 0.0).then_some(max_loop_duration),
                approx_loop_start: (approx_loop_start > 0.0).then_some(approx_loop_start),
                approx_loop_end: (approx_loop_end > 0.0).then_some(approx_loop_end),
                brute_force: brute_force != 0,
                disable_pruning: disable_pruning != 0,
            };
            detect_loop_points_from_bytes(bytes, &params)
        }
    };

    let json = match result {
        Ok(output) => serde_json::to_string(&output)
            .unwrap_or_else(|_| r#"{"meta":null,"pairs":[],"error":"serialization failed"}"#.into()),
        Err(e) => format!(
            r#"{{"meta":null,"pairs":[],"error":{}}}"#,
            serde_json::to_string(&e.to_string()).unwrap_or_else(|_| "\"error\"".into())
        ),
    };

    let mut buf = json.into_bytes();
    buf.push(0);
    Box::into_raw(Box::new(buf)) as *mut c_char
}

/// Free a string returned by [`rustmusiclooper_analyze`].
#[unsafe(no_mangle)]
pub extern "C" fn rustmusiclooper_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr as *mut u8));
    }
}
