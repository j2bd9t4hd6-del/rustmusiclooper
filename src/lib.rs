pub mod analysis;
pub mod audio;
pub mod dsp;
pub mod model;

use anyhow::Result;
use analysis::Params;
use model::{AnalysisMeta, AnalysisOutput, LoopPairResponse};

/// Load audio and detect loop points. Returns pairs sorted by score (desc).
pub fn detect_loop_points(path: &str, params: &Params) -> Result<AnalysisOutput> {
    let audio = audio::load(path)?;
    let pairs = analysis::find_loop_pairs(&audio, params);
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
