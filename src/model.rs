use serde::Serialize;

/// One detected loop pair, matching the requested
/// `LoopPairResponse` schema. Results are returned sorted by `score`
/// (descending, best first).
#[derive(Debug, Clone, Serialize)]
pub struct LoopPairResponse {
    pub loop_start_sample: i64,
    pub loop_end_sample: i64,
    pub loop_start_seconds: f64,
    pub loop_end_seconds: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisMeta {
    pub rate: i64,
    pub n_channels: i64,
    pub total_samples: i64,
    pub total_duration: f64,
    pub trim_offset: i64,
    pub n_pairs: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisOutput {
    pub meta: AnalysisMeta,
    /// Sorted by `score` descending.
    pub pairs: Vec<LoopPairResponse>,
}
