use anyhow::Result;
use clap::Parser;
use rustmusiclooper::analysis::Params;
use rustmusiclooper::detect_loop_points;

#[derive(Parser, Debug)]
#[command(name = "rustmusiclooper", about = "Detect music loop points (PyMusicLooper re-implementation)")]
struct Cli {
    /// Path to the audio file (m4a/mp4/aac/wav/...).
    path: String,

    /// Minimum loop duration as a multiplier of the track duration.
    #[arg(long, default_value_t = 0.35)]
    min_duration_multiplier: f32,

    /// Minimum loop duration in seconds (overrides --min-duration-multiplier).
    #[arg(long)]
    min_loop_duration: Option<f32>,

    /// Maximum loop duration in seconds.
    #[arg(long)]
    max_loop_duration: Option<f32>,

    /// Approximate loop start in seconds (requires --approx-loop-end).
    #[arg(long)]
    approx_loop_start: Option<f32>,

    /// Approximate loop end in seconds (requires --approx-loop-start).
    #[arg(long)]
    approx_loop_end: Option<f32>,

    /// Check the entire track instead of detected beats (slow).
    #[arg(long)]
    brute_force: bool,

    /// Return all candidate loop points without filtering.
    #[arg(long)]
    disable_pruning: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let params = Params {
        min_duration_multiplier: cli.min_duration_multiplier,
        min_loop_duration: cli.min_loop_duration,
        max_loop_duration: cli.max_loop_duration,
        approx_loop_start: cli.approx_loop_start,
        approx_loop_end: cli.approx_loop_end,
        brute_force: cli.brute_force,
        disable_pruning: cli.disable_pruning,
    };

    let output = detect_loop_points(&cli.path, &params)?;

    // Emit the array of LoopPairResponse (score-desc) on stdout.
    let pairs_json = serde_json::to_string(&output.pairs)?;
    println!("{}", pairs_json);

    // Emit metadata on stderr so it doesn't pollute the JSON stream.
    eprintln!(
        "meta: rate={} ch={} total_samples={} total_duration={} trim_offset={} n_pairs={}",
        output.meta.rate,
        output.meta.n_channels,
        output.meta.total_samples,
        format!("{:.3}", output.meta.total_duration),
        output.meta.trim_offset,
        output.meta.n_pairs
    );

    Ok(())
}
