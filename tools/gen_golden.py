"""Generate golden reference data from PyMusicLooper for the Rust port.

Runs find_loop_pairs with disable_pruning=True to capture ALL candidate loop
points (after scoring/filtering by zero-crossing), and dumps them as JSON so
the Rust implementation can be validated against the Python reference.
"""
import json
import sys

from pymusiclooper.core import MusicLooper


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: gen_golden.py <audio_path> <out_json>", file=sys.stderr)
        sys.exit(2)

    audio_path = sys.argv[1]
    out_path = sys.argv[2]

    looper = MusicLooper(audio_path)

    # Capture all candidate pairs (disable_pruning returns every scored pair).
    pairs = looper.find_loop_pairs(
        min_duration_multiplier=0.35,
        min_loop_duration=None,
        max_loop_duration=None,
        approx_loop_start=None,
        approx_loop_end=None,
        brute_force=False,
        disable_pruning=True,
    )

    records = []
    for p in pairs:
        records.append(
            {
                "loop_start_sample": int(p.loop_start),
                "loop_end_sample": int(p.loop_end),
                "loop_start_seconds": float(p.loop_start) / float(looper.mlaudio.rate),
                "loop_end_seconds": float(p.loop_end) / float(looper.mlaudio.rate),
                "score": float(p.score),
                "note_distance": float(p.note_distance),
                "loudness_difference": float(p.loudness_difference),
            }
        )

    # Sort by score descending (matches find_best_loop_points output order).
    records.sort(key=lambda r: r["score"], reverse=True)

    meta = {
        "filepath": audio_path,
        "rate": int(looper.mlaudio.rate),
        "n_channels": int(looper.mlaudio.n_channels),
        "total_samples": int(looper.mlaudio.length),
        "total_duration": float(looper.mlaudio.total_duration),
        "trim_offset": int(looper.mlaudio.trim_offset),
        "n_pairs": len(records),
    }

    payload = {"meta": meta, "pairs": records}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(
        f"rate={meta['rate']} ch={meta['n_channels']} "
        f"total_samples={meta['total_samples']} trim_offset={meta['trim_offset']} "
        f"n_pairs={meta['n_pairs']}",
        file=sys.stderr,
    )
    if records:
        top = records[0]
        print(
            "top: start_sample={} end_sample={} start_s={:.3f} end_s={:.3f} score={:.6f}".format(
                top["loop_start_sample"],
                top["loop_end_sample"],
                top["loop_start_seconds"],
                top["loop_end_seconds"],
                top["score"],
            ),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
