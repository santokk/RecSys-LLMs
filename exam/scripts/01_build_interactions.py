"""Build airings and per-airing user interactions for weeks 11-30.

Streams the view log in bounded chunks. Reads the training and validation
schedules only. Writes the `preprocess` stage to the external artifacts folder.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweettv.artifacts import save_stage, source_hashes, write_result
from sweettv.config import RESULTS_DIR, load_config
from sweettv.pipeline import build_interactions
from sweettv.runlog import RunTimer
from sweettv.split import window_bounds
from sweettv.timeutil import WallClock


def main():
    timer = RunTimer("01_build_interactions")
    cfg = load_config()
    source_hashes(cfg)
    clock = WallClock(cfg["timezone"])
    airings, interactions, coverage, users, audit = build_interactions(cfg, clock, window_bounds(cfg, clock))
    arrays = {"airings": airings, "interactions": interactions, "coverage": coverage}
    manifest = save_stage(cfg, "preprocess", arrays, {"users.txt": "\n".join(users) + "\n"})
    write_result(RESULTS_DIR / "preprocess_audit.json", audit)
    timer.finish(artifact_bytes=sum(entry["bytes"] for entry in manifest["files"].values()))


if __name__ == "__main__":
    main()
