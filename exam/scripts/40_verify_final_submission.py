"""Check the designated final file against its source candidate and the template.

Uses only the standard library. The file is a byte-for-byte copy of an existing
candidate; nothing here regenerates or edits rows.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exam"))

from sweettv.artifacts import load_arrays, stage_dir, write_result
from sweettv.config import RESULTS_DIR, data_dir, load_config

EXPECTED_SHA256 = "334ea1a56901b31e98eb178030f0f0e0b798e97d9a0ad427f9067fcc74e97ace"
ROW = re.compile(r"^(\d+),([1-9]\d*(?: [1-9]\d*){4})$")


def main():
    cfg = load_config()
    final = ROOT / "exam" / "submission" / "final_submission.csv"
    source = stage_dir(cfg, "adaptive_candidates") / "history_adaptive_submission.csv"
    raw, source_raw = final.read_bytes(), source.read_bytes()
    template = [line.split(",")[0] for line in (data_dir(cfg) / cfg["files"]["submission_template"]).read_text(encoding="utf-8").split("\n")[1:] if line]
    pool = set(load_arrays(cfg, "test_pool", "pool")["candidates"].tolist())
    text = raw.decode("ascii")
    rows = text[:-1].split("\n")[1:]
    parsed = [ROW.match(row) for row in rows]
    ids = [[int(i) for i in match.group(2).split(" ")] for match in parsed if match]
    checks = {
        "sha256_equals_expected": hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256,
        "bytes_equal_source_candidate": raw == source_raw,
        "header_exact": text.startswith("user_id,tv_show_id\n"),
        "exactly_one_final_newline": text.endswith("\n") and not text.endswith("\n\n"),
        "rows_1260": len(rows) == 1260,
        "every_row_matches_schema": all(parsed),
        "exact_template_user_order": [match.group(1) for match in parsed if match] == template,
        "unique_users_1260": len({match.group(1) for match in parsed if match}) == 1260,
        "five_distinct_ids_per_row": all(len(set(row)) == 5 for row in ids),
        "all_ids_in_test_pool": all(set(row) <= pool for row in ids),
    }
    write_result(RESULTS_DIR / "final_submission_check.json", {
        "file": "exam/submission/final_submission.csv",
        "source_candidate": "adaptive_candidates/history_adaptive_submission.csv",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "post_kaggle_choice": True,
        "checks": checks,
    })
    print(json.dumps(checks, indent=1))
    if not all(checks.values()):
        raise SystemExit("final submission check failed")


if __name__ == "__main__":
    main()
