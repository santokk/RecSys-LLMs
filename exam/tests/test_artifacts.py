import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sweettv import artifacts, csvio
from sweettv.config import load_config

VIEWS = '"user_id","vsetv_id","start_time","stop_time","duraton"\n' \
        '18417531283109304442,7,2020-03-09 00:00:02,2020-03-09 00:01:53,111\n' \
        '3398461054087191302,6,2020-03-09 00:00:05,2020-03-09 00:01:26,81\n' \
        '3398461054087191302,6,2020-03-09 00:02:05,2020-03-09 00:03:26,81\n'


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.data, self.store = root / "data", root / "store"
        self.data.mkdir()
        self.view_file = self.data / "views.csv"
        self.view_file.write_text(VIEWS, encoding="utf-8")
        self.cfg = load_config()
        self.cfg["files"] = {"view_log": "views.csv", "schedule_test": "hidden.csv"}
        self.cfg["expected_sha256"] = {"views.csv": hashlib.sha256(VIEWS.encode()).hexdigest()}
        self.cfg["chunk_rows"] = 2
        self.previous = {k: os.environ.get(k) for k in ("SWEETTV_DATA_DIR", "SWEETTV_ARTIFACT_DIR")}
        os.environ["SWEETTV_DATA_DIR"] = str(self.data)
        os.environ["SWEETTV_ARTIFACT_DIR"] = str(self.store)
        artifacts._source_hashes.clear()

    def tearDown(self):
        for key, value in self.previous.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
        artifacts._source_hashes.clear()
        self.folder.cleanup()


class StaleGuard(Sandbox):
    def build(self):
        artifacts.save_stage(self.cfg, "stage", {"table": {"x": np.arange(3)}})

    def test_round_trip(self):
        self.build()
        self.assertEqual(artifacts.load_arrays(self.cfg, "stage", "table")["x"].tolist(), [0, 1, 2])

    def test_missing_manifest_is_refused(self):
        with self.assertRaises(artifacts.StaleArtifactError):
            artifacts.load_manifest(self.cfg, "stage")

    def test_config_change_is_refused(self):
        self.build()
        self.cfg["watched"]["primary_seconds"] = 301
        with self.assertRaises(artifacts.StaleArtifactError):
            artifacts.load_arrays(self.cfg, "stage", "table")

    def test_altered_artifact_is_refused(self):
        self.build()
        (artifacts.stage_dir(self.cfg, "stage") / "table.npz").write_bytes(b"tampered")
        with self.assertRaises(artifacts.StaleArtifactError):
            artifacts.load_arrays(self.cfg, "stage", "table")

    def test_changed_source_is_refused(self):
        self.build()
        self.view_file.write_text(VIEWS + "1,1,2020-03-09 00:00:00,2020-03-09 00:01:00,60\n", encoding="utf-8")
        artifacts._source_hashes.clear()
        with self.assertRaises(artifacts.StaleArtifactError):
            artifacts.load_arrays(self.cfg, "stage", "table")

    def test_code_change_is_refused(self):
        self.build()
        path = artifacts.stage_dir(self.cfg, "stage") / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["code_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(artifacts.StaleArtifactError):
            artifacts.load_arrays(self.cfg, "stage", "table")

    def test_rebuilt_dependency_is_refused(self):
        self.build()
        artifacts.save_stage(self.cfg, "child", {"table": {"x": np.arange(2)}}, depends_on=("stage",))
        artifacts.save_stage(self.cfg, "stage", {"table": {"x": np.arange(4)}})
        with self.assertRaises(artifacts.StaleArtifactError):
            artifacts.load_arrays(self.cfg, "child", "table")

    def test_paths_are_not_part_of_the_fingerprint(self):
        before = artifacts.config_hash(self.cfg)
        self.cfg["paths"]["data_dir"] = "elsewhere"
        self.assertEqual(before, artifacts.config_hash(self.cfg))


class StreamingAccess(Sandbox):
    def test_chunks_are_bounded(self):
        sizes = [len(chunk) for chunk in csvio.iter_view_chunks(self.cfg)]
        self.assertEqual(sizes, [2, 1])

    def test_user_ids_stay_strings(self):
        chunk = next(csvio.iter_view_chunks(self.cfg))
        self.assertEqual(chunk[0][0], "18417531283109304442")
        self.assertIsInstance(chunk[0][0], str)

    def test_source_bytes_are_untouched_by_reading(self):
        before = self.view_file.read_bytes()
        list(csvio.iter_view_chunks(self.cfg))
        self.assertEqual(self.view_file.read_bytes(), before)

    def test_unexpected_header_is_refused(self):
        self.view_file.write_text("a,b\n1,2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(csvio.iter_view_chunks(self.cfg))

    def test_test_schedule_is_refused_for_fitting(self):
        with self.assertRaises(PermissionError):
            list(csvio.iter_fit_schedule(self.cfg, "schedule_test"))


if __name__ == "__main__":
    unittest.main()
