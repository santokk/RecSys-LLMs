"""Runtime and peak-memory records for each numbered script."""
import json
import platform
import resource
import sys
import time

import numpy as np

from .config import RESULTS_DIR


class RunTimer:
    def __init__(self, name):
        self.name = name
        self.started = time.perf_counter()

    def finish(self, **extra):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        path = RESULTS_DIR / "runtime.json"
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        records["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                                  "platform": f"{platform.system()} {platform.machine()}"}
        records[self.name] = {"seconds": round(time.perf_counter() - self.started, 1),
                              "peak_rss_mib": round(peak_bytes / (1 << 20), 1), **extra}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
        print(f"{self.name}: {records[self.name]}")
