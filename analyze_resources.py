#!/usr/bin/env python3
"""Resource analysis is performed by analyze_energy.py from the same active samples."""
from pathlib import Path
import json
from common import ROOT

required = [ROOT / "resources/resource_summary.json", ROOT / "temperature/temperature_summary.json", ROOT / "throttling/throttling_summary.json"]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing resource outputs: {missing}")
print(json.dumps({"status": "PASS", "verified": [str(path) for path in required]}, indent=2))
