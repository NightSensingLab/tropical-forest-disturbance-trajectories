#!/usr/bin/env python3
"""Run Stage 09 on the current full-history Stage 08 package."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from pipeline.run_subannual_response import run_pipeline
from pipeline.subannual_response import SubannualSettings


STAGE08 = RUN_ROOT / "stage08_current_full_history"
OUT = RUN_ROOT / "stage09_current_full_history"
PAIR_FILE = PROJECT_ROOT / "runs/20260904-expanded-control-pool/fixed_matched_pairs.parquet"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    settings = SubannualSettings(
        stage08_signature="current_full_history_1990_stage08_20260909",
        stage07_source_sha256=sha256(PAIR_FILE),
        expected_pair_links=50198,
        expected_treatments=16747,
        expected_unit_events=52248,
        expected_monthly_rows=5642784,
        expected_sixteen_day_rows=10815336,
    )
    config = {"input_stage_08": str(STAGE08), "output_stage_09": str(OUT)}
    OUT.mkdir(parents=True, exist_ok=True)
    report = run_pipeline(config, settings=settings, force_rebuild=False)
    print(json.dumps({"stage09_pass": report["stage09_pass"], "sample_flow": report["sample_flow"], "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
