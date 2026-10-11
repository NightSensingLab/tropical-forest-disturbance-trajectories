#!/usr/bin/env python3
"""Check selected code provenance/imports and optional frozen reference inputs."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"
CURRENT = Path("runs/20260909-extended-sensitivities")
EXTRACTED = {
    "run_analysis_diagnostics.py": {
        "current_dynamic_outcomes", "negative_fraction_bootstrap", "event_id",
        "extreme_distance_current_outcomes",
    },
    "run_composition_diagnostics.py": {"weighted_moments", "weighted_smd", "stage09_composition"},
    "run_matching_diagnostics.py": {
        "read_year_features", "event_id", "raw_mad_scales", "reconstruct_distances",
        "distance_diagnostics", "concentration_diagnostics", "balance_from_moments_local",
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RemoveAlignment(ast.NodeTransformer):
    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "require_matplotlib_panel_alignment"):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("ee-zheng"):
            return ast.Constant(value="TFDT_EE_PROJECT")
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os" and node.func.value.attr == "environ"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "TFDT_EE_PROJECT"):
            return ast.Constant(value="TFDT_EE_PROJECT")
        return node


def function_bodies(path):
    tree = RemoveAlignment().visit(ast.parse(path.read_text(encoding="utf-8")))
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_references(source_root, modules):
    import numpy as np
    import pandas as pd

    run = source_root / CURRENT
    analysis = run / "analysis_current_full_history"
    expected_counts = {
        "history_qualified_candidates.parquet": 24406,
        "matched_treatments.parquet": 16747,
        "fixed_matched_pairs.parquet": 50198,
        "common_support_weights.parquet": 10970,
    }
    counts = {}
    for filename, expected in expected_counts.items():
        counts[filename] = len(pd.read_parquet(analysis / filename))
        if counts[filename] != expected:
            raise AssertionError(f"{filename}: {counts[filename]} != {expected}")
    primary = modules[str(CURRENT / "recompute_current_standardized.py")]
    primary.OUT = analysis
    primary.STAGE08 = run / "stage08_current_full_history"
    primary.STAGE09 = run / "stage09_current_full_history"
    frame = primary.prepare_frame()
    saved = pd.read_csv(analysis / "standardized_event_type_contrast.csv")
    endpoints = {}
    for year, count in [(0, 6581), (5, 5000)]:
        group = frame.loc[frame.relative_year.eq(year)]
        if len(group) != count:
            raise AssertionError(f"t{year} cohort count {len(group)} != {count}")
        estimates = {}
        for event_type, rows in group.groupby("event_type"):
            estimate = np.average(rows.pair_effect_evi_mean, weights=rows.standardized_event_weight)
            reference = saved.loc[saved.relative_year.eq(year) & saved.event_type.eq(event_type)].iloc[0]
            np.testing.assert_allclose(estimate, reference.estimate, atol=1e-12, rtol=1e-10)
            if len(rows) != int(reference.treatment_count):
                raise AssertionError(f"t{year}/{event_type} count differs from reference")
            estimates[event_type] = float(estimate)
        contrast = estimates["deforestation"] - estimates["degradation"]
        reference = saved.loc[saved.relative_year.eq(year)
                              & saved.event_type.eq("deforestation_minus_degradation")].iloc[0]
        np.testing.assert_allclose(contrast, reference.estimate, atol=1e-12, rtol=1e-10)
        endpoints[f"t{year}"] = {"n": len(group), "contrast": contrast}
    classifier = modules[str(CURRENT / "recompute_current_classification.py")]
    classifier.RUN_ROOT = run
    cohort = classifier.cohort()
    class_counts = cohort.event_type.value_counts().to_dict()
    if len(cohort) != 3881 or class_counts != {"degradation": 2677, "deforestation": 1204}:
        raise AssertionError(f"Classification cohort differs: {class_counts}")
    fixed = modules["revision_v28_audit/run_fixed_cohort_sensitivity.py"]
    fixed.ANALYSIS = analysis
    panel, _ = fixed.load_panel()
    fixed_counts = {}
    for label, years, expected in [("A1", list(range(6)), 2469), ("A2", [0, 3, 5], 3468)]:
        _, keys = fixed.subset_cohort(panel, years)
        if len(keys) != expected:
            raise AssertionError(f"{label} cohort {len(keys)} != {expected}")
        fixed_counts[label] = len(keys)
    return {"input_counts": counts, "primary_endpoints": endpoints,
            "classification_counts": class_counts, "fixed_cohort_counts": fixed_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Original project, for source hash/AST checks")
    parser.add_argument("--reference-check", action="store_true", help="Read existing inputs; do not fit or write results")
    args = parser.parse_args()
    source_root = args.source_root.resolve() if args.source_root else None
    errors = []
    with (ROOT / "docs/SOURCE_MANIFEST.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    files = sorted(ROOT.rglob("*.py"))
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            compile(tree, str(path), "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pipeline."):
                    dependency = PIPELINE / (node.module.replace(".", "/") + ".py")
                    if not dependency.is_file():
                        errors.append(f"Missing local dependency: {node.module}")
        except Exception as exc:
            errors.append(f"Syntax {path.relative_to(ROOT)}: {exc}")
    compared = 0
    for record in records:
        destination = ROOT / record["destination"]
        if not destination.is_file() or digest(destination) != record["packaged_sha256"]:
            errors.append(f"Packaged hash mismatch: {record['destination']}")
        if source_root:
            source = source_root / record["source"]
            if not source.is_file() or digest(source) != record["source_sha256"]:
                errors.append(f"Source hash mismatch: {record['source']}")
                continue
            if destination.suffix == ".py":
                before, after = function_bodies(source), function_bodies(destination)
                selected = EXTRACTED.get(destination.name, set(before))
                for name in selected:
                    if destination.name == "generate_figure4.py" and name == "parenthesized_label_anchor":
                        continue  # Removed machine-local presentation audit hook.
                    if destination.name == "submit_current_modis_increment.py" and name == "_import_ee":
                        continue  # Explicit project-configuration guard; recorded in manifest.
                    if after.get(name) != before[name]:
                        errors.append(f"Function changed: {record['destination']}::{name}")
                    compared += 1
    forbidden = re.compile(r"/Users/|/home/[^ /]+/|\.codex/|ee-zheng|-----BEGIN .*PRIVATE KEY|AIza[0-9A-Za-z_-]{30,}")
    allowed = {".py", ".md", ".csv", ".json", ".txt", ".parquet", ".xlsx"}
    for path in ROOT.rglob("*"):
        if any(part == ".git" for part in path.relative_to(ROOT).parts):
            continue
        if path.is_dir():
            if path.name in {"__pycache__", ".venv"}:
                errors.append(f"Unwanted directory: {path.relative_to(ROOT)}")
            continue
        if path.name != ".gitignore" and path.suffix not in allowed:
            errors.append(f"Unexpected file type: {path.relative_to(ROOT)}")
        if path == Path(__file__).resolve() or path.name == "check_publication_data.py":
            continue  # The checker contains private-path detection patterns.
        if path.suffix not in {".parquet", ".xlsx"} and forbidden.search(path.read_text(encoding="utf-8")):
            errors.append(f"Private path/key pattern: {path.relative_to(ROOT)}")
    modules = {}
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PIPELINE))
    sys.path.insert(0, str(ROOT / "runs/20260904-expanded-control-pool"))
    sys.path.insert(0, str(ROOT / "03_publication/manuscript/TropicalRainforest_manuscript/02_pipeline/tools"))
    with tempfile.TemporaryDirectory(prefix="tfdt-import-check-") as cache:
        cache_root = Path(cache).resolve()
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["MPLBACKEND"] = "Agg"
        checks_enabled = [True]

        def read_only_imports(event, arguments):
            if not checks_enabled[0]:
                return
            if event == "open":
                mode, flags = arguments[1:3]
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+"))
                writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
                if writing and isinstance(arguments[0], (str, bytes, os.PathLike)):
                    if not Path(arguments[0]).resolve().is_relative_to(cache_root):
                        raise RuntimeError("Import attempted to write outside temporary cache")
            if event in {"socket.connect", "subprocess.Popen", "os.remove", "os.rename"}:
                if event == "subprocess.Popen" and arguments[1] in (
                    ["fc-list", "--help"], ["fc-list", "--format=%{file}\\n"],
                    ["system_profiler", "-xml", "SPFontsDataType"],
                ):
                    return  # Matplotlib's read-only system font queries.
                if event.startswith("os.") and Path(arguments[0]).resolve().is_relative_to(cache_root):
                    return
                raise RuntimeError(f"Forbidden during read-only checks: {event}")
            if event == "os.mkdir":
                target = Path(arguments[0]).resolve()
                if not target.exists() and not target.is_relative_to(cache_root):
                    raise RuntimeError("Import attempted to create output directory")

        sys.addaudithook(read_only_imports)
        for index, path in enumerate(files):
            if path == Path(__file__).resolve():
                continue
            relative = str(path.relative_to(ROOT))
            name = "pipeline." + path.stem if path.parent == PIPELINE / "pipeline" else f"tfdt_check_{index}"
            try:
                modules[relative] = load_module(path, name)
            except Exception as exc:
                errors.append(f"Import {relative}: {type(exc).__name__}: {exc}")
        reference = None
        if args.reference_check and not errors:
            try:
                reference = check_references(source_root or ROOT, modules)
            except Exception as exc:
                errors.append(f"Reference check: {type(exc).__name__}: {exc}")
        checks_enabled[0] = False
    report = {"python_files": len(files), "selected_manifest_files": len(records),
              "successful_imports": len(modules), "source_functions_compared": compared,
              "reference_checks": reference, "errors": errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
