#!/usr/bin/env python3
"""Check data provenance, display coverage, workbook cells and release hashes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
XML = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.reader(stream))


def workbook_cells(path):
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(t.text or "" for t in item.findall(".//s:t", XML))
                      for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        relationships = {item.attrib["Id"]: item.attrib["Target"]
                         for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        result = {}
        for sheet in workbook.find("s:sheets", XML):
            link = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = relationships[link]
            filename = target.lstrip("/") if target.startswith("/") else "xl/" + target
            cells = {}
            for cell in ET.fromstring(archive.read(filename)).findall(".//s:sheetData/s:row/s:c", XML):
                if cell.find("s:f", XML) is not None:
                    raise AssertionError("Unexpected formula in manuscript table workbook")
                kind = cell.attrib.get("t", "n")
                value = cell.find("s:v", XML)
                value = value.text if value is not None else ""
                if kind == "s":
                    value = shared[int(value)]
                elif kind == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//s:t", XML))
                elif kind == "n" and value:
                    value = float(value)
                elif kind == "e":
                    raise AssertionError(f"Workbook error cell: {sheet.attrib['name']}!{cell.attrib['r']}")
                cells[cell.attrib["r"]] = value
            result[sheet.attrib["name"]] = cells
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Original project for source hash verification")
    args = parser.parse_args()
    with (ROOT / "docs/DATA_MANIFEST.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    with (ROOT / "source_data/DISPLAY_INDEX.csv").open(newline="", encoding="utf-8") as stream:
        displays = list(csv.DictReader(stream))
    errors = []
    destinations = set()
    for record in manifest:
        path = ROOT / record["destination"]
        if record["destination"] in destinations:
            errors.append("Duplicate manifest destination: " + record["destination"])
        destinations.add(record["destination"])
        if not path.is_file() or sha256(path) != record["packaged_sha256"]:
            errors.append("Data hash mismatch: " + record["destination"])
        if args.source_root:
            source = args.source_root / record["source"]
            if not source.is_file() or sha256(source) != record["source_sha256"]:
                errors.append("Original data hash mismatch: " + record["source"])
    expected = {"Table 1", *[f"Table S{i}" for i in range(1, 16)],
                *[f"Figure {i}" for i in range(1, 5)], *[f"Figure S{i}" for i in range(1, 7)]}
    if len(displays) != 26 or {r["display"] for r in displays} != expected:
        errors.append("Display index must contain exactly 26 current figures/tables")
    for display in displays:
        paths = display["data_files"].split(";")
        if not paths or any(p not in destinations for p in paths):
            errors.append("Missing indexed data: " + display["display"])
    csv_count = 0
    for relative in destinations:
        path = ROOT / relative
        if path.suffix == ".csv":
            csv_count += 1
            try:
                rows = read_csv(path)
                if not rows or not rows[0] or len({len(row) for row in rows}) != 1:
                    errors.append("Empty/nonrectangular CSV: " + relative)
                if re.search(r"/Users/|/home/[^ /]+/|\.codex/", path.read_text(encoding="utf-8")):
                    errors.append("Private path in CSV: " + relative)
            except Exception as exc:
                errors.append(f"CSV {relative}: {exc}")
    cells = workbook_cells(ROOT / "source_data/tables/manuscript_tables.xlsx")
    checked_cells = 0
    for table in sorted(p for p in destinations if p.startswith("source_data/tables/Table_") and p.endswith("_manuscript.csv")):
        sheet = Path(table).stem.removesuffix("_manuscript").replace("_", " ")
        rows = read_csv(ROOT / table)
        for r, row in enumerate(rows):
            for c, original in enumerate(row):
                expected_value = original
                normalized = original.replace("\u2212", "-").replace(",", "")
                if r > 0 and re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
                    expected_value = float(normalized)
                address = f"{chr(65+c)}{r+1}"
                if cells[sheet].get(address, "") != expected_value:
                    errors.append(f"Workbook/CSV mismatch: {sheet}!{address}")
                checked_cells += 1
    if len(cells) != 17:
        errors.append("Expected 16 manuscript tables plus workbook index")
    checksum_count = 0
    if (ROOT / "SHA256SUMS.txt").exists():
        for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
            expected_hash, relative = line.split("  ", 1)
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected_hash:
                errors.append("Release hash mismatch: " + relative)
            checksum_count += 1
    report = {"data_manifest_files": len(manifest), "source_csvs": csv_count,
              "figures": 10, "tables": 16, "workbook_sheets": len(cells),
              "workbook_cells_compared": checked_cells, "release_hashes_checked": checksum_count,
              "errors": errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
