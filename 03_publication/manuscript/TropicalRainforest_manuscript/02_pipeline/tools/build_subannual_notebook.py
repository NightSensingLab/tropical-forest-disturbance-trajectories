import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_SHA256 = "493359a28b473d4a3213c4dceee4dad3b5cbc13476f87c0d860cb811f18a7672"
EVENT_YEARS = tuple(range(2004, 2019))
TRANSFER_SENTINEL = -9999.0
PARQUET_COMPRESSION = "zstd"
UNIT_EVENT_COLUMNS = (
    "grid_id",
    "event_year",
    "role",
    "lon",
    "lat",
    "macroregion",
    "source_signature",
)
MATCH_REQUIRED_COLUMNS = (
    "treatment_grid_id",
    "control_grid_id",
    "event_year",
    "control_event_year",
    "macroregion",
    "control_macroregion",
    "treatment_lon",
    "treatment_lat",
    "control_lon",
    "control_lat",
    "geographic_distance_km",
    "within_set_weight",
    "search_radius_km",
)


def sorted_python_ints(values):
    return tuple(sorted(int(value) for value in values))


def _require_columns(frame, required, label):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def split_unit_event_catalog(catalog, event_year, max_unit_events=750):
    _require_columns(catalog, UNIT_EVENT_COLUMNS, "unit-event catalog")
    if (
        isinstance(max_unit_events, (bool, np.bool_))
        or not isinstance(max_unit_events, (int, np.integer))
        or int(max_unit_events) <= 0
    ):
        raise ValueError("max_unit_events must be a positive integer")

    event_year = int(event_year)
    part = catalog.loc[catalog["event_year"].eq(event_year)].copy(deep=True)
    if part.empty:
        raise ValueError(f"No unit-events found for event year {event_year}")
    key_columns = ["grid_id", "event_year", "role"]
    if part.duplicated(key_columns).any():
        raise ValueError(f"Duplicate unit-event keys for event year {event_year}")

    part = part.sort_values(["role", "grid_id"], kind="stable").reset_index(drop=True)
    shard_size = int(max_unit_events)
    shards = [
        part.iloc[start : start + shard_size].copy(deep=True).reset_index(drop=True)
        for start in range(0, len(part), shard_size)
    ]
    combined = pd.concat(shards, ignore_index=True)
    if len(combined) != len(part) or combined.duplicated(key_columns).any():
        raise RuntimeError("Unit-event shard coverage is incomplete or overlapping")
    if not combined.loc[:, key_columns].equals(part.loc[:, key_columns]):
        raise RuntimeError("Unit-event shard order differs from the stable catalogue order")
    return shards


def aggregate_record_state(record):
    shard_tasks = record.get("shard_tasks") or []
    if not shard_tasks:
        return record.get("task_state", "NOT_SUBMITTED")
    states = [task.get("task_state", "NOT_SUBMITTED") for task in shard_tasks]
    if all(
        state == "COMPLETED" and bool(task.get("drive_files"))
        for state, task in zip(states, shard_tasks)
    ):
        return "COMPLETED"
    if any(state in {"FAILED", "CANCELLED"} for state in states):
        return "FAILED"
    if any(state == "RUNNING" for state in states):
        return "RUNNING"
    if any(
        task.get("task_id") or state != "NOT_SUBMITTED"
        for task, state in zip(shard_tasks, states)
    ):
        return "READY"
    return "NOT_SUBMITTED"


def _reject_conflicting_attributes(frame, keys, attributes, label):
    conflicts = frame.groupby(list(keys), dropna=False)[list(attributes)].nunique(
        dropna=False
    )
    if conflicts.gt(1).any(axis=None):
        raise ValueError(f"Found conflicting {label} attributes within a unit-event")


def build_unit_event_catalog(pairs, source_signature=SOURCE_SHA256):
    _require_columns(pairs, MATCH_REQUIRED_COLUMNS, "Stage 07 matched pairs")
    frame = pairs.copy(deep=True)
    if frame.empty:
        raise ValueError("Stage 07 matched pairs are empty")

    numeric_columns = (
        "treatment_lon",
        "treatment_lat",
        "control_lon",
        "control_lat",
        "geographic_distance_km",
        "within_set_weight",
        "search_radius_km",
    )
    if not np.isfinite(frame.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Stage 07 matched pairs contain non-finite numeric values")
    if not frame["event_year"].isin(EVENT_YEARS).all():
        raise ValueError("Stage 07 event years fall outside 2004-2018")
    if not frame["control_event_year"].eq(frame["event_year"]).all():
        raise ValueError("Control event years do not match treatment event years")
    if not frame["macroregion"].eq(frame["control_macroregion"]).all():
        raise ValueError("Treatment and control macroregion values differ")
    if (frame["geographic_distance_km"] < 0).any() or (
        frame["geographic_distance_km"] > frame["search_radius_km"] + 1e-9
    ).any():
        raise ValueError("Matched controls violate the recorded search radius")
    if (frame["within_set_weight"] <= 0).any():
        raise ValueError("Stage 07 within-set weights must be positive")

    link_key = ["treatment_grid_id", "control_grid_id", "event_year"]
    if frame.duplicated(link_key).any():
        raise ValueError("Stage 07 contains duplicate treatment-control links")
    weight_sums = frame.groupby(
        ["treatment_grid_id", "event_year"], sort=False
    )["within_set_weight"].sum()
    if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Stage 07 within-set weights do not sum to one")

    _reject_conflicting_attributes(
        frame,
        ("treatment_grid_id", "event_year"),
        ("treatment_lon", "treatment_lat", "macroregion"),
        "treatment",
    )
    _reject_conflicting_attributes(
        frame,
        ("control_grid_id", "event_year"),
        ("control_lon", "control_lat", "control_macroregion"),
        "control",
    )

    treatments = frame.loc[
        :,
        [
            "treatment_grid_id",
            "event_year",
            "treatment_lon",
            "treatment_lat",
            "macroregion",
        ],
    ].rename(
        columns={
            "treatment_grid_id": "grid_id",
            "treatment_lon": "lon",
            "treatment_lat": "lat",
        }
    )
    treatments["role"] = "treatment"
    controls = frame.loc[
        :,
        [
            "control_grid_id",
            "event_year",
            "control_lon",
            "control_lat",
            "control_macroregion",
        ],
    ].rename(
        columns={
            "control_grid_id": "grid_id",
            "control_lon": "lon",
            "control_lat": "lat",
            "control_macroregion": "macroregion",
        }
    )
    controls["role"] = "control"

    catalog = pd.concat([treatments, controls], ignore_index=True)
    catalog = catalog.drop_duplicates(["grid_id", "event_year", "role"])
    catalog["source_signature"] = str(source_signature)
    catalog = catalog.loc[:, UNIT_EVENT_COLUMNS]
    return catalog.sort_values(
        ["event_year", "role", "grid_id"], kind="stable"
    ).reset_index(drop=True)


def observation_metadata(observation_date, event_year):
    timestamp = pd.Timestamp(observation_date)
    return {
        "observation_date": timestamp.strftime("%Y-%m-%d"),
        "calendar_year": int(timestamp.year),
        "calendar_month": int(timestamp.month),
        "day_of_year": int(timestamp.dayofyear),
        "relative_year": int(timestamp.year - event_year),
    }


def validate_collection_calendar(calendar, analysis_years=range(2001, 2024)):
    _require_columns(calendar, ("observation_date",), "MOD13Q1 calendar")
    frame = calendar.copy(deep=True)
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], errors="raise"
    )
    if frame["observation_date"].duplicated().any():
        raise RuntimeError("MOD13Q1 collection calendar changed: duplicate date")
    counts = frame.groupby(frame["observation_date"].dt.year).size()
    expected = set(int(year) for year in analysis_years)
    if set(counts.index) != expected or not counts.eq(23).all():
        raise RuntimeError("MOD13Q1 collection calendar changed")
    return frame.sort_values("observation_date").reset_index(drop=True)


SIXTEEN_DAY_REQUIRED_COLUMNS = (
    "grid_id",
    "event_year",
    "role",
    "lon",
    "lat",
    "macroregion",
    "observation_date",
    "calendar_year",
    "calendar_month",
    "day_of_year",
    "relative_year",
    "modis_image_id",
    "system_time_start",
    "evi",
    "valid_pixel_fraction",
    "source_signature",
    "analysis_version",
)
SIXTEEN_DAY_PRIMARY_KEY = (
    "grid_id",
    "event_year",
    "role",
    "observation_date",
)


def _expected_period_keys(catalog, calendar):
    parts = []
    calendar_dates = pd.to_datetime(calendar["observation_date"], errors="raise")
    for event_year, units in catalog.groupby("event_year", sort=False):
        dates = calendar_dates.loc[
            calendar_dates.dt.year.between(event_year - 3, event_year + 5)
        ]
        if dates.empty:
            raise ValueError(f"No audited calendar dates for event year {event_year}")
        unit_keys = units.loc[:, ["grid_id", "event_year", "role"]].reset_index(
            drop=True
        )
        unit_keys["_join"] = 1
        date_keys = pd.DataFrame({"observation_date": dates.to_numpy(), "_join": 1})
        parts.append(unit_keys.merge(date_keys, on="_join").drop(columns="_join"))
    if not parts:
        raise ValueError("Expected unit-event catalog is empty")
    return pd.concat(parts, ignore_index=True)


def validate_16day_frame(frame, expected_catalog, calendar):
    _require_columns(frame, SIXTEEN_DAY_REQUIRED_COLUMNS, "16-day frame")
    _require_columns(expected_catalog, UNIT_EVENT_COLUMNS, "unit-event catalog")
    _require_columns(
        calendar,
        ("observation_date", "modis_image_id", "system_time_start"),
        "MOD13Q1 calendar",
    )
    result = frame.copy(deep=True)
    catalog = expected_catalog.copy(deep=True)
    audited_calendar = calendar.copy(deep=True)
    result["observation_date"] = pd.to_datetime(
        result["observation_date"], errors="raise"
    ).dt.normalize()
    audited_calendar["observation_date"] = pd.to_datetime(
        audited_calendar["observation_date"], errors="raise"
    ).dt.normalize()

    if result.duplicated(list(SIXTEEN_DAY_PRIMARY_KEY)).any():
        raise ValueError("Duplicate 16-day primary keys")
    if catalog.duplicated(["grid_id", "event_year", "role"]).any():
        raise ValueError("Duplicate unit-event catalog keys")
    if not result["role"].isin(("treatment", "control")).all():
        raise ValueError("Invalid role in 16-day frame")
    if not result["relative_year"].between(-3, 5).all():
        raise ValueError("Relative year falls outside -3..5")

    expected_keys = _expected_period_keys(catalog, audited_calendar)
    observed_keys = result.loc[:, SIXTEEN_DAY_PRIMARY_KEY].sort_values(
        list(SIXTEEN_DAY_PRIMARY_KEY), ignore_index=True
    )
    expected_keys = expected_keys.loc[:, SIXTEEN_DAY_PRIMARY_KEY].sort_values(
        list(SIXTEEN_DAY_PRIMARY_KEY), ignore_index=True
    )
    if len(observed_keys) != len(expected_keys) or not observed_keys.equals(expected_keys):
        raise ValueError("16-day rows do not match the expected Cartesian product")

    catalog_attributes = catalog.rename(
        columns={
            "lon": "expected_lon",
            "lat": "expected_lat",
            "macroregion": "expected_macroregion",
            "source_signature": "expected_source_signature",
        }
    )
    checked = result.merge(
        catalog_attributes,
        on=["grid_id", "event_year", "role"],
        how="left",
        validate="many_to_one",
    )
    if checked["expected_lon"].isna().any():
        raise ValueError("16-day frame contains unknown unit-events")
    if not np.allclose(checked["lon"], checked["expected_lon"], atol=1e-7, rtol=0):
        raise ValueError("16-day longitude differs from the unit-event catalog")
    if not np.allclose(checked["lat"], checked["expected_lat"], atol=1e-7, rtol=0):
        raise ValueError("16-day latitude differs from the unit-event catalog")
    if not checked["macroregion"].eq(checked["expected_macroregion"]).all():
        raise ValueError("16-day macroregion differs from the unit-event catalog")
    if not checked["source_signature"].eq(
        checked["expected_source_signature"]
    ).all():
        raise ValueError("16-day source signature differs from the unit-event catalog")

    expected_year = result["observation_date"].dt.year
    expected_month = result["observation_date"].dt.month
    expected_day = result["observation_date"].dt.dayofyear
    expected_relative = expected_year - result["event_year"].astype(int)
    if not result["calendar_year"].astype(int).eq(expected_year).all():
        raise ValueError("calendar_year does not match observation_date")
    if not result["calendar_month"].astype(int).eq(expected_month).all():
        raise ValueError("calendar_month does not match observation_date")
    if not result["day_of_year"].astype(int).eq(expected_day).all():
        raise ValueError("day_of_year does not match observation_date")
    if not result["relative_year"].astype(int).eq(expected_relative).all():
        raise ValueError("relative_year does not match event_year")

    calendar_lookup = audited_calendar.loc[
        :, ["observation_date", "modis_image_id", "system_time_start"]
    ].drop_duplicates("observation_date")
    checked_calendar = result.merge(
        calendar_lookup,
        on="observation_date",
        suffixes=("", "_expected"),
        how="left",
        validate="many_to_one",
    )
    if checked_calendar["modis_image_id_expected"].isna().any():
        raise ValueError("16-day frame contains a date outside the audited calendar")
    if not checked_calendar["modis_image_id"].eq(
        checked_calendar["modis_image_id_expected"]
    ).all():
        raise ValueError("modis_image_id differs from the audited calendar")
    if not checked_calendar["system_time_start"].eq(
        checked_calendar["system_time_start_expected"]
    ).all():
        raise ValueError("system_time_start differs from the audited calendar")

    evi = pd.to_numeric(result["evi"], errors="coerce")
    if np.isinf(evi.to_numpy(dtype=float)).any() or (
        evi.notna() & ~evi.between(-0.1, 1.0)
    ).any():
        raise ValueError("EVI outside the accepted -0.1..1.0 domain")
    valid_fraction = pd.to_numeric(
        result["valid_pixel_fraction"], errors="coerce"
    )
    if valid_fraction.isna().any() or not valid_fraction.between(0, 1).all():
        raise ValueError("Invalid valid-pixel fraction outside 0..1")

    result["evi"] = evi.astype(float)
    result["valid_pixel_fraction"] = valid_fraction.astype(float)
    result["evi_valid"] = result["evi"].notna() & result[
        "valid_pixel_fraction"
    ].gt(0)
    return result.sort_values(list(SIXTEEN_DAY_PRIMARY_KEY)).reset_index(drop=True)


def build_monthly_summary(frame):
    required = set(SIXTEEN_DAY_REQUIRED_COLUMNS) | {"evi_valid"}
    _require_columns(frame, required, "validated 16-day frame")
    data = frame.copy(deep=True)
    data["observation_date"] = pd.to_datetime(
        data["observation_date"], errors="raise"
    )
    data["evi_valid"] = data["evi"].notna() & data[
        "valid_pixel_fraction"
    ].gt(0)
    data["valid_evi"] = data["evi"].where(data["evi_valid"])

    group_columns = [
        "grid_id",
        "event_year",
        "role",
        "lon",
        "lat",
        "macroregion",
        "calendar_year",
        "relative_year",
        "calendar_month",
        "source_signature",
        "analysis_version",
    ]
    monthly = (
        data.groupby(group_columns, sort=True, dropna=False)
        .agg(
            evi_mean=("valid_evi", "mean"),
            evi_median=("valid_evi", "median"),
            valid_composite_count=("evi_valid", "sum"),
            expected_composite_count=("observation_date", "size"),
            mean_valid_pixel_fraction=("valid_pixel_fraction", "mean"),
        )
        .reset_index()
    )
    monthly["valid_composite_count"] = monthly[
        "valid_composite_count"
    ].astype("int16")
    monthly["expected_composite_count"] = monthly[
        "expected_composite_count"
    ].astype("int16")
    monthly["coverage_fraction"] = (
        monthly["valid_composite_count"] / monthly["expected_composite_count"]
    )

    baseline_keys = ["grid_id", "event_year", "role", "calendar_month"]
    baseline = monthly.loc[
        monthly["relative_year"].isin((-3, -2, -1)), baseline_keys + ["evi_mean"]
    ]
    baseline = (
        baseline.groupby(baseline_keys, sort=False, dropna=False)["evi_mean"]
        .agg(baseline_evi="mean", baseline_year_count="count")
        .reset_index()
    )
    baseline.loc[baseline["baseline_year_count"] < 2, "baseline_evi"] = np.nan
    monthly = monthly.merge(
        baseline, on=baseline_keys, how="left", validate="many_to_one"
    )
    monthly["baseline_year_count"] = monthly["baseline_year_count"].fillna(0).astype(
        "int8"
    )
    monthly["evi_anomaly"] = monthly["evi_mean"] - monthly["baseline_evi"]
    return monthly.sort_values(
        ["event_year", "role", "grid_id", "relative_year", "calendar_month"]
    ).reset_index(drop=True)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_event_year_part(
    csv_paths,
    catalog,
    calendar,
    output_16day,
    output_monthly,
    transfer_sentinel=TRANSFER_SENTINEL,
):
    csv_paths = [Path(path) for path in csv_paths]
    if not csv_paths or not all(path.is_file() for path in csv_paths):
        raise FileNotFoundError("One or more Earth Engine CSV shards are missing")
    output_16day = Path(output_16day)
    output_monthly = Path(output_monthly)
    if output_16day.exists() or output_monthly.exists():
        raise FileExistsError("Assembly refuses to overwrite an existing Parquet product")

    raw = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)
    _require_columns(raw, SIXTEEN_DAY_REQUIRED_COLUMNS, "Earth Engine CSV export")
    numeric_evi = pd.to_numeric(raw["evi"], errors="coerce")
    invalid_evi_text = raw["evi"].notna() & numeric_evi.isna()
    if invalid_evi_text.any():
        raise ValueError("Earth Engine CSV contains non-numeric EVI values")
    raw["evi"] = numeric_evi
    raw.loc[np.isclose(raw["evi"], transfer_sentinel), "evi"] = np.nan
    numeric_fraction = pd.to_numeric(raw["valid_pixel_fraction"], errors="coerce")
    invalid_fraction_text = raw["valid_pixel_fraction"].notna() & numeric_fraction.isna()
    if invalid_fraction_text.any():
        raise ValueError("Earth Engine CSV contains non-numeric valid-pixel fractions")
    raw["valid_pixel_fraction"] = numeric_fraction.fillna(0.0)

    validated = validate_16day_frame(raw, catalog, calendar)
    monthly = build_monthly_summary(validated)
    output_16day.parent.mkdir(parents=True, exist_ok=True)
    output_monthly.parent.mkdir(parents=True, exist_ok=True)
    temporary_16day = output_16day.with_name(output_16day.name + ".tmp")
    temporary_monthly = output_monthly.with_name(output_monthly.name + ".tmp")
    validated.to_parquet(
        temporary_16day, index=False, compression=PARQUET_COMPRESSION
    )
    monthly.to_parquet(
        temporary_monthly, index=False, compression=PARQUET_COMPRESSION
    )
    os.replace(temporary_16day, output_16day)
    os.replace(temporary_monthly, output_monthly)
    return {
        "rows_16day": len(validated),
        "rows_monthly": len(monthly),
        "size_16day": output_16day.stat().st_size,
        "size_monthly": output_monthly.stat().st_size,
        "sha256_16day": sha256_file(output_16day),
        "sha256_monthly": sha256_file(output_monthly),
        "csv_sha256": [sha256_file(path) for path in csv_paths],
    }


def _source_lines(source):
    return source.strip("\n").splitlines(keepends=True)


def _cell_id(prefix, index, source):
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{index:02d}-{digest}"


def markdown_cell(source, prefix, index):
    return {
        "cell_type": "markdown",
        "id": _cell_id(prefix, index, source),
        "metadata": {},
        "source": _source_lines(source),
    }


def code_cell(source, prefix, index, tags=None):
    metadata = {"tags": list(tags or [])} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(prefix, index, source),
        "metadata": metadata,
        "outputs": [],
        "source": _source_lines(source),
    }


def _notebook_helper_source():
    functions = [
        sorted_python_ints,
        _require_columns,
        split_unit_event_catalog,
        aggregate_record_state,
        _reject_conflicting_attributes,
        build_unit_event_catalog,
        observation_metadata,
        validate_collection_calendar,
        _expected_period_keys,
        validate_16day_frame,
        build_monthly_summary,
    ]
    extracted = "\n\n".join(inspect.getsource(function) for function in functions)
    return f'''\
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

UNIT_EVENT_COLUMNS = {UNIT_EVENT_COLUMNS!r}
MATCH_REQUIRED_COLUMNS = {MATCH_REQUIRED_COLUMNS!r}
SIXTEEN_DAY_REQUIRED_COLUMNS = {SIXTEEN_DAY_REQUIRED_COLUMNS!r}
SIXTEEN_DAY_PRIMARY_KEY = {SIXTEEN_DAY_PRIMARY_KEY!r}

{extracted}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

def atomic_write_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression=PARQUET_COMPRESSION)
    os.replace(temporary, path)

def file_record(path, rows=None):
    path = Path(path)
    record = {{
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }}
    if rows is not None:
        record["rows"] = int(rows)
    return record

def paths_overlap(left, right):
    left = Path(left).resolve()
    right = Path(right).resolve()
    return left == right or left in right.parents or right in left.parents

def reusable_part(path, record, signature):
    path = Path(path)
    return (
        path.is_file()
        and record.get("analysis_signature") == signature
        and record.get("size_bytes") == path.stat().st_size
        and record.get("sha256") == sha256_file(path)
    )
'''


CONFIG_SOURCE = r'''
from pathlib import Path

ANALYSIS_ROOT = Path("/content/drive/MyDrive/TropicalRainforest_analysis")
MATCH_PATH = ANALYSIS_ROOT / "07_matched_control" / "matched_pairs.parquet"
BASE_OUT_DIR = ANALYSIS_ROOT / "08_subannual_matched"

# Required: enter the Google Cloud project registered for Earth Engine.
GEE_PROJECT = ""
GEE_DRIVE_FOLDER = "stage08_mod13q1_493359a28b47"

MODIS_COLLECTION = "MODIS/061/MOD13Q1"
ACCEPTED_SUMMARY_QA = (0, 1)
EVI_SCALE = 0.0001
EVI_RANGE = (-0.1, 1.0)
USE_SRTM_LAND_MASK = True
SRTM_COLLECTION = "CGIAR/SRTM90_V4"
GRID_CRS = "EPSG:4326"
GRID_TRANSFORM = [0.05, 0, -180, 0, -0.05, 23.5]
RELATIVE_YEARS = tuple(range(-3, 6))
EVENT_YEARS = tuple(range(2004, 2019))
TRANSFER_SENTINEL = -9999.0

EXPECTED_SOURCE_SIZE = 1_502_573
EXPECTED_SOURCE_SHA256 = "493359a28b473d4a3213c4dceee4dad3b5cbc13476f87c0d860cb811f18a7672"
SOURCE_SHA256 = EXPECTED_SOURCE_SHA256
EXPECTED_PAIR_LINKS = 46_984
EXPECTED_TREATMENT_UNIT_EVENTS = 15_674
EXPECTED_CONTROL_UNIT_EVENTS = 33_523
EXPECTED_UNIT_EVENTS = 49_197
EXPECTED_UNIQUE_GRIDS = 42_098
EXPECTED_PERIODS_PER_UNIT_EVENT = 207
EXPECTED_16DAY_ROWS = 10_183_779
FAILED_TASK_SHARD_UNIT_EVENTS = 750

ANALYSIS_VERSION = "2026-07-25-stage08-subannual-matched-v1"
PARQUET_COMPRESSION = "zstd"
POLL_SECONDS = 60
EXPORT_TILE_SCALE = 4
FORCE_REBUILD = False
REBUILD_NAMESPACE = ""

if FORCE_REBUILD:
    if not REBUILD_NAMESPACE.strip():
        raise ValueError("Set REBUILD_NAMESPACE when FORCE_REBUILD=True.")
    OUT_DIR = BASE_OUT_DIR / "runs" / REBUILD_NAMESPACE.strip()
else:
    OUT_DIR = BASE_OUT_DIR

GEE_EXPORT_DIR = OUT_DIR / GEE_DRIVE_FOLDER
PARTS_16DAY_DIR = OUT_DIR / "parts_16day"
PARTS_MONTHLY_DIR = OUT_DIR / "parts_monthly"
TASK_MANIFEST_PATH = OUT_DIR / "task_manifest.json"

print({
    "match_input": str(MATCH_PATH),
    "output": str(OUT_DIR),
    "event_years": EVENT_YEARS,
    "nominal_rows": EXPECTED_16DAY_ROWS,
    "gee_project_configured": bool(GEE_PROJECT.strip()),
})
'''


INSTALL_SOURCE = r'''
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "ee": "earthengine-api>=1.4",
    "numpy": "numpy>=1.26",
    "pandas": "pandas>=2.1",
    "pyarrow": "pyarrow>=15",
}
missing = [
    specification
    for module, specification in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(module) is None
]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
print("Dependencies ready.")
'''


INITIALIZE_SOURCE = r'''
from google.colab import drive
import ee

drive.mount("/content/drive", force_remount=False)
if not GEE_PROJECT.strip() or GEE_PROJECT.strip().lower() in {
    "your-gcp-project-id",
    "your-project-id",
}:
    raise ValueError("Set GEE_PROJECT to the Cloud Project ID registered for Earth Engine.")
ee.Authenticate()
ee.Initialize(project=GEE_PROJECT)
print("Google Drive mounted and Earth Engine initialized.")
'''


INPUT_GATE_SOURCE = r'''
if paths_overlap(MATCH_PATH, OUT_DIR):
    raise RuntimeError("Stage 07 input and Stage 08 output paths overlap.")
if not MATCH_PATH.is_file():
    raise FileNotFoundError(f"Upload the approved Stage 07 file to {MATCH_PATH}")
if MATCH_PATH.stat().st_size != EXPECTED_SOURCE_SIZE:
    raise RuntimeError(
        f"Stage 07 size mismatch: {MATCH_PATH.stat().st_size} != {EXPECTED_SOURCE_SIZE}"
    )
observed_source_sha256 = sha256_file(MATCH_PATH)
if observed_source_sha256 != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(f"Stage 07 SHA-256 mismatch: {observed_source_sha256}")

matched_links = pd.read_parquet(MATCH_PATH)
if len(matched_links) != EXPECTED_PAIR_LINKS:
    raise RuntimeError(f"Pair-link count mismatch: {len(matched_links)}")
unit_event_catalog = build_unit_event_catalog(matched_links)
treatment_count = int(unit_event_catalog["role"].eq("treatment").sum())
control_count = int(unit_event_catalog["role"].eq("control").sum())
unique_grid_count = int(unit_event_catalog["grid_id"].nunique())
observed_event_years = sorted_python_ints(unit_event_catalog["event_year"].unique())
if treatment_count != EXPECTED_TREATMENT_UNIT_EVENTS:
    raise RuntimeError(f"Treatment unit-event mismatch: {treatment_count}")
if control_count != EXPECTED_CONTROL_UNIT_EVENTS:
    raise RuntimeError(f"Control unit-event mismatch: {control_count}")
if len(unit_event_catalog) != EXPECTED_UNIT_EVENTS:
    raise RuntimeError(f"Total unit-event mismatch: {len(unit_event_catalog)}")
if unique_grid_count != EXPECTED_UNIQUE_GRIDS:
    raise RuntimeError(f"Unique-grid mismatch: {unique_grid_count}")
if observed_event_years != EVENT_YEARS:
    raise RuntimeError(f"Event-year mismatch: {observed_event_years}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
for directory in (GEE_EXPORT_DIR, PARTS_16DAY_DIR, PARTS_MONTHLY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

links_output = OUT_DIR / "matched_links.parquet"
if links_output.is_file():
    if sha256_file(links_output) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Existing matched-links copy differs: {links_output}")
else:
    temporary = links_output.with_name(links_output.name + ".tmp")
    with MATCH_PATH.open("rb") as source, temporary.open("wb") as target:
        while chunk := source.read(8 * 1024 * 1024):
            target.write(chunk)
    os.replace(temporary, links_output)

catalog_output = OUT_DIR / "unit_event_catalog.parquet"
if catalog_output.is_file():
    existing_catalog = pd.read_parquet(catalog_output)
    pd.testing.assert_frame_equal(existing_catalog, unit_event_catalog, check_dtype=False)
else:
    atomic_write_parquet(unit_event_catalog, catalog_output)

input_audit = {
    "audited_at_utc": utc_now(),
    "source_path": str(MATCH_PATH),
    "source_size_bytes": MATCH_PATH.stat().st_size,
    "source_sha256": observed_source_sha256,
    "pair_links": len(matched_links),
    "treatment_unit_events": treatment_count,
    "control_unit_events": control_count,
    "unit_events": len(unit_event_catalog),
    "unique_grids": unique_grid_count,
    "event_years": list(observed_event_years),
}
atomic_write_json(input_audit, OUT_DIR / "input_audit.json")
print(input_audit)
'''


CALENDAR_AUDIT_SOURCE = r'''
collection = (
    ee.ImageCollection(MODIS_COLLECTION)
    .filterDate("2001-01-01", "2024-01-01")
    .sort("system:time_start")
)
first_band_names = set(collection.first().bandNames().getInfo())
missing_bands = {"EVI", "SummaryQA"} - first_band_names
if missing_bands:
    raise RuntimeError(f"MOD13Q1 lacks required bands: {sorted(missing_bands)}")

image_ids = collection.aggregate_array("system:index").getInfo()
time_starts = collection.aggregate_array("system:time_start").getInfo()
collection_calendar = pd.DataFrame(
    {
        "modis_image_id": image_ids,
        "system_time_start": time_starts,
    }
)
collection_calendar["observation_date"] = pd.to_datetime(
    collection_calendar["system_time_start"], unit="ms", utc=True
).dt.tz_localize(None).dt.normalize()
collection_calendar = validate_collection_calendar(
    collection_calendar,
    analysis_years=range(2001, 2024),
)
collection_calendar = collection_calendar[
    ["observation_date", "modis_image_id", "system_time_start"]
]
if len(collection_calendar) != 23 * 23:
    raise RuntimeError(f"Unexpected MOD13Q1 calendar length: {len(collection_calendar)}")

calendar_output = OUT_DIR / "collection_calendar.csv"
if calendar_output.is_file():
    existing_calendar = pd.read_csv(calendar_output, parse_dates=["observation_date"])
    pd.testing.assert_frame_equal(
        existing_calendar,
        collection_calendar,
        check_dtype=False,
    )
else:
    atomic_write_csv(collection_calendar, calendar_output)

signature_payload = {
    "analysis_version": ANALYSIS_VERSION,
    "source_sha256": EXPECTED_SOURCE_SHA256,
    "modis_collection": MODIS_COLLECTION,
    "calendar": collection_calendar.assign(
        observation_date=collection_calendar["observation_date"].dt.strftime("%Y-%m-%d")
    ).to_dict("records"),
    "accepted_summary_qa": list(ACCEPTED_SUMMARY_QA),
    "evi_scale": EVI_SCALE,
    "evi_range": list(EVI_RANGE),
    "use_srtm_land_mask": USE_SRTM_LAND_MASK,
    "grid_crs": GRID_CRS,
    "grid_transform": GRID_TRANSFORM,
    "relative_years": list(RELATIVE_YEARS),
    "transfer_sentinel": TRANSFER_SENTINEL,
    "output_columns": list(SIXTEEN_DAY_REQUIRED_COLUMNS) + ["evi_valid"],
}
ANALYSIS_SIGNATURE = hashlib.sha256(
    json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

native_projection = collection.first().select("EVI").projection().getInfo()
target_projection = ee.Projection(GRID_CRS, GRID_TRANSFORM)
grid_audit = {
    "audited_at_utc": utc_now(),
    "collection_rows": len(collection_calendar),
    "periods_per_year": 23,
    "native_projection": native_projection,
    "target_crs": GRID_CRS,
    "target_transform": GRID_TRANSFORM,
    "analysis_signature": ANALYSIS_SIGNATURE,
}
atomic_write_json(grid_audit, OUT_DIR / "grid_audit.json")
print(grid_audit)
'''


MANIFEST_SOURCE = r'''
def export_description(event_year):
    return f"stage08_mod13q1_{event_year}_{ANALYSIS_SIGNATURE[:12]}"

def export_prefix(event_year):
    return f"stage08_mod13q1_event_{event_year}_{ANALYSIS_SIGNATURE[:12]}"

def shard_export_description(event_year, shard_index, shard_count):
    return (
        f"stage08_mod13q1_{event_year}_s{shard_index:02d}of{shard_count:02d}_"
        f"{ANALYSIS_SIGNATURE[:12]}"
    )

def shard_export_prefix(event_year, shard_index, shard_count):
    return (
        f"stage08_mod13q1_event_{event_year}_{ANALYSIS_SIGNATURE[:12]}_"
        f"shard_{shard_index:02d}-of-{shard_count:02d}"
    )

def fresh_task_record(event_year):
    units = int(unit_event_catalog["event_year"].eq(event_year).sum())
    return {
        "event_year": int(event_year),
        "expected_unit_events": units,
        "expected_rows": units * EXPECTED_PERIODS_PER_UNIT_EVENT,
        "task_description": export_description(event_year),
        "output_prefix": export_prefix(event_year),
        "task_id": None,
        "task_state": "NOT_SUBMITTED",
        "last_checked_utc": None,
        "drive_files": [],
        "validation_status": "pending",
    }

def discover_drive_files(task_record):
    prefix = task_record["output_prefix"]
    return [str(path) for path in sorted(GEE_EXPORT_DIR.glob(prefix + "*.csv"))]

def reconcile_task_record(task_record, statuses_by_id, tasks_by_description=None):
    task_id = task_record.get("task_id")
    if not task_id and tasks_by_description is not None:
        matches = tasks_by_description.get(task_record["task_description"], [])
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple Earth Engine tasks found for {task_record['task_description']}"
            )
        if len(matches) == 1:
            task_id = matches[0].get("id")
            task_record["task_id"] = task_id
            task_record["task_state"] = matches[0].get("state", "UNKNOWN")
    if task_id and task_id in statuses_by_id:
        status = statuses_by_id[task_id]
        task_record["task_state"] = status.get("state", "UNKNOWN")
        if status.get("error_message"):
            task_record["error_message"] = status["error_message"]
    task_record["drive_files"] = discover_drive_files(task_record)
    task_record["last_checked_utc"] = utc_now()

if TASK_MANIFEST_PATH.is_file():
    task_manifest = json.loads(TASK_MANIFEST_PATH.read_text(encoding="utf-8"))
    if task_manifest.get("analysis_signature") != ANALYSIS_SIGNATURE:
        raise RuntimeError(
            "Existing task manifest has different settings. Use a new FORCE_REBUILD namespace."
        )
else:
    task_manifest = {
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "analysis_signature": ANALYSIS_SIGNATURE,
        "source_signature": EXPECTED_SOURCE_SHA256,
        "parts": {str(year): fresh_task_record(year) for year in EVENT_YEARS},
    }

earth_engine_tasks = ee.data.getTaskList()
tasks_by_description = {}
for task_info in earth_engine_tasks:
    description = task_info.get("description")
    if description:
        tasks_by_description.setdefault(description, []).append(task_info)

tracked_records = []
for event_year in EVENT_YEARS:
    key = str(event_year)
    record = task_manifest["parts"].setdefault(key, fresh_task_record(event_year))
    if record.get("shard_tasks"):
        tracked_records.extend(record["shard_tasks"])
    else:
        tracked_records.append(record)

tracked_task_ids = sorted(
    {task["task_id"] for task in tracked_records if task.get("task_id")}
)
task_statuses = ee.data.getTaskStatus(tracked_task_ids) if tracked_task_ids else []
statuses_by_id = {
    status.get("id"): status for status in task_statuses if status.get("id")
}

for event_year in EVENT_YEARS:
    record = task_manifest["parts"][str(event_year)]
    if record.get("shard_tasks"):
        for shard_task in record["shard_tasks"]:
            reconcile_task_record(shard_task, statuses_by_id, tasks_by_description)
        record["drive_files"] = sorted(
            {
                path
                for shard_task in record["shard_tasks"]
                for path in shard_task.get("drive_files", [])
            }
        )
        record["task_state"] = aggregate_record_state(record)
        record["last_checked_utc"] = utc_now()
    else:
        reconcile_task_record(record, statuses_by_id, tasks_by_description)

task_manifest["updated_at_utc"] = utc_now()
atomic_write_json(task_manifest, TASK_MANIFEST_PATH)
reconcile_rows = []
for record in task_manifest["parts"].values():
    shard_tasks = record.get("shard_tasks") or []
    reconcile_rows.append(
        {
            "event_year": record["event_year"],
            "expected_rows": record["expected_rows"],
            "task_state": record["task_state"],
            "recovery_mode": record.get("recovery_mode"),
            "completed_shards": sum(
                task.get("task_state") == "COMPLETED" and bool(task.get("drive_files"))
                for task in shard_tasks
            ),
            "shard_count": len(shard_tasks),
            "drive_file_count": len(record.get("drive_files", [])),
        }
    )
display(pd.DataFrame(reconcile_rows).sort_values("event_year"))
'''


SUBMISSION_SOURCE = r'''
land_mask = ee.Image(SRTM_COLLECTION).select("elevation").mask().gt(0)

def catalog_to_feature_collection(catalog_part):
    features = []
    for row in catalog_part.itertuples(index=False):
        properties = {
            "grid_id": int(row.grid_id),
            "event_year": int(row.event_year),
            "role": str(row.role),
            "lon": float(row.lon),
            "lat": float(row.lat),
            "macroregion": str(row.macroregion),
            "source_signature": str(row.source_signature),
            "analysis_version": ANALYSIS_VERSION,
        }
        features.append(
            ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), properties)
        )
    return ee.FeatureCollection(features)

def aggregate_modis_image(image):
    raw_evi = image.select("EVI")
    scaled_evi = raw_evi.multiply(EVI_SCALE)
    qa = image.select("SummaryQA")
    qa_ok = qa.eq(ACCEPTED_SUMMARY_QA[0]).Or(qa.eq(ACCEPTED_SUMMARY_QA[1]))
    range_ok = scaled_evi.gte(EVI_RANGE[0]).And(scaled_evi.lte(EVI_RANGE[1]))
    valid_mask = qa_ok.And(range_ok)
    if USE_SRTM_LAND_MASK:
        valid_mask = valid_mask.And(land_mask)

    accepted_evi = scaled_evi.updateMask(valid_mask)
    native_valid = valid_mask.unmask(0).toFloat()
    if USE_SRTM_LAND_MASK:
        native_valid = native_valid.updateMask(land_mask)

    evi_005 = (
        accepted_evi.reduceResolution(
            reducer=ee.Reducer.mean(), bestEffort=False, maxPixels=65535
        )
        .reproject(crs=GRID_CRS, crsTransform=GRID_TRANSFORM)
        .rename("evi")
        .unmask(TRANSFER_SENTINEL)
    )
    valid_fraction_005 = (
        native_valid.reduceResolution(
            reducer=ee.Reducer.mean(), bestEffort=False, maxPixels=65535
        )
        .reproject(crs=GRID_CRS, crsTransform=GRID_TRANSFORM)
        .rename("valid_pixel_fraction")
        .unmask(0)
    )
    return evi_005.addBands(valid_fraction_005)

def sample_one_observation(points, calendar_row, event_year):
    image = ee.Image(f"{MODIS_COLLECTION}/{calendar_row.modis_image_id}")
    aggregate = aggregate_modis_image(image)
    relative_year = int(calendar_row.observation_date.year - event_year)
    metadata = {
        "observation_date": calendar_row.observation_date.strftime("%Y-%m-%d"),
        "calendar_year": int(calendar_row.observation_date.year),
        "calendar_month": int(calendar_row.observation_date.month),
        "day_of_year": int(calendar_row.observation_date.dayofyear),
        "relative_year": relative_year,
        "modis_image_id": str(calendar_row.modis_image_id),
        "system_time_start": int(calendar_row.system_time_start),
    }
    sampled = aggregate.reduceRegions(
        collection=points,
        reducer=ee.Reducer.first(),
        crs=GRID_CRS,
        crsTransform=GRID_TRANSFORM,
        tileScale=EXPORT_TILE_SCALE,
    )
    return sampled.map(lambda feature: feature.set(metadata))

def build_event_year_export(event_year, catalog_part=None):
    if catalog_part is None:
        catalog_part = unit_event_catalog.loc[
            unit_event_catalog["event_year"].eq(event_year)
        ].copy()
    else:
        catalog_part = catalog_part.copy(deep=True)
    if catalog_part.empty or not catalog_part["event_year"].eq(event_year).all():
        raise RuntimeError(f"Invalid catalogue part for event year {event_year}")
    points = catalog_to_feature_collection(catalog_part)
    start = pd.Timestamp(event_year - 3, 1, 1)
    stop = pd.Timestamp(event_year + 6, 1, 1)
    dates = collection_calendar.loc[
        collection_calendar["observation_date"].between(start, stop, inclusive="left")
    ]
    if len(dates) != EXPECTED_PERIODS_PER_UNIT_EVENT:
        raise RuntimeError(f"Event year {event_year} has {len(dates)} calendar slots")
    nested = [
        sample_one_observation(points, row, event_year)
        for row in dates.itertuples(index=False)
    ]
    return ee.FeatureCollection(nested).flatten()

EXPORT_SELECTORS = [
    "grid_id", "event_year", "role", "lon", "lat", "macroregion",
    "observation_date", "calendar_year", "calendar_month", "day_of_year",
    "relative_year", "modis_image_id", "system_time_start", "evi",
    "valid_pixel_fraction", "source_signature", "analysis_version",
]
LIVE_STATES = {"READY", "RUNNING"}
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED"}

def create_recovery_plan(record, event_year):
    catalog_shards = split_unit_event_catalog(
        unit_event_catalog,
        event_year,
        max_unit_events=FAILED_TASK_SHARD_UNIT_EVENTS,
    )
    shard_count = len(catalog_shards)
    record["recovery_parent_task"] = {
        "task_description": record["task_description"],
        "output_prefix": record["output_prefix"],
        "task_id": record.get("task_id"),
        "task_state": record.get("task_state"),
        "error_message": record.get("error_message"),
        "last_checked_utc": record.get("last_checked_utc"),
        "drive_files": list(record.get("drive_files", [])),
    }
    record["recovery_mode"] = "unit_event_shards"
    record["recovery_created_at_utc"] = utc_now()
    record["shard_unit_events"] = FAILED_TASK_SHARD_UNIT_EVENTS
    record["shard_tasks"] = []
    for shard_index, catalog_shard in enumerate(catalog_shards, start=1):
        units = len(catalog_shard)
        record["shard_tasks"].append(
            {
                "shard_index": shard_index,
                "shard_count": shard_count,
                "expected_unit_events": units,
                "expected_rows": units * EXPECTED_PERIODS_PER_UNIT_EVENT,
                "task_description": shard_export_description(
                    event_year, shard_index, shard_count
                ),
                "output_prefix": shard_export_prefix(
                    event_year, shard_index, shard_count
                ),
                "task_id": None,
                "task_state": "NOT_SUBMITTED",
                "last_checked_utc": None,
                "drive_files": [],
            }
        )
    record["drive_files"] = []
    record["task_state"] = "NOT_SUBMITTED"

def checked_recovery_shards(record, event_year):
    catalog_shards = split_unit_event_catalog(
        unit_event_catalog,
        event_year,
        max_unit_events=FAILED_TASK_SHARD_UNIT_EVENTS,
    )
    shard_tasks = sorted(record["shard_tasks"], key=lambda task: task["shard_index"])
    if len(shard_tasks) != len(catalog_shards):
        raise RuntimeError(f"Recovery shard-count mismatch for {event_year}")
    for shard_index, (task_record, catalog_shard) in enumerate(
        zip(shard_tasks, catalog_shards), start=1
    ):
        expected_rows = len(catalog_shard) * EXPECTED_PERIODS_PER_UNIT_EVENT
        if (
            int(task_record["shard_index"]) != shard_index
            or int(task_record["shard_count"]) != len(catalog_shards)
            or int(task_record["expected_unit_events"]) != len(catalog_shard)
            or int(task_record["expected_rows"]) != expected_rows
        ):
            raise RuntimeError(f"Recovery shard manifest differs for {event_year}")
    return list(zip(shard_tasks, catalog_shards))

def start_export_task(task_record, event_year, catalog_part):
    export_collection = build_event_year_export(event_year, catalog_part=catalog_part)
    task = ee.batch.Export.table.toDrive(
        collection=export_collection,
        description=task_record["task_description"],
        folder=GEE_DRIVE_FOLDER,
        fileNamePrefix=task_record["output_prefix"],
        fileFormat="CSV",
        selectors=EXPORT_SELECTORS,
    )
    task.start()
    task_record["task_id"] = task.id
    task_record["task_state"] = "READY"
    task_record["submitted_at_utc"] = utc_now()
    task_record["last_checked_utc"] = utc_now()
    return task

submitted = []
for event_year in EVENT_YEARS:
    record = task_manifest["parts"][str(event_year)]
    if not record.get("shard_tasks"):
        record["drive_files"] = discover_drive_files(record)
        if record["drive_files"] or record.get("task_state") in LIVE_STATES:
            continue
        if record.get("task_state") == "FAILED":
            create_recovery_plan(record, event_year)
            task_manifest["updated_at_utc"] = utc_now()
            atomic_write_json(task_manifest, TASK_MANIFEST_PATH)
        elif record.get("task_state") in {"COMPLETED", "CANCELLED"}:
            if record.get("task_state") == "CANCELLED":
                raise RuntimeError(f"Cancelled task requires review for event year {event_year}")
            continue
        else:
            catalog_part = unit_event_catalog.loc[
                unit_event_catalog["event_year"].eq(event_year)
            ]
            task = start_export_task(record, event_year, catalog_part)
            submitted.append(
                {"event_year": event_year, "shard_index": None, "task_id": task.id}
            )
            task_manifest["updated_at_utc"] = utc_now()
            atomic_write_json(task_manifest, TASK_MANIFEST_PATH)
            continue

    for shard_task, catalog_shard in checked_recovery_shards(record, event_year):
        shard_task["drive_files"] = discover_drive_files(shard_task)
        if shard_task["drive_files"] or shard_task.get("task_state") in LIVE_STATES:
            continue
        if shard_task.get("task_state") in TERMINAL_STATES:
            raise RuntimeError(
                f"Recovery shard {shard_task['shard_index']} for {event_year} is "
                f"{shard_task['task_state']}; review it before any resubmission."
            )
        task = start_export_task(shard_task, event_year, catalog_shard)
        submitted.append(
            {
                "event_year": event_year,
                "shard_index": shard_task["shard_index"],
                "shard_count": shard_task["shard_count"],
                "task_id": task.id,
            }
        )
        record["drive_files"] = sorted(
            {
                path
                for task_record in record["shard_tasks"]
                for path in task_record.get("drive_files", [])
            }
        )
        record["task_state"] = aggregate_record_state(record)
        record["last_checked_utc"] = utc_now()
        task_manifest["updated_at_utc"] = utc_now()
        atomic_write_json(task_manifest, TASK_MANIFEST_PATH)

print({"submitted_now": len(submitted), "tasks": submitted})
print("Close Colab if desired. Rerun through monitoring after Earth Engine finishes.")
'''


MONITOR_SOURCE = r'''
tracked_task_ids = sorted(
    {
        task_record["task_id"]
        for record in task_manifest["parts"].values()
        for task_record in (
            record.get("shard_tasks")
            or ([record] if record.get("task_id") else [])
        )
        if task_record.get("task_id")
    }
)
task_statuses = ee.data.getTaskStatus(tracked_task_ids) if tracked_task_ids else []
statuses_by_id = {
    status.get("id"): status for status in task_statuses if status.get("id")
}

status_rows = []
for event_year in EVENT_YEARS:
    record = task_manifest["parts"][str(event_year)]
    shard_tasks = record.get("shard_tasks") or []
    if shard_tasks:
        for shard_task in shard_tasks:
            reconcile_task_record(shard_task, statuses_by_id)
        record["drive_files"] = sorted(
            {
                path
                for shard_task in shard_tasks
                for path in shard_task.get("drive_files", [])
            }
        )
        record["task_state"] = aggregate_record_state(record)
    else:
        reconcile_task_record(record, statuses_by_id)
    record["last_checked_utc"] = utc_now()
    status_rows.append(
        {
            "event_year": event_year,
            "expected_rows": record["expected_rows"],
            "task_state": record["task_state"],
            "recovery_mode": record.get("recovery_mode"),
            "completed_shards": sum(
                task.get("task_state") == "COMPLETED" and bool(task.get("drive_files"))
                for task in shard_tasks
            ),
            "shard_count": len(shard_tasks),
            "drive_file_count": len(record.get("drive_files", [])),
        }
    )

task_manifest["updated_at_utc"] = utc_now()
atomic_write_json(task_manifest, TASK_MANIFEST_PATH)
task_status = pd.DataFrame(status_rows).sort_values("event_year")
display(task_status)
state_counts = task_status["task_state"].value_counts(dropna=False).to_dict()
print({"task_states": state_counts, "all_completed": bool(task_status.task_state.eq("COMPLETED").all())})
'''


ASSEMBLE_SOURCE = r'''
incomplete = [
    int(year)
    for year, record in task_manifest["parts"].items()
    if record.get("task_state") != "COMPLETED" or not record.get("drive_files")
]
if incomplete:
    raise RuntimeError(
        f"Earth Engine exports are not ready for event years {incomplete}. Rerun monitoring later."
    )

for event_year in EVENT_YEARS:
    key = str(event_year)
    record = task_manifest["parts"][key]
    output_path = PARTS_16DAY_DIR / f"event_year={event_year}" / "part.parquet"
    previous = record.get("part_16day", {})
    if reusable_part(output_path, previous, ANALYSIS_SIGNATURE):
        if previous.get("rows") != record["expected_rows"]:
            raise RuntimeError(f"Recorded row count mismatch for {output_path}")
        record["validation_status"] = "reused"
        continue
    if output_path.exists():
        raise RuntimeError(
            f"Existing unvalidated part blocks overwrite: {output_path}. Use FORCE_REBUILD."
        )

    csv_paths = [Path(value) for value in record["drive_files"]]
    if not csv_paths or not all(path.is_file() for path in csv_paths):
        raise RuntimeError(f"Missing CSV shard for event year {event_year}")
    raw = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)
    _require_columns(raw, SIXTEEN_DAY_REQUIRED_COLUMNS, "Earth Engine CSV export")
    raw["evi"] = pd.to_numeric(raw["evi"], errors="coerce")
    raw.loc[np.isclose(raw["evi"], TRANSFER_SENTINEL), "evi"] = np.nan
    raw["valid_pixel_fraction"] = pd.to_numeric(
        raw["valid_pixel_fraction"], errors="coerce"
    ).fillna(0.0)
    catalog_part = unit_event_catalog.loc[
        unit_event_catalog["event_year"].eq(event_year)
    ]
    validated = validate_16day_frame(raw, catalog_part, collection_calendar)
    if len(validated) != record["expected_rows"]:
        raise RuntimeError(
            f"Validated row mismatch for {event_year}: {len(validated)} != {record['expected_rows']}"
        )
    atomic_write_parquet(validated, output_path)
    part_record = file_record(output_path, len(validated))
    part_record.update(
        {
            "analysis_signature": ANALYSIS_SIGNATURE,
            "csv_inputs": [file_record(path) for path in csv_paths],
            "validated_at_utc": utc_now(),
        }
    )
    record["part_16day"] = part_record
    record["validation_status"] = "complete"
    task_manifest["updated_at_utc"] = utc_now()
    atomic_write_json(task_manifest, TASK_MANIFEST_PATH)
    print(f"Validated {event_year}: {len(validated):,} rows")
'''


MONTHLY_SOURCE = r'''
for event_year in EVENT_YEARS:
    record = task_manifest["parts"][str(event_year)]
    input_path = PARTS_16DAY_DIR / f"event_year={event_year}" / "part.parquet"
    if not reusable_part(input_path, record.get("part_16day", {}), ANALYSIS_SIGNATURE):
        raise RuntimeError(f"Missing validated 16-day input for {event_year}")
    output_path = PARTS_MONTHLY_DIR / f"event_year={event_year}" / "part.parquet"
    previous = record.get("part_monthly", {})
    if reusable_part(output_path, previous, ANALYSIS_SIGNATURE):
        continue
    if output_path.exists():
        raise RuntimeError(
            f"Existing unvalidated monthly part blocks overwrite: {output_path}. Use FORCE_REBUILD."
        )
    sixteen_day = pd.read_parquet(input_path)
    monthly = build_monthly_summary(sixteen_day)
    atomic_write_parquet(monthly, output_path)
    monthly_record = file_record(output_path, len(monthly))
    monthly_record.update(
        {
            "analysis_signature": ANALYSIS_SIGNATURE,
            "source_16day_sha256": sha256_file(input_path),
            "validated_at_utc": utc_now(),
        }
    )
    record["part_monthly"] = monthly_record
    task_manifest["updated_at_utc"] = utc_now()
    atomic_write_json(task_manifest, TASK_MANIFEST_PATH)
    print(f"Summarized {event_year}: {len(monthly):,} monthly rows")
'''


FINALIZE_SOURCE = r'''
parts_16day = []
parts_monthly = []
for event_year in EVENT_YEARS:
    record = task_manifest["parts"][str(event_year)]
    path_16day = PARTS_16DAY_DIR / f"event_year={event_year}" / "part.parquet"
    path_monthly = PARTS_MONTHLY_DIR / f"event_year={event_year}" / "part.parquet"
    if not reusable_part(path_16day, record.get("part_16day", {}), ANALYSIS_SIGNATURE):
        raise RuntimeError(f"Invalid final 16-day part: {path_16day}")
    if not reusable_part(path_monthly, record.get("part_monthly", {}), ANALYSIS_SIGNATURE):
        raise RuntimeError(f"Invalid final monthly part: {path_monthly}")
    parts_16day.append(record["part_16day"])
    parts_monthly.append(record["part_monthly"])

total_16day_rows = sum(int(record["rows"]) for record in parts_16day)
total_monthly_rows = sum(int(record["rows"]) for record in parts_monthly)
if total_16day_rows != EXPECTED_16DAY_ROWS:
    raise RuntimeError(f"Final 16-day row mismatch: {total_16day_rows}")
if len(parts_16day) != len(EVENT_YEARS) or len(parts_monthly) != len(EVENT_YEARS):
    raise RuntimeError("Final part-count mismatch")

final_manifest = {
    "created_at_utc": utc_now(),
    "analysis_version": ANALYSIS_VERSION,
    "analysis_signature": ANALYSIS_SIGNATURE,
    "source_signature": EXPECTED_SOURCE_SHA256,
    "input": input_audit,
    "collection_calendar": file_record(OUT_DIR / "collection_calendar.csv", len(collection_calendar)),
    "unit_event_catalog": file_record(OUT_DIR / "unit_event_catalog.parquet", len(unit_event_catalog)),
    "matched_links": file_record(OUT_DIR / "matched_links.parquet", len(matched_links)),
    "parts_16day": parts_16day,
    "parts_monthly": parts_monthly,
    "total_16day_rows": total_16day_rows,
    "total_monthly_rows": total_monthly_rows,
    "event_years": list(EVENT_YEARS),
    "relative_years": list(RELATIVE_YEARS),
}
atomic_write_json(final_manifest, OUT_DIR / "subannual_manifest.json")

report = {
    "stage08_pass": True,
    "created_at_utc": utc_now(),
    "analysis_signature": ANALYSIS_SIGNATURE,
    "pair_links": EXPECTED_PAIR_LINKS,
    "unit_events": EXPECTED_UNIT_EVENTS,
    "periods_per_unit_event": EXPECTED_PERIODS_PER_UNIT_EVENT,
    "rows_16day": total_16day_rows,
    "rows_monthly": total_monthly_rows,
    "event_years": list(EVENT_YEARS),
    "interpretation_boundary": (
        "TMF supplies event year only; calendar month and day of year are not an exact disturbance date."
    ),
}
atomic_write_json(report, OUT_DIR / "subannual_report.json")
report_markdown = f"""# Stage 08 subannual matched extraction report

- Status: PASS
- Analysis signature: `{ANALYSIS_SIGNATURE}`
- Pair links: {EXPECTED_PAIR_LINKS:,}
- Role-specific unit-events: {EXPECTED_UNIT_EVENTS:,}
- Validated 16-day rows: {total_16day_rows:,}
- Monthly rows: {total_monthly_rows:,}
- Event-year window: t-3 through t+5

TMF identifies an event year, not an exact disturbance date. Calendar-month results must be
interpreted as within-year trajectories conditional on annual event classification.
"""
temporary_report = (OUT_DIR / "subannual_report.md").with_name("subannual_report.md.tmp")
temporary_report.write_text(report_markdown, encoding="utf-8")
os.replace(temporary_report, OUT_DIR / "subannual_report.md")
print(json.dumps(report, indent=2))
'''


def build_notebook():
    prefix = "08_MODIS_TMF_subannual_matched_extraction"
    cell_specs = [
        (
            "md",
            """## 1. Purpose and interpretation boundary

Extract quality-controlled MODIS/061/MOD13Q1 16-day EVI for the validated Stage 07 matched
cohort only. The output covers relative years t-3 through t+5 and preserves calendar date,
month and day of year. TMF identifies an event year, not an exact event date; this notebook never
constructs a false event-relative month. Final estimation and figures remain local.""",
            None,
        ),
        (
            "md",
            """## 2. Editable configuration

Set `GEE_PROJECT` to the Google Cloud Project registered for Earth Engine. Defaults otherwise
implement the approved Stage 08 source, QA, grid, time window, identity gates and Drive layout.""",
            None,
        ),
        ("code", CONFIG_SOURCE, ["stage08-config"]),
        (
            "md",
            """## 3. Minimal dependencies

Install only packages absent from the current Colab runtime.""",
            None,
        ),
        ("code", INSTALL_SOURCE, ["stage08-install"]),
        (
            "md",
            """## 4. Drive mount and Earth Engine initialization

Authentication is interactive. Initialization stops before any export when `GEE_PROJECT` is not
explicitly configured.""",
            None,
        ),
        ("code", INITIALIZE_SOURCE, ["stage08-initialize"]),
        (
            "md",
            """## 5. Pure helpers and schemas

These functions enforce catalogue, calendar, 16-day and monthly contracts and use atomic writes
for manifests and Parquet products.""",
            None,
        ),
        ("code", _notebook_helper_source(), ["stage08-helpers"]),
        (
            "md",
            """## 6. Stage 07 input gate and unit-event catalogue

Verify the approved 1.5 MB match file by size, SHA-256, schema and fixed counts. Preserve all
46,984 original links and deduplicate controls only by `(grid_id, event_year)`.""",
            None,
        ),
        ("code", INPUT_GATE_SOURCE, ["stage08-preflight"]),
        (
            "md",
            """## 7. MOD13Q1 calendar and grid audit

Query the authoritative collection calendar before exports. Every year from 2001 through 2023
must contain exactly 23 composites. QA and range masks are applied at native resolution before
aggregation to the fixed 0.05-degree grid.""",
            None,
        ),
        ("code", CALENDAR_AUDIT_SOURCE, ["stage08-calendar-audit"]),
        (
            "md",
            """## 8. Restart manifest and task reconciliation

Reconcile deterministic event-year keys with existing Earth Engine tasks and Drive CSV files.
Changed settings require a separate rebuild namespace; no source or completed task is removed.""",
            None,
        ),
        ("code", MANIFEST_SOURCE, ["stage08-reconcile"]),
        (
            "md",
            """## 9. Submit missing event-year exports

Submit one primary task for each event year 2004-2018. Existing CSV exports and live or completed
tasks are reused. A failed primary task is recovered with deterministic catalogue shards of at
most 750 unit-events; failed recovery shards are never resubmitted automatically.""",
            None,
        ),
        ("code", SUBMISSION_SOURCE, ["stage08-submit"]),
        (
            "md",
            """## 10. Monitor task status

Rerun this cell after Earth Engine processing. It records current primary or recovery-shard states,
requires a Drive file for every completed shard, and never cancels tasks or deletes exports.""",
            None,
        ),
        ("code", MONITOR_SOURCE, ["stage08-monitor"]),
        (
            "md",
            """## 11. Assemble validated 16-day Parquet parts

Assembly starts only after all event years are complete and every required CSV, including recovery
shards, is visible. Transfer sentinels become missing EVI, the full unit-event by audited-date
skeleton is enforced, and one Zstandard Parquet part is written per event year.""",
            None,
        ),
        ("code", ASSEMBLE_SOURCE, ["stage08-assemble-16day"]),
        (
            "md",
            """## 12. Monthly summaries and baseline anomalies

Derive monthly means, medians, coverage and same-calendar-month baselines from validated 16-day
parts. Baselines use t-3, t-2 and t-1 and require at least two finite years.""",
            None,
        ),
        ("code", MONTHLY_SOURCE, ["stage08-monthly"]),
        (
            "md",
            """## 13. Final manifest and validation report

Re-hash every final part, reconcile the nominal 10,183,779-row 16-day skeleton and write machine-
readable plus Markdown reports. This stage creates no publication figures.""",
            None,
        ),
        ("code", FINALIZE_SOURCE, ["stage08-finalize"]),
    ]
    cells = []
    for index, (kind, source, tags) in enumerate(cell_specs):
        if kind == "md":
            cells.append(markdown_cell(source, prefix, index))
        else:
            cells.append(code_cell(source, prefix, index, tags=tags))
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": f"{prefix}.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv=None):
    default_output = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "08_MODIS_TMF_subannual_matched_extraction.ipynb"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args(argv)
    output = write_notebook(args.output)
    print(output)


if __name__ == "__main__":
    main()
