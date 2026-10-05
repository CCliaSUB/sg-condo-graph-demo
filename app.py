"""FastAPI backend for the PGIM Singapore graph demonstration.

All data is read from existing artifacts; nothing is retrained or regenerated.
Four datasets are served (``?dataset=``):

* ``R260903`` / ``S260903`` – full Singapore rental / sale graphs. Predictions come
  from ``outputs/predictions/sg_full_260903_final`` (exported by
  ``scripts/export_sg_full.py`` from the fixed-epoch ``final_model`` checkpoints).
  The graph shown is the one the model used: ``size_project`` plus dynamic
  ``same_mrt`` and dynamic ``similar_price``, one snapshot per month.
* ``R260611`` / ``S260611`` – the earlier 804/693-project graphs with six static
  relations and the seed-42 ``report_best`` artifacts.

Every prediction is a held-out *evaluation* prediction for an observed test month,
not a forecast beyond the end of the dataset.
"""

from __future__ import annotations

import json
import math
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(os.environ.get("PGIM_REPO_ROOT") or Path(__file__).resolve().parents[2])
PREDICTIONS = REPO_ROOT / "outputs" / "predictions"
HORIZONS = (1, 12, 24)
SIZE_PROJECT_RELATION = "size_project"
SIZE_LABELS = {
    1: "SZ1 · smallest floor-area quintile",
    2: "SZ2 · second quintile",
    3: "SZ3 · middle quintile",
    4: "SZ4 · fourth quintile",
    5: "SZ5 · largest floor-area quintile",
}

# Colors are paired with a dash pattern so relation types never rely on color alone.
STATIC_RELATIONS = {
    "dist_250": {"label": "Within 250 m", "color": "#d9632b", "dash": "", "level": "project"},
    "same_mrt_250": {"label": "Same MRT within 250 m", "color": "#2567b3", "dash": "8 4", "level": "project"},
    "same_mrt_dist_eps_5": {"label": "Similar MRT distance (±5 m)", "color": "#2e8b57", "dash": "2 4", "level": "project"},
    "same_planning_area": {"label": "Same planning area", "color": "#7b4fa8", "dash": "12 4 2 4", "level": "project"},
    "same_age": {"label": "Same age at transaction", "color": "#a67c00", "dash": "4 2", "level": "project"},
    "same_school_dist_eps_0p01": {"label": "Similar school distance (±0.01 km)", "color": "#c0364a", "dash": "1 3", "level": "project"},
}
SNAPSHOT_RELATIONS = {
    "same_mrt": {
        "label": "Same nearest MRT (≤250 m, opened stations)", "short": "Same MRT", "color": "#2567b3", "dash": "8 4", "level": "project",
        "description": "Projects within 250 m of the same MRT station, using only stations open in the selected month.",
    },
    "similar_price": {
        "label": "Similar price (room-size nodes)", "short": "Similar price", "color": "#c0364a", "dash": "2 3", "level": "size",
        "description": "Room-size nodes whose imputed price per sqft is within 10% (log scale) in the selected month; "
                       "at most 5 neighbours per node. Links room-size nodes, drawn here between their projects.",
    },
}


@dataclass(frozen=True)
class Dataset:
    id: str
    task: str
    coverage: str
    label: str
    unit: str
    data_root: Path
    prediction_template: str  # relative to PREDICTIONS, formatted with horizon
    graph_mode: str  # "static" | "snapshot"
    relations: dict = field(hash=False)
    snapshot_file: str | None = None
    manifest_file: str | None = None
    model_note: str = ""

    def prediction_path(self, horizon: int) -> Path:
        return PREDICTIONS / self.prediction_template.format(h=horizon)

    @property
    def snapshot_path(self) -> Path | None:
        return PREDICTIONS / self.snapshot_file if self.snapshot_file else None


DATASETS = {
    "R260903": Dataset(
        id="R260903", task="rent", coverage="full", label="Rental · full Singapore (R260903)", unit="SGD / sqft / month",
        data_root=REPO_ROOT / "dataset" / "database_R260903",
        prediction_template="sg_full_260903_final/rent_h{h}_seed42_final.csv", graph_mode="snapshot",
        relations=SNAPSHOT_RELATIONS, snapshot_file="sg_full_260903_final/rent_graph_snapshots.csv.gz",
        manifest_file="sg_full_260903_final/manifest.json",
        model_note="Fixed-epoch refit on pre-2020 data (final_model); the test period was not used for model or epoch selection.",
    ),
    "S260903": Dataset(
        id="S260903", task="sale", coverage="full", label="Sale · full Singapore (S260903)", unit="SGD / sqft",
        data_root=REPO_ROOT / "dataset" / "database_S260903",
        prediction_template="sg_full_260903_final/sale_h{h}_seed42_final.csv", graph_mode="snapshot",
        relations=SNAPSHOT_RELATIONS, snapshot_file="sg_full_260903_final/sale_graph_snapshots.csv.gz",
        manifest_file="sg_full_260903_final/manifest.json",
        model_note="Fixed-epoch refit on pre-2020 data (final_model); the test period was not used for model or epoch selection.",
    ),
    "R260611": Dataset(
        id="R260611", task="rent", coverage="ccr", label="Rental · CCR, 804 projects (R260611)", unit="SGD / sqft / month",
        data_root=REPO_ROOT / "dataset" / "database_R260611",
        prediction_template="report_best/seed42/rent_h{h}_seed42_best.csv", graph_mode="static",
        relations=STATIC_RELATIONS, model_note="report_best seed-42 export.",
    ),
    "S260611": Dataset(
        id="S260611", task="sale", coverage="ccr", label="Sale · CCR, 693 projects (S260611)", unit="SGD / sqft",
        data_root=REPO_ROOT / "dataset" / "database_S260611",
        prediction_template="report_best/seed42/sale_h{h}_seed42_best.csv", graph_mode="static",
        relations=STATIC_RELATIONS, model_note="report_best seed-42 export.",
    ),
}
if os.environ.get("PGIM_DATASETS"):  # e.g. "R260903,S260903" to serve a subset on a small host
    DATASETS = {key: DATASETS[key] for key in os.environ["PGIM_DATASETS"].split(",")}
PREFERRED_DEFAULTS = tuple(key for key in ("R260903", "R260611") if key in DATASETS) or tuple(DATASETS)[:1]


# --------------------------------------------------------------------------- helpers
def json_safe(value: Any) -> Any:
    """Recursively convert NumPy/pandas scalars and NaN to plain JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if value is pd.NaT or value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _require(path: Path) -> Path:
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Required demo asset is missing: {_rel(path)}")
    return path


def _round(value: Any, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def dataset_available(ds: Dataset) -> dict:
    missing = [_rel(p) for p in (ds.data_root / "node_id.csv", ds.data_root / "project_info.csv") if not p.exists()]
    missing += [_rel(ds.prediction_path(h)) for h in HORIZONS if not ds.prediction_path(h).exists()]
    if ds.snapshot_path and not ds.snapshot_path.exists():
        missing.append(_rel(ds.snapshot_path))
    return {"available": not missing, "missing": missing}


def default_dataset() -> str:
    for key in PREFERRED_DEFAULTS:
        if dataset_available(DATASETS[key])["available"]:
            return key
    return PREFERRED_DEFAULTS[-1]


def get_dataset(dataset: str | None) -> Dataset:
    key = dataset or default_dataset()
    if key not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset {key!r}. Valid: {list(DATASETS)}")
    return DATASETS[key]


def _validate_horizon(horizon: int) -> int:
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail=f"Horizon must be one of {list(HORIZONS)}; got {horizon}.")
    return horizon


def _parse_relations(ds: Dataset, relations: str | None) -> list[str]:
    """``None`` means every relation; an empty string means no relations."""
    if relations is None:
        return list(ds.relations)
    requested = list(dict.fromkeys(item.strip() for item in relations.split(",") if item.strip()))
    unknown = sorted(set(requested) - set(ds.relations))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown relations for {ds.id}: {unknown}. Valid: {list(ds.relations)}")
    return [item for item in ds.relations if item in requested]  # stable, canonical order


# --------------------------------------------------------------------------- graph tables
@lru_cache(maxsize=None)
def project_table(ds_id: str) -> pd.DataFrame:
    """One deterministic row per project.

    In R/S260611 ``project_info.csv`` repeats a project once per distinct
    ``Condo_Age_2026`` value, which is the renamed age *at transaction*. Static
    fields are constant per project; the age is summarised as a range.
    """
    ds = DATASETS[ds_id]
    raw = pd.read_csv(_require(ds.data_root / "project_info.csv"))
    age_col = "condo_age_at_transaction" if "condo_age_at_transaction" in raw.columns else "Condo_Age_2026"
    # Projects without coordinates (99 in S260903) stay listed: they are graph nodes
    # with predictions, they just cannot be placed on the map.
    raw = raw.dropna(subset=["project_id"])
    raw["project_id"] = raw["project_id"].astype(int)
    raw = raw.sort_values(["project_id", age_col], kind="mergesort")
    table = raw.groupby("project_id", sort=True).agg(
        project_name=("project_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        planning_area=("Planning Area", "first"),
        age_min=(age_col, "min"),
        age_max=(age_col, "max"),
    )
    table.index.name = None
    if not raw["project_id"].duplicated().any():
        # 260903 keeps a single, unspecified transaction's age per project; showing it
        # as "the" age would be misleading, so no age is reported there.
        table[["age_min", "age_max"]] = np.nan
    table["project_id"] = table.index
    predicted = set()
    for horizon in HORIZONS:
        try:
            predicted |= set(prediction_table(ds_id, horizon)["project_id"].unique().tolist())
        except HTTPException:
            pass
    table["has_predictions"] = table.index.isin(predicted)
    return table.sort_values(["project_name", "project_id"], kind="mergesort")


@lru_cache(maxsize=None)
def node_table(ds_id: str) -> pd.DataFrame:
    nodes = pd.read_csv(_require(DATASETS[ds_id].data_root / "node_id.csv"))
    return nodes[["project_id", "size_tier", "node_id"]].astype(int)


@lru_cache(maxsize=None)
def node_lookup(ds_id: str) -> dict[int, tuple[int, int]]:
    """graph node_id -> (project_id, size_tier)."""
    nodes = node_table(ds_id)
    return dict(zip(nodes["node_id"].tolist(), zip(nodes["project_id"].tolist(), nodes["size_tier"].tolist())))


@lru_cache(maxsize=None)
def project_node_maps(ds_id: str) -> tuple[dict[int, int], dict[int, int]]:
    """Graph node IDs differ from project IDs; edges use node IDs."""
    project_nodes = node_table(ds_id).loc[node_table(ds_id)["size_tier"].eq(0)]
    project_to_node = dict(zip(project_nodes["project_id"].tolist(), project_nodes["node_id"].tolist()))
    return project_to_node, {node: project for project, node in project_to_node.items()}


def size_nodes_of(ds_id: str, project_id: int) -> pd.DataFrame:
    nodes = node_table(ds_id)
    return nodes.loc[nodes["project_id"].eq(project_id) & nodes["size_tier"].ne(0)].sort_values("size_tier")


@lru_cache(maxsize=None)
def relation_neighbours(ds_id: str, relation: str) -> dict[int, list[int]]:
    """Unique, undirected project neighbours for one static relation, keyed by project ID.

    The raw edge CSVs contain self-loops and many exact duplicates, so they are
    canonicalised here.
    """
    ds = DATASETS[ds_id]
    edges = pd.read_csv(_require(ds.data_root / "edges" / f"{relation}.csv"), usecols=["node_id", "neig_node_id"])
    _, node_to_project = project_node_maps(ds_id)
    pairs = pd.DataFrame({"a": edges["node_id"].map(node_to_project), "b": edges["neig_node_id"].map(node_to_project)})
    pairs = pairs.dropna().astype(int)
    pairs = pairs.loc[pairs["a"].ne(pairs["b"])]
    pairs = pd.DataFrame({"a": np.minimum(pairs["a"], pairs["b"]), "b": np.maximum(pairs["a"], pairs["b"])}).drop_duplicates()
    both = pd.concat([pairs, pairs.rename(columns={"a": "b", "b": "a"})], ignore_index=True)
    both = both.sort_values(["a", "b"], kind="mergesort")
    return {int(key): group.tolist() for key, group in both.groupby("a", sort=True)["b"]}


@lru_cache(maxsize=None)
def snapshots(ds_id: str) -> dict[str, pd.DataFrame]:
    """Per-month edge lists (graph node IDs, undirected, deduplicated) keyed by ISO month."""
    ds = DATASETS[ds_id]
    frame = pd.read_csv(
        _require(ds.snapshot_path),
        dtype={"relation": "category", "node_id": "int32", "neig_node_id": "int32", "timestep": "int16", "date": "category"},
    )
    return {str(month): group.drop(columns=["date"]).reset_index(drop=True) for month, group in frame.groupby("date", sort=True, observed=True)}


def available_months(ds: Dataset) -> list[str]:
    return list(snapshots(ds.id)) if ds.graph_mode == "snapshot" else []


def _haversine_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


def project_payload(row: pd.Series) -> dict:
    return {
        "project_id": int(row["project_id"]),
        "name": str(row["project_name"]),
        "latitude": None if pd.isna(row["latitude"]) else float(row["latitude"]),
        "longitude": None if pd.isna(row["longitude"]) else float(row["longitude"]),
        "planning_area": None if pd.isna(row["planning_area"]) else str(row["planning_area"]),
        "age_at_transaction": None if pd.isna(row["age_min"]) else {
            "min": _round(row["age_min"], 1), "max": _round(row["age_max"], 1),
        },
        "has_predictions": bool(row["has_predictions"]),
    }


def _get_project(ds: Dataset, project_id: int) -> pd.Series:
    table = project_table(ds.id)
    if project_id not in table.index:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found in {ds.id}.")
    return table.loc[project_id]


# --------------------------------------------------------------------------- predictions
BASE_COLUMNS = ["dataset", "seed", "trial", "forecast_horizon", "target_timestep", "date",
                "node_id", "project_id", "size_tier", "ground_truth", "prediction"]
OPTIONAL_COLUMNS = ["prediction_source", "checkpoint"]


@lru_cache(maxsize=None)
def prediction_table(ds_id: str, horizon: int) -> pd.DataFrame:
    """Load one artifact once. Multiple seeds/trials/horizons are rejected, never merged."""
    _validate_horizon(horizon)
    path = _require(DATASETS[ds_id].prediction_path(horizon))
    header = pd.read_csv(path, nrows=0).columns
    frame = pd.read_csv(path, usecols=BASE_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in header])
    for column in ("seed", "trial", "forecast_horizon", "dataset"):
        if frame[column].nunique(dropna=False) > 1:
            raise HTTPException(status_code=503, detail=f"{path.name} mixes several values of '{column}'.")
    if int(frame["forecast_horizon"].iloc[0]) != horizon:
        raise HTTPException(status_code=503, detail=f"{path.name} does not contain horizon {horizon}.")
    if str(frame["dataset"].iloc[0]) != DATASETS[ds_id].task:
        raise HTTPException(status_code=503, detail=f"{path.name} is not a {DATASETS[ds_id].task} artifact.")
    if frame.duplicated(["node_id", "date"]).any():
        raise HTTPException(status_code=503, detail=f"{path.name} has duplicate node/date rows.")
    # Keep dates as ISO strings: no timezone conversion anywhere.
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["date"])
    frame = frame.sort_values(["project_id", "size_tier", "date"], kind="mergesort").reset_index(drop=True)
    # One distinct value per artifact: categories keep small hosts under their memory limit.
    for column in ["dataset", "trial"] + [c for c in OPTIONAL_COLUMNS if c in frame]:
        frame[column] = frame[column].astype("category")
    # ISO dates sort chronologically, so an ordered category keeps min()/max() correct.
    frame["date"] = pd.Categorical(frame["date"], categories=sorted(frame["date"].unique()), ordered=True)
    return frame


@lru_cache(maxsize=None)
def prediction_index(ds_id: str, horizon: int) -> dict[int, pd.DataFrame]:
    return {int(key): group for key, group in prediction_table(ds_id, horizon).groupby("project_id", sort=False)}


@lru_cache(maxsize=None)
def manifest(ds_id: str) -> dict:
    ds = DATASETS[ds_id]
    path = PREDICTIONS / ds.manifest_file if ds.manifest_file else None
    return json.loads(path.read_text()) if path and path.exists() else {}


@lru_cache(maxsize=None)
def provenance(ds_id: str, horizon: int) -> dict:
    ds = DATASETS[ds_id]
    frame = prediction_table(ds_id, horizon)
    first = frame.iloc[0]
    err = frame["prediction"] - frame["ground_truth"]
    ss_res, ss_tot = float((err ** 2).sum()), float(((frame["ground_truth"] - frame["ground_truth"].mean()) ** 2).sum())
    source = first.get("checkpoint") if "checkpoint" in frame else None
    if source is None or pd.isna(source) or not str(source):
        source = first.get("prediction_source")
    extra = manifest(ds_id).get(f"{ds.task}_h{horizon}", {})
    return {
        "dataset": ds.id,
        "target": ds.task,
        "seed": int(first["seed"]),
        "trial": str(first["trial"]),
        "horizon_months": horizon,
        "artifact": _rel(ds.prediction_path(horizon)),
        "prediction_source": None if source is None or pd.isna(source) else str(source),
        "kind": "held-out test-set evaluation predictions",
        "selection": (extra.get("selection") or ds.model_note).rstrip(".") + ".",
        "date_range": [frame["date"].min(), frame["date"].max()],
        "rows": len(frame),
        "test_metrics": {
            "rmse": round(math.sqrt(ss_res / len(frame)), 4),
            "mae": round(float(err.abs().mean()), 4),
            "mape_pct": round(float((err.abs() / frame["ground_truth"].abs()).mean() * 100), 2),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
        },
    }


@lru_cache(maxsize=None)
def horizon_coverage(ds_id: str) -> dict:
    """Check whether the horizon artifacts evaluate the same node/date cells."""
    keys, summary = {}, {}
    for horizon in HORIZONS:
        try:
            frame = prediction_table(ds_id, horizon)
        except HTTPException as error:
            summary[str(horizon)] = {"available": False, "detail": error.detail}
            continue
        keys[horizon] = set(zip(frame["node_id"].tolist(), frame["date"].tolist()))
        summary[str(horizon)] = {
            "available": True, "rows": len(frame), "nodes": int(frame["node_id"].nunique()),
            "projects": int(frame["project_id"].nunique()),
            "date_range": [frame["date"].min(), frame["date"].max()],
        }
    shared = set.intersection(*keys.values()) if keys else set()
    union = set.union(*keys.values()) if keys else set()
    return {
        "per_horizon": summary,
        "shared_cells": len(shared),
        "union_cells": len(union),
        "identical_cells": bool(keys) and len(shared) == len(union),
    }


# --------------------------------------------------------------------------- app
def warm_caches() -> None:
    """Load tables in the background so first requests are not slow. Default dataset first."""
    order = [default_dataset()] + [key for key in DATASETS if key != default_dataset()]
    for key in order:
        ds = DATASETS[key]
        if not dataset_available(ds)["available"]:
            continue
        try:
            project_table(key)
            if ds.graph_mode == "static":
                for relation in ds.relations:
                    relation_neighbours(key, relation)
            else:
                snapshots(key)
            for horizon in HORIZONS:
                prediction_index(key, horizon)
                provenance(key, horizon)
            horizon_coverage(key)
        except HTTPException:
            pass  # surfaced as 503 on the relevant endpoint


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("PGIM_SKIP_WARMUP") != "1":
        threading.Thread(target=warm_caches, name="pgim-warmup", daemon=True).start()
    yield


app = FastAPI(title="PGIM Graph Map API", version="3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DatasetParam = Annotated[str | None, Query(description=f"One of {list(DATASETS)}; defaults to the best available")]


@app.get("/health")
def health():
    return json_safe({
        "status": "ok", "default_dataset": default_dataset(),
        "datasets": {key: dataset_available(ds)["available"] for key, ds in DATASETS.items()},
    })


@app.get("/datasets")
def datasets():
    return json_safe({
        "default": default_dataset(),
        "datasets": [
            {"id": ds.id, "task": ds.task, "coverage": ds.coverage, "label": ds.label, "unit": ds.unit,
             "graph_mode": ds.graph_mode, **dataset_available(ds)}
            for ds in DATASETS.values()
        ],
    })


@app.get("/metadata")
def metadata(dataset: DatasetParam = None):
    ds = get_dataset(dataset)
    status = dataset_available(ds)
    if not status["available"]:
        raise HTTPException(status_code=503, detail=f"{ds.id} is not available; missing: {status['missing']}")
    return json_safe({
        "dataset": ds.id,
        "dataset_label": ds.label,
        "task": ds.task,
        "coverage": ds.coverage,
        "unit": ds.unit,
        "graph_mode": ds.graph_mode,
        "months": available_months(ds),
        "relations": [{"id": key, **value} for key, value in ds.relations.items()],
        "size_relation": {"id": SIZE_PROJECT_RELATION, "label": "Room-size node → project (size_project)"},
        "horizons": list(HORIZONS),
        "size_tiers": [{"id": key, "label": value} for key, value in SIZE_LABELS.items()],
        "model_note": ds.model_note,
        "coverage_check": horizon_coverage(ds.id),
    })


@app.get("/projects")
def projects(
    dataset: DatasetParam = None,
    q: Annotated[str | None, Query(max_length=100)] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 5000,
):
    ds = get_dataset(dataset)
    table = project_table(ds.id)
    if q and q.strip():
        table = table.loc[table["project_name"].str.contains(q.strip(), case=False, na=False, regex=False)]
    return json_safe({"dataset": ds.id, "total": len(table), "projects": [project_payload(row) for _, row in table.head(limit).iterrows()]})


def _nearest(ds: Dataset, selected: pd.Series, candidates: list[int], limit: int) -> list[int]:
    """Nearest neighbours first, ties broken by project ID: stable across requests.

    Projects without coordinates sort last (by project ID).
    """
    if not candidates:
        return []
    rows = project_table(ds.id).loc[candidates]
    dist = _haversine_m(selected["latitude"], selected["longitude"], rows["latitude"].to_numpy(dtype=float), rows["longitude"].to_numpy(dtype=float))
    keyed = [(math.isnan(d), 0.0 if math.isnan(d) else d, pid) for d, pid in zip(dist.tolist(), candidates)]
    return [pid for *_, pid in sorted(keyed)[:limit]]


def _static_neighbours(ds: Dataset, project_id: int, relation: str) -> tuple[list[int], dict]:
    return relation_neighbours(ds.id, relation).get(project_id, []), {}


def _snapshot_neighbours(ds: Dataset, project_id: int, relation: str, month_edges: pd.DataFrame, size_ids: set[int]):
    """Neighbouring projects for one relation in one monthly snapshot.

    For ``similar_price`` (size ↔ size) the size-level pairs are kept so the UI
    can say which room-size nodes are linked.
    """
    lookup = node_lookup(ds.id)
    edges = month_edges.loc[month_edges["relation"].eq(relation)]
    if ds.relations[relation]["level"] == "project":
        selected_node = project_node_maps(ds.id)[0][project_id]
        hit = edges.loc[edges["node_id"].eq(selected_node) | edges["neig_node_id"].eq(selected_node)]
        others = np.where(hit["node_id"].to_numpy() == selected_node, hit["neig_node_id"].to_numpy(), hit["node_id"].to_numpy())
        return sorted({lookup[int(n)][0] for n in others if int(n) in lookup} - {project_id}), {}
    hit = edges.loc[edges["node_id"].isin(size_ids) | edges["neig_node_id"].isin(size_ids)]
    pairs: dict[int, list[dict]] = {}
    for a, b in zip(hit["node_id"].tolist(), hit["neig_node_id"].tolist()):
        own, other = (a, b) if a in size_ids else (b, a)
        other_project, other_tier = lookup[other]
        pairs.setdefault(other_project, []).append({
            "from_node": own, "from_tier": lookup[own][1], "to_node": other, "to_tier": other_tier,
        })
    for items in pairs.values():
        items.sort(key=lambda item: (item["from_tier"], item["to_tier"], item["to_node"]))
    internal = pairs.pop(project_id, [])
    return sorted(pairs), {"pairs": pairs, "internal": internal}


@app.get("/graph/{project_id}")
def graph(
    project_id: int,
    dataset: DatasetParam = None,
    relations: Annotated[str | None, Query(description="Comma-separated relation IDs; omit for all, empty for none")] = None,
    month: Annotated[str | None, Query(description="Snapshot month (YYYY-MM-01) for dynamic graphs; defaults to latest")] = None,
    limit_per_relation: Annotated[int, Query(ge=1, le=200)] = 40,
):
    ds = get_dataset(dataset)
    selected = _get_project(ds, project_id)
    requested = _parse_relations(ds, relations)
    projects_df = project_table(ds.id)
    project_to_node, _ = project_node_maps(ds.id)
    size_nodes = size_nodes_of(ds.id, project_id)
    size_ids = set(size_nodes["node_id"].tolist())

    month_edges = None
    if ds.graph_mode == "snapshot":
        months = available_months(ds)
        month = month or months[-1]
        if month not in snapshots(ds.id):
            raise HTTPException(status_code=400, detail=f"Month must be one of {months[0]} … {months[-1]} (first of month); got {month!r}.")
        month_edges = snapshots(ds.id)[month]
    else:
        month = None

    neighbour_relations: dict[int, list[str]] = {}
    size_pairs: dict[int, list[dict]] = {}
    per_size_node: dict[int, int] = {}
    summary = []
    for relation in requested:
        if month_edges is None:
            candidates, extra = _static_neighbours(ds, project_id, relation)
        else:
            candidates, extra = _snapshot_neighbours(ds, project_id, relation, month_edges, size_ids)
        candidates = [pid for pid in candidates if pid in projects_df.index]
        shown = _nearest(ds, selected, candidates, limit_per_relation)
        for pid in shown:
            neighbour_relations.setdefault(pid, []).append(relation)
        if "pairs" in extra:
            for pid in shown:
                size_pairs[pid] = extra["pairs"][pid]
            for items in list(extra["pairs"].values()) + [extra["internal"]]:
                for item in items:
                    per_size_node[item["from_node"]] = per_size_node.get(item["from_node"], 0) + 1
        summary.append({
            "id": relation, "level": ds.relations[relation]["level"],
            "neighbours_total": len(candidates), "neighbours_shown": len(shown),
            **({"size_edges_total": sum(len(v) for v in extra["pairs"].values()) + len(extra["internal"]),
                "within_project_edges": len(extra["internal"])} if "pairs" in extra else {}),
        })

    nodes = []
    for pid in sorted(neighbour_relations):
        payload = project_payload(projects_df.loc[pid])
        payload["node_id"] = project_to_node.get(pid)
        payload["relations"] = neighbour_relations[pid]
        nodes.append(payload)
    edges = []
    for pid in sorted(neighbour_relations):
        for relation in neighbour_relations[pid]:
            edge = {"source": project_id, "target": pid, "relation": relation}
            if ds.relations[relation]["level"] == "size":
                edge["pairs"] = [{"from_tier": p["from_tier"], "to_tier": p["to_tier"]} for p in size_pairs.get(pid, [])]
            edges.append(edge)

    # size_project edges: static in *260611; in snapshots they exist only once the project is built.
    active_size_edges = set()
    if month_edges is not None:
        sp = month_edges.loc[month_edges["relation"].eq(SIZE_PROJECT_RELATION)]
        sp = sp.loc[sp["node_id"].isin(size_ids) | sp["neig_node_id"].isin(size_ids)]
        active_size_edges = set(sp["node_id"].tolist()) | set(sp["neig_node_id"].tolist())
    predicted_nodes = set()
    for horizon in HORIZONS:
        try:
            group = prediction_index(ds.id, horizon).get(project_id)
        except HTTPException:
            continue
        if group is not None:
            predicted_nodes |= set(group["node_id"].tolist())

    selected_payload = project_payload(selected)
    selected_payload["node_id"] = project_to_node.get(project_id)
    size_payload = []
    for row in size_nodes.itertuples(index=False):
        node_id = int(row.node_id)
        size_payload.append({
            "node_id": node_id, "size_tier": int(row.size_tier), "label": SIZE_LABELS.get(int(row.size_tier)),
            "has_predictions": node_id in predicted_nodes,
            "size_project_active": True if month_edges is None else node_id in active_size_edges,
            "similar_price_neighbours": per_size_node.get(node_id, 0) if "similar_price" in requested and month_edges is not None else None,
        })
    return json_safe({
        "dataset": ds.id,
        "graph_mode": ds.graph_mode,
        "month": month,
        "selected": selected_payload,
        "relations": requested,
        "relation_summary": summary,
        "limit_per_relation": limit_per_relation,
        "nodes": nodes,
        "edges": edges,
        "size_nodes": size_payload,
        "size_edges": [
            {"source_node": item["node_id"], "target_node": project_to_node.get(project_id), "relation": SIZE_PROJECT_RELATION,
             "active": item["size_project_active"]}
            for item in size_payload
        ],
    })


@app.get("/predictions/{project_id}")
def predictions(project_id: int, dataset: DatasetParam = None, horizon: Annotated[int, Query()] = 12):
    ds = get_dataset(dataset)
    _validate_horizon(horizon)
    selected = _get_project(ds, project_id)
    group = prediction_index(ds.id, horizon).get(project_id)

    series = []
    for node in size_nodes_of(ds.id, project_id).itertuples(index=False):
        rows = group.loc[group["node_id"].eq(node.node_id)] if group is not None else None
        points = []
        if rows is not None:
            for row in rows.itertuples(index=False):
                actual, predicted = _round(row.ground_truth), _round(row.prediction)
                pct = None if actual in (None, 0) or predicted is None else round((predicted - actual) / actual * 100, 2)
                points.append({
                    "date": row.date, "target_timestep": int(row.target_timestep),
                    "actual": actual, "prediction": predicted, "pct_error": pct,
                })
        series.append({
            "node_id": int(node.node_id), "size_tier": int(node.size_tier),
            "label": SIZE_LABELS.get(int(node.size_tier), f"SZ{node.size_tier}"),
            "n_points": len(points), "last_evaluated": points[-1] if points else None, "points": points,
        })
    return json_safe({
        "dataset": ds.id,
        "project_id": project_id,
        "project_name": str(selected["project_name"]),
        "horizon": horizon,
        "unit": ds.unit,
        "has_predictions": any(item["n_points"] for item in series),
        "series": series,
        "provenance": provenance(ds.id, horizon),
    })


# --------------------------------------------------------------------------- single-process deployment
# ``uvicorn app:site`` serves the built frontend at ``/`` and this API under ``/api``,
# the same paths the dev-server proxy uses. A mounted app's lifespan does not run,
# so ``site`` starts the warm-up itself.
from fastapi.staticfiles import StaticFiles  # noqa: E402

STATIC_DIR = Path(os.environ.get("PGIM_STATIC_DIR") or Path(__file__).resolve().parents[1] / "dist")
site = FastAPI(title="PGIM Graph Map", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
site.mount("/api", app)
if STATIC_DIR.is_dir():
    site.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
