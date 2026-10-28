"""UbiGraph CoSimHeat Eq. 12 with geographic and optional semantic dual graphs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.spatial import cKDTree

from .config import Config
from .utils import write_json


@dataclass
class SpatialResult:
    garages: list[str]
    adjacency: np.ndarray
    kernel: np.ndarray
    similarity: np.ndarray
    distances_m: np.ndarray
    details: dict


def haversine(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2
            or not np.isfinite(coordinates).all()
            or (np.abs(coordinates[:, 0]) > 90).any() or (np.abs(coordinates[:, 1]) > 180).any()):
        raise ValueError("Coordinates must be finite (latitude, longitude) in degrees")
    lat, lon = np.radians(coordinates).T
    delta_lat = lat[:, None] - lat[None, :]
    delta_lon = lon[:, None] - lon[None, :]
    q = np.sin(delta_lat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(delta_lon / 2) ** 2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(np.clip(q, 0, 1)))


def cosimheat(adjacency: np.ndarray, diffusion: float = 1.0, theta: float = 0.5) -> np.ndarray:
    """Matrix exponentials, not elementwise exp or cosine similarity.

    Weighted graph convention: m=sum(A) is total directed edge mass. T0's second
    term is (1-theta)/n * I (the supplied draft omits I in the displayed T0 but
    defines it in prose). Isolated vertices have absorbing stochastic rows.
    """
    a = np.asarray(adjacency, dtype=np.float64)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
            or not np.isfinite(a).all() or (a < 0).any()):
        raise ValueError("CoSimHeat needs a finite non-negative square adjacency")
    if diffusion < 0 or not 0 <= theta <= 1:
        raise ValueError("Need lambda>=0 and theta in [0,1]")
    degree = a.sum(axis=1)
    n, mass = len(a), degree.sum()
    p = np.divide(a, degree[:, None], out=np.zeros_like(a), where=degree[:, None] > 0)
    isolated = np.flatnonzero(degree == 0)
    p[isolated, isolated] = 1
    t0 = (1 - theta) / n * np.eye(n)
    if mass > 0:
        t0 = t0 + theta * np.outer(degree / mass, degree / mass)
    elif theta == 1:
        t0 = np.eye(n) / n  # explicit degenerate-graph convention
    transition = expm(diffusion / 2 * (p - np.eye(n)))
    s = transition.T @ t0 @ transition
    return np.maximum((s + s.T) / 2, 0)  # roundoff only


def normalized_similarity(s: np.ndarray) -> np.ndarray:
    denominator = np.sqrt(np.maximum(np.diag(s), 1e-15))
    similarity = s / np.outer(denominator, denominator)
    return np.clip(similarity, 0, 1)


def geographic_adjacency(distances: np.ndarray, neighbors: int) -> np.ndarray:
    n = len(distances)
    if n == 1:
        return np.zeros((1, 1))
    k = min(neighbors, n - 1)
    indices = np.argsort(distances + np.eye(n) * 1e20, axis=1)[:, :k]
    selected_distances = np.take_along_axis(distances, indices, axis=1)
    scale = max(float(np.median(selected_distances)), 1.0)
    a = np.zeros_like(distances)
    a[np.arange(n)[:, None], indices] = np.exp(-selected_distances / scale)
    return np.maximum(a, a.T)


def dbscan_keep(coordinates: np.ndarray, radius_m: float, min_samples: int) -> np.ndarray:
    """Density clusters with deterministic border assignment; retain non-noise POIs.

    A per-garage all-noise input falls back to keeping all its POIs, recorded by
    the caller. This helper uses local metric coordinates for sub-city regions.
    """
    radians = np.radians(coordinates)
    xy = np.column_stack([radians[:, 1] * np.cos(radians[:, 0].mean()), radians[:, 0]]) * 6371008.8
    neighborhoods = cKDTree(xy).query_ball_point(xy, radius_m)
    core = np.array([len(ids) >= min_samples for ids in neighborhoods])
    keep = core.copy()
    for i in np.flatnonzero(core):
        keep[neighborhoods[i]] = True
    return keep


def semantic_adjacency(poi: pd.DataFrame) -> np.ndarray:
    """Block dual graph: within-region same-category 1/d; across-region same-category 1."""
    required = {"garagecode", "poi_id", "category", "latitude", "longitude"}
    if not required.issubset(poi) or poi[list(required)].isna().any().any():
        raise ValueError(f"POI table must have complete columns {sorted(required)}")
    if poi.duplicated(["garagecode", "poi_id"]).any():
        raise ValueError("Duplicate per-garage poi_id")
    d = haversine(poi[["latitude", "longitude"]].to_numpy())
    codes, categories = poi.garagecode.to_numpy(), poi.category.to_numpy()
    same_category = categories[:, None] == categories[None, :]
    same_region = codes[:, None] == codes[None, :]
    a = np.where(same_region, 1 / np.maximum(d, 1.0), 1.0) * same_category
    np.fill_diagonal(a, 0)
    return a


def build_spatial(meta: pd.DataFrame, cfg: Config, output: Path | None = None) -> SpatialResult:
    names = sorted(meta.index)
    meta = meta.loc[names]
    distance = haversine(meta[["latitude", "longitude"]].to_numpy())
    details = {"mode": "geographic_adaptation", "uses_target_parking_labels": False,
               "theta": cfg.spatial_theta, "lambda": cfg.spatial_lambda,
               "T0_identity_interpretation": True, "weighted_mass_convention": "sum of directed weights"}
    if cfg.poi_csv:
        poi = pd.read_csv(cfg.path("poi_csv"))
        required = {"garagecode", "poi_id", "category", "latitude", "longitude"}
        if not required.issubset(poi) or poi[list(required)].isna().any().any():
            raise ValueError(f"POI table must have complete columns {sorted(required)}")
        if set(poi.garagecode) != set(names):
            raise ValueError("POI data must cover exactly the metadata garages")
        selected, stats = [], []
        for garage, group in poi.groupby("garagecode", sort=True):
            keep = dbscan_keep(group[["latitude", "longitude"]].to_numpy(),
                               cfg.poi_cluster_radius_m, cfg.poi_min_samples)
            fallback = not keep.any()
            if fallback:
                keep[:] = True
            selected.append(group.loc[keep])
            stats.append({"garage": garage, "input_pois": len(group),
                          "retained_pois": int(keep.sum()), "all_noise_fallback": fallback})
        poi = pd.concat(selected, ignore_index=True)
        # Exact dense expm is intentional for this small research input.
        if len(poi) > 2000:
            raise ValueError("Exact semantic CoSimHeat limited to 2000 POIs; use a sparse approximation for larger data")
        a = semantic_adjacency(poi)
        node_s = cosimheat(a, cfg.spatial_lambda, cfg.spatial_theta)
        membership = np.column_stack([(poi.garagecode == g).to_numpy(float) for g in names])
        membership /= membership.sum(axis=0, keepdims=True)
        kernel = membership.T @ node_s @ membership
        details.update(mode="semantic_dual_graph_adaptation", poi_counts=stats,
                       pooling="mean block kernel; diagonal normalization for region scores")
    else:
        a = geographic_adjacency(distance, cfg.spatial_neighbors)
        kernel = cosimheat(a, cfg.spatial_lambda, cfg.spatial_theta)
    similarity = normalized_similarity(kernel)
    result = SpatialResult(names, a, kernel, similarity, distance, details)
    if output:
        output.mkdir(parents=True, exist_ok=True)
        for filename, array in (("kernel", kernel), ("similarity", similarity), ("distance_m", distance)):
            pd.DataFrame(array, index=names, columns=names).to_csv(output / f"{filename}.csv", index_label="garage")
        np.save(output / "spatial_adjacency.npy", a)
        write_json(output / "spatial_details.json", details)
    return result
