from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _estimate_point_spacing(coordinates: np.ndarray) -> tuple[float, float]:
    if len(coordinates) < 2:
        return 0.0, 0.0

    neighbor_count = min(8, len(coordinates))
    model = NearestNeighbors(n_neighbors=neighbor_count)
    model.fit(coordinates)
    distances, _ = model.kneighbors(coordinates)

    nearest = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
    bridge = distances[:, -1]
    return float(np.median(nearest)), float(np.quantile(bridge, 0.9))


def _build_cluster_models(coordinates: np.ndarray, labels: np.ndarray) -> dict[int, dict]:
    models: dict[int, dict] = {}
    for sector_id in sorted({int(label) for label in labels if int(label) != -1}):
        members = coordinates[labels == sector_id]
        if len(members) < 3:
            continue

        center = members.mean(axis=0)
        centered = members - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        major_axis = vh[0]
        minor_axis = vh[1]
        projected_major = centered @ major_axis
        projected_minor = centered @ minor_axis

        models[sector_id] = {
            "center": center,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "major_min": float(projected_major.min()),
            "major_max": float(projected_major.max()),
            "minor_q05": float(np.quantile(projected_minor, 0.05)),
            "minor_q95": float(np.quantile(projected_minor, 0.95)),
        }
    return models


def _reassign_noise_points(coordinates: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if len(coordinates) == 0 or np.all(labels == -1):
        return labels

    nearest_spacing, bridge_spacing = _estimate_point_spacing(coordinates)
    if nearest_spacing <= 0:
        return labels

    cross_axis_limit = max(nearest_spacing * 3.0, 1e-9)
    along_axis_limit = max(bridge_spacing * 3.25, cross_axis_limit)
    updated = labels.copy()
    cluster_models = _build_cluster_models(coordinates, labels)
    if not cluster_models:
        return updated

    for point_index in np.where(labels == -1)[0]:
        point = coordinates[point_index]
        best_match: tuple[float, int] | None = None

        for sector_id, model in cluster_models.items():
            offset = point - model["center"]
            major_position = float(offset @ model["major_axis"])
            minor_position = float(offset @ model["minor_axis"])

            major_gap = 0.0
            if major_position < model["major_min"]:
                major_gap = model["major_min"] - major_position
            elif major_position > model["major_max"]:
                major_gap = major_position - model["major_max"]

            minor_gap = 0.0
            if minor_position < model["minor_q05"]:
                minor_gap = model["minor_q05"] - minor_position
            elif minor_position > model["minor_q95"]:
                minor_gap = minor_position - model["minor_q95"]

            score = (minor_gap / cross_axis_limit) ** 2 + (major_gap / along_axis_limit) ** 2
            if minor_gap <= cross_axis_limit and major_gap <= along_axis_limit:
                if best_match is None or score < best_match[0]:
                    best_match = (score, sector_id)

        if best_match is not None:
            updated[point_index] = best_match[1]

    return updated


def _relabel_sectors_by_position(labels: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    sector_ids = sorted({int(label) for label in labels if int(label) != -1})
    if not sector_ids:
        return labels

    ordering: list[tuple[float, float, int]] = []
    for sector_id in sector_ids:
        members = coordinates[labels == sector_id]
        center = members.mean(axis=0)
        ordering.append((-float(center[1]), float(center[0]), sector_id))

    mapping = {old_sector_id: new_sector_id for new_sector_id, (_, _, old_sector_id) in enumerate(sorted(ordering))}
    return np.array([mapping.get(int(label), -1) if int(label) != -1 else -1 for label in labels], dtype=np.int32)


def cluster_plants(centroids: list[dict], min_cluster_size: int = 10) -> list[dict]:
    if not centroids:
        return []

    coordinates = np.array([[float(item["geo_x"]), float(item["geo_y"])] for item in centroids], dtype=np.float64)
    try:
        try:
            import hdbscan

            labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(coordinates)
        except ModuleNotFoundError:
            from sklearn.cluster import HDBSCAN

            labels = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(coordinates)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to cluster plant centroids with HDBSCAN (min_cluster_size={min_cluster_size})."
        ) from exc

    labels = _reassign_noise_points(coordinates, np.asarray(labels, dtype=np.int32))
    labels = _relabel_sectors_by_position(labels, coordinates)

    clustered: list[dict] = []
    for centroid, label in zip(centroids, labels, strict=False):
        updated = dict(centroid)
        updated["sector_id"] = int(label)
        clustered.append(updated)
    return clustered


def _estimate_axis_tolerance(axis_values: np.ndarray) -> float:
    if len(axis_values) < 2:
        return 0.0

    nearest_distances: list[float] = []
    for axis_value in axis_values:
        deltas = np.abs(axis_values - axis_value)
        non_zero = deltas[deltas > 0]
        if non_zero.size > 0:
            nearest_distances.append(float(non_zero.min()))

    if not nearest_distances:
        return 0.0
    return float(np.median(nearest_distances) * 1.5)


def assign_row_col_within_sector(centroids: list[dict]) -> list[dict]:
    if not centroids:
        return []

    grouped: dict[int, list[dict]] = defaultdict(list)
    for centroid in centroids:
        grouped[int(centroid.get("sector_id", -1))].append(dict(centroid))

    output: list[dict] = []
    for sector_id, group in grouped.items():
        if sector_id == -1:
            for item in group:
                item["row_index"] = None
                item["col_index"] = None
                output.append(item)
            continue

        coordinates = np.array([[float(item["geo_x"]), float(item["geo_y"])] for item in group], dtype=np.float64)
        center = coordinates.mean(axis=0)
        centered = coordinates - center

        if len(group) >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            major_axis = vh[0]
            minor_axis = vh[1]
        else:
            major_axis = np.array([1.0, 0.0], dtype=np.float64)
            minor_axis = np.array([0.0, 1.0], dtype=np.float64)

        local_major = centered @ major_axis
        local_minor = centered @ minor_axis
        minor_tolerance = _estimate_axis_tolerance(local_minor)
        if minor_tolerance <= 0:
            minor_tolerance = 1e-9

        oriented_group: list[dict] = []
        for item, major_value, minor_value in zip(group, local_major, local_minor, strict=False):
            updated = dict(item)
            updated["_local_major"] = float(major_value)
            updated["_local_minor"] = float(minor_value)
            oriented_group.append(updated)

        sorted_group = sorted(oriented_group, key=lambda item: (-float(item["_local_minor"]), float(item["_local_major"])))

        rows: list[list[dict]] = []
        current_row: list[dict] = []
        current_anchor_minor: float | None = None

        for item in sorted_group:
            minor_value = float(item["_local_minor"])
            if current_anchor_minor is None or abs(current_anchor_minor - minor_value) <= minor_tolerance:
                current_row.append(item)
                if current_anchor_minor is None:
                    current_anchor_minor = minor_value
            else:
                rows.append(current_row)
                current_row = [item]
                current_anchor_minor = minor_value

        if current_row:
            rows.append(current_row)

        for row_index, row_items in enumerate(rows, start=1):
            for col_index, item in enumerate(sorted(row_items, key=lambda entry: float(entry["_local_major"])), start=1):
                item.pop("_local_major", None)
                item.pop("_local_minor", None)
                item["row_index"] = row_index
                item["col_index"] = col_index
                output.append(item)

    return output


def format_sector_label(sector_id: int) -> str:
    if sector_id == -1:
        return "NOISE"
    return f"S{sector_id:02d}"
