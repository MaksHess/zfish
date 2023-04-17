from typing import Any

import numpy as np
import polars as pl


def _napari_centroids(df: pl.DataFrame, centroid: str = "Centroid") -> np.array:
    DIMS = ("z", "y", "x")
    ndim = len(df["Centroid"].struct.fields)
    sel = [pl.col(d) for d in DIMS[-ndim:]]
    return df.select(centroid).unnest(centroid).select(sel).to_numpy()


def napari_centroids(
    df: pl.DataFrame,
    features=["PhysicalSize", "Roundness", "Elongation"],
) -> dict[str, Any]:
    return {
        "data": _napari_centroids(df),
        "size": 5,
        "face_color": features[2],
        "edge_color": "white",
        # "shading": "spherical",
        "features": df.select(features).to_pandas().copy(),
    }


def napari_principal_axes(
    df: pl.DataFrame, scaler=lambda x: 2 * np.sqrt(x)
) -> tuple[dict[str, Any], ...]:
    ndims = len(df["PrincipalMoments"].struct.fields)
    if ndims == 2:
        colors = ["blue", "red"]
        pa_colss = (("a-y", "a-x"), ("b-y", "b-x"))
        pm_cols = ("x", "y")
    else:
        colors = ["blue", "green", "red"]
        pa_colss = (("a-z", "a-y", "a-x"), ("b-z", "b-y", "b-x"), ("c-z", "c-y", "c-x"))
        pm_cols = ("x", "y", "z")
    axes = []
    for color, pa_cols, pm_col in zip(colors, pa_colss, pm_cols):
        direction = (
            df.select("PrincipalAxes")
            .unnest("PrincipalAxes")
            .select([pl.col(e) for e in pa_cols])
            .to_numpy()
        )
        centroids = _napari_centroids(df)
        length_scaler = scaler(
            df.select("PrincipalMoments")
            .unnest("PrincipalMoments")
            .select(pm_col)
            .to_numpy()
        )
        print(centroids.shape, length_scaler.shape)
        data = np.stack(
            np.broadcast_arrays(centroids, direction * length_scaler), axis=1
        )
        axes.append({"data": data, "edge_color": color})
    return tuple(axes)
