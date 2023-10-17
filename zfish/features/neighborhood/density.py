from typing import TYPE_CHECKING, Literal

import polars as pl

from zfish.features.neighborhood import aggregation_functions as agg_funcs
from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject

if TYPE_CHECKING:
    from zfish.features.types import LabelImage

RADIUS_NEIGHBORHOODS = tuple([10, 20, 30, 40, 50, 80, 100, 150, 200, 250])
RADIUS_AGGFUNCS = tuple([agg_funcs.Count])

DISTANCE_TO_CLOSEST_NEIGHBOR = True

KNN_DISTANCE_NEIGHBORHOODS = tuple([2, 5, 10, 20, 50, 100, 200])
KNN_AGGFUNCS = tuple([agg_funcs.Mean, agg_funcs.Max])

DELAUNAY_NEIGHBORHOODS = tuple([1])
DELAUNAY_AGGFUNCS = tuple([agg_funcs.Count])

TOUCH_NEIGHBORHOODS = tuple([1])
TOUCH_AGGFUNCS = tuple([agg_funcs.Count])

DISTANCE_AGGFUNCS = tuple([agg_funcs.Mean, agg_funcs.Max])
ADJACENCY_AGGFUNCS = tuple([agg_funcs.Count])


def get_density_features(
    label_image: "LabelImage",
    delaunay_mask_label: "LabelImage | None" = None,
    radius: tuple[float, ...] = RADIUS_NEIGHBORHOODS,
    knn_distance: tuple[int, ...] = KNN_DISTANCE_NEIGHBORHOODS,
    distance_to_closest_neighbor: bool = True,
    delaunay: tuple[int, ...] = DELAUNAY_NEIGHBORHOODS,
    touch: tuple[int, ...] = TOUCH_NEIGHBORHOODS,
    distance_aggfuncs: tuple[int, ...] = DISTANCE_AGGFUNCS,
    adjacency_aggfuncs: tuple[int, ...] = ADJACENCY_AGGFUNCS,
    index_columns: tuple[Literal["label", "label_image"], ...] = ("label",),
) -> "pl.DataFrame":
    nq = NeighborhoodQueryObject.from_labelimage(label_image, delaunay_mask_label)
    results = []

    # Compute object counts in radius
    results.append(
        nq.radius(radius, self_loops=False, distance=False).aggregate(
            adjacency_aggfuncs
        )
    )

    if distance_to_closest_neighbor:
        # Compute distance to closest neighbor
        results.append(
            nq.knn(k=1, self_loops=False, distance=True).aggregate_weights(
                agg_funcs.Max
            )
        )

    # Compute distances to closest neighbors
    results.append(
        nq.knn(knn_distance, self_loops=False, distance=True).aggregate_weights(
            distance_aggfuncs
        )
    )

    # TODO: implement thresholding
    # Compute delaunay neighbor counts
    results.append(
        nq.delaunay(delaunay, self_loops=False).aggregate(adjacency_aggfuncs)
    )

    # TODO: implement thresholding
    # Compute touch neighbor counts
    results.append(nq.touch(touch, self_loops=False).aggregate(adjacency_aggfuncs))

    df = pl.concat(
        results,
        how="horizontal",
    )
    
    if "label" in index_columns:
        df = df.with_columns(nq.label.select(pl.col("label")))
        
    if "label_image" in index_columns:
        df = df.with_columns(pl.lit(label_image['l'].item()).alias("label_image"))

    return df.select(pl.col(index_columns), pl.exclude(index_columns))