from typing import TYPE_CHECKING

import polars as pl

from zfish.features.neighborhood import aggregation_functions as agg_funcs
from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject

if TYPE_CHECKING:
    from zfish.features.types import LabelImage

RADIUS_NEIGHBORHOODS = [10, 20, 30, 40, 50, 80, 100, 150, 200, 250]

DISTANCE_TO_CLOSEST_NEIGHBOR = True

KNN_DISTANCE_NEIGHBORHOODS = [2, 5, 10, 20, 50, 100, 200]
KNN_AGGFUNCS = [agg_funcs.Mean, agg_funcs.Max]

DELAUNAY_NEIGHBORHOODS = [1]

TOUCH_NEIGHBORHOODS = [1]


def get_density_features(
    lbl_img: "LabelImage", mask_img: "LabelImage | None" = None
) -> "pl.DataFrame":
    nq = NeighborhoodQueryObject.from_labelimage(lbl_img, mask_img)

    # Compute object counts in radius
    radius_counts = nq.radius(
        RADIUS_NEIGHBORHOODS, self_loops=False, distance=False
    ).aggregate(agg_funcs.Count)

    # Compute distance to closest neighbor
    distance_to_closest_neighbor = nq.knn(
        k=1, self_loops=False, distance=True
    ).aggregate_weights(agg_funcs.Max)

    # Compute distances to closest neighbors
    knn_distance_features = nq.knn(
        KNN_DISTANCE_NEIGHBORHOODS, self_loops=False, distance=True
    ).aggregate_weights(KNN_AGGFUNCS)

    # TODO: implement thresholding
    # Compute delaunay neighbor counts
    delaunay_counts = nq.delaunay(DELAUNAY_NEIGHBORHOODS, self_loops=False).aggregate(
        agg_funcs.Count
    )

    # TODO: implement thresholding
    # Compute touch neighbor counts
    touch_counts = nq.touch(
        TOUCH_NEIGHBORHOODS, self_loops=False
    ).aggregate(agg_funcs.Count)
    return pl.concat(
        [
            nq.label,
            radius_counts,
            distance_to_closest_neighbor,
            knn_distance_features,
            delaunay_counts,
            touch_counts,
        ],
        how = 'horizontal'
    )
