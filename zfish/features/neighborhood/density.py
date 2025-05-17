# %%
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from pydantic import BaseModel

from zfish.features.constants import DensityParams
from zfish.features.neighborhood import aggregation_functions
from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject
from zfish.features.queries import FeatureQuery

if TYPE_CHECKING:
    from zfish.features.typing import LabelImage
    from zfish.roi.spatial_roi import Roi


class DensityQuery(BaseModel, FeatureQuery):
    label_image: str
    delaunay_mask_label: str | None = None
    params: DensityParams = DensityParams()

    def load_resources(self, roi: "Roi") -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "delaunay_mask_label": None
            if self.delaunay_mask_label is None
            else roi.sel(l=self.delaunay_mask_label).drop_dim("c").labels.compute(),
            **asdict(self.params),
        }

    def compute(self, roi: "Roi") -> "pl.DataFrame":
        return get_density_features(**self.load_resources(roi))

default_params = DensityParams()

def get_density_features(
    label_image: "LabelImage",
    delaunay_mask_label: "LabelImage | None" = None,
    radius: tuple[float, ...] = default_params.radius,
    knn_distance: tuple[int, ...] = default_params.knn_distance,
    distance_to_closest_neighbor: bool = default_params.distance_to_closest_neighbor,
    delaunay: tuple[int, ...] = default_params.delaunay,
    touch: tuple[int, ...] = default_params.touch,
    distance_aggfuncs: tuple[int, ...] = default_params.distance_aggfuncs,
    adjacency_aggfuncs: tuple[int, ...] = default_params.adjacency_aggfuncs,
    index_columns: tuple[Literal["label", "label_image"], ...] = ("label",),
) -> "pl.DataFrame":
    nq = NeighborhoodQueryObject.from_labelimage(label_image, delaunay_mask_label)
    results = []
    distance_aggfuncs = [getattr(aggregation_functions, f) for f in distance_aggfuncs]
    adjacency_aggfuncs = [getattr(aggregation_functions, f) for f in adjacency_aggfuncs]

    # Compute object counts in radius
    results.append(
        nq.radius(radius, self_loops=False, distance=False).aggregate(
            adjacency_aggfuncs
        )
    )

    # Compute distance to closest neighbor
    if distance_to_closest_neighbor:
        results.append(
            nq.knn(k=1, self_loops=False, distance=True).aggregate_weights(
                aggregation_functions.Max
            )
        )

    # Compute distances to closest knn neighbors
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
        df = df.with_columns(pl.lit(label_image["l"].item()).alias("label_image"))

    return df.select(pl.col(index_columns), pl.exclude(index_columns))
