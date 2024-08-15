from itertools import combinations
from typing import Literal

import numpy as np
import polars as pl
from spatial_image import to_spatial_image

from zfish.features.constants import (
    ColocalizationFeature,
    DistanceFeature,
    DistanceFunction,
    IntensityFeature,
    LabelFeature,
)
from zfish.multi_table.raster_io import (
    ImageQueryDask,
    LabelImageQueryDask,
    _bbx_scaler_to_m,
)
from zfish.multi_table.tables_io import Resources
from zfish.polars.column_nesting import unnest

DASK_CHUNKS = (-1, -1, -1)
LABEL_FEATURE = pl.Enum(list(LabelFeature))
DISTANCE_FEATURE = pl.Enum(list(DistanceFeature))
DISTANCE_TRANSFORM = pl.Enum(list(DistanceFunction))
INTENSITY_FEATURE = pl.Enum(list(IntensityFeature))
COLOCALIZATION_FEATURE = pl.Enum(list(ColocalizationFeature))


label_feature_queries = {
    "idx.m": pl.UInt8,
    "idx.roi": pl.Categorical,
    "idx.o": pl.Categorical,
    "features": pl.List(LABEL_FEATURE),
}

intensity_feature_queries = {
    "idx.m": pl.UInt8,
    "idx.roi": pl.Categorical,
    "idx.o": pl.Categorical,
    "idx.c": pl.Categorical,
    "features": pl.List(INTENSITY_FEATURE),
}

colocalization_feature_queries = {
    "idx.m": pl.UInt8,
    "idx.roi": pl.Categorical,
    "idx.o": pl.Categorical,
    "idx.c.0": pl.Categorical,
    "idx.c.1": pl.Categorical,
    "features": pl.List(COLOCALIZATION_FEATURE),
}

distance_feature_queries = {
    "idx.m": pl.UInt8,
    "idx.roi": pl.Categorical,
    "idx.o": pl.Categorical,
    "idx.o.to": pl.Categorical,
    "idx.label.to": pl.Categorical,
    "idx.dtf": DISTANCE_TRANSFORM,
    "features": pl.List(DISTANCE_FEATURE),
}

nhood_type = pl.Enum(["KNN", "RADIUS", "DELAUNAY", "TOUCH"])
nhood_r = pl.Float64
nhood_k = pl.Int32
nhood_n_removed = pl.Int32
nhood_self_loop = pl.Boolean
nhood_distance = pl.Boolean
nhood_kernel_function = pl.Enum(["uniform", "triangular", "gaussian"])

neighborhood_queries = {
    "type": nhood_type,
    "self_loop": pl.Boolean,
    "radius": pl.Float64,  # Optional
    "k": pl.Int32,  # Optional
    "distance": pl.Boolean,  # Optional
    "kernel_function": nhood_kernel_function,  # Optional
    "n_removed": pl.Int32,  # Optional
}

density_count_feature_queries = {
    "idx.m": pl.UInt8,
    "idx.roi": pl.Categorical,
    "idx.nhood": pl.Categorical,
}

density_distance_feature_queries = {
    "idx.m": pl.UInt8,
    "idx.roi": pl.Categorical,
    "idx.nhood": pl.Categorical,
}


def _create_virtual_label_images(df_label_images, df_multiscale_levels):
    persistent_levels = df_label_images["idx.m"].unique().to_list()
    assert len(persistent_levels) == 1, "`df_label_images` contains multiple levels!"
    persistent_level = persistent_levels[0]
    target_levels = df_multiscale_levels["idx.m"].to_list()
    lowest_level = max(target_levels)
    df_resample_scale_factors = (
        _bbx_scaler_to_m(df_multiscale_levels, persistent_level, lowest_level)
        .pipe(unnest)
        .select(
            pl.col("idx.m"),
            pl.concat_list(pl.col(f"bbx.scaler.{d}") for d in ["z", "y", "x"]).alias(
                "resample_scale_factors"
            ),
        )
        .with_columns(
            pl.when(pl.col("idx.m") == persistent_level)
            .then(None)
            .otherwise(pl.col("resample_scale_factors"))
            .alias("resample_scale_factors")
        )
    )

    dfs = []
    for target_level in target_levels:
        dfs.append(
            df_label_images.with_columns(
                pl.col("idx.m")
                .replace({persistent_level: target_level})
                .cast(df_label_images["idx.m"].dtype)
            ).drop("resample_scale_factors")
        )
    return pl.concat(dfs).join(df_resample_scale_factors, on="idx.m", coalesce=True)


def _construct_correlation_queries(
    resources: Resources,
    levels: tuple[int, ...] | None = None,
    wells: tuple[str, ...] | None = None,
    rois: tuple[str, ...] | None = None,
    object_types: tuple[str, ...] | None = None,
    channel_pairs: tuple[tuple[str, str], ...] | None = None,
    features: tuple[str, ...] | None = None,
    return_queries: Literal["invalid", "any_valid", "all_valid"] = "invalid",
):
    if levels is None:
        levels = resources.multiscale_levels["idx.m"].to_list()
    if wells is None:
        wells = resources.wells["idx.well"].to_list()
    if rois is None:
        rois = resources.rois.filter(pl.col("well").is_in(wells))["idx.roi"].to_list()
    if object_types is None:
        object_types = resources.object_types["idx.o"].to_list()
    if channel_pairs is None:
        channel_pairs = list(combinations(resources.channels["idx.c"], 2))
    if features is None:
        features = tuple(COLOCALIZATION_FEATURE.categories)

    df_resources = (
        resources.images.group_by(["idx.m", "idx.roi"], maintain_order=True)
        .agg("idx.c")
        .join(
            resources.label_images.group_by(
                ["idx.m", "idx.roi"], maintain_order=True
            ).agg("idx.o"),
            on=["idx.m", "idx.roi"],
            coalesce=True,
            validate="1:1",
        )
        .select(
            pl.col("idx.m", "idx.roi"),
            pl.col("idx.c").alias("c.available"),
            pl.col("idx.o").alias("o.available"),
        )
    )

    df_possible_queries = (
        pl.DataFrame(
            data=[
                (
                    levels,
                    rois,
                    object_types,
                    [c[0] for c in channel_pairs],
                    [c[1] for c in channel_pairs],
                    features,
                )
            ],
            schema=list(colocalization_feature_queries.keys()),
            orient="row",
        )
        .explode("idx.m")
        .explode("idx.roi")
        .explode("idx.o")
        .explode("idx.c.0", "idx.c.1")
        # .explode("idx.label:s")
        # .explode("idx.bbx")
        .cast(colocalization_feature_queries)
    )

    df_queries = (
        df_possible_queries.join(df_resources, on=["idx.m", "idx.roi"], how="left")
        .with_columns(
            pl.col("o.available").list.contains(pl.col("idx.o")),
            pl.col("c.available")
            .list.set_intersection(pl.concat_list("idx.c.0", "idx.c.1"))
            .list.len()
            == 2,
        )
        .with_columns(
            (pl.col("c.available") & pl.col("o.available")).alias("r.available")
        )
        # .filter(pl.col("r.available"))
        .group_by(["idx.m", "idx.roi"], maintain_order=True)
        .agg(
            *[e for e in df_possible_queries.columns if e not in ["idx.m", "idx.roi"]],
            pl.col("r.available"),
            pl.col("r.available").any().alias("any_queries"),
            pl.col("r.available").all().alias("all_queries"),
        )
        .explode(
            [e for e in df_possible_queries.columns if e not in ["idx.m", "idx.roi"]]
            + ["r.available"]
        )
        # .filter(pl.col("r.available"))
        # .filter(pl.col("all_queries"))
    )
    if return_queries == "invalid":
        df_out = df_queries
    elif return_queries == "any_valid":
        df_out = df_queries.filter(pl.col("r.available"))
    elif return_queries == "all_valid":
        df_out = df_queries.filter(pl.col("r.available")).filter(pl.col("all_queries"))
    return df_out.select(
        pl.col("idx.m"),
        pl.col("idx.roi"),
        pl.col("idx.o"),
        pl.col("idx.c.0"),
        pl.col("idx.c.1"),
        pl.col("features"),
    )


MIGRATION = {
    "path.h5.image": "c.path",
    "path.h5.label_image": "o.path",
    "fidx.z_model": "idx.z_model",
    "path.z_model": "z_model.path",
    "z_model.o.mask": "z_model.o",
    "fidx.t_model": "idx.t_model",
    "correction_factor": "t_model.correction_factor",
    "fidx.o": "idx.o",
    "path.root": "roi.path",
}


def rename(df: pl.DataFrame, rename_map=MIGRATION) -> pl.DataFrame:
    migration = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(migration)


def _aggregate_image_paths2(r, df_image_ids, z_model, t_model):
    df_images = df_image_ids.join(
        r.images, on=["idx.m", "idx.roi", "idx.c"], how="left"
    ).join(r.rois.select(pl.col("idx.roi", "roi.path")), on=["idx.roi"])

    df_z_models = r.z_models.filter(pl.col("idx.z_model") == z_model)
    df_t_model_corr_factors = r.t_models.filter(pl.col("idx.t_model") == t_model)

    df_images_corr = (
        df_images.join(
            df_z_models.with_columns(pl.col("idx.c").cast(pl.Categorical)).select(
                pl.col("idx.c"),
                pl.col("idx.z_model"),
                pl.col("z_model.o").cast(pl.Categorical),
                pl.col("z_model.path").str.replace(".json", ".pkl"),
            ),
            on="idx.c",
            how="left",
        )
        .join(
            r.label_images.select(
                pl.col("idx.m"),
                pl.col("idx.roi"),
                pl.col("idx.o"),
                pl.col("o.path").alias("z_model.o.path"),
                pl.col("resample_scale_factors").alias(
                    "z_model.resample_scale_factors"
                ),
            ),
            left_on=["idx.m", "idx.roi", "z_model.o"],
            right_on=["idx.m", "idx.roi", "idx.o"],
            how="left",
        )
        .join(
            df_t_model_corr_factors.select(
                ["idx.c", "idx.roi", "idx.t_model", "t_model.correction_factor"]
            ),
            on=["idx.c", "idx.roi"],
            how="left",
        )
    )
    return df_images_corr


def _aggregate_label_paths2(r, df_label_image_ids):
    df_labels = df_label_image_ids.join(
        r.label_images,
        on=["idx.m", "idx.roi", "idx.o"],
    ).join(r.rois.select(pl.col("idx.roi", "roi.path")), on="idx.roi")
    return df_labels


def _lazy_load_images(
    df_images,
    columns=(
        "idx.m",
        "idx.roi",
        "idx.c",
        "c.path",
        "roi.path",
        "idx.z_model",
        "z_model.o",
        "z_model.path",
        "z_model.o.path",
        "z_model.resample_scale_factors",
        "idx.t_model",
        "t_model.correction_factor",
    ),
    dask_tuple=False,
):
    images = {}
    for (
        m,
        roi,
        c,
        c_path,
        roi_path,
        z_model_,
        z_model_o,
        z_model_path,
        z_model_o_path,
        z_model_o_resample,
        t_model_,
        t_model_cf,
    ) in df_images.select(columns).iter_rows():
        if z_model_ is not None:
            z_model_lbl_q = LabelImageQueryDask(
                root_path=roi_path,
                object_type_path=z_model_o_path,
                resample_scale_factors=z_model_o_resample,
                chunks=DASK_CHUNKS,
            )
        else:
            z_model_lbl_q = None

        iq = ImageQueryDask(
            root_path=roi_path,
            channel_path=c_path,
            z_model_path=z_model_path,
            z_model_label_image_query=z_model_lbl_q,
            t_model_correction_factor=t_model_cf,
            mask_query=None,
            chunks=DASK_CHUNKS,
        )
        if dask_tuple:
            images[(m, roi, c)] = iq.lazy()
        else:
            images[(m, roi, c)] = _to_si(iq.lazy())
    return images


def _lazy_load_labels(
    df_labels,
    columns=(
        "idx.m",
        "idx.roi",
        "idx.o",
        "o.path",
        "resample_scale_factors",
        "roi.path",
    ),
    dask_tuple=False,
):
    labels = {}
    for m, roi, o, lbl_path, scale_factors, roi_path in df_labels.select(
        columns
    ).iter_rows():
        lq = LabelImageQueryDask(
            root_path=roi_path,
            object_type_path=lbl_path,
            resample_scale_factors=scale_factors,
            chunks=DASK_CHUNKS,
        )
        if dask_tuple:
            labels[(m, roi, o)] = lq.lazy()
        else:
            labels[(m, roi, o)] = _to_si(lq.lazy())
    return labels


def _to_si(img_w_meta):
    img, meta = img_w_meta
    scale = dict(zip(meta.dims, meta.scale))
    translation = dict(zip(meta.dims, meta.scale))
    if meta.type_ == "intensity":
        return to_spatial_image(
            np.expand_dims(img, 0),
            dims=("c",) + meta.dims,
            scale=scale,
            translation=translation,
            c_coords=(meta.name,),
        ).squeeze()
    elif meta.type_ == "label":
        return (
            to_spatial_image(
                np.expand_dims(img, 0),
                dims=("c",) + meta.dims,
                scale=scale,
                translation=translation,
                c_coords=(meta.name,),
            )
            .rename({"c": "l"})
            .squeeze()
        )
    elif meta.type_ == "multichannel_intensity":
        return to_spatial_image(
            img,
            dims=meta.dims,
            scale=scale,
            translation=translation,
            c_coords=meta.name,
        )
    else:
        raise ValueError(f"Unknown image type {meta.type_!r}")
