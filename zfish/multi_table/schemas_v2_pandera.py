"""Ditch the pandera stuff, but the nested regoions table looks interesting."""

# %%
from itertools import combinations, islice, permutations, product
from typing import Callable, Literal, Self, Sequence, Type, TypeAlias

import napari
import numpy as np
import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from pandera import Check as C
from pandera.typing.polars import DataFrame, LazyFrame

from zfish.multi_table.schemas_v2 import CAT_TYPE
from zfish.preprocessing.sliding_window_samples import stratified_sample
from zfish.preprocessing.types import AnyFrame, AnyFrameT

ParseFunc: TypeAlias = Callable[[AnyFrameT], AnyFrameT]
ROI_TYPES = ["root", "plate", "well", "site", "label_object", "grid"]
RoiType: TypeAlias = Literal[*ROI_TYPES]


class BaseTable(pa.DataFrameModel):
    class Config:
        coerce = True
        add_missing_columns = True
        # strict = "filter"


class VirtualMixin:
    _virtual_parsers: tuple[ParseFunc, ...] = ()

    @classmethod
    def add_parser(cls, parser: ParseFunc) -> Type[Self]:
        cls._virtual_parsers = cls._virtual_parsers + (parser,)
        return cls

    @classmethod
    def set_parsers(cls, parsers: Sequence[ParseFunc]) -> Type[Self]:
        cls._virtual_parsers = tuple(parsers)
        return cls

    @classmethod
    def reset_parsers(cls, parsers: Sequence[ParseFunc]) -> Type[Self]:
        cls._virtual_parsers = ()
        return cls

    @classmethod
    def parse(cls, df: AnyFrame) -> AnyFrame:
        for parser in cls._virtual_parsers:
            df = parser(df)
        return df


# %%
DIMS = ["z", "y", "x"]

from dataclasses import dataclass

import pandera.polars as pa
import polars as pl


@dataclass
class VirtualCol:
    name: str
    comps: tuple[str, ...] = ()
    exprs: tuple[pl.Expr, ...] = ()


{
    f"extent.{dim}": pl.col(f"scale.{m}.i{dim}") * pl.col(f"shape.{m}.i{dim}")
    for m, dim in product(range(5), ["z", "y", "x"])
}


# %%
class Region(BaseTable):
    type: CAT_TYPE = pa.Field(isin=ROI_TYPES)
    o: CAT_TYPE = pa.Field(nullable=True)
    # full_type: CAT_TYPE
    name: CAT_TYPE = pa.Field(unique=True)
    type__parent: CAT_TYPE
    name__parent: CAT_TYPE
    origin__z: pl.Float64 = pa.Field(default=0.0)
    origin__y: pl.Float64 = pa.Field(default=0.0)
    origin__x: pl.Float64 = pa.Field(default=0.0)
    extent__z: pl.Float64
    extent__y: pl.Float64
    extent__x: pl.Float64

    @pa.dataframe_check
    def _origin_0_if_its_own_parent(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.filter(pl.col("name") == pl.col("name__parent")).select(
            [pl.col(f"origin__{dim}") == 0 for dim in DIMS]
        )

    class Config:
        metadata = {
            "pk": ("region", ("name",)),
            "fk": ("name__parent", "Region.name"),
            "ck": (),
        }


class RegionV(VirtualMixin, Region):
    bound__z__upper: pl.Float64
    bound__y__upper: pl.Float64
    bound__x__upper: pl.Float64
    centroid__z: pl.Float64
    centroid__y: pl.Float64
    centroid__x: pl.Float64


def regionv_create_virtual_columns(df: DataFrame[Region]) -> DataFrame[RegionV]:
    return df.with_columns(
        [
            (pl.col(f"origin__{dim}") + pl.col(f"extent__{dim}")).alias(
                f"bound__{dim}__upper"
            )
            for dim in DIMS
        ]
        + [
            (pl.col(f"origin__{dim}") + pl.col(f"extent__{dim}") / 2).alias(
                f"centroid__{dim}"
            )
            for dim in DIMS
        ]
    )


RegionV.add_parser(regionv_create_virtual_columns)


def is_singleton(data: pa.PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(data.key).unique().len() == 1)


class SampledRegion(Region):
    m: pl.UInt8
    scale__m_z: pl.Float64
    scale__m_y: pl.Float64
    scale__m_x: pl.Float64
    shape__m_iz: pl.Int32
    shape__m_iy: pl.Int32
    shape__m_ix: pl.Int32
    iorigin__m_z: pl.Float32
    iorigin__m_y: pl.Float32
    iorigin__m_x: pl.Float32

    @pa.check("m")
    def _is_singleton(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return is_singleton(data)


class MultiscaleLevels(BaseTable):
    m: pl.UInt8 = pa.Field(unique=True)
    scale__z: pl.Float64 = pa.Field(gt=0.0)
    scale__y: pl.Float64 = pa.Field(gt=0.0)
    scale__x: pl.Float64 = pa.Field(gt=0.0)


class MultiscaleLevelsVirtual(MultiscaleLevels):
    iorigin__z: pl.Float64
    iorigin__y: pl.Float64
    iorigin__x: pl.Float64


def sample_region(
    df_reg: DataFrame[Region], df_m: DataFrame[MultiscaleLevels], m: int = 0
) -> DataFrame[SampledRegion]:
    return df_reg.with_columns(pl.lit(m).alias("m")).join(df_m, on="m")


SampledCoordinates = pa.DataFrameSchema(
    columns={"m": pa.Column(pl.UInt8, checks=[pa.Check(is_singleton)])}
)


class Loadable(BaseTable):
    path: pl.List(pl.String) = pa.Field(default=[])
    return_type: pl.String = pa.Field(
        isin=[
            "table",
            "model",
            "ccp_model",
            "clf_model",
            "canvas",
            "image",
            "label_image",
            "binary_mask",
            "dft",
            "ms_image",
            "ms_label_image",
        ],
        nullable=True,
    )


class LoadableRegion(Loadable, Region):
    pass


class Roots(Region):
    name__parent: CAT_TYPE = pa.Field(nullable=True)
    type: CAT_TYPE = pa.Field(default="root")
    path: pl.String
    t__start: pl.Datetime = pa.Field(nullable=True)


class Plates(Region):
    type: CAT_TYPE = pa.Field(default="plate")
    path: pl.String

    class Config:
        metadata = metadata = {
            "pk": ("plate", ("name",)),
            "fks": (("name__parent", "roots.name"),),
            "cks": (),
        }


class Wells(Region):
    type: CAT_TYPE
    name: pl.String = pa.Field(unique=True, str_matches="^[A-P]\d\d$")
    row: pl.String = pa.Field(str_matches="^[A-H]$")
    col: pl.String = pa.Field(str_matches="^\d\d$")
    is_control_well: pl.Boolean
    control_well_for_acquisition: pl.UInt16

    class Config:
        metadata = metadata = {
            "pk": ("well", ("name",)),
            "fks": (("name__parent", "plates.name"),),
            "cks": ("well", ("row", "", "col")),
        }


class Rois(Region):
    type: CAT_TYPE = pa.Field(default="roi")
    site: CAT_TYPE
    path: pl.String

    class Config:
        metadata = {
            "pk": ("roi", ("name",)),
            "fks": (("name__parent", "wells.name"),),
            "cks": ("roi", ("name__parent", "_", "site")),
        }


class LabelObjects(Region):
    type: CAT_TYPE = pa.Field(default="label_object")
    name: CAT_TYPE = pa.Field(unique=True)
    roi: CAT_TYPE
    o: CAT_TYPE
    lbl: pl.UInt16

    class Config:
        metadata = {
            "pk": ("label_object", ("roi", "o", "lbl")),
            "fks": (
                ("roi", "rois.name"),
                ("o", "object_types.o", "hierarchy.o.parent", "hierarchy.o.child"),
            ),
            "cks": (),
        }


class Images(BaseTable):
    m: pl.UInt8
    c: CAT_TYPE
    roi: CAT_TYPE
    path: pl.String

    class Config:
        metadata = {
            "pk": (
                "image",
                (
                    "m",
                    "c",
                    "roi",
                ),
            ),
            "fks": (
                ("m", "multiscale_levels.m"),
                ("c", "channels.c"),
                ("roi", "rois.roi"),
            ),
            "cks": ("roi", ("well", "_", "site")),
        }


class LabelImages(BaseTable):
    m: pl.UInt8
    o: CAT_TYPE
    roi: CAT_TYPE
    path: pl.String

    class Config:
        metadata = {
            "pk": (
                "label_image",
                (
                    "m",
                    "o",
                    "roi",
                ),
            ),
            "fks": (
                ("m", "multiscale_levels.m"),
                ("o", "object_types.o"),
                ("roi", "rois.roi"),
            ),
            "cks": ("roi", ("well", "_", "site")),
        }


def _default_origin(ndim=3):
    return [0.0 for _ in range(ndim)]


def _default_extent(ndim=3, extent=1.0):
    return [extent for _ in range(ndim)]


def _default_basis_vecs(ndim=3):
    bvecs = []
    for i in range(ndim):
        bvec = [1.0 if i == j else 0.0 for j in range(ndim)]
        bvecs.append(bvec)
    return bvecs


def example_tables(object_types=("embryoRaw", "nucleiRaw3")):
    from zfish.multi_table.tables_io import (
        TABLE_NAME_MAP_,
        LazyResources,
        Resources,
        scan_resources,
    )

    r0 = Resources.from_schemas()

    r = scan_resources().collect()
    n_plates = 1
    well_extent = 7_000.0

    df_root = (
        pl.DataFrame(
            {
                "name": ["zfish_cluster", "zfish_local"],
                "name__parent": ["zfish_cluster", "zfish_local"],
                "type": ["root"] * 2,
                # "full_type": ["root"] * 2,
                "type__parent": ["root"] * 2,
                "extent__z": [271] * 2,
                "extent__y": [100_000] * 2,
                "extent__x": [100_000] * 2,
                "path": [
                    "/data/active/hmax/zfish",
                    r"C:\Users\hessm\Documents\zfish_local",
                ],
            }
        )
        .with_columns(pl.Series("by", ["/", "\\"]))
        .with_columns(pl.col("path").str.split(by=pl.col("by")))
    )

    df_plates = pl.DataFrame(
        {
            "name": [f"plate{i}" for i in range(n_plates)],
            "name__parent": ["zfish_local"],
            "type": ["plate"] * n_plates,
            "type__parent": ["root"] * n_plates,
            "extent__z": [271] * n_plates,
            "extent__y": [60_000] * n_plates,
            "extent__x": [100_000] * n_plates,
            "path": ["img"],
        }
    )

    df_wells = r.wells.select(
        pl.col("well").alias("name"),
        pl.lit("plate0").alias("name__parent"),
        pl.lit("well").alias("type"),
        pl.lit("plate").alias("type__parent"),
        pl.lit(271.0).alias("extent__z"),
        pl.lit(well_extent).alias("extent__y"),
        pl.lit(well_extent).alias("extent__x"),
        *[pl.col(f"plate.{dim}").alias(f"origin__{dim}") for dim in ["y", "x"]],
        pl.col("row", "col", "is_control_well", "control_well_for_acquisition"),
    )

    df_rois = r.rois.select(
        pl.col("roi").alias("name"),
        pl.lit("site").alias("type"),
        pl.col("well").alias("name__parent"),
        pl.lit("well").alias("type__parent"),
        pl.col("roi").cast(pl.String).str.split("_").list.get(0).alias("site"),
        pl.lit(271.0).alias("extent__z"),
        pl.lit(650.0).alias("extent__y"),
        pl.lit(650.0).alias("extent__x"),
        *[pl.col(f"well.{dim}").alias(f"origin__{dim}") for dim in ["y", "x"]],
        pl.col("path").str.split("\\").list.slice(-1, 1).list.explode().alias("path"),
    ).with_columns(
        pl.col("origin__y") + well_extent / 2, pl.col("origin__x") + well_extent / 2
    )

    df_label_objects = (
        r.label_objects.filter(pl.col("o").is_in(object_types))
        .join(r.multiscale_levels, on="m")
        .select(
            pl.concat_str(["roi", "o", "label"], separator="__").alias("name"),
            pl.lit("label_object").alias("type"),
            pl.col("roi").alias("name__parent"),
            pl.col("o"),
            pl.lit("site").alias("type__parent"),
            *[
                (pl.col(f"m.i{dim}.lower") * pl.col(f"scale.i{dim}")).alias(
                    f"origin__{dim}"
                )
                for dim in ["z", "y", "x"]
            ],
            *[
                (
                    (pl.col(f"m.i{dim}.upper") - pl.col(f"m.i{dim}.lower"))
                    * pl.col(f"scale.i{dim}")
                ).alias(f"extent__{dim}")
                for dim in ["z", "y", "x"]
            ],
        )
        .pipe(Region)
    )

    df_regions = pl.concat(
        LoadableRegion.validate(e)
        for e in [df_root, df_plates, df_wells, df_rois, df_label_objects]
    )
    return df_regions.with_columns(
        pl.concat_list(pl.col("type").cast(pl.String), pl.col("o").cast(pl.String))
        .list.drop_nulls()
        .list.join(":")
        .cast(CAT_TYPE)
    )


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def bbx_corners_to_vertices(c_lower, c_upper):
    c_arr = np.array([c_lower, c_upper])
    vertices = []
    for v_indices in (
        (np.arange(2**3) & (2 ** np.arange(3)).reshape(-1, 1)).clip(0, 1)
    ).T:
        vertices.append(c_arr[v_indices, np.arange(3)])
    return np.array(vertices)


def bbx_vertices_to_edges(vertices):
    ndim = vertices.shape[1]

    edges = []
    for i in range(ndim):
        n = 2**i
        for j in range(n):
            for edge in batched(vertices[j::n], 2):
                edges.append(edge)

    return np.array(edges)


def bbx_corners_to_edges(c_lower, c_upper):
    return bbx_vertices_to_edges(bbx_corners_to_vertices(c_lower, c_upper))


def _transform_coordinate_systems_rec(df: pl.DataFrame) -> pl.LazyFrame:
    dd = df.lazy().select(
        [
            "type",
            "name",
            pl.col("name__parent").fill_null(pl.col("name")),
            pl.col("type__parent").fill_null("root"),
            cs.starts_with("origin"),
            cs.starts_with("extent"),
        ]
    )

    return dd.join(
        dd, left_on=["name__parent", "type__parent"], right_on=["name", "type"]
    ).select(
        pl.col("type"),
        pl.col("name"),
        pl.col("name__parent_right").alias("name__parent"),
        pl.col("type__parent_right").alias("type__parent"),
        *[
            pl.col(f"origin__{dim}") + pl.col(f"origin__{dim}_right")
            for dim in ["z", "y", "x"]
        ],
        *[pl.col(f"extent__{dim}") for dim in ["z", "y", "x"]],
    )


def _transform_coordinate_systems(df_coordinate_systems, to_type="root"):
    order = ["root", "plate", "well", "site", "label_object"]
    allowed_types = order[: order.index(to_type) + 1]
    print(allowed_types)
    df_transformed = df_coordinate_systems.lazy()

    while True:
        available_types = (
            df_transformed.select("type__parent").collect()["type__parent"].unique()
        )
        print(available_types)
        if all([e in allowed_types for e in available_types]):
            break
        df_transformed = _transform_coordinate_systems_rec(df_transformed)
    return df_transformed


def visualize_nested_regions(
    subsample_objects_per_site=2000, object_types=("embryoRaw", "cells", "nucleiRaw3")
):
    df_regions = example_tables(object_types)
    df = pl.concat(
        [
            df_regions.filter(
                pl.col("type").cast(pl.String).str.starts_with("label_object").not_()
            ),
            df_regions.filter(pl.col("o").is_not_null()).filter(
                stratified_sample(
                    by=("type", "name__parent"), n=subsample_objects_per_site
                )
            ),
        ]
    )
    # return df

    bbxs_verts = {}
    bbxs_edges = {}
    df_out = (
        df.pipe(_transform_coordinate_systems_rec)
        .pipe(_transform_coordinate_systems_rec)
        .collect()
    ).filter(pl.col("type") != "root")
    # return df_out

    for name, df_level in df_out.group_by(["type"], maintain_order=True):
        bbxs_verts[name] = []
        bbxs_edges[name] = []
        for rw in df_level.select(
            cs.starts_with("origin"), cs.starts_with("extent")
        ).rows():
            c_lower = np.array(rw[:3])
            c_upper = c_lower + np.array(rw[3:])
            bbx_verts = bbx_corners_to_vertices(c_lower, c_upper)
            bbxs_verts[name].append(bbx_verts)
            bbxs_edges[name].append(bbx_vertices_to_edges(bbx_verts))

    def _lines_to_vecs(bbx_edges):
        bbx_vecs = bbx_edges.copy()
        bbx_vecs[:, 1, :] = bbx_vecs[:, 1, :] - bbx_vecs[:, 0, :]
        return bbx_vecs

    PROPS = {
        "plate": {"edge_width": 70, "edge_color": "purple"},
        "well": {"edge_width": 70, "edge_color": "green"},
        "site": {"edge_width": 50, "edge_color": "orange"},
        "label_object:embryoRaw": {"edge_width": 20, "edge_color": "yellow"},
        "label_object:cells": {"edge_width": 2, "edge_color": "red"},
        "label_object:nucleiRaw3": {"edge_width": 2, "edge_color": "blue"},
    }

    viewer = napari.Viewer()

    for k, bbx_edges in bbxs_edges.items():
        name = "".join(k)
        viewer.add_vectors(
            _lines_to_vecs(np.concatenate(bbx_edges)),
            name=name,
            vector_style="line",
            **PROPS[name],
        )


def df_sites_from_resources(df_images, df_ms, df_rois):
    return (
        df_images.filter(pl.col("m") == 0)
        .group_by(("roi",), maintain_order=True)
        .agg(pl.col("m").first())
        .join(df_ms, on="m")
        .join(
            df_rois.select(["roi", "well", "well.x", "well.y"]),
            on="roi",
        )
        .with_columns(
            (pl.col(f"scale.i{dim}") / 2).alias(f"iorigin__{dim}")
            for dim in ["z", "y", "x"]
        )
        .with_columns(pl.lit(0.0).alias(f"origin__{dim}") for dim in ["z", "y", "x"])
        .with_columns(
            pl.lit(271).alias("shape__z"),
            pl.lit(2000).alias("shape__y"),
            pl.lit(2000).alias("shape__x"),
        )
        .with_columns(
            (
                pl.col(f"origin__{dim}")
                + (pl.col(f"scale.i{dim}") * pl.col(f"shape__{dim}"))
            ).alias(f"extent__{dim}")
            for dim in ["z", "y", "x"]
        )
        .select(
            pl.col("roi").alias("name"),
            pl.lit("site").alias("type"),
            cs.starts_with("origin"),
            cs.starts_with("extent"),
            *[
                ((pl.col(f"extent__{dim}") - pl.col(f"origin__{dim}")) / 2).alias(
                    f"centroid__{dim}"
                )
                for dim in ["z", "y", "x"]
            ],
            pl.col("well").alias("name__parent"),
            pl.lit(0.0).alias("translate__parent__z"),
            pl.col("well.y").alias("translate__parent__y"),
            pl.col("well.x").alias("translate__parent__x"),
        )
    )


visualize_nested_regions()
# %%
