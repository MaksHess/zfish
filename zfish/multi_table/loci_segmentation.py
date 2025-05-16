# %%
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from polars._typing import SelectorType
from scipy.ndimage import generate_binary_structure, label, maximum_filter
from skimage.feature import blob_dog, blob_log

from zfish.multi_table.feature_query_schemas import _to_si
from zfish.multi_table.raster_io import (
    LabelObjectQueryDask,
    _aggregate_label_object_paths,
    _images_to_queries,
    _label_objects_to_queries,
)
from zfish.multi_table.table_base_mixin import AnyFrame, safe_lazy
from zfish.multi_table.tables_io import (
    TABLES_PATH,
    read_resources,
    read_resources_and_features,
)
from zfish.roi.spatial_roi import Roi, apply_z_decay_models_to_roi, read_models
from zfish.roi.visualize import imshow_di, imshow_roi, imshow_si

lazy_roi = Roi.from_file(
    r"C:\Users\hessm\Documents\zfish_local\imgs\F03_px-2043_py+0598.h5", level=1
)

model_fld = r"C:\Users\hessm\Documents\zfish_local\models_v3\z_decay\Exp(features=MediumPath;EmbryoPath, loss=huber, pos_offset=True)"

models = read_models(model_fld)

lazy_roi_corr = apply_z_decay_models_to_roi(
    models, lazy_roi, two_step_label="embryoRaw"
)


roi = lazy_roi_corr.sel(
    l="nucleiRaw3", c=["Pol-II-S2P.0", "Pol-II-S5P.2", "FLAG.0", "DAPI.1"]
).compute()

r, f = read_resources_and_features()

r_roi = r.filter(pl.col("idx.roi") == roi.name)


# df_imgs, df_label_objects = _aggregate_label_object_paths(
#     r_roi,
#     "nucleiRaw3",
#     channels=["DAPI.1", "Pol-II-S2P.0", "bCatenin.1"],
#     channel_masks=["embryoRaw", "nucleiRaw3", "embryoRaw"],
#     z_model="Exp-2D",
#     multiscale_level=1,
# )
# # object_queries = _label_objects_to_queries(df_label_objects, df_imgs, strategy="memory")
# img, meta = list(_images_to_queries(df_imgs, dask_chunk_size=(100, -1, -1)).values())[
#     0
# ].compute()
# %%
imshow_roi(roi)

# %%
QUICKSAVE = r""
df = pl.read_parquet()


# %%

arr_pol2 = img[1]
blobs = blob_dog(
    arr_pol2 / arr_pol2.max(), min_sigma=0.5, max_sigma=3, threshold=0.05, overlap=0.3
)

# %%
import napari

viewer = napari.Viewer()
viewer.add_image(img, channel_axis=0, scale=meta.scale)
viewer.add_points(
    blobs[:, :3],
    scale=meta.scale,
    size=2 * blobs[:, 3] * np.sqrt(3),
    out_of_slice_display=True,
    opacity=0.8,
    face_color="red",
)
# %%


viewer = napari.Viewer()
imshow_di((nuc_channels, meta.from_template(origin=(0, 0, 0))), viewer=viewer)
viewer.add_labels(detect_local_maxima_3d(nuc_pol2), scale=meta.scale)
viewer.add_points(blobs[:, :3], scale=meta.scale)
# %%
from dataclasses import field
from functools import cached_property, reduce
from typing import Any, ClassVar, Literal, Sequence, TypeAlias

CONFIG: dict[str, Any] = {
    "column_set_name_separator": ":",
    "column_set_name_prefix": "id.",
    "column_set_col_pk_prefix": "idx.",
    "column_set_col_default_prefix": "",
}

MissingPrefixStrategy = Literal[
    "ignore", "drop_columns", "drop_prefix", "add_prefix", "raise", "strict"
]


# %%
@dataclass(frozen=True, slots=True)
class ColumnSet:
    column_names: tuple[str, ...] = ()
    name: str = None
    auto_name_sep: str = field(default=":", repr=False)

    def _validate(self):
        if self.name is None:
            object.__setattr__(self, "name", self.auto_name_sep.join(self.column_names))

    def __post_init__(self):
        self._validate()

    @classmethod
    def from_columns(
        cls,
        column_names: Sequence[str],
        name: str = None,
        auto_name_sep: str = ":",
    ):
        return cls(
            column_names=tuple(column_names),
            name=name,
            auto_name_sep=auto_name_sep,
        )


@dataclass(frozen=True, slots=True)
class PrimaryKey:
    column_set: ColumnSet
    column_prefix: str

    @classmethod
    def from_columns(
        cls,
        column_names: Sequence[str],
        name: str = None,
        auto_name_sep: str = ":",
        column_prefix: str = "idx.",
    ):
        return cls(
            column_set=ColumnSet(
                column_names=tuple(column_names),
                name=name,
                auto_name_sep=auto_name_sep,
            ),
            column_prefix=column_prefix,
        )

    @classmethod
    def from_df(
        cls,
        df: AnyFrame,
        name: str = None,
        auto_name_sep: str = ":",
        column_prefix: str = "idx.",
        remove_prefix: bool = True,
    ):
        idx_columns = (
            pl.LazyFrame(df)
            .select(cs.starts_with(column_prefix))
            .collect_schema()
            .names()
        )
        if remove_prefix:
            idx_column_names = tuple(e.removeprefix(column_prefix) for e in idx_columns)
        else:
            idx_column_names = idx_columns
        return cls.from_columns(
            column_names=idx_column_names,
            name=name,
            auto_name_sep=auto_name_sep,
            column_prefix=column_prefix,
        )

    @property
    def name(self):
        return self.column_set.name

    @property
    def column_names(self):
        return self.column_set.column_names

    @property
    def columns(self):
        return tuple(f"{self.column_prefix}{cn}" for cn in self.column_names)


@dataclass(frozen=True, slots=True)
class ForeingKey:
    column_set: ColumnSet
    other_column_set: ColumnSet
    name: str = None

    def _validate(self):
        if self.name is None:
            object.__setattr__(
                self, "name", " -> ".join([self.table_name, self.other_table_name])
            )

    def __post_init__(self):
        self._validate()

    @property
    def table_name(self):
        return self.column_set.name

    @property
    def column_names(self):
        return self.column_set.column_names

    @property
    def other_table_name(self):
        return self.other_column_set.name

    @property
    def other_column_names(self):
        return self.other_column_set.column_names


fk = ForeingKey(
    column_set=ColumnSet.from_columns(column_names=("well",), name="rois"),
    other_column_set=ColumnSet.from_columns(column_names=("idx.well",), name="wells"),
)

pk = PrimaryKey.from_df(r.rois)

pk2 = PrimaryKey.from_df(r.wells)

print(pk)
print(pk2)
# %%


@dataclass(frozen=True, slots=True)
class ForeignKeyConstraint:
    table_id: str
    column_set: ColumnSet
    foreign_table_id: str
    foreign_column_set: ColumnSet | None = None

    def _validate(self):
        if self.foreign_column_set is None:
            object.__setattr__(self, "foreign_column_set", self.column_set)
        assert len(self.column_set) == len(self.foreign_column_set)

    def __post_init__(self):
        self._validate()


ColumnSet_: TypeAlias = tuple[str, ...]
NamedColumnSet_: TypeAlias = tuple[str, ColumnSet_]

ForeignKey_: TypeAlias = tuple[NamedColumnSet_, NamedColumnSet_]


@dataclass(frozen=True, slots=True)
class Table:
    df: pl.DataFrame
    pk: PrimaryKey  # PK

    @classmethod
    def from_polars(cls, df, pk=None, name=None):
        if pk is None:
            pk = PrimaryKey.from_df(df, name=name)
        return cls(df=pl.DataFrame(df), pk=pk)

    @classmethod
    def from_df(cls, df, pk=None, name=None):
        return cls.from_polars(df)

    @property
    def name(self) -> str:
        return self.pk.name

    @property
    def idx(self) -> pl.DataFrame:
        return self.df.select(self.pk.columns)


@dataclass(frozen=True, slots=True)
class LazyTable:
    df: pl.LazyFrame
    pk: PrimaryKey  # PK

    @classmethod
    def from_polars(cls, df, pk=None, name=None):
        if pk is None:
            pk = PrimaryKey.from_df(df, name=name)
        return cls(df=pl.LazyFrame(df), pk=pk)

    @classmethod
    def from_df(cls, df, pk=None, name=None):
        return cls.from_polars(df)

    @property
    def name(self) -> str:
        return self.pk.name

    @property
    def idx(self) -> pl.LazyFrame:
        return self.df.select(self.pk.columns)


@dataclass(frozen=True, slots=True)
class Tables:
    _tables: dict[str, Table] = field(default_factory=dict)
    fks: tuple[ForeignKey_, ...] = ()


# %%
tbls = [Table.from_df(df, name) for name, df in r._tables.items()]
ltbls = [LazyTable.from_df(df, name) for name, df in r._tables.items()]

# %%
r.z_models["idx.z_model"].unique()
# %%
r.t_models["idx.t_model"].unique()
