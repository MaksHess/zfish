# %%
import io
from collections import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import chain
from os import PathLike
from pathlib import Path
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
)
from weakref import ProxyType, proxy

import h5py
import numpy as np
import polars as pl
import xarray as xr

from zfish.features.types import LabelImage, SpatialImage
from zfish.image.image import to_si
from zfish.intensity_normalization.models import apply_model_to_channel
from zfish.io import h5
from zfish.roi._roi_formatting import SORT_KEY, _coordinates_repr

if TYPE_CHECKING:
    import polars as pl

    from zfish.intensity_normalization.models import Model

# pl.enable_string_cache(True)

Image: TypeAlias = LabelImage | SpatialImage
Element: TypeAlias = Image | pl.DataFrame

EXPERIMENT_INDEX = ("roi", "object", "label")
ROI_INDEX = ("object", "label")
OBJECT_INDEX = ("label",)


# def _load_roi(
#     root_path: PathLike,
#     attrs_select: dict[str, str | int | tuple[str | int, ...]],
# ) -> list[SpatialImage]:
#     f = h5py.File(root_path)
#     dsets = [to_si(dset) for dset in h5.select(f, attrs_select)]
#     return dsets
#     # return xr.concat(dsets, dim="c", combine_attrs="drop")


# def load_channels(root_path: PathLike, level: int | None = None) -> SpatialImage:
#     f = h5py.File(root_path)
#     attrs_select = {"img_type": "intensity"}
#     if level is None:
#         level = sorted(h5.attrs_set(f, "level", attrs_select=attrs_select))[0]
#     attrs_select = {**attrs_select, **{"level": level}}
#     return xr.concat(_load_roi(root_path=root_path, attrs_select=attrs_select), dim="c")


# def load_labels(root_path: PathLike, level: int | None = None) -> LabelImage:
#     f = h5py.File(root_path)
#     attrs_select = {"img_type": "label"}
#     if level is None:
#         level = sorted(h5.attrs_set(f, "level", attrs_select=attrs_select))[0]
#     attrs_select = {**attrs_select, **{"level": level}}
#     return xr.concat(_load_roi(root_path=root_path, attrs_select=attrs_select), dim="c")

def _load_roi(
    f: h5py.File,
    attrs_select: dict[str, str | int | tuple[str | int, ...]],
) -> list[SpatialImage]:
    dsets = [to_si(dset) for dset in h5.select(f, attrs_select)]
    return dsets
    # return xr.concat(dsets, dim="c", combine_attrs="drop")


def load_channels(f: h5py.File, level: int | None = None) -> SpatialImage:
    attrs_select = {"img_type": "intensity"}
    if level is None:
        level = sorted(h5.attrs_set(f, "level", attrs_select=attrs_select))[0]
    attrs_select = {**attrs_select, **{"level": level}}
    return xr.concat(_load_roi(f=f, attrs_select=attrs_select), dim="c")


def load_labels(f: h5py.File, level: int | None = None) -> LabelImage:
    attrs_select = {"img_type": "label"}
    if level is None:
        level = sorted(h5.attrs_set(f, "level", attrs_select=attrs_select))[0]
    attrs_select = {**attrs_select, **{"level": level}}
    return xr.concat(_load_roi(f=f, attrs_select=attrs_select), dim="c")


def load_roi_tables(
    root_path: PathLike, lazy: bool = False
) -> dict[str, "pl.DataFrame"]:
    import polars as pl

    tables = {}
    for fn in Path(root_path).glob("*.parquet"):
        structure = fn.stem
        if lazy:
            tables[structure] = pl.scan_parquet(fn)
        else:
            tables[structure] = pl.read_parquet(fn)
    return tables


def load_roi_table(root_path: PathLike, id_column="object") -> "pl.DataFrame":
    import polars as pl

    tables = load_roi_tables(root_path)

    return pl.concat(
        [
            table.with_columns(pl.lit(object_name).alias(id_column))
            for object_name, table in tables.items()
        ],
        how='diagonal'
    )

def load_roi_map_table(root_path: PathLike, id_column="roi") -> "pl.DataFrame":
    import polars as pl

    tables = []

    for roi_root in Path(root_path).glob('*'):
        if roi_root.is_dir():
            tables.append(load_roi_table(roi_root).with_columns(pl.lit(roi_root.name).alias(id_column)))
    
    return pl.concat(tables, how='diagonal')

# def load_ill_corr_models(root_path: PathLike) -> dict[str, dict[str, dict[str, "Model"]]]:
#     """
#     Load illumination correction model from a directory with the following structure:
#     - root_path
#         - z_decay
#             - stain0
#                 - Model0
#                 - Model1
#                 - Model2
#             - stain1
#                 - Model0
#         - t_decay
#     ...
#     """
#     models = dict()
#     for model_fn in Path(root_path).rglob("*.pkl"):
#         rel_fn = model_fn.relative_to(root_path)
#         keys = rel_fn.parts[:-1] + (rel_fn.stem,)
#         print(keys)
#         level = models
#         for key in keys:
#             print(key)
#             level = level[key]
#         level[key] = Model.load(model_fn)
#     return models


# def write_ill_corr_models(
#     root_path: PathLike, models_root: dict[str, dict[str, dict[str, "Model"]]]
# ) -> None:
#     for model_type, models in models_root.items():
#         for channel, channel_models in models.items():
#             for model_type, channel_model in channel_models.items():
#                 if channel_model is None:
#                     continue
#                 channel_model.save(
#                     directory=Path(root_path) / channel,
#                     file_name=f"{str(channel_model)}",
#                     mode="wb",
#                 )

import os

from zfish.intensity_normalization.models import Model


def write_models(models, root: Path | str):
    """
    Write a nested dict to a nested file structure.
    """
    for key, value in models.items():
        # Create a directory for the current key
        current_path = os.path.join(root, str(key))
        os.makedirs(current_path, exist_ok=True)

        # Recursively write sub-dictionaries or write the value to a file
        if isinstance(value, dict):
            write_models(value, current_path)
        elif isinstance(value, Model):
            value.save(directory=current_path, file_name="model", mode="wb")


def read_models(root: Path | str | None):
    """
    Read a nested file structure to a nested dict.
    """
    result = {}
    if root is None:
        return result
    if not Path(root).exists():
        return result
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            # Recursively process subdirectories
            result[item] = read_models(item_path)
        elif os.path.isfile(item_path) and item_path.endswith(".pkl"):
            result = Model.load(item_path)
    return result


def get_bounding_box_slices(
    df: "pl.DataFrame", index_col: str | tuple[str, str] = "label"
):
    BOUNDING_BOX_STRUCT_COLUMN = "BoundingBox"
    BOUNDING_BOX_COLUMNS = [
        "lower-x",
        "upper-x",
        "lower-y",
        "upper-y",
        "lower-z",
        "upper-z",
    ]

    return (
        df.select(pl.col(index_col), pl.col(BOUNDING_BOX_STRUCT_COLUMN))
        .unnest(BOUNDING_BOX_STRUCT_COLUMN)
        .select(
            [
                pl.col(index_col),
                pl.concat_list(pl.col(BOUNDING_BOX_COLUMNS[:2]).alias("x")),
                pl.concat_list(pl.col(BOUNDING_BOX_COLUMNS[2:4]).alias("y")),
                pl.concat_list(pl.col(BOUNDING_BOX_COLUMNS[4:]).alias("z")),
            ]
        )
        .to_pandas()
        .set_index("label")
        .applymap(lambda x: slice(x[0], x[1], None))
        .T.to_dict()
    )


def get_bounding_box(
    df: "pl.DataFrame", index_col: str | tuple[str, str] = "label"
):  # -> dict[str, slice]:
    BOUNDING_BOX_STRUCT_COLUMN = "BoundingBox"
    BOUNDING_BOX_COLUMNS = [
        "lower-x",
        "upper-x",
        "lower-y",
        "upper-y",
        "lower-z",
        "upper-z",
    ]

    return (
        df.select(pl.col(BOUNDING_BOX_STRUCT_COLUMN))
        .unnest(BOUNDING_BOX_STRUCT_COLUMN)
        .select(
            [
                # pl.col('label').min().alias('lower-label'),
                # pl.col('label').max().alias('upper-label'),
                pl.col("^lower.*$").min(),
                pl.col("^upper.*$").max(),
            ]
        )
        .select(
            [
                # pl.col(index_col),
                pl.concat_list(pl.col(BOUNDING_BOX_COLUMNS[:2]).alias("x")),
                pl.concat_list(pl.col(BOUNDING_BOX_COLUMNS[2:4]).alias("y")),
                pl.concat_list(pl.col(BOUNDING_BOX_COLUMNS[4:]).alias("z")),
            ]
        )
        .to_pandas()
        .applymap(lambda x: slice(x[0], x[1], None))
        .T.to_dict()[0]
    )


# TODO: Return empty SpatialImage
def empty_data():
    return {
        "labels": xr.DataArray(),
        "images": xr.DataArray(),
    }


# TODO: Maybe use a StrEnum (available from 3.11)
class TableWriteStrategy(Enum):
    OVERWRITE = auto()
    MERGE = auto()
    RAISE = auto()


def _get_channels_safe(img: LabelImage | SpatialImage) -> set[str]:
    """
    Custom function to extract channel info from `SpatialImage`. Using the obvious
    `set(img.c.values)` seemed to work initially but lead to unexpected behaviour when
    trying to serialize using yaml.dump. Must be some weird interaction w/ numpy/xarray
    under the hood, explicitly converting to `str` in a comprehension does the trick.
    """
    if "l" in list(img.coords.keys()):
        if "l" in img.dims:
            return set([str(e) for e in img.l.values])
        else:
            return set([str(img.l.item())])
    elif "c" in list(img.coords.keys()):
        if "c" in img.dims:
            return set([str(e) for e in img.c.values])
        else:
            return set([str(img.c.item())])
    else:
        raise ValueError("No channel axis  ('l' or 'c') found!")


SEL_KWARGS = ("method", "tolerance", "drop")




class Roi(abc.Mapping):
    _f: list[h5py.File] = []
    _LABELS_KEY: str = "labels"
    _IMAGES_KEY: str = "images"
    _FEATURES_KEY: str = "features"
    _ILL_CORR_KEY: str = "models"
    _TABLES_IDX: tuple[str, ...] | str = "label"
    _TWO_STEP_ILL_CORR_LABEL: str | None = "embryoRaw"

    def __init__(
            self,
            _fp: ProxyType,
            data: Mapping[str, Any] | None = None,
            tables: Mapping[str, Any] | None = None,
            models: Mapping[str, Mapping[str, Mapping[str, Model]]] | None = None,
            paths: Mapping[str, Path] | None = None,
            name: str = ""
            ):
        self._fp = _fp
        if data is None:
            self.data = empty_data()
        else:
            self.data = data
        if tables is None:
            self.tables = dict()
        else:
            self.tables = tables
        if models is None:
            self.models = dict()
        else:
            self.models = models
        if paths is None:
            self.paths = dict()
        else:
            self.paths = paths
        self.name = name
        

    @classmethod
    def from_file(
        cls,
        root: PathLike[str],
        level: int,
        features_root: PathLike[str] | None = None,
        ill_corr_root: PathLike[str] | None = None,
        lazy_tables: bool = False,
    ) -> "Roi":
        root = Path(root)
        name = root.stem
        if features_root is None:
            features_root = root.parent.parent / "features" / name
        if ill_corr_root is None:
            ill_corr_root = root.parent.parent / "models"
        paths = {
            "root": root,
            cls._FEATURES_KEY: features_root,
            cls._ILL_CORR_KEY: ill_corr_root,
        }
        tables = load_roi_tables(features_root, lazy=lazy_tables)
        models = read_models(Path(ill_corr_root))
        cls._f.append(h5py.File(root))
        _fp = proxy(cls._f[-1])
        labels = load_labels(_fp, level=level)
        channels = load_channels(_fp, level=level)


        print(f"loading {name}...")
        return Roi(
            _fp=_fp,
            data={
                cls._LABELS_KEY: labels.rename({"c": "l"}),
                cls._IMAGES_KEY: channels,
            },
            tables=tables,
            models=models,
            paths=paths,
            name=name,
        )
    
    @classmethod
    def close_all_files(
        cls
    ):
        for f in cls._f:
            f.close()
        cls._f.clear()

    def write_tables(
        self,
        strategy: TableWriteStrategy = TableWriteStrategy.MERGE,
        root_path: Path | str | None = None,
    ) -> None:
        assert self._FEATURES_KEY in self.paths or root_path is not None
        if root_path is None:
            root_path = self.paths[self._FEATURES_KEY]

        if self.tables == {}:
            return

        for structure, df in self.tables.items():
            out_file = Path(root_path) / f"{structure}.parquet"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            match strategy:
                case TableWriteStrategy.RAISE:
                    if out_file.exists():
                        raise FileExistsError(
                            f"There's a file at {out_file}. Use `TableWriteStrategy.MERGE` or `TableWriteStrategy.OVERWRITE` to merge/overwrite."
                        )
                    new_table = df
                case TableWriteStrategy.MERGE:
                    old_table = (
                        pl.read_parquet(out_file)
                        if out_file.exists()
                        else pl.DataFrame(df[self._TABLES_IDX])
                    )
                    shared_feature_columns = tuple(
                        set(old_table.drop(self._TABLES_IDX).columns).intersection(
                            df.drop(self._TABLES_IDX).columns
                        )
                    )
                    new_table = old_table.drop(shared_feature_columns).join(
                        df, on=self._TABLES_IDX, how="outer"
                    )
                case TableWriteStrategy.OVERWRITE:
                    new_table = df
                case _:
                    raise ValueError(
                        f"Unknown strategy: `{strategy}`, pick one of `{TableWriteStrategy.__members__}`"
                    )
            new_table.write_parquet(out_file)

    def rename(self, name: str) -> "Roi":
        return Roi(
            **{
                **{
                    attr: getattr(self, attr)
                    for attr in self.__dataclass_fields__
                    if not attr.startswith("_")
                },
                "name": name,
                "_f": proxy(self._f) if isinstance(self._f, h5py.File) else self._f
            }
        )

    # TODO: Make sure the right kwargs are passed on (currently none).
    # TODO: Refactor the following functions DRY
    # TODO: .pick to extract a single element from a categorical dim.
    def pick(self, **kwargs) -> "Element":
        """
        Extract element from roi.
        """
        raise NotImplementedError()

    def sel(self, **kwargs) -> "Roi":
        out = dict()
        for k, elem in self.data.items():
            valid_keys = set(kwargs.keys()).intersection(elem.dims + SEL_KWARGS)
            out[k] = elem.sel(**{k: kwargs[k] for k in valid_keys})
        return Roi(
            _fp=self._fp,
            data=out,
            tables=self.tables,
            models=self.models,
            name=self.name,
            paths=self.paths,
        )

    def isel(self, **kwargs) -> "Roi":
        out = dict()
        for k, elem in self.data.items():
            valid_keys = set(kwargs.keys()).intersection(elem.dims)
            out[k] = elem.isel(**{k: kwargs[k] for k in valid_keys})
        return Roi(
            _fp=self._fp,
            data=out,
            tables=self.tables,
            models=self.models,
            name=self.name,
            paths=self.paths,
        )

    def drop_sel(self, **kwargs) -> "Roi":
        out = dict()
        for k, elem in self.data.items():
            valid_keys = set(kwargs.keys()).intersection(elem.dims)
            out[k] = elem.drop_sel(**{k: kwargs[k] for k in valid_keys})
        return Roi(
            _fp=self._fp,
            data=out,
            tables=self.tables,
            models=self.models,
            name=self.name,
            paths=self.paths,
        )

    def drop_dim(self, dim: Literal["l", "c"]) -> "Roi":
        assert dim in ["l", "c"]
        if dim == "l":
            return Roi(
            _fp=self._fp,
                data={
                    self._LABELS_KEY: empty_data()[self._LABELS_KEY],
                    self._IMAGES_KEY: self[self._IMAGES_KEY],
                },
                tables=self.tables,
                models=self.models,
                name=self.name,
                paths=self.paths,
            )
        else:
            return Roi(
            _fp=self._fp,
                data={
                    self._LABELS_KEY: self[self._LABELS_KEY],
                    self._IMAGES_KEY: empty_data()[self._IMAGES_KEY],
                },
                tables=self.tables,
                models=self.models,
                name=self.name,
                paths=self.paths,
            )

    def map(self, func: Callable[[SpatialImage], SpatialImage], **kwargs) -> "Roi":
        return Roi(
            _fp=self._fp,
            data={k: func(v, **kwargs) for k, v in self.data.items()},
            tables=self.tables,
            models=self.models,
            name=self.name,
            paths=self.paths,
        )

    @property
    def dims(self) -> tuple[str]:
        return tuple(
            sorted(set(self.labels.dims).union(self.images.dims), key=SORT_KEY.dims)
        )

    @property
    def coords(self) -> dict[str, xr.DataArray]:
        coords = {**self.images.coords, **self.labels.coords}
        return {k: coords[k] for k in sorted(coords, key=SORT_KEY.dims)}

    @property
    def sizes(self) -> dict[Hashable, int]:
        sizes = {**dict(self.images.sizes), **dict(self.labels.sizes)}
        return {k: sizes[k] for k in sorted(sizes, key=SORT_KEY.dims)}

    @property
    def labels(self) -> SpatialImage:
        return self.data[self._LABELS_KEY]

    @property
    def images(self) -> SpatialImage:
        return self.data[self._IMAGES_KEY]

    def table(
        self,
        _objects: Sequence[str] | str | None = None,
        columns: Sequence[str] | str = "*",
    ) -> pl.DataFrame:
        if _objects is None:
            tables = self.tables
        elif isinstance(_objects, str):
            tables = {_objects: self.tables[_objects]}
        else:
            tables = {k: self.tables[k] for k in _objects}

        if isinstance(columns, str):
            columns_expr = pl.col(columns)
        else:
            columns_expr = [pl.col(c) for c in columns]
        idx_expr = [pl.col(idx) for idx in OBJECT_INDEX]

        filt_tables = []
        for name, table in tables.items():
            idx = table.select(idx_expr)
            data = table.select(columns_expr)
            filt_tables.append(
                data.select(pl.exclude(set(data.columns).intersection(ROI_INDEX)))
                .hstack(idx)
                .with_columns(pl.lit(name).alias("object"))
            )

        # tables = [table.select(pl.col(columns)).with_columns(pl.lit(name).alias('object'))
        #           for name, table in tables.items()]
        if filt_tables == []:
            return pl.DataFrame()
        return pl.concat(filt_tables, how="diagonal").select(
            [
                pl.col(["object", "label"]),
                pl.exclude(["object", "label"]),
            ]
        )

    @property
    def resources(self) -> dict[str, set[str]]:
        return {
            "channels": _get_channels_safe(self.images),
            "label_images": _get_channels_safe(self.labels),
        }

    # @property
    # def tables(self) -> pl.DataFrame:
    # return self.tables

    def __getitem__(self, key: str) -> SpatialImage:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        for k in self.data:
            yield (k)

    def collect(self) -> "Roi":
        """
        Load `pl.LazyFrames` into memory. Use `Roi.compute()` to also load raster data.
        """
        return Roi(
            _fp=self._fp,
            data=self.data,
            tables={
                k: (v.collect() if isinstance(v, pl.LazyFrame) else v)
                for k, v in self.tables.items()
            },
            models=self.models,
            name=self.name,
            paths=self.paths,
        )

    def compute(self, ill_corr_model: str | None = None) -> "Roi":
        """
        Load `Roi` into memory. In case of two-step models use
        `self._TWO_STEP_ILL_CORR_LABEL` as the foreground label image.

        Parameters
        ----------
        ill_corr_model : str, optional
            Apply an illumination correction model to channel stored under `Roi.models`.
            If `None` no correction is applied.

        Returns
        -------
        Roi
            In memory `Roi` with illumination correction applied.

        Raises
        ------
        ValueError
            If the required `ill_corr_model` was not found.
        """
        if ill_corr_model is not None:
            for channel in _get_channels_safe(self.images):
                if channel not in self.models["z_decay"][ill_corr_model]:
                    raise ValueError(
                        f"Model `{ill_corr_model}` not found for channel `{channel}`."
                    )

        data = {}
        for k, v in self.data.items():
            if k == self._IMAGES_KEY and ill_corr_model is not None:
                channels = []
                if (
                    "c" in list(self.data[k].coords.keys())
                    and "c" not in self.data[k].dims
                ):
                    channel_images = list(self.data[k].expand_dims("c"))
                else:
                    channel_images = list(self.data[k])
                for channel_image in channel_images:
                    model = self.models["z_decay"][ill_corr_model][
                        channel_image.c.item()
                    ]
                    if len(model._feature_names) == 2:
                        if "l" not in self.dims:
                            label_image = self.labels
                            assert (
                                label_image.l.item() == self._TWO_STEP_ILL_CORR_LABEL
                            ), f"`{self._TWO_STEP_ILL_CORR_LABEL}` not found!"
                        else:
                            label_image = self.labels.sel(
                                l=self._TWO_STEP_ILL_CORR_LABEL
                            )
                    else:
                        label_image = None
                    channels.append(
                        apply_model_to_channel(model, channel_image, label_image)
                    )
                if (
                    "c" in list(self.data[k].coords.keys())
                    and "c" not in self.data[k].dims
                ):
                    data[k] = xr.concat(channels, dim="c").compute().squeeze("c")
                else:
                    data[k] = xr.concat(channels, dim="c").compute()
            else:
                data[k] = v.compute()
        return Roi(
            _fp=self._fp,
            data=data,
            tables={
                k: (v.collect() if isinstance(v, pl.LazyFrame) else v)
                for k, v in self.tables.items()
            },
            models=self.models,
            name=self.name,
            paths=self.paths,
        )

    def __str__(self) -> str:
        return pformat(self)

    def _repr_small(self) -> str:
        return _repr_roi_small(_get_roi_meta(self))

    def __repr__(self, collapse=False, indent=0) -> str:
        from xarray.core.formatting import dim_summary

        rep = [f"< {self.__class__.__name__} {self.name!r}   ({dim_summary(self)}) >"]
        if not collapse:
            rep.append(_coordinates_repr(self))
        rep = map((lambda x: f"{' '*indent}{x}"), rep)
        # rep.append(coords_repr(self.labels.coords, col_width=6).split("\n")[1])
        # rep.extend(coords_repr(self.images.coords, col_width=6).split("\n")[1:])
        return "\n".join(rep)


def apply_z_decay_models_to_roi(
    models: Mapping[str, Model | None] | None,
    roi: Roi,
    two_step_label: str | None = None,
    suffix: str = "",
) -> Roi:
    if models is None:
        return roi

    data = {}
    for k, v in roi.data.items():
        if k == roi._IMAGES_KEY:
            channels = []
            if "c" in list(roi.data[k].coords.keys()) and "c" not in roi.data[k].dims:
                channel_images = list(roi.data[k].expand_dims("c"))
            else:
                channel_images = list(roi.data[k])
            for channel_image in channel_images:
                model = models[channel_image.c.item()]
                if model is None:
                    channel_image_corr = channel_image
                else:
                    assert model._feature_names is not None, "Model not fit."
                    if len(model._feature_names) == 2:
                        if "l" not in roi.dims:
                            label_image = roi.labels
                            assert (
                                label_image.l.item() == two_step_label
                            ), f"`{two_step_label}` not found!"
                        else:
                            label_image = roi.labels.sel(l=two_step_label)
                    else:
                        label_image = None
                    channel_image_corr = apply_model_to_channel(
                        model, channel_image, label_image
                    )
                channels.append(channel_image_corr)
            if "c" in list(roi.data[k].coords.keys()) and "c" not in roi.data[k].dims:
                data[k] = xr.concat(channels, dim="c").squeeze("c")
            else:
                data[k] = xr.concat(channels, dim="c")
        else:
            data[k] = v
    return Roi(
        data=data,
        tables={
            k: (v.collect() if isinstance(v, pl.LazyFrame) else v)
            for k, v in roi.tables.items()
        },
        models=roi.models,
        name=roi.name,
        paths=roi.paths,
    )


@dataclass(frozen=True, slots=True)
class SpatialDimMeta:
    shape: int = 0
    coord_range: tuple[float, float] = (
        np.inf,
        -np.inf,
    )  # Min & max values of dimensions in all arrays of the dset.
    origin: float = 0.0
    translation: float = 0.0

    def aggregate(self, other: "SpatialDimMeta") -> "SpatialDimMeta":
        return SpatialDimMeta(
            shape=max(self.shape, other.shape),
            coord_range=(
                min(self.coord_range[0], other.coord_range[0]),
                max(self.coord_range[1], other.coord_range[1]),
            ),
        )


@dataclass(frozen=True, slots=True)
class CatDimMeta:
    shape: int = 0
    coords: set = field(
        default_factory=set
    )  # Union of all coordinates across the dataset.

    def aggregate(self, other: "CatDimMeta") -> "CatDimMeta":
        return CatDimMeta(
            shape=max(self.shape, other.shape),
            coords=self.coords.union(other.coords),
        )


DimMeta: TypeAlias = SpatialDimMeta | CatDimMeta
RoiMeta: TypeAlias = dict[str, DimMeta]


def _get_roi_meta(roi: Roi) -> RoiMeta:
    roi_meta = dict()
    roi_meta["roi"] = CatDimMeta(shape=1, coords=set([roi.name]))
    for dim, coord in chain(roi.images.coords.items(), roi.labels.coords.items()):
        if dim in SPATIAL_DIMS:
            if len(coord.dims) == 0:
                roi_meta[dim] = SpatialDimMeta()
            roi_meta[dim] = SpatialDimMeta(
                shape=len(coord), coord_range=(coord.min().item(), coord.max().item())
            )
        else:
            roi_meta[dim] = CatDimMeta(shape=len(coord), coords=set(coord.values))
    return roi_meta


def _repr_roi_small(roi_meta: RoiMeta, indent=21, col_width=12, n_cols=4):
    FSTRING_ROI_SMALL = """{roi_coord} (roi):
    labels:  (l: {l_shape:>2}) {l_coord}
    images:  (c: {c_shape:>2}) {c_coord}
    spatial: (z: {z_shape}, y: {y_shape}, x: {x_shape})
    """

    return FSTRING_ROI_SMALL.format(
        roi_coord=list(roi_meta["roi"].coords)[0],
        c_coord=format_sequence_to_columns(
            sorted(roi_meta["c"].coords),
            indent=indent,
            col_width=col_width,
            n_cols=n_cols,
        ),
        c_shape=roi_meta["c"].shape,
        l_coord=format_sequence_to_columns(
            sorted(roi_meta["l"].coords),
            indent=indent,
            col_width=col_width,
            n_cols=n_cols,
        ),
        l_shape=roi_meta["l"].shape,
        z_shape=roi_meta["z"].shape,
        y_shape=roi_meta["y"].shape,
        x_shape=roi_meta["x"].shape,
    )


# TODO: .drop_dim, .drop_sel
@dataclass
class RoiMap(abc.Mapping):
    rois: Mapping[str, Roi] = field(default_factory=dict)
    name: str | None = field(default=None)

    @classmethod
    def from_files(
        cls,
        fns: list[PathLike[str]],
        level: int,
        features_root: PathLike[str] | None = None,
    ) -> "RoiMap":
        rois = {}
        for fn in fns:
            fn = Path(fn)
            roi = Roi.from_file(
                fn,
                level=level,
                features_root=None
                if features_root is None
                else Path(features_root) / fn.stem,
            )
            rois[fn.stem] = roi

        name = (fn.parent / "*").as_posix()
        return RoiMap(rois=rois, name=name)

    # TODO: fix and use this.
    def accumulate_site_info(self):
        aggs = {}
        for k in self:
            roi = self[k]
            aggs[k] = _get_roi_meta(roi)
        return aggs

    # def show(self, **kwargs):
    #     aggs = self.accumulate_site_info()
    #     print(
    #         self._ROI_MAP_FSTRING.format(
    #             name=self.__class__.__name__,
    #             title=self.name,
    #             dim_shape=f"(roi: {len(self)})",
    #         )
    #     )
    #     for k, v in aggs.items():
    #         print(_repr_roi_small(v))
    def _index_to_name(self, index: int) -> str:
        return sorted(list(self.keys()))[index]

    def isel(self, **kwargs) -> "RoiMap | Roi":
        if "roi" in kwargs:
            roi_indices = kwargs.pop("roi")
            if isinstance(roi_indices, list):
                roi_names = list(map(self._index_to_name, roi_indices))
            elif isinstance(roi_indices, slice | int):
                roi_names = sorted(list(self.keys()))[roi_indices]
            else:
                raise TypeError("Only `list[int]`, `slice` or `int` allowed.")
        else:
            roi_names = sorted(list(self.keys()))

        if isinstance(roi_names, str):
            return self.__getitem__(roi_names).isel(**kwargs)
        return RoiMap(
            rois={
                roi_name: self.__getitem__(roi_name).isel(**kwargs)
                for roi_name in roi_names
            },
            name=self.name,
        )
        #     RoiMap(rois=self.rois, name=self.name).sel(roi=roi_names).isel(**kwargs)
        #     return self.sel(roi=roi_names).isel(**kwargs)
        # else:
        #     return RoiMap()

    def sel(self, **kwargs) -> "RoiMap | Roi":
        if "roi" in kwargs:
            roi_names = kwargs.pop("roi")
            if isinstance(roi_names, str):
                return self.__getitem__(roi_names).sel(**kwargs)
            selected_rois = {
                roi_name: self.__getitem__(roi_name) for roi_name in roi_names
            }
        else:
            selected_rois = self.rois

        sliced_rois = {}
        for roi_name, roi in selected_rois.items():
            try:
                sliced_roi = roi.sel(**kwargs)
            except KeyError:
                continue
            sliced_rois[roi_name] = sliced_roi

        return RoiMap(rois=sliced_rois, name=self.name)

    # TODO: Aggregate those from all rois, not just first.
    @property
    def dims(self) -> tuple[str]:
        return ("roi", *self.first().dims)

    @property
    def sizes(self) -> dict[Hashable, int]:
        return {"roi": len(self), **self.first().sizes}

    def table(
        self,
        _objects: Sequence[str] | str | None = None,
        columns: Sequence[str] | str = "*",
    ) -> "pl.DataFrame":
        return pl.concat(
            [
                roi.table(_objects=_objects, columns=columns).with_columns(
                    pl.lit(name).alias("roi")
                )
                for name, roi in self.items()
            ],
            how="diagonal",
        ).select([pl.col(EXPERIMENT_INDEX), pl.exclude(EXPERIMENT_INDEX)])

    def first(self) -> "Roi":
        try:
            roi_name = list(self.rois.keys())[0]
            return self.rois[roi_name]
        except IndexError:
            return Roi()

    def __getitem__(self, key: str) -> "Roi":
        return self.rois[key]

    def map(self, func: Callable[[Roi], Roi]):
        pass

    def __len__(self) -> int:
        return len(self.rois)

    def __iter__(self) -> Iterator[str]:
        for k in self.rois:
            yield (k)

    def items(self) -> Iterator[tuple[str, Roi]]:
        for k in self:
            yield k, self[k]

    def compute(self) -> "RoiMap":
        return RoiMap(
            rois={k: v.compute() for k, v in self.rois.items()}, name=self.name
        )

    def __repr__(self) -> str:
        from xarray.core.formatting import dim_summary

        rep = [f"<<< {self.__class__.__name__} {self.name!r} ({dim_summary(self)}) >>>"]
        for name, roi in self.items():
            rep.append(f"{roi.__repr__(collapse=True, indent=2)}")
        return "\n".join(rep)


# %%
# def check(rows):
#     print(rows)
#     return rows


# def format_sequence_to_columns(
#     raw_sequence: Sequence[Any],
#     n_cols: int = 3,
#     col_width: int = 10,
#     indent: int = 0,
#     first_indent: bool = True,
#     sep: str = " | ",
#     left_padding: str | None = "| ",
#     right_padding: str | None = " |",
#     is_bracket_padding: bool = True,
#     column_names: Sequence[str] | None = None,
#     transpose: bool = False,
# ) -> str:

#     n_rows = int(np.ceil(len(raw_sequence) / n_cols))
#     if left_padding is None:
#         left_padding = ""
#     if right_padding is None:
#         right_padding = ""
#     else:
#         space_left = " " * len(left_padding)
#         space_right = " " * len(right_padding)
#         # print(repr(brace_left))
#         # print(repr(brace_right))

#     batches_of_rows = lambda x: t.partition(n=n_cols, seq=x, pad="")
#     batches_of_transposed_rows = lambda seq: [
#         islice(seq, i, None, n_cols) for i in range(n_cols)
#     ]

#     add_header_row = lambda rows: chain((column_names,), (row for row in rows))

#     replace_too_long_element = (
#         lambda x: x if len(x) <= col_width else f"{x[:(col_width-1)]}."
#     )

#     format_element_left = partial("{:<{col_width}}".format, col_width=col_width)
#     format_element_right = partial("{:>{col_width}}".format, col_width=col_width)
#     format_elements_in_row_left = c.map(format_element_left)
#     format_elements_in_row_right = c.map(format_element_right)

#     sequence_of_row_formatters = chain(
#         (format_elements_in_row_right,), repeat(format_elements_in_row_left)
#     )
#     sequence_of_indents = (
#         repeat(indent) if first_indent else chain((0,), repeat(indent))
#     )

#     format_rows_by_sequence = lambda rows: [
#         formatter(row) for formatter, row in zip(sequence_of_row_formatters, rows)
#     ]
#     compute = lambda rows: [list(row) for row in rows]

#     format_elements_in_rows = c.map(format_elements_in_row_left)

#     join_elements_to_rows = c.map(partial(conditional_separator_join, sep=sep))
#     # join_elements_old = c.map(sep.join)
#     add_brackets = lambda rows: [
#         "{left}{content}{right}".format(left=bleft, content=row, right=bright)
#         for row, bleft, bright in zip(
#             rows,
#             chain((left_padding,), repeat(space_left)),
#             # chain(repeat(space_right, len(list(rows)) - 2), (right_padding)),
#             chain(repeat(space_right, n_rows - 2), repeat(right_padding)),
#         )
#     ]
#     add_padding = lambda rows: [
#         "{left}{content}{right}".format(left=bleft, content=row, right=bright)
#         for row, bleft, bright in zip(
#             rows,
#             repeat(left_padding),
#             repeat(right_padding),
#         )
#     ]

#     add_row_indentation = lambda rows: [
#         "{pad_char:>{indent}}{content}".format(pad_char="", indent=indent, content=row)
#         for row, indent in zip(rows, sequence_of_indents)
#     ]
#     join_rows = "\n".join
#     # c.map(partial("{ {pad_value}:>{indent}}{0:}".format, indent=indent, pad_value=' '))

#     rows = t.pipe(
#         raw_sequence,
#         c.map(replace_too_long_element),
#         batches_of_transposed_rows if transpose else batches_of_rows,
#         compute,
#         format_elements_in_rows,
#         # add_header_row,
#         # format_rows_by_sequence,
#         # check,
#         compute,  # Following `join_elements_to_rows` acts weird if we don't load to memory.Might have something to do with lazy evaluation, the old ver
#         # check,
#         join_elements_to_rows,
#         list,
#         (add_brackets if is_bracket_padding else add_padding),
#         # check,
#         add_row_indentation,
#         "\n".join,
#         # check,
#     )
#     # return list(map(list, list(rows)))
#     return rows


# import io
# from itertools import islice, pairwise


# def conditional_separator_join(
#     seq: Sequence[str],
#     sep: str = ", ",
#     false_sep: str | None = None,
#     condition: Callable = lambda a, b: not (str.isspace(a) and str.isspace(b)),
# ) -> str:
#     if false_sep is None:
#         false_sep = " " * len(sep)
#     res = io.StringIO()
#     res.write(next(iter(seq)))
#     # for a, b in zip(seq, islice(seq, 1, None, None)):
#     for a, b in pairwise(seq):
#         if condition(a, b):
#             res.write(sep)
#         else:
#             res.write(false_sep)
#         res.write(b)
#     return res.getvalue()


# res = format_sequence_to_columns(
#     sorted(raw_sequence, key=lambda x: (int(x.split("-")[-1]), x)), col_width=10
# )
# # list(res)
# print(res)
# # %%
# def get_canvas(x=100, y=20, symbols=" "):
#     return [[random.choice(symbols) for _ in range(x)] for _ in range(y)]


# # %%
# X, Y = 25, 10
# CANVAS = get_canvas(X, Y)
# FLIP_MAP = {"|": "-", "-": "|"}


# def transpose():
#     global CANVAS
#     global X
#     global Y
#     res = []
#     for i in range(X):
#         row = []
#         for j in range(Y):
#             row.append(FLIP_MAP.get(CANVAS[j][i], CANVAS[j][i]))
#         res.append(row)
#     CANVAS = res
#     Y, X, = (
#         X,
#         Y,
#     )
#     show()


# def draw(x, y, symbol="*"):
#     CANVAS[y][x] = symbol


# def write(x, y, text=None):
#     if text is None:
#         print("Nothing to write....")

#     dx = len(text)

#     CANVAS[y][x : x + dx] = list(text)


# def write_vertical(x, y, text):
#     dy = len(text)
#     for y, letter in zip(range(y, y + dy), text):
#         draw(x, y, letter)


# def hline(y, x_min=0, x_max=len(CANVAS[0]), symbol="-"):
#     write(x_min, y, symbol * (x_max - x_min))


# def vline(x, y_min=0, y_max=len(CANVAS), symbol="|"):
#     write_vertical(x, y_min, symbol * (y_max - y_min))


# # def show():
# #     print("-" * len(CANVAS[0]))
# #     print("\n".join(map("".join, CANVAS)))
# #     print("-" * len(CANVAS[0]))


# def show():
#     fspec_side = "|{}|"
#     fspec_corner = "+{}+"

#     v_line = "-" * X
#     rows = map("".join, CANVAS)

#     for i, row in enumerate(chain((v_line,), rows, (v_line,))):
#         if i == 0 or i == Y + 1:
#             fspec = fspec_corner
#         else:
#             fspec = fspec_side
#         print(fspec.format(row))


# def clear():
#     global CANVAS
#     CANVAS = get_canvas(X, Y)
#     show()


# # draw(0, 0)
# write(3, 0, "hello")
# hline(7)
# hline(3)
# # write(3, 2, "hello")
# # write(3, 3, "hello")
# # write_vertical(3, 3, "hello")
# # # vline(0)
# vline(8)
# # debug()
# # canvas[0]
# show()
# # %%
# import random
# import string

# compute = lambda rows: [list(row) for row in rows]

# random.seed(42)


# def random_string(size: int | tuple[int, int] = (4, 8), empty=False):
#     if empty:
#         choose_from = " "
#     else:
#         choose_from = string.ascii_letters
#     return "".join(
#         [random.choice(choose_from) for _ in range(random.randint(*sorted(size)))]
#     )


# def random_strings(n=20, size: int | tuple[int, int] = (4, 8), empty_perc: float = 0.4):
#     return [
#         random_string(size=size, empty=random.random() < empty_perc) for _ in range(n)
#     ]


# def random_strings_lazy(
#     n=20, size: int | tuple[int, int] = (4, 8), empty_perc: float = 0.4
# ):
#     return (
#         random_string(size=size, empty=random.random() < empty_perc) for _ in range(n)
#     )


# def groups_of_strings(
#     n_groups: int = 20,
#     n_per_group: int = 8,
#     size: int | tuple[int, int] = (5, 5),
#     empty_perc: float = 0.2,
# ):
#     return [
#         random_strings_lazy(n=n_per_group, size=size, empty_perc=empty_perc)
#         for _ in range(n_groups)
#     ]


# def groups_of_strings_lazy(
#     n_groups: int = 20,
#     n_per_group: int = 8,
#     size: int | tuple[int, int] = (5, 5),
#     empty_perc: float = 0.2,
# ):
#     return (
#         random_strings_lazy(n=n_per_group, size=size, empty_perc=empty_perc)
#         for _ in range(n_groups)
#     )


# res = t.pipe(
#     groups_of_strings(empty_perc=0.5, n_per_group=4, n_groups=5),
#     c.map(list),
#     # list,
#     # c.map(conditional_separator_join),
#     c.map(sep.join),
#     list,
# )
# # list(random_strings_lazy())
# # groups_of_strings()
# # compute(groups_of_strings_lazy())
# # random_strings()
# # list(map(list, [range(5) for _ in range(10)]))
# # random_strings()
# # compute([range(5) for _ in range(10)])
# res
# # %%
# seq = [list(roi_info["c"])[:14], list(roi_info["c"])[:14], list(roi_info["c"])[:14]]

# list(map(conditional_separator_join, seq))

# # %%
# sep.join(raw_sequence)
# # %%
# dater = [
#     ["asdf", "", "asdf", "    ", "     ", "    "],
#     ["asdfasd", "asdfasd", "asdfsadf", "sadfsadf", ""],
#     ["      ", "asdfasd", "asdfas", "asdfsad", "     ", " ", "a"],
# ]
# list(map(conditional_separator_join, dater))
# # list(map(sep.join, dater))
# # %%
# join_elements_to_row = c.map(partial(conditional_separator_join, sep=", "))
# join_elements_old = c.map(sep.join)
# element_fmt = partial("{:>{col_width}}".format, col_width=12)
# format_batched_elems = c.map(c.map(element_fmt))

# dater = [
#     ["asdf", "", "asdf", "    ", "     ", "    "],
#     ["asdfasd", "asdfasd", "asdfsadf", "sadfsadf", ""],
#     ["      ", "asdfasd", "asdfas", "asdfsad", "     ", " ", "a"],
# ]
# dater = t.pipe(dater, format_batched_elems)
# # list(map(list, list(dater)))
# # dater2 = list(map(list, dater))
# # list(join_elements_old(dater))
# list(join_elements_to_row(dater))
# # %%
# f = c.map(c.filter(lambda x: x != ""))
# j = c.map(sep.join)
# t = c.compose(j, f)

# dater = [["asdf", "asdf", "sdfg", "     ", "     ", "     "]] * 3
# res = j(f(dater))
# res2 = t(dater)
# list(map(conditional_separator_join, sequence))
# # def format_sequence_column(
# #     raw_strings: Sequence[str],
# #     n_cols: int = 4,
# #     col_width: int = 20,
# #     indent: int = 0,
# #     separator=", ",
# # ) -> tuple[str, ...]:
# #     # padded_strings = pad_strings(raw_strings, width=col_width)

# #     row_gen = partial(t.partition, n=n_cols, pad="")
# #     element_fmt = partial("{:>{col_width}}".format, col_width=col_width)
# #     row_fmt = lambda x: separator.join(x)

# #     return element_fmt, row_gen, row_fmt


# # %%
# element_fmt, row_gen, row_fmt = format_sequence_column(
#     raw_strings, col_width=10, n_cols=5
# )
# rows = c.flip(row_gen)(raw_strings)
# list(map(element_fmt, next(rows)))
# # %%
# # @dataclass(slots=True)
# # class DictRoi2(abc.Mapping):
# #     labels: SpatialImage
# #     images: SpatialImage

# #     @classmethod
# #     def from_file(cls, fn: str, level: int = 1) -> "DictRoi2":
# #         return DictRoi2(
# #             labels=xr.concat(load_labels(fn, level=1), dim="c").rename({"c": "l"}),
# #             images=xr.concat(load_channels(fn, level=level), dim="c"),
# #         )

# #     def sel(self, **kwargs) -> "DictRoi2":
# #         return DictRoi2(
# #             labels=self.labels.sel(
# #                 **{
# #                     k: kwargs[k]
# #                     for k in set(kwargs.keys()).intersection(self.labels.dims)
# #                 }
# #             ),
# #             images=self.images.sel(
# #                 **{
# #                     k: kwargs[k]
# #                     for k in set(kwargs.keys()).intersection(self.images.dims)
# #                 }
# #             ),
# #         )

# #     def __getitem__(self, key: str) -> SpatialImage:
# #         return getattr(self, key)

# #     def __len__(self) -> int:
# #         return len(self.labels) + len(self.images)

# #     def __iter__(self) -> Iterator[SpatialImage]:
# #         for e in chain.from_iterable([self.labels, self.images]):
# #             yield e

# #     def compute(self) -> "DictRoi2":
# #         return DictRoi2(
# #             **{k: getattr(self, k).compute() for k in self.__dataclass_fields__.keys()}
# #         )


# # @dataclass
# # class RoiList(abc.Sequence):
# #     rois: list[Roi]

# #     def __post_init__(self):
# #         print("hi")
# #         return self

# #     @classmethod
# #     def from_files(cls, fns: list[str], level: int) -> "RoiList":
# #         return RoiList(rois=load_rois_stacked(fns, level=level))

# #     def sel(self, *args, **kwargs) -> "RoiList":
# #         return RoiList(rois=sel(self.rois, *args, **kwargs))

# #     def first(self) -> "Roi":
# #         return self.rois[0]

# #     def __getitem__(self, index: int) -> "Roi":
# #         return self.rois[index]

# #     def __len__(self) -> int:
# #         return len(self.rois)