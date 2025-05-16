# %%
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import polars as pl
import polars.selectors as cs
from polars._typing import JoinStrategy, JoinValidation, SelectorType
from typing_extensions import Self
from zfish.features.polars_utils import split_and_melt_column_names_on
from zfish.multi_table.schemas_id_v2 import FrameId
from zfish.multi_table.schemas_v2 import (
    IDX_SEL,
    LABEL_OBJECT_SEL,
    PARSED_SCHEMA,
    build_lazy_tables,
    sel,
)
from zfish.multi_table.table_base_mixin import (
    AnyFrame,
    MFrameBaseMixin,
    join,
    safe_collect,
    safe_lazy,
)
from zfish.preprocessing.types import (
    IntoPath,
    StrMapDict,
    StrMapFunc,
    id_,
)

if TYPE_CHECKING:
    from typing import Callable

    import pandera as pa
    from zfish.multi_table.schema_metadata import FrameMeta

pl.enable_string_cache()

WriteStrategy: TypeAlias = Literal["overwrite_all", "overwrite_non_empty", "raise"]

USE_PYARROW = False

TABLES_PATH = Path(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\multi_table\data"
)

NEW_TABLES_PATH = (
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\multi_table\mframe"
)

DEFAULT_MFRAME_PATH = TABLES_PATH


TABLE_NAME_MAP_ = {
    "wells": "Resources.wells",
    "rois": "Resources.rois",
    "channels": "Resources.channels",
    "object_types": "Resources.object_types",
    "multiscale_levels": "Resources.multiscale_levels",
    "images": "Resources.images",
    # "decay_models": "params.decay_model",
    "z_models": "Resources.z_models",
    "t_models": "Resources.t_models",
    "label_images": "Resources.label_images",
    "label_objects": "Resources.label_objects",
    "hierarchy": "Resources.hierarchy",
    "classifiers": "Resources.classifiers",
    "label": "Features.label",
    "distance": "Features.distance",
    "intensity": "Features.intensity",
    "correlation": "Features.correlation",
    "density_count": "Features.density_count",
    "density_distance": "Features.density_distance",
    "classifier": "Features.classifier",
}

TABLE_NAME_MAP = {
    # "roots": "resources.roots.v2",
    # "plates": "Resources.plates.v2",
    **TABLE_NAME_MAP_,
    "wells": "Resources.wells.v2",
    "rois": "Resources.rois.v2",
    # "images": "resources.images.v2",
    "label_objects": "Resources.label_objects.v2",
}

MFRAMES_NAME_MAP: dict[tuple[str, str], [str]] = {
    (v.split(".")[0], ".".join(v.split(".")[1:])): k for k, v in TABLE_NAME_MAP.items()
}

RESOURCES_NAME_MAP: dict[str, [str]] = {
    k[1]: v for k, v in MFRAMES_NAME_MAP if k == "Resources"
}
FEATURES_NAME_MAP: dict[str, [str]] = {
    k[1]: v for k, v in MFRAMES_NAME_MAP if k == "Features"
}


def scan_tables(
    path: str,
    table_names: tuple[str, ...] | None = None,
    parse_ids: bool = True,
    **polars_kwargs,
) -> dict[str, pl.LazyFrame]:
    if table_names is not None:
        tbls = {
            v.stem: pl.scan_parquet(v, **polars_kwargs)
            for v in Path(path).rglob("*.parquet")
            if v.stem in table_names
        }
    else:
        tbls = {
            v.stem: pl.scan_parquet(v, **polars_kwargs)
            for v in Path(path).rglob("*.parquet")
        }
    if parse_ids:
        return parse_table_name_to_ids(tbls)
    return tbls


def parse_table_name_to_ids(tables: dict[str, AnyFrame]) -> dict[FrameId, AnyFrame]:
    return {FrameId.from_filename(name): table for name, table in tables.items()}


def rename_tables_on_disk(
    root_path: IntoPath,
    rename_map: StrMapDict | None = None,
    rename_func: StrMapFunc | None = None,
) -> dict[str, str]:
    status_dict = {}
    if rename_func is None:
        rename_func = id_
    if rename_map is None:
        rename_map = {}
    for name_path in Path(root_path).glob("*.parquet"):
        name = name_path.stem
        new_name = rename_map.get(name, name)
        new_name = rename_func(new_name)

        if new_name != name:
            out_path = name_path.parent / f"{new_name}{name_path.suffix}"
            name_path.rename(out_path)
            status_dict[name] = new_name
    return status_dict


def write_tables(
    path: Path | str,
    tables: dict[str, pl.DataFrame],
    overwrite: bool = False,
    write_empty: bool = False,
    use_pyarrow: bool = USE_PYARROW,
):
    if not write_empty:
        tables = {k: v for k, v in tables.items() if v.height > 0}
    if not overwrite:
        for name in tables.keys():
            table_path = Path(path) / f"{name}.parquet"
            if table_path.exists():
                raise IOError(
                    f"Can't write tables, {name!r} already exists at {table_path!r}"
                )
    for name, table in tables.items():
        table_path = Path(path) / f"{name}.parquet"
        table_path.parent.mkdir(exist_ok=True)
        table.write_parquet(table_path, use_pyarrow=use_pyarrow)
        # if write_empty:
        #     table.write_parquet(table_path, use_pyarrow=use_pyarrow)
        # else:
        #     if table.height > 0:
        #         table.write_parquet(table_path, use_pyarrow=use_pyarrow)


class LazyMFrameBase(MFrameBaseMixin):
    @classmethod
    def from_schemas(cls) -> Self:
        dfs = {
            k.to_localname(): v
            for k, v in build_lazy_tables(cls.__mframe_schema__).items()
        }
        # return dfs
        return cls(**dfs)

    @classmethod
    def from_tables(
        cls,
        tables: dict[FrameId, AnyFrame],
        validate_schema: bool = True,
        strict: bool = False,
    ) -> Self:
        correct_tables = cls._validate_right_frames_in_tables(
            tables,
            strict=strict,
        )
        valid_tables = cls._validate_table_schemas(
            correct_tables, validate_schema=validate_schema
        )
        dfs = {k.to_localname(): safe_lazy(v) for k, v in valid_tables.items()}
        return cls(**dfs)

    @classmethod
    def from_path(
        cls,
        path: str = NEW_TABLES_PATH,
        validate_schema: bool = True,
        strict: bool = False,
    ) -> Self:
        tables = scan_tables(
            path,
        )
        return cls.from_tables(tables, validate_schema=validate_schema, strict=strict)


class MFrameBase(MFrameBaseMixin):
    @classmethod
    def from_schemas(cls) -> Self:
        dfs = {
            k.to_localname(): safe_collect(v)
            for k, v in build_lazy_tables(cls.__mframe_schema__).items()
        }
        return cls(**dfs)

    @classmethod
    def from_tables(
        cls,
        tables: dict[FrameId, AnyFrame],
        validate_schema: bool = True,
        strict: bool = False,
    ) -> Self:
        correct_tables = cls._validate_right_frames_in_tables(
            tables,
            strict=strict,
        )
        valid_tables = cls._validate_table_schemas(
            correct_tables, validate_schema=validate_schema
        )
        dfs = {k.to_localname(): safe_collect(v) for k, v in valid_tables.items()}
        return cls(**dfs)

    @classmethod
    def from_path(
        cls,
        path: str = NEW_TABLES_PATH,
        validate_schema: bool = True,
        strict: bool = False,
    ) -> Self:
        tables = scan_tables(
            path,
        )
        return cls.from_tables(tables, validate_schema=validate_schema, strict=strict)

    def write_tables(
        self,
        path: str,
        overwrite: bool = False,
        write_empty: bool = False,
        use_pyarrow: bool = USE_PYARROW,
    ):
        write_tables(
            path=path,
            tables={
                f"{self.__class__.__name__}.{table_name}": table
                for table_name, table in self._tables.items()
            },
            overwrite=overwrite,
            write_empty=write_empty,
            use_pyarrow=use_pyarrow,
        )


FrameBackend: TypeAlias = Literal["polars", "custom"]


def safe_collect_to(df: AnyFrame, backend: FrameBackend = "polars"):
    if backend == "polars":
        return safe_collect(df)
    else:
        raise NotImplementedError(f"{backend!r} not yet implemented!")


def safe_lazy_to(df: AnyFrame, backend: FrameBackend = "polars"):
    if backend == "polars":
        return safe_lazy(df)
    else:
        raise NotImplementedError(f"{backend!r} not yet implemented!")


def make_table_dataclasses_from_schema(
    name,
    table_schemas: dict[FrameId, "pa.Schema"],
    lazy_name: str | None = None,
    cardinality_repr: bool = True,
    data_frame_namespace: dict[str, Any] | None = None,
    lazy_frame_namespace: dict[str, Any] | None = None,
    shared_namespace: dict[str, Any] | None = None,
    version: str | None = None,
    # tables: tuple[AnyFrame, ...],
    backend: FrameBackend = "polars",
) -> tuple[type[MFrameBaseMixin], type[MFrameBaseMixin]]:
    """ "Make a pair of dataclasses with tables attached. Names will be f'{name}' & f'Lazy{name}'"""

    if lazy_name is None:
        lazy_name = f"Lazy{name}"
    if data_frame_namespace is None:
        data_frame_namespace = {}
    if lazy_frame_namespace is None:
        lazy_frame_namespace = {}
    if shared_namespace is None:
        shared_namespace = {}

    if backend == "polars":
        data_frame_type = pl.DataFrame
        lazy_frame_type = pl.LazyFrame
    else:
        raise NotImplementedError(f"{backend!r} not yet implemented!")

    TableDataClass = make_dataclass(
        name,
        fields=[
            (table_name.to_localname(), data_frame_type) for table_name in table_schemas
        ],
        bases=(MFrameBase,),
        namespace={
            **{
                "lazy": lambda self: LazyTableDataClass(
                    **{k: safe_lazy_to(df, backend) for k, df in self._tables.items()}
                ),
                "collect": lambda self: self,
            },
            **data_frame_namespace,
            **shared_namespace,
        },
        frozen=True,
        repr=False,
    )
    TableDataClass.__mframe_name__ = name
    TableDataClass.__mframe_schema__ = table_schemas
    TableDataClass.__mframe_meta__ = {
        k: v.get_metadata().get(k.to_filename(), v.get_metadata().get(None))[
            "dataframe"
        ]
        for k, v in table_schemas.items()
    }

    LazyTableDataClass = make_dataclass(
        lazy_name,
        fields=[
            (table_name.to_localname(), lazy_frame_type) for table_name in table_schemas
        ],
        bases=(LazyMFrameBase,),
        namespace={
            **{
                "collect": lambda self: TableDataClass(
                    **{
                        k: safe_collect_to(df, backend)
                        for k, df in self._tables.items()
                    }
                ),
                "lazy": lambda self: self,
            },
            **lazy_frame_namespace,
            **shared_namespace,
        },
        frozen=True,
        repr=False,
    )
    LazyTableDataClass.__mframe_name__ = name
    LazyTableDataClass.__mframe_schema__ = TableDataClass.__mframe_schema__
    LazyTableDataClass.__mframe_meta__ = TableDataClass.__mframe_meta__
    return TableDataClass, LazyTableDataClass


Resources, LazyResources = make_table_dataclasses_from_schema(
    "Resources",
    {k: v for k, v in PARSED_SCHEMA.items() if k.mframe == "Resources"},
)

Features, LazyFeatures = make_table_dataclasses_from_schema(
    "Features",
    {k: v for k, v in PARSED_SCHEMA.items() if k.mframe == "Features"},
)

Regions, LazyRegions = make_table_dataclasses_from_schema(
    "Regions",
    {k: v for k, v in PARSED_SCHEMA.items() if k.mframe == "Regions"},
)

Dims, LazyDims = make_table_dataclasses_from_schema(
    "Dims",
    {
        k: v
        for k, v in PARSED_SCHEMA.items()
        if k.mframe == "Resources"
        and k.frame in ["rois", "multiscale_levels", "channels", "object_types"]
    },
)

Root, LazyRoot = make_table_dataclasses_from_schema(
    "Root",
    {k: v for k, v in PARSED_SCHEMA.items() if k.mframe == "Root"},
)


def map_schema_meta(
    schema: "pa.Schema", func: "Callable[[FrameMeta], FrameMeta]"
) -> "pa.Schema":
    new_schema = schema.add_columns({})
    schema_meta = schema.get_metadata()
    new_schema_meta = {**schema_meta}
    root_key = list(new_schema_meta.keys())[0]
    frame_meta = new_schema_meta[root_key]["dataframe"]
    new_frame_meta = func(frame_meta)
    # new_schema_meta[root_key]["dataframe"] = new_frame_meta
    new_schema.metadata = new_frame_meta
    return new_schema


def derive_table_dataclasses(
    cls: LazyMFrameBase | MFrameBase,
    **kwargs,
):
    return derive_tables_dataclass(cls, **kwargs)


def derive_tables_dataclass(
    cls: LazyMFrameBase | MFrameBase,
    table_schemas: dict[FrameId, "pa.Schema"] = None,
    # map_table_schemas: "dict[FrameId | None, Callable[[pa.Schema], pa.Schema]]" = None,
    metadata_mapping: "dict[FrameId | None, Callable[[FrameMeta], FrameMeta]] | None" = None,
    name: str | None = None,
    lazy_name: str | None = None,
    cardinality_repr: bool = True,
    data_frame_namespace: dict[str, Any] | None = None,
    lazy_frame_namespace: dict[str, Any] | None = None,
    shared_namespace: dict[str, Any] | None = None,
    how: Literal["add", "replace"] = "add",
) -> tuple[type[MFrameBase], type[LazyMFrameBase]]:
    if table_schemas is None:
        table_schemas = {}
    if name is None:
        name = cls.__name__
        if lazy_name is not None:
            raise ValueError("Provide either none or both of `name` and `lazy_name`")
        lazy_name = f"Lazy{name}"
    if data_frame_namespace is None:
        data_frame_namespace = {}
    if lazy_frame_namespace is None:
        lazy_frame_namespace = {}
    if shared_namespace is None:
        shared_namespace = {}

    old_schemas = {
        k: v.add_columns({}) for k, v in cls._schemas.items()
    }  # just to be save and have a copy
    if how == "add":
        new_schemas = {**old_schemas, **table_schemas}
    elif how == "replace":
        new_schemas = table_schemas
    else:
        raise ValueError(f"Unknown strategy: {how}")
    if metadata_mapping is not None:
        if None in metadata_mapping:
            default_map = metadata_mapping[None]
        else:
            default_map = id_
    else:
        metadata_mapping = {}
        default_map = id_

    new_schemas = {
        name: map_schema_meta(
            map_schema_meta(schema, default_map), metadata_mapping.get(name, id_)
        )
        for name, schema in new_schemas.items()
    }

    return make_table_dataclasses_from_schema(
        name=name,
        table_schemas=new_schemas,
        lazy_name=lazy_name,
        cardinality_repr=cardinality_repr,
        data_frame_namespace=data_frame_namespace,
        lazy_frame_namespace=lazy_frame_namespace,
        shared_namespace=shared_namespace,
    )


FeaturesWide, LazyFeaturesWide = derive_table_dataclasses(
    Features,
    name="FeaturesWide",
    metadata_mapping={
        None: lambda x: x.from_template(
            pk={"comps": ("roi", "o", "label"), "name": "label_object"}
        )
    },
)

R2, LazyR2 = derive_table_dataclasses(
    Resources,
    name="R2",
    metadata_mapping={
        FrameId("Resources", "label_objects"): lambda x: x.from_template(
            pk={"comps": ("roi", "o", "label"), "name": "label_object"}
        )
    },
)


def rename_columns(df: AnyFrame, rename_map: dict[str, str]) -> AnyFrame:
    return df.rename({k: v for k, v in rename_map.items() if k in df})


def replace(
    df: AnyFrame, rename_map: dict[str, str], out_dtype: pl.DataType | None = None
) -> AnyFrame:
    return df.rename({k: v for k, v in rename_map.items() if k in df})


# def make_table_dataclasses(
#     name,
#     table_names: tuple[str, ...],
#     lazy_name: str | None = None,
#     cardinality_repr: bool = True,
#     data_frame_namespace: dict[str, Any] | None = None,
#     lazy_frame_namespace: dict[str, Any] | None = None,
#     shared_namespace: dict[str, Any] | None = None,
#     # tables: tuple[AnyFrame, ...],
#     backend: FrameBackend = "polars",
# ):
#     """ "Make a pair of dataclasses with tables attached. Names will be f'{name}' & f'Lazy{name}'"""
#     if name.startswith("Lazy"):
#         raise ValueError("Name must not start with 'Lazy'!")
#     if lazy_name is None:
#         lazy_name = f"Lazy{name}"
#     if data_frame_namespace is None:
#         data_frame_namespace = {}
#     if lazy_frame_namespace is None:
#         lazy_frame_namespace = {}
#     if shared_namespace is None:
#         shared_namespace = {}

#     if backend == "polars":
#         data_frame_type = pl.DataFrame
#         lazy_frame_type = pl.LazyFrame
#     else:
#         raise NotImplementedError(f"{backend!r} not yet implemented!")

#     TableDataClass = make_dataclass(
#         name,
#         fields=[(table_name, data_frame_type) for table_name in table_names],
#         bases=(MFrameBase,),
#         namespace={
#             **{
#                 "lazy": lambda self: LazyTableDataClass(
#                     **{k: safe_lazy_to(df, backend) for k, df in self._tables.items()}
#                 ),
#                 "collect": lambda self: self,
#             },
#             **data_frame_namespace,
#             **shared_namespace,
#         },
#         frozen=True,
#         repr=False,
#     )

#     LazyTableDataClass = make_dataclass(
#         lazy_name,
#         fields=[(table_name, lazy_frame_type) for table_name in table_names],
#         bases=(LazyMFrameBase,),
#         namespace={
#             **{
#                 "collect": lambda self: TableDataClass(
#                     **{
#                         k: safe_collect_to(df, backend)
#                         for k, df in self._tables.items()
#                     }
#                 ),
#                 "lazy": lambda self: self,
#             },
#             **lazy_frame_namespace,
#             **shared_namespace,
#         },
#         frozen=True,
#         repr=False,
#     )
#     return TableDataClass, LazyTableDataClass


# Resources_, LazyResources_ = make_table_dataclasses(
#     "Resources",
#     table_names=[
#         # single,
#         "plates",
#         "wells",
#         "rois",
#         "object_types",
#         "channels",
#         "multiscale_levels",
#         # compound
#         # intensity
#         "images",
#         "z_models",
#         "t_models",
#         # objects
#         "label_images",
#         "label_objects",
#         # stratifying objects
#         "classifiers",
#         # object relationships
#         "hierarchy",
#         # "nhood",
#     ],
# )

# Features_, LazyFeatures_ = make_table_dataclasses(
#     "Features",
#     table_names=[
#         "label",
#         "distance",
#         "intensity",
#         "correlation",
#         "density_count",
#         "density_distance",
#         "classifier",
#     ],
# )


# @dataclass(frozen=True, repr=False)
# class LazyResources_(LazyMFrameBase):
#     wells: pl.LazyFrame
#     rois: pl.LazyFrame
#     channels: pl.LazyFrame
#     object_types: pl.LazyFrame
#     multiscale_levels: pl.LazyFrame
#     images: pl.LazyFrame
#     z_models: pl.LazyFrame
#     t_models: pl.LazyFrame
#     label_images: pl.LazyFrame
#     label_objects: pl.LazyFrame
#     hierarchy: pl.LazyFrame
#     classifiers: pl.LazyFrame

#     def collect(self) -> "Resources":
#         return Resources(**{k: safe_collect(v) for k, v in self._tables.items()})

#     def from_template(self, new_tables: dict[str, AnyFrame]):
#         for table_name in new_tables.keys():
#             if table_name not in self._table_names:
#                 raise ValueError(
#                     f"Invalid table name: {table_name} not in {self._table_names}"
#                 )
#         return LazyResources(
#             **{**self._tables, **{k: safe_lazy(v) for k, v in new_tables.items()}}
#         )


# @dataclass(frozen=True, repr=False)
# class Resources_(MFrameBase):
#     wells: pl.DataFrame
#     rois: pl.DataFrame
#     channels: pl.DataFrame
#     object_types: pl.DataFrame
#     multiscale_levels: pl.DataFrame
#     images: pl.DataFrame
#     z_models: pl.DataFrame
#     t_models: pl.DataFrame
#     label_images: pl.DataFrame
#     label_objects: pl.DataFrame
#     hierarchy: pl.DataFrame
#     classifiers: pl.DataFrame

#     def lazy(self) -> "LazyResources":
#         return LazyResources(**{k: safe_lazy(v) for k, v in self._tables.items()})

#     def from_template(self, new_tables: dict[str, AnyFrame]):
#         for table_name in new_tables.keys():
#             if table_name not in self._table_names:
#                 raise ValueError(
#                     f"Invalid table name: {table_name} not in {self._table_names}"
#                 )
#         return Resources(
#             **{**self._tables, **{k: safe_collect(v) for k, v in new_tables.items()}}
#         )


# @dataclass(frozen=True, repr=False)
# class LazyFeatures_(LazyMFrameBase):
#     label: pl.LazyFrame
#     distance: pl.LazyFrame
#     intensity: pl.LazyFrame
#     correlation: pl.LazyFrame
#     density_count: pl.LazyFrame
#     density_distance: pl.LazyFrame
#     classifier: pl.LazyFrame

#     def collect(self) -> "Features":
#         return Features(**{k: safe_collect(v) for k, v in self._tables.items()})

#     def from_template(self, new_tables: dict[str, AnyFrame]) -> "LazyFeatures":
#         for table_name in new_tables.keys():
#             if table_name not in self._table_names:
#                 raise ValueError(
#                     f"Invalid table name: {table_name} not in {self._table_names}"
#                 )
#         return LazyFeatures(
#             **{**self._tables, **{k: safe_lazy(v) for k, v in new_tables.items()}}
#         )


# @dataclass(frozen=True, repr=False)
# class Features_(MFrameBase):
#     label: pl.DataFrame
#     distance: pl.DataFrame
#     intensity: pl.DataFrame
#     correlation: pl.DataFrame
#     density_count: pl.DataFrame
#     density_distance: pl.DataFrame
#     classifier: pl.DataFrame

#     def lazy(self) -> "LazyFeatures":
#         return LazyFeatures(**{k: safe_collect(v) for k, v in self._tables.items()})

#     def from_template(self, new_tables: dict[str, AnyFrame]) -> "Features":
#         for table_name in new_tables.keys():
#             if table_name not in self._table_names:
#                 raise ValueError(
#                     f"Invalid table name: {table_name} not in {self._table_names}"
#                 )
#         return Features(
#             **{**self._tables, **{k: safe_collect(v) for k, v in new_tables.items()}}
#         )

#     def to_wide(
#         self, on_sel=IDX_SEL - LABEL_OBJECT_SEL, index_sel=LABEL_OBJECT_SEL
#     ) -> "Features":
#         return self.pipe_tables(to_wide, on_sel=on_sel, index_sel=index_sel)

#     def to_tall(self, **kwargs) -> "Features":
#         return self.pipe_tables(to_tall, **kwargs)

#     def to_table(self) -> pl.DataFrame:
#         return join(
#             *list(self.to_wide()._tables.values()),
#             column_selector=IDX_SEL,
#             how="full",
#             coalesce=True,
#         )


def scan_resources(
    root: str = NEW_TABLES_PATH,
    validate_schema: bool = True,
    strict: bool = False,
) -> LazyResources:
    lazy_resources = LazyResources.from_path(
        root, strict=strict, validate_schema=validate_schema
    )
    return lazy_resources


def scan_features(
    root: str = NEW_TABLES_PATH,
    validate_schema: bool = True,
    strict: bool = False,
) -> LazyFeatures:
    lazy_features = LazyFeatures.from_path(
        root, strict=strict, validate_schema=validate_schema
    )
    return lazy_features


def scan_resources_and_features(
    root: str = NEW_TABLES_PATH,
    validate_schema: bool = True,
    strict: bool = False,
) -> tuple[LazyResources, LazyFeatures]:
    return (
        scan_resources(root=root, strict=strict, validate_schema=validate_schema),
        scan_features(root=root, strict=strict, validate_schema=validate_schema),
    )


# %%


def read_resources(
    root: str = TABLES_PATH,
    name_map: dict[str, str] = TABLE_NAME_MAP,
    use_pyarrow: bool = USE_PYARROW,
    validate_schema: bool = True,
    include_tables: tuple[str, ...] | None = None,
) -> Resources:
    resources = Resources.from_path(
        root,
        name_map=name_map,
        use_pyarrow=use_pyarrow,
        validate_schema=validate_schema,
        include_tables=include_tables,
    )
    return resources


def read_resources_and_features(
    root: str = TABLES_PATH,
    name_map: dict[str, str] = TABLE_NAME_MAP,
    use_pyarrow: bool = USE_PYARROW,
    validate_schema: bool = True,
    include_resource_tables: tuple[str, ...] | None = None,
    include_feature_tables: tuple[str, ...] | None = None,
) -> tuple[Resources, Features]:
    with pl.StringCache():
        resources = Resources.from_path(
            root,
            name_map=name_map,
            use_pyarrow=use_pyarrow,
            validate_schema=validate_schema,
            include_tables=include_resource_tables,
        )
        features = Features.from_path(
            root,
            name_map=name_map,
            use_pyarrow=use_pyarrow,
            validate_schema=validate_schema,
            include_tables=include_feature_tables,
        )
    return resources, features


def _update_table_from_table(
    resources: "Resources",
    from_table: str,
    to_tables: tuple[str, ...],
    strategy: Literal["column", "selector_intersection"] = "selector_intersection",
    selector=cs.starts_with(""),
) -> "Resources":
    df_from = getattr(resources, from_table)
    if strategy == "selector_intersection":
        return resources.pipe_tables(
            join_with,
            df_from,
            include_tables=to_tables,
            how="semi",
            column_selector=selector,
        )
    else:
        raise NotImplementedError()


INDEX_COLUMN_PREFIXES = ("", "fidx.", "sidx.")

DEFAULT_INDEX_PREFIX = INDEX_COLUMN_PREFIXES[0]
# def _set_index(
#     df: pl.DataFrame,
#     columns: tuple[str, ...] = (),
#     prefix: str = "",
#     index_prefixes: tuple[str, ...] = INDEX_COLUMN_PREFIXES,
#     assert_unique: bool = True,
# ):
#     for prefix in index_prefixes:
#         df


def set_index(df: AnyFrame, columns=(), index_prefix=DEFAULT_INDEX_PREFIX) -> AnyFrame:
    return df.select(_reset_index_selector_single(index_prefix=index_prefix)).select(
        _set_index_selector_single(columns=columns, index_prefix=index_prefix)
    )


def _reset_index_selector(
    index_prefixes=INDEX_COLUMN_PREFIXES,
):
    return [
        _reset_index_selector_single(prefix, passthrough=False)
        for prefix in index_prefixes
    ] + [~cs.starts_with(index_prefixes)]


def _reset_index_selector_single(
    index_prefix: str = "",
    passthrough: bool = True,
):
    if passthrough:
        return [
            cs.starts_with(index_prefix).name.map(
                lambda x: x.removeprefix(index_prefix)
            ),
            (~cs.starts_with(index_prefix)),
        ]
    return cs.starts_with(index_prefix).name.map(lambda x: x.removeprefix(index_prefix))


def _set_index_selector_single(
    columns: tuple[str, ...] | str = (),
    index_prefix: str = "",
    passthrough: bool = True,
):
    if passthrough:
        return [cs.by_name(columns).name.prefix(index_prefix), (~cs.by_name(columns))]
    return cs.by_name(columns).name.prefix(index_prefix)


def __reset_index(
    df: pl.DataFrame,
    index_prefixes: tuple[str, ...] = INDEX_COLUMN_PREFIXES,
):
    return [
        cs.starts_with(prefix).name.map(lambda x: x.removeprefix(prefix))
        for prefix in index_prefixes
    ]
    return df.select(
        *[
            cs.starts_with(prefix).name.map(lambda x: x.removeprefix(prefix))
            for prefix in index_prefixes
        ],
        ~cs.starts_with(index_prefixes),
    )


def _update_dims(
    resources: "Resources", strategy: Literal["intersection", "union"] = "intersection"
) -> "Resources":
    available_objects = resources.label_images["o"].unique()
    objects_with_features = resources.label_objects["o"].unique()
    df_o = resources.object_types.filter(
        pl.col("o").is_in(available_objects)
    ).with_columns(pl.col("o").is_in(objects_with_features).alias("has_features"))

    available_channels = resources.images["c"].unique()
    df_c = resources.channels.filter(pl.col("c").is_in(available_channels))

    if strategy == "intersection":
        available_m_levels = set(resources.images["m"].unique()).intersection(
            resources.label_images["m"].unique()
        )
        available_rois = set(resources.images["roi"].unique()).intersection(
            resources.label_images["roi"].unique()
        )

    else:
        available_m_levels = set(resources.images["m"].unique()).union(
            resources.label_images["m"].unique()
        )
        available_rois = set(resources.images["roi"].unique()).union(
            resources.label_images["roi"].unique()
        )

    df_m = resources.multiscale_levels.filter(pl.col("m").is_in(available_m_levels))
    df_roi = resources.rois.filter(pl.col("roi").is_in(available_rois))

    available_wells = df_roi["well"].unique()
    df_well = resources.wells.filter(pl.col("well").is_in(available_wells))

    return Resources(
        **{
            **{k: safe_collect(v) for k, v in resources._tables.items()},
            **{
                "object_types": df_o,
                "channels": df_c,
                "multiscale_levels": df_m,
                "rois": df_roi,
                "wells": df_well,
            },
        }
    )


def _update_resources(resources: "Resources") -> "Resources":
    well = resources.wells["well"]
    roi = resources.rois["roi"]
    c = resources.channels["c"]
    o = resources.object_types["o"]
    m = resources.multiscale_levels["m"]
    return Resources(
        **{
            **{k: safe_collect(v) for k, v in resources._tables.items()},
            **{
                "rois": resources.rois.filter(pl.col("well").is_in(well)),
                "images": resources.images.filter(
                    pl.col("roi").is_in(roi)
                    & pl.col("c").is_in(c)
                    & pl.col("m").is_in(m)
                ),
                "label_images": resources.label_images.filter(
                    pl.col("roi").is_in(roi)
                    & pl.col("o").is_in(o)
                    & pl.col("m").is_in(m)
                ),
                "label_objects": resources.label_objects.filter(
                    pl.col("roi").is_in(roi), pl.col("o").is_in(o)
                ),
                "hierarchy": resources.hierarchy.filter(
                    pl.col("roi").is_in(roi)
                ).filter(pl.col("o.parent").is_in(o) | pl.col("o.child").is_in(o)),
            },
        }
    )


def _update(resources: "Resources", strategy="intersection") -> "Resources":
    return resources.pipe(_update_dims, strategy=strategy).pipe(_update_resources)


def _join(dfs, on, how, coalesce=None):
    df = dfs[0]
    for df2 in dfs[1:]:
        df = df.join(df2, on=on, how=how, coalesce=coalesce)
    return df


def _unstack_column_to_column_name(
    df: pl.DataFrame,
    column_selector: SelectorType = cs.starts_with("fidx."),
    index_selector: SelectorType = cs.starts_with("idx."),
    feature_selector: "SelectorType | None" = None,
    sep: str = "_",
    prefix: bool = True,
    fidx_sep: str = "-",
) -> pl.DataFrame:
    """Slightly more ergonomic than the version in zfish.features.polars_utils w/ the same name."""

    if feature_selector is None:
        feature_selector = ~(column_selector | index_selector)

    index_names = cs.expand_selector(df, index_selector)
    feature_names = cs.expand_selector(df, feature_selector)

    column_names = cs.expand_selector(df, column_selector)
    if len(column_names) == 0:
        return df.select(index_selector | feature_selector)
    elif len(column_names) > 1:
        df = df.with_columns(
            pl.concat_str(
                [pl.col(e).cast(pl.String) for e in column_names],
                separator=fidx_sep,
                ignore_nulls=True,
            )
        )
    column_name = column_names[0]

    channel_names = (
        df.select(column_name).unique(maintain_order=True)[column_name].to_list()
    )
    dfs = []
    for channel_name in channel_names:
        if prefix:
            feature_expr = pl.col(feature_names).name.prefix(f"{channel_name}{sep}")
        else:
            feature_expr = pl.col(feature_names).name.suffix(f"{sep}{channel_name}")

        dfs.append(
            df.filter(pl.col(column_name) == channel_name).select(
                [
                    *index_names,
                    feature_expr,
                ]
            )
        )
    return _join(dfs, on=list(index_names), how="full", coalesce=True)


def to_wide(
    df,
    on_sel=IDX_SEL - LABEL_OBJECT_SEL,
    index_sel=LABEL_OBJECT_SEL,
    values_sel=None,
    squeeze=False,
    prefix=True,
    sep="_",
    fidx_sep="-",
):
    df = safe_collect(df)
    if df.height == 0:
        return df.select(index_sel)
    if squeeze:
        raise NotImplementedError

    return _unstack_column_to_column_name(
        df,
        sep=sep,
        column_selector=on_sel,
        index_selector=index_sel,
        feature_selector=values_sel,
        prefix=prefix,
        fidx_sep=fidx_sep,
    )


def to_tall(df, index_sel, pattern_before=r"^[^_]*", new_column_name="fidx", sep="_"):
    return split_and_melt_column_names_on(
        df,
        sep=sep,
        pattern_before=pattern_before,
        new_column_name=new_column_name,
        index_columns=index_sel,
    )
