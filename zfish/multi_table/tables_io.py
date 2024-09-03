from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import polars as pl
import polars.selectors as cs
from polars.type_aliases import JoinStrategy, JoinValidation, SelectorType
from typing_extensions import Self

from zfish.features.polars_utils import split_and_melt_column_names_on
from zfish.multi_table.schemas import EMPTY_TABLES, IDX_SEL, LABEL_OBJECT_SEL
from zfish.multi_table.table_base_mixin import (
    AnyFrame,
    TablesBaseMixin,
    safe_collect,
    safe_lazy,
)

WriteStrategy: TypeAlias = Literal["overwrite_all", "overwrite_non_empty", "raise"]

USE_PYARROW = False

TABLES_PATH = Path(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\multi_table\data"
)


TABLE_NAME_MAP = {
    "wells": "resources.wells",
    "rois": "resources.rois",
    "channels": "resources.channels",
    "object_types": "resources.object_types",
    "multiscale_levels": "resources.multiscale_levels",
    "images": "resources.images",
    "z_models": "resources.z_models",
    "t_models": "resources.t_models",
    "label_images": "resources.label_images",
    "label_objects": "resources.label_objects",
    "hierarchy": "resources.hierarchy",
    "classifiers": "resources.classifiers",
    "label": "features.label",
    "distance": "features.distance",
    "intensity": "features.intensity",
    "correlation": "features.correlation",
    "density_count": "features.density_count",
    "density_distance": "features.density_distance",
    "classifier": "features.classifier",
}
TABLE_NAME_MAP_OLD = {
    "wells": "wells",
    "rois": "rois",
    "channels": "channels",
    "object_types": "object_types",
    "multiscale_levels": "multiscale_levels",
    "images": "images",
    "z_models": "z_models",
    "t_models": "t_models",
    "label_images": "label_images",
    "label_objects": "label_objects",
    "hierarchy": "label_objects.hierarchy",
    "classifiers": "classifiers.v2",
    "label": "features.label",
    "distance": "features.distance.v2",
    "intensity": "features.intensity",
    "correlation": "features.correlation",
    "density_count": "features.density.count",
    "density_distance": "features.density.distance",
    "classifier": "classifier_inference.v2",
}

DEFAULT_TABLES = {k: EMPTY_TABLES[TABLE_NAME_MAP[k]] for k in TABLE_NAME_MAP}


def scan_tables(
    path: str, table_names: tuple[str, ...] | None = None, **kwargs
) -> dict[str, pl.LazyFrame]:
    if table_names is not None:
        return {
            v.stem: pl.scan_parquet(v, **kwargs)
            for v in Path(path).rglob("*.parquet")
            if v.stem in table_names
        }
    return {v.stem: pl.scan_parquet(v, **kwargs) for v in Path(path).rglob("*.parquet")}


def read_tables(
    path: str,
    table_names: tuple[str, ...] | None = None,
    use_pyarrow: bool = USE_PYARROW,
    **kwargs,
) -> dict[str, pl.DataFrame]:
    if table_names is not None:
        return {
            v.stem: pl.read_parquet(v, use_pyarrow=use_pyarrow, **kwargs)
            for v in Path(path).rglob("*.parquet")
            if v.stem in table_names
        }
    return {
        v.stem: pl.read_parquet(v, use_pyarrow=use_pyarrow, **kwargs)
        for v in Path(path).rglob("*.parquet")
    }


def write_tables(
    path: Path | str,
    tables: dict[str, pl.DataFrame],
    overwrite: bool = False,
    write_empty: bool = False,
    use_pyarrow: bool = USE_PYARROW,
):
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
        if write_empty:
            table.write_parquet(table_path, use_pyarrow=use_pyarrow)
        else:
            if table.height > 0:
                table.write_parquet(table_path, use_pyarrow=use_pyarrow)


class LazyTablesBase(TablesBaseMixin):
    @classmethod
    def from_tables(
        cls,
        tables: dict[str, AnyFrame],
    ) -> Self:
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in tables:
                table = tables[field_name]
            else:
                table = DEFAULT_TABLES[field_name]
            kwargs[field_name] = safe_lazy(table)
        return cls(**kwargs)

    @classmethod
    def from_path(
        cls,
        path: str,
        name_map: dict[str, str] = TABLE_NAME_MAP,
        include_tables: tuple[str, ...] | None = None,
        cast_schema: bool = True,
    ) -> Self:
        if include_tables is None:
            load_table_names = cls._table_names
        else:
            for t in include_tables:
                if t not in cls._table_names:
                    raise ValueError(f"Unknown table {t!r} not in {cls._table_names}")
            load_table_names = include_tables
        tables = scan_tables(
            path,
            table_names=tuple(
                [name_map[table_name] for table_name in load_table_names]
            ),
        )
        if cast_schema:
            renamed_tables = {
                k: tables[name_map.get(k, k)].cast(DEFAULT_TABLES[k].schema)
                for k in load_table_names
            }
        else:
            renamed_tables = {k: tables[name_map.get(k, k)] for k in load_table_names}
        return cls.from_tables(renamed_tables)


class TablesBase(TablesBaseMixin):
    @classmethod
    def from_tables(
        cls,
        tables: dict[str, AnyFrame],
    ) -> Self:
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in tables:
                table = tables[field_name]
            else:
                table = DEFAULT_TABLES[field_name]
            kwargs[field_name] = safe_collect(table)
        return cls(**kwargs)

    @classmethod
    def from_path(
        cls,
        path: str,
        name_map: dict[str, str] = TABLE_NAME_MAP,
        include_tables: tuple[str, ...] | None = None,
        cast_schema: bool = True,
        use_pyarrow: bool = True,
    ) -> Self:
        if include_tables is None:
            load_table_names = cls._table_names
        else:
            for t in include_tables:
                if t not in cls._table_names:
                    raise ValueError(f"Unknown table {t!r} not in {cls._table_names}")
            load_table_names = include_tables
        tables = read_tables(
            path,
            table_names=tuple(
                [name_map[table_name] for table_name in load_table_names]
            ),
            use_pyarrow=use_pyarrow,
        )
        if cast_schema:
            renamed_tables = {
                k: tables[name_map.get(k, k)].cast(DEFAULT_TABLES[k].schema)
                for k in load_table_names
            }
        else:
            renamed_tables = {k: tables[name_map.get(k, k)] for k in load_table_names}
        return cls.from_tables(renamed_tables)

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
                f"{self.__class__.__name__.lower()}.{table_name}": table
                for table_name, table in self._tables.items()
            },
            overwrite=overwrite,
            write_empty=write_empty,
            use_pyarrow=use_pyarrow,
        )


@dataclass(frozen=True, repr=False)
class LazyResources(LazyTablesBase):
    wells: pl.LazyFrame
    rois: pl.LazyFrame
    channels: pl.LazyFrame
    object_types: pl.LazyFrame
    multiscale_levels: pl.LazyFrame
    images: pl.LazyFrame
    z_models: pl.LazyFrame
    t_models: pl.LazyFrame
    label_images: pl.LazyFrame
    label_objects: pl.LazyFrame
    hierarchy: pl.LazyFrame
    classifiers: pl.LazyFrame

    def collect(self) -> "Resources":
        return Resources(**{k: safe_collect(v) for k, v in self._tables.items()})

    def from_template(self, new_tables: dict[str, AnyFrame]):
        for table_name in new_tables.keys():
            if table_name not in self._table_names:
                raise ValueError(
                    f"Invalid table name: {table_name} not in {self._table_names}"
                )
        return LazyResources(
            **{**self._tables, **{k: safe_lazy(v) for k, v in new_tables.items()}}
        )


@dataclass(frozen=True, repr=False)
class Resources(TablesBase):
    wells: pl.DataFrame
    rois: pl.DataFrame
    channels: pl.DataFrame
    object_types: pl.DataFrame
    multiscale_levels: pl.DataFrame
    images: pl.DataFrame
    z_models: pl.DataFrame
    t_models: pl.DataFrame
    label_images: pl.DataFrame
    label_objects: pl.DataFrame
    hierarchy: pl.DataFrame
    classifiers: pl.DataFrame

    def lazy(self) -> "LazyResources":
        return LazyResources(**{k: safe_lazy(v) for k, v in self._tables.items()})

    def from_template(self, new_tables: dict[str, AnyFrame]):
        for table_name in new_tables.keys():
            if table_name not in self._table_names:
                raise ValueError(
                    f"Invalid table name: {table_name} not in {self._table_names}"
                )
        return Resources(
            **{**self._tables, **{k: safe_collect(v) for k, v in new_tables.items()}}
        )


@dataclass(frozen=True, repr=False)
class LazyFeatures(LazyTablesBase):
    label: pl.LazyFrame
    distance: pl.LazyFrame
    intensity: pl.LazyFrame
    correlation: pl.LazyFrame
    density_count: pl.LazyFrame
    density_distance: pl.LazyFrame
    classifier: pl.LazyFrame

    def collect(self) -> "Features":
        return Features(**{k: safe_collect(v) for k, v in self._tables.items()})

    def from_template(self, new_tables: dict[str, AnyFrame]) -> "LazyFeatures":
        for table_name in new_tables.keys():
            if table_name not in self._table_names:
                raise ValueError(
                    f"Invalid table name: {table_name} not in {self._table_names}"
                )
        return LazyFeatures(
            **{**self._tables, **{k: safe_lazy(v) for k, v in new_tables.items()}}
        )


@dataclass(frozen=True, repr=False)
class Features(TablesBase):
    label: pl.DataFrame
    distance: pl.DataFrame
    intensity: pl.DataFrame
    correlation: pl.DataFrame
    density_count: pl.DataFrame
    density_distance: pl.DataFrame
    classifier: pl.DataFrame

    def lazy(self) -> "LazyFeatures":
        return LazyFeatures(**{k: safe_collect(v) for k, v in self._tables.items()})

    def to_wide(
        self, on_sel=IDX_SEL - LABEL_OBJECT_SEL, index_sel=LABEL_OBJECT_SEL
    ) -> "Features":
        return self.pipe_tables(to_wide, on_sel=on_sel, index_sel=index_sel)

    def to_tall(self, **kwargs) -> "Features":
        return self.pipe_tables(to_tall, **kwargs)

    def to_table(self) -> pl.DataFrame:
        return join(
            list(self.to_wide()._tables.values()), on=IDX_SEL, how="outer_coalesce"
        )

    def from_template(self, new_tables: dict[str, AnyFrame]) -> "Features":
        for table_name in new_tables.keys():
            if table_name not in self._table_names:
                raise ValueError(
                    f"Invalid table name: {table_name} not in {self._table_names}"
                )
        return Features(
            **{**self._tables, **{k: safe_collect(v) for k, v in new_tables.items()}}
        )


def scan_resources_and_features(
    root: str = TABLES_PATH,
    name_map: dict[str, str] = TABLE_NAME_MAP,
    cast_schema: bool = True,
) -> tuple[LazyResources, LazyFeatures]:
    lazy_resources = LazyResources.from_path(
        root, name_map=TABLE_NAME_MAP, cast_schema=cast_schema
    )
    lazy_features = LazyFeatures.from_path(
        root, name_map=TABLE_NAME_MAP, cast_schema=cast_schema
    )
    return lazy_resources, lazy_features


def read_resources(
    root: str = TABLES_PATH,
    name_map: dict[str, str] = TABLE_NAME_MAP,
    use_pyarrow: bool = USE_PYARROW,
    cast_schema: bool = True,
    include_tables: tuple[str, ...] | None = None,
) -> Resources:
    resources = Resources.from_path(
        root,
        name_map=TABLE_NAME_MAP,
        use_pyarrow=use_pyarrow,
        cast_schema=cast_schema,
        include_tables=include_tables,
    )
    return resources


def read_resources_and_features(
    root: str = TABLES_PATH,
    name_map: dict[str, str] = TABLE_NAME_MAP,
    use_pyarrow: bool = USE_PYARROW,
    cast_schema: bool = True,
    include_resource_tables: tuple[str, ...] | None = None,
    include_feature_tables: tuple[str, ...] | None = None,
) -> tuple[Resources, Features]:
    with pl.StringCache():
        resources = Resources.from_path(
            root,
            name_map=TABLE_NAME_MAP,
            use_pyarrow=USE_PYARROW,
            cast_schema=cast_schema,
            include_tables=include_resource_tables,
        )
        features = Features.from_path(
            root,
            name_map=TABLE_NAME_MAP,
            use_pyarrow=USE_PYARROW,
            cast_schema=cast_schema,
            include_tables=include_feature_tables,
        )
    return resources, features


def _update_table_from_table(
    resources: "Resources",
    from_table: str,
    to_tables: tuple[str, ...],
    strategy: Literal["column", "selector_intersection"] = "selector_intersection",
    selector=cs.starts_with("idx."),
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


def _update_dims(
    resources: "Resources", strategy: Literal["intersection", "union"] = "intersection"
) -> "Resources":
    available_objects = resources.label_images["idx.o"].unique()
    objects_with_features = resources.label_objects["idx.o"].unique()
    df_o = resources.object_types.filter(
        pl.col("idx.o").is_in(available_objects)
    ).with_columns(pl.col("idx.o").is_in(objects_with_features).alias("has_features"))

    available_channels = resources.images["idx.c"].unique()
    df_c = resources.channels.filter(pl.col("idx.c").is_in(available_channels))

    if strategy == "intersection":
        available_m_levels = set(resources.images["idx.m"].unique()).intersection(
            resources.label_images["idx.m"].unique()
        )
        available_rois = set(resources.images["idx.roi"].unique()).intersection(
            resources.label_images["idx.roi"].unique()
        )

    else:
        available_m_levels = set(resources.images["idx.m"].unique()).union(
            resources.label_images["idx.m"].unique()
        )
        available_rois = set(resources.images["idx.roi"].unique()).union(
            resources.label_images["idx.roi"].unique()
        )

    df_m = resources.multiscale_levels.filter(pl.col("idx.m").is_in(available_m_levels))
    df_roi = resources.rois.filter(pl.col("idx.roi").is_in(available_rois))

    available_wells = df_roi["well"].unique()
    df_well = resources.wells.filter(pl.col("idx.well").is_in(available_wells))

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
    roi = resources.roi["idx.roi"]
    c = resources.c["idx.c"]
    o = resources.o["idx.o"]
    m = resources.m["idx.m"]
    return Resources(
        **{
            **{k: safe_collect(v) for k, v in resources._tables.items()},
            **{
                "images": resources.images.filter(
                    pl.col("idx.roi").is_in(roi)
                    & pl.col("idx.c").is_in(c)
                    & pl.col("idx.m").is_in(m)
                ),
                "label_images": resources.label_images.filter(
                    pl.col("idx.roi").is_in(roi)
                    & pl.col("idx.o").is_in(o)
                    & pl.col("idx.m").is_in(m)
                ),
                "label_objects": resources.label_objects.filter(
                    pl.col("idx.roi").is_in(roi), pl.col("idx.o").is_in(o)
                ),
                "hierarchy": resources.hierarchy.filter(
                    pl.col("idx.roi").is_in(roi)
                ).filter(pl.col("idx.o.parent").is_in(o)),
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


def join(
    *dfs,
    column_selector=IDX_SEL,
    how: JoinStrategy = "full",
    validate: JoinValidation = "m:m",
    coalesce: bool | None = True,
):
    df_out = dfs[0]
    if len(dfs) > 1:
        for df2 in dfs[1:]:
            df_out = join_with(
                df_out,
                df2,
                column_selector=column_selector,
                how=how,
                validate=validate,
                coalesce=coalesce,
            )
    return df_out


def join_with(
    df0: pl.DataFrame,
    df1: pl.DataFrame,
    column_selector: SelectorType = cs.starts_with("idx."),
    how: JoinStrategy = "inner",
    validate: JoinValidation = "m:m",
    coalesce: bool | None = None,
) -> pl.DataFrame:
    """Join DataFrame's on intersection of columns matching `column_selector`."""
    join_columns0 = cs.expand_selector(df0, selector=column_selector)
    join_columns1 = cs.expand_selector(df1, selector=column_selector)
    join_columns = tuple(e for e in join_columns0 if e in join_columns1)
    return df0.join(df1, on=join_columns, how=how, validate=validate, coalesce=coalesce)


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
                    # pl.exclude(list(index) + [column_name]).prefix(f"{channel_name}{sep}"),
                ]
            )
        )
    return _join(dfs, on=list(index_names), how="full", coalesce=True)


def to_wide(
    df,
    on_sel=IDX_SEL - LABEL_OBJECT_SEL,
    index_sel=LABEL_OBJECT_SEL,
    values_sel=None,
    prefix=True,
    sep="_",
    fidx_sep="-",
):
    if df.height == 0:
        return df.select(index_sel)
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
