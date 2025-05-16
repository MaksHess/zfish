# %%
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Literal,
    Mapping,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

import pandera as pa
import polars as pl
import polars.selectors as cs
from polars._typing import JoinStrategy, JoinValidation, SelectorType
from polars.exceptions import ColumnNotFoundError, ComputeError
from typing_extensions import Self

from zfish.multi_table.schema_metadata import FrameMeta
from zfish.multi_table.schemas_id_v2 import FrameId
from zfish.multi_table.schemas_v2 import LAZY_TABLES_EMPTY, sel
from zfish.preprocessing.types import (
    AnyFrame,
    FrameID,
    FrameName,
    MFrameName,
    SelectorType,
    id_,
)

if TYPE_CHECKING:
    from typing import Any, Callable

    import pandera.polars as pa

    from zfish.multi_table.schema_metadata import FrameMeta

REPR_PRINT_CONFIG = {
    "tbl_formatting": "UTF8_FULL_CONDENSED",
    "tbl_hide_column_data_types": True,
    "tbl_hide_dataframe_shape": True,
    "tbl_rows": 20,
    "fmt_str_lengths": 1000,
    # "tbl_width_chars": 10_000,
}

SingletonStrategy: TypeAlias = Literal["drop", "1", "first_item", "ignore"]


# TODO: @classmethod, @property breaks in python 3.13
class MFrameBaseMixin:
    """Base class for a dataclasses with `pl.LazyFrame` or `pl.DataFrame` as fields."""

    __mframe_name__: MFrameName
    __mframe_schema__: Mapping[FrameId, "pa.DataFrameSchema"]
    __mframe_meta__: Mapping[FrameId, dict[str, "Any"]]
    __repr_cache__: Mapping[FrameId, pl.DataFrame]
    __mframe_meta_is_stale__: bool

    @classmethod
    def derive(cls, name: str) -> type:
        raise NotImplementedError

    @classmethod
    @property
    def _schemas(cls) -> dict[FrameId, "pa.DataFrameSchema"] | None:
        return cls.__mframe_schema__

    @classmethod
    @property
    def _table_ids(cls) -> tuple[FrameId, ...]:
        return tuple(cls.__mframe_schema__)

    @classmethod
    @property
    def _table_names(cls) -> tuple[FrameName, ...]:
        return tuple(map(lambda x: x.to_localname(), cls._table_ids))

    @classmethod
    @property
    def _table_qualnames(cls) -> tuple[FrameID, ...]:
        return tuple(map(lambda x: x.to_qualname(), cls._table_ids))

    @classmethod
    @property
    def _table_filenames(cls) -> tuple[FrameID, ...]:
        return tuple(map(lambda x: x.to_qualname(), cls._table_ids))

    @classmethod
    @property
    def _table_metas(cls) -> dict[FrameName, tuple[str, ...]]:
        return {k.frame: v for k, v in cls.__mframe_meta__.items()}

    @classmethod
    @property
    def _table_metas_id(cls) -> dict[FrameId, tuple[str, ...]]:
        return {k: v for k, v in cls.__mframe_meta__.items()}

    @classmethod
    @property
    def _table_metas_qual(cls) -> dict[FrameID, tuple[str, ...]]:
        return {FrameID((k.mframe, k.frame)): v for k, v in cls.__mframe_meta__.items()}

    @property
    def _tables(self) -> dict[FrameName, AnyFrame]:
        tables: dict[str, AnyFrame] = {}
        potential_tables = {k: getattr(self, k) for k in self._table_names}
        for k, table in potential_tables.items():
            if isinstance(table, pl.DataFrame) or isinstance(table, pl.LazyFrame):
                tables = {**tables, k: table}
            else:
                tables = {**tables, **table._tables}
        return tables

    @property
    def _tables_qual(self) -> dict[FrameID, AnyFrame]:
        qual_name_map = dict(zip(self._table_names, self._table_qualnames))
        return {qual_name_map[name]: table for name, table in self._tables.items()}

    def from_template(self, new_tables: dict[FrameName, AnyFrame]) -> Self:
        return type(self)(**{**self._tables, **new_tables})

    def pipe(self, func, *args, **kwargs) -> Self:
        """Pipe a function of func(t: TableBase) -> TableBase"""
        return func(self, *args, **kwargs)

    def map_metas(
        self,
        func: Callable[[FrameMeta], FrameMeta],
        *args,
        exclude: tuple[str, ...] = (),
        include: tuple[str, ...] | None = None,
    ):
        raise NotImplementedError
        if include is not None:
            if exclude != ():
                raise ValueError("Can only pass one of `exclude` or `include`")
            if not all(e in self._table_names for e in include):
                raise ValueError(
                    f"one of {include} was not found in {self._table_names}"
                )
            exclude = tuple(e for e in self._table_names if e not in include)

    def pipe_table(self, func, table_name: str, *args, **kwargs):
        return self.pipe_tables(func, include_tables=(table_name,), *args, **kwargs)

    def map_tables(
        self,
        func,
        *args,
        exclude_tables: tuple[str, ...] = (),
        include_tables: tuple[str, ...] | None = None,
        _meta_funcs: "Callable[[FrameMeta], FrameMeta] | None" = None,
        **kwargs,
    ):
        return self.pipe_tables(
            func,
            *args,
            exclude_tables=exclude_tables,
            include_tables=include_tables,
            _meta_funcs=_meta_funcs,
        )

    def pipe_tables(
        self,
        func,
        *args,
        exclude_tables: tuple[str, ...] = (),
        include_tables: tuple[str, ...] | None = None,
        _meta_funcs: "Callable[[FrameMeta], FrameMeta] | None" = None,
        **kwargs,
    ) -> Self:
        if include_tables is not None:
            if exclude_tables != ():
                raise ValueError(
                    "Can only pass one of `exclude_tabels` or `include_tables`"
                )

            if not all(e in self._table_names for e in include_tables):
                raise ValueError(
                    f"one of {include_tables} was not found in {self._table_names}"
                )
            exclude_tables = tuple(
                e for e in self._table_names if e not in include_tables
            )

        out_tables = {}
        for table_name, table in self._tables.items():
            # print(table_name)
            if table_name in exclude_tables:
                out_tables[table_name] = table
            else:
                try:
                    out_tables[table_name] = table.pipe(func, *args, **kwargs)
                except (ColumnNotFoundError, NameError):
                    out_tables[table_name] = table

        return self.__class__(**out_tables)
        # return self.__class__(
        #     **{k: v.pipe(func, *args, **kwargs) for k, v in self._tables.items()}
        # )

    # FIXME: should not touch index
    def select(self, *sel: SelectorType, **named_sel: SelectorType) -> Self:
        return self.__class__(**{k: v.select(sel) for k, v in self._tables.items()})

    def join(self, include=(), exclude=(), how="left") -> pl.DataFrame:
        if include != () and exclude != ():
            raise ValueError("only one of `include` and `exclude` allowed")

        if exclude != ():
            include = tuple(e for e in self._table_names if e not in exclude)

        out_tables = {}
        for frame_name in include:
            out_tables[frame_name] = dict(zip(self._table_names, self._tables))
        return join(*out_tables.values(), how=how)

    # TODO: pl.any_horizontal by default?
    def filter(self, predicate: pl.Expr):
        out_tables = {}
        for table_name, table in self._tables.items():
            try:
                out_tables[table_name] = table.filter(predicate)
            except (ColumnNotFoundError, ComputeError) as e:
                out_tables[table_name] = table

        return self.__class__(**out_tables)

    def _validate_tables_debug(self):
        for name, schema in self._schemas.items():
            table = self._tables[name.frame]
            print(name)
            table.pipe(schema.validate)
            print("valid.")

    @classmethod
    def _validate_right_frames_in_tables(
        cls,
        tables: dict[FrameId, AnyFrame],
        strict: bool,
    ) -> dict[FrameName, AnyFrame]:
        valid_tables = {}
        for frame_id, df in tables.items():
            mframe_name = frame_id.mframe
            frame_name = frame_id.frame
            if mframe_name != cls.__mframe_name__ or frame_name not in cls._table_names:
                continue
            valid_tables[frame_id] = df

        selected_tables = {}
        for frame_id in cls._table_ids:
            if frame_id not in valid_tables:
                if strict:
                    raise ValueError(
                        f"Table {frame_name!r} not found in {list(valid_tables.keys())!r}"
                    )
                else:
                    # print(f"{frame_name=}")
                    # print(f"{mframe_name=}")
                    # print(f"{LAZY_TABLES_EMPTY.keys()}")
                    # print(f"{frame_name=}")
                    selected_tables[frame_id] = LAZY_TABLES_EMPTY[frame_id]
            else:
                selected_tables[frame_id] = valid_tables[frame_id]
        return selected_tables

    @classmethod
    def _validate_table_schemas(
        cls,
        tables: dict[FrameName, AnyFrame],
        validate_schema: bool,
    ) -> dict[FrameName, AnyFrame]:
        if not validate_schema:
            return tables
        else:
            schemas = cls._schemas
            for n, tbl in tables.items():
                tbl_schema = schemas[n]
                try:
                    tbl.pipe(tbl_schema)
                except:
                    print(n, tbl.columns, tbl_schema.columns)
                return {n: tbl.pipe(schemas[n].validate) for (n, tbl) in tables.items()}

    @classmethod
    def _validate(
        cls,
        tables: dict[FrameID, AnyFrame] | dict[FrameName, AnyFrame],
        strict: bool,
        validate_schema: bool,
    ):
        correct_tables = cls._validate_right_frames_in_tables(tables, strict=strict)
        return cls._validate_right_frames_in_tables(correct_tables, strict=strict)

    # @lru_cache(maxsize=20)
    def _repr_table(
        self,
        hidden: tuple[str, ...] = (),
        singleton_strategy: SingletonStrategy = "1",
        index_meta_keys=("pk", "fks"),
        cardinality: bool = True,
        cardinality_exclude: tuple[str, ...] = ("label", "label.parent", "label.child"),
        return_components=False,
    ):
        header_info = f"< {self.__class__.__name__} >"
        # print(header_info)
        if singleton_strategy == "drop":
            additional_hidden = tuple(
                k for k, e in self._tables.items() if e.height == 1
            )
        else:
            additional_hidden = ()
        all_hidden = tuple(hidden) + tuple(additional_hidden)
        print_tables = {
            name: table
            for name, table in self._tables.items()
            if name not in all_hidden
        }
        print_index_meta = {
            name.to_localname(): meta.get_key_components_dict()
            for name, meta in self._table_metas_id.items()
        }

        if return_components:
            return _new_repr_v2_components(
                header_info,
                dfs=print_tables,
                index_meta_keys=print_index_meta,
                singleton_strategy=singleton_strategy,
                cardinality=cardinality,
                cardinality_exclude=cardinality_exclude,
            )
        repr_string = _new_repr_v2(
            header_info,
            dfs=print_tables,
            index_meta_keys=print_index_meta,
            singleton_strategy=singleton_strategy,
            cardinality=cardinality,
            cardinality_exclude=cardinality_exclude,
        )
        # print(repr_string)
        # return safe_collect(repr_string)
        return repr_string

    def __repr__(
        self,
        hidden: tuple[str, ...] = (),  # Not used
        singleton_strategy: SingletonStrategy = "1",
        index_meta_keys=("pk", "fks"),
        cardinality: bool = True,
        cardinality_exclude: tuple[str, ...] = (
            "label",
            "label.parent",
            "label.child",
            "lbl",
            "lbl.parent",
            "lbl.child",
        ),  # DEFAULTS instead
    ):
        return self._repr_table(**MFRAME_REPR_DEFAULTS)


MFRAME_REPR_DEFAULTS = {
    "hidden": (),
    "singleton_strategy": "1",
    "index_meta_keys": ("pk", "fks"),
    "cardinality": True,
    "cardinality_exclude": (
        "label",
        "label.parent",
        "label.child",
        "lbl",
        "lbl.parent",
        "lbl.child",
    ),
}


def join(
    *dfs,
    column_selector=sel.idx,
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
    column_selector: SelectorType = sel.idx,
    how: JoinStrategy = "inner",
    validate: JoinValidation = "m:m",
    coalesce: bool | None = None,
) -> pl.DataFrame:
    """Join DataFrame's on intersection of columns matching `column_selector`."""
    join_columns0 = cs.expand_selector(df0, selector=column_selector)
    join_columns1 = cs.expand_selector(df1, selector=column_selector)
    join_columns = tuple(e for e in join_columns0 if e in join_columns1)
    return df0.join(df1, on=join_columns, how=how, validate=validate, coalesce=coalesce)


def _safe_shape(
    df: AnyFrame, exclude_prefixes: tuple[str, ...] = ("idx.",)
) -> tuple[int | str, str]:
    return (
        getattr(df, "height", "?"),
        df.collect_schema().len()
        - (
            len(cs.expand_selector(df, cs.starts_with(exclude_prefixes)))
            if len(exclude_prefixes) > 0
            else 0
        ),
    )


def safe_shape(
    df: AnyFrame, index_columns: tuple[str, ...] = ()
) -> tuple[int | str, str]:
    return (
        getattr(df, "height", "?"),
        df.collect_schema().len()
        - (
            len(cs.expand_selector(df, cs.by_name(index_columns)))
            if len(index_columns) > 0
            else 0
        ),
    )


def _is_lazy_frame(df: AnyFrame) -> TypeGuard[pl.LazyFrame]:
    if isinstance(df, pl.LazyFrame):
        return True
    return False


def _is_data_frame(df: AnyFrame) -> TypeGuard[pl.DataFrame]:
    if isinstance(df, pl.DataFrame):
        return True
    return False


def safe_lazy(df: AnyFrame) -> pl.LazyFrame:
    if _is_lazy_frame(df):
        return df
    else:
        return df.lazy()


def safe_collect(df: AnyFrame) -> pl.DataFrame:
    if _is_data_frame(df):
        return df
    elif _is_lazy_frame(df):
        return df.collect()
    else:
        raise ValueError("df can only be DataFrame or LazyFrame.")


T = TypeVar("T")


def safe_index(lst: list[T], element: T) -> int:
    try:
        return lst.index(element)
    except ValueError as e:
        return len(lst)


def _format_index(columns, sep="."):
    current_idx = None
    for e in columns:
        split = e.split(sep)
        idx = split[0]
        if current_idx != idx:
            print(f"{idx}: ", end="")
            current_idx = idx
        print(f"({sep.join(split[1:])})", end=" ")


def table_title(title, padding=1):
    name_with_sides = f"┆{'':<{padding}}{title}{'':<{padding}}┆"
    top_bar = (len(title) + 2 * padding) * "─"
    top_border = f"┌{top_bar}┐"
    return "\n".join([top_border, name_with_sides])


def human_format_safe(num):
    """https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings"""
    if isinstance(num, str):
        return num

    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def is_empty(col: str) -> pl.Expr:
    return pl.col(col).len() == 0


def is_singleton(col: str) -> pl.Expr:
    return pl.col(col).unique().len() == 1


def is_full_rank(col: str) -> pl.Expr:
    return pl.col(col).unique().len() == pl.col(col).len()


def first_item(col: str) -> pl.Expr:
    return pl.col(col).first()


# FIXME: first() should not be necessary in the next three functions. bug.
def empty(col: str):
    return pl.lit(None).cast(pl.String).first().alias(col)


# FIXME: first() should not be necessary in the next three functions. bug.
def component(col: str):
    return pl.lit(col).first().alias(col)


def component_singleton_first_item_v2(col: str, max_item_length: int | None = 8):
    if max_item_length is None:
        return pl.concat_str(component(col), pl.lit("="), pl.col(col).first())
    else:
        return pl.concat_str(
            component(col),
            pl.lit("="),
            pl.when(
                pl.col(col).cast(pl.String).first().str.len_chars() > max_item_length
            )
            .then(
                pl.concat_str(
                    pl.col(col)
                    .cast(pl.String)
                    .first()
                    .str.slice(0, max_item_length - 2),
                    pl.lit(".."),
                )
            )
            .otherwise(pl.col(col).cast(pl.String).first()),
        )


def component_with_cardinality_v2(col: str) -> pl.Expr:
    return pl.concat_str(component(col), pl.lit(":"), pl.col(col).unique().len())


def _item_repr_v2(
    col: str,
    singleton_strategy: SingletonStrategy = "1",
    cardinality: bool = True,
    first_item_string_cap: int | None = None,
):
    if cardinality:
        if singleton_strategy == "drop":
            singleton_expr = empty(col)
        elif singleton_strategy == "first_item":
            singleton_expr = component_singleton_first_item_v2(
                col, max_item_length=first_item_string_cap
            )
        elif singleton_strategy == "1":
            singleton_expr = component_with_cardinality_v2(col)
        else:
            singleton_expr = component(col)
        return (
            pl.when(is_singleton(col))
            .then(singleton_expr)
            .when(is_full_rank(col))
            .then(component(col))
            .otherwise(component_with_cardinality_v2(col))
        )
    else:
        return component(col)


def _index_repr_v2(
    df: pl.DataFrame,
    index_columns: tuple[str, ...] | None = None,  # all if None
    singleton_strategy: SingletonStrategy = "1",
    cardinality: bool | tuple[bool, ...] = True,
    first_item_string_cap: int | None = None,
):
    if index_columns is None:
        index_columns = df.collect_schema().names()
    if isinstance(cardinality, bool):
        col_cardinality = (cardinality,) * len(index_columns)
    else:
        col_cardinality = cardinality
    return df.select(
        _item_repr_v2(
            index_column,
            singleton_strategy=singleton_strategy,
            cardinality=car,
            first_item_string_cap=first_item_string_cap,
        )
        for index_column, car in zip(index_columns, col_cardinality)
    )


def __index_repr_multi_v2(
    dfs: dict[FrameName, pl.DataFrame],
    index_meta_keys: dict[FrameName, dict[str, tuple[str, ...]]],
    singleton_strategy: SingletonStrategy = "1",
    cardinality: bool = True,
    cardinality_exclude: tuple[str, ...] = (),
    first_item_string_cap: int | None = None,
):
    res = []
    for name, df in dfs.items():
        meta = index_meta_keys[name]
        key_names = list(meta)

        pk_name = key_names[0]
        columns_visited = list(
            meta[pk_name]
        )  # don't use them again if they're primary key
        repr_comps = []
        for key_name, columns in meta.items():
            if key_name != pk_name:
                columns = tuple(c for c in columns if c not in columns_visited)
                columns_visited.extend(columns)
            if len(columns) == 0:
                continue
            col_cardinality = (cardinality,) * len(columns)
            col_cardinality = tuple(
                car if col not in cardinality_exclude else False
                for car, col in zip(col_cardinality, columns)
            )
            key_info = safe_collect(
                _index_repr_v2(
                    df.select(columns),
                    singleton_strategy=singleton_strategy,
                    cardinality=col_cardinality,
                    first_item_string_cap=first_item_string_cap,
                ).select(
                    pl.concat_str(pl.all(), separator=", ", ignore_nulls=True).alias(
                        key_name
                    )
                )
            )
            repr_comps.append(key_info)

        rows, cols = safe_shape(df, index_columns=columns_visited)
        repr_comps.append(
            pl.DataFrame().with_columns(
                pl.lit(name).alias("table"),
                pl.lit(rows).alias("rows"),
                pl.lit(cols).alias("cols"),
            )
        )
        res.append(pl.concat(repr_comps, how="horizontal"))
    info = pl.concat(res, how="diagonal").fill_null("")
    info_table = safe_collect(info)
    all_key_names = [
        e for e in info_table.columns if e not in ["table", "rows", "cols"]
    ]
    return info_table.select(
        "table",
        "rows",
        *all_key_names,
        "cols",
    )


def _new_repr_v2(
    multi_table_name: str,
    dfs: dict[str, pl.DataFrame],
    index_meta_keys: dict[str, dict[str, tuple[str, ...]]],
    cardinality: bool = True,
    cardinality_exclude: tuple[str, ...] = (),
    singleton_strategy: SingletonStrategy = "1",
    short_numbers: bool = True,
    first_item_string_cap=10,
):
    header = table_title(multi_table_name)
    body_table = __index_repr_multi_v2(
        dfs,
        index_meta_keys=index_meta_keys,
        singleton_strategy=singleton_strategy,
        cardinality=cardinality,
        cardinality_exclude=cardinality_exclude,
        first_item_string_cap=first_item_string_cap,
    )
    if short_numbers:
        body_table = body_table.with_columns(
            pl.col("rows").map_elements(human_format_safe, return_dtype=pl.String)
        )
    with pl.Config(**REPR_PRINT_CONFIG) as cfg:
        result = "\n".join([header, repr(body_table)])
    return result


def _new_repr_v2_components(
    multi_table_name: str,
    dfs: dict[str, pl.DataFrame],
    index_meta_keys: dict[str, dict[str, tuple[str, ...]]],
    cardinality: bool = True,
    cardinality_exclude: tuple[str, ...] = (),
    singleton_strategy: SingletonStrategy = "1",
    short_numbers: bool = True,
    first_item_string_cap=None,
):
    header = table_title(multi_table_name)
    body_table = __index_repr_multi_v2(
        dfs,
        index_meta_keys=index_meta_keys,
        singleton_strategy=singleton_strategy,
        cardinality=cardinality,
        cardinality_exclude=cardinality_exclude,
        first_item_string_cap=first_item_string_cap,
    )
    return body_table


class NonUniquePrimaryKeyError(Exception):
    """Raised upon finding non-unique values in a unique declared set of columns."""

    pass


class NullInPrimaryKeyError(Exception):
    """Raised upon finding non-unique values in a unique declared set of columns."""

    pass


@dataclass(frozen=True, slots=True)
class Frame:
    __exclude_from_template__: ClassVar[tuple[str, ...]] = (
        "_index",
        "_index_unique_components",
        "_schema",
        "_columns",
    )
    name: str
    pk: tuple[str, ...]
    _lf: pl.LazyFrame
    _compute_cardinalities: tuple[bool, ...] | bool = False
    _squeezed: tuple[tuple[str, str, pl.DataType], ...] = ()
    _index: pl.DataFrame = field(default=None, repr=False, init=False)
    _index_unique_components: pl.DataFrame = field(default=None, repr=False, init=False)
    _schema: pl.Schema = field(default=None, repr=False, init=False)
    _columns: tuple[str, ...] = field(default=None, init=False)

    def __post_init__(self):
        cardinalities = self._compute_cardinalities
        if isinstance(cardinalities, bool):
            object.__setattr__(
                self,
                "_compute_cardinalities",
                tuple(cardinalities for _ in range(len(self.pk))),
            )
        all_columns = self._lf.collect_schema().names()
        columns = tuple([e for e in all_columns if e not in self.pk])
        object.__setattr__(self, "_columns", columns)

    def from_template(
        self,
        **kwargs,
    ):
        old_fields = {
            k: v
            for k, v in asdict(self).items()
            if k not in self.__exclude_from_template__
        }
        new_fields = {
            **old_fields,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        return self.__class__(**new_fields)

    def filter(self, expr: "pl.Expr") -> "Frame":
        return self.from_template(_lf=self._lf.filter(expr))

    def select(self, expr: "pl.Expr") -> "Frame":
        print(self.index_columns)
        return self.from_template(
            _lf=pl.concat(
                [
                    self._lf.select(self.index_columns),
                    self._lf.select(pl.exclude(self.index_columns)).select(expr),
                ],
                how="horizontal",
            )
        )

    def pl(
        self,
        squeeze_strategy: Literal[
            "drop", "tidy", "col_names", "col_names_short"
        ] = "drop",
        sep="__",
        sep_inner="-",
    ):
        def column_name_remap(col: str) -> str:
            if col not in self.pk and col not in self.squeezed_columns:
                col_comps = []
                for e in self._squeezed:
                    if squeeze_strategy == "col_names_short":
                        col_comp = f"{e[1]}"
                    else:
                        col_comp = f"{e[0]}={e[1]}"
                    col_comps.append(col_comp)
                return f"{sep_inner.join(col_comps)}{sep}{col}"
            else:
                return col

        if squeeze_strategy == "drop":
            return self._lf.select(pl.exclude(self.squeezed_columns))
        elif squeeze_strategy == "tidy":
            return self._lf
        elif squeeze_strategy == "col_names":
            # return column_name_remap
            return self._lf.select(pl.exclude(self.squeezed_columns)).rename(
                column_name_remap
            )
        elif squeeze_strategy == "col_names_short":
            # return column_name_remap
            return self._lf.select(pl.exclude(self.squeezed_columns)).rename(
                column_name_remap
            )
        else:
            raise ValueError(f"unknown strategy {squeeze_strategy}")

    @property
    def squeezed_columns(self) -> tuple[str, ...]:
        return tuple(e[0] for e in self._squeezed)

    @property
    def index(self):
        if self._index is None:
            object.__setattr__(self, "_index", self._lf.select(self.pk).collect())
        return self._index

    @property
    def index_columns(self):
        return tuple(self.index.columns) + self.squeezed_columns

    @property
    def features(self):
        self.columns

    @property
    def index_unique_components(self):
        if self._index_unique_components is None:
            object.__setattr__(
                self,
                "_index_unique_components",
                self.index.group_by(None).agg(pl.all().unique()).drop("literal"),
            )
        return self._index_unique_components

    def squeeze(self) -> Self:
        squeezed_columns = []
        index_columns = []
        for index_col, index_cardinality in self.cardinality.row(0, named=True).items():
            if index_cardinality == 1:
                squeezed_columns.append(
                    (
                        index_col,
                        self.index_unique_components[index_col][0].item(),
                        self.index.schema[index_col],
                    )
                )
            else:
                index_columns.append(index_col)
        return self.from_template(
            pk=tuple(index_columns),
            _squeezed=tuple(squeezed_columns),
            # lf=self.lf.drop(squeezed_columns),
        )

    def expand_dims(self) -> Self:
        new_pk = self.pk + tuple([e[0] for e in self._squeezed])
        new_frame = self.lf.with_columns(
            [
                pl.lit(value).cast(type_).alias(column_name)
                for column_name, value, type_ in self._squeezed
            ]
        )

        return self.from_template(
            pk=new_pk, _lf=new_frame.select(new_pk, pl.exclude(new_pk)), _squeezed=()
        )

    @property
    def cardinality(self):
        return self.index_unique_components.select(pl.all().list.len())

    @property
    def schema(self):
        if self._schema is None:
            object.__setattr__(self, "_schema", self._lf.collect_schema())
        return self._schema

    @property
    def columns(self):
        return self._columns

    def _validate(self):
        if self._schema:
            self.lf.pipe(self._schema.validate)
        for pk_col in self.pk:
            if pk_col not in self.pk:
                raise ValueError(
                    f"Primary key: {pk_col!r} not in dataframe columns: {self.lf.collect_schema().names()}"
                )
            if self.index.unique().height != self.index.height:
                raise NonUniquePrimaryKeyError(
                    "Primary key contains non-unique values."
                )
            for col, any_nulls in (
                self.index.select(pl.all().is_null().any()).row(0, named=True).items()
            ):
                if any_nulls:
                    raise NullInPrimaryKeyError(f"pk column {col} contains nulls.")

    def _get_repr_components(self):
        table_comps = pl.DataFrame({"table": [self.name], "rows": [self.index.height]})
        index_comps = _index_repr_v2(
            self.index, cardinality=self._compute_cardinalities
        )
        column_comps = pl.DataFrame(
            {"cols": len(self.columns), "col_names": [list(self.columns)]}
        )
        return pl.concat([table_comps, index_comps, column_comps], how="horizontal")

    def get_repr_components(self, cast_to_str=True, short_numbers=True):
        if cast_to_str:
            caster = 9
        if short_numbers:
            formatter = human_format_safe
        else:
            formatter = id_
        table_comps = pl.DataFrame(
            {"table": [self.name], "rows": [formatter(self.index.height)]}
        )
        index_comps = self.index_unique_components
        column_comps = pl.DataFrame(
            {"cols": formatter(len(self.columns)), "f": [list(self.columns)]}
        )
        return pl.concat([table_comps, index_comps, column_comps], how="horizontal")

    def repr_minimal(self):
        return f"{self.__class__.__name__}(name={self.name!r}, ...)"

    def __debug_repr__(self):
        pass


def is_sub_index(col: str, sep: str = ".") -> pl.Expr:
    return len(col.split(sep)) > 2


def component_after_idx(col: str, sep="."):
    return pl.lit(sep.join(col.split(sep)[1:])).first().alias(col)


def component_last(col: str, sep="."):
    return pl.lit(col.split(sep)[-1]).first().alias(col)


def component_singleton_first_item(col: str, sep=".", max_item_length: int | None = 8):
    if max_item_length is None:
        return pl.concat_str(
            component_after_idx(col, sep=sep), pl.lit("="), pl.col(col).first()
        )
    else:
        return pl.concat_str(
            component_after_idx(col, sep=sep),
            pl.lit("="),
            pl.when(
                pl.col(col).cast(pl.String).first().str.len_chars() > max_item_length
            )
            .then(
                pl.concat_str(
                    pl.col(col)
                    .cast(pl.String)
                    .first()
                    .str.slice(0, max_item_length - 2),
                    pl.lit(".."),
                )
            )
            .otherwise(pl.col(col).cast(pl.String).first()),
        )


def component_with_cardinality(col: str, sep=".") -> pl.Expr:
    return pl.concat_str(
        component_after_idx(col, sep=sep), pl.lit(":"), pl.col(col).unique().len()
    )


def _item_repr(
    col: str,
    sep: str = ".",
    singleton_strategy: SingletonStrategy = "1",
    cardinality: bool = True,
):
    if cardinality:
        if singleton_strategy == "drop":
            singleton_expr = empty(col)
        elif singleton_strategy == "first_item":
            singleton_expr = component_singleton_first_item(col, sep=sep)
        elif singleton_strategy == "1":
            singleton_expr = component_with_cardinality(col, sep=sep)
        else:
            singleton_expr = component_after_idx(col, sep=sep)
        return (
            pl.when(is_singleton(col))
            .then(singleton_expr)
            .when(is_full_rank(col))
            .then(component_after_idx(col, sep=sep))
            .otherwise(component_with_cardinality(col, sep=sep))
        )
    else:
        return component_after_idx(col, sep=sep)


def _index_repr(
    df: pl.DataFrame,
    index_prefixes: tuple[str, ...] = ("idx", "fidx"),
    sep: str = ".",
    singleton_strategy: SingletonStrategy = "1",
    cardinality: bool = True,
):
    index_columns = cs.expand_selector(df, cs.starts_with(index_prefixes))
    return df.select(
        _item_repr(
            index_column,
            sep=sep,
            singleton_strategy=singleton_strategy,
            cardinality=cardinality,
        )
        for index_column in index_columns
    )


def __index_repr_multi(
    dfs: dict[str, pl.DataFrame],
    index_prefixes: tuple[str, ...] = ("idx", "fidx"),
    sep: str = ".",
    singleton_strategy: SingletonStrategy = "1",
    cardinality: bool = True,
):
    res = []
    for name, df in dfs.items():
        rows, cols = _safe_shape(df)
        res.append(
            _index_repr(
                df,
                index_prefixes=index_prefixes,
                sep=sep,
                singleton_strategy=singleton_strategy,
                cardinality=cardinality,
            ).with_columns(
                pl.lit(name).alias("table"),
                pl.lit(rows).alias("rows"),
                pl.lit(cols).alias("cols"),
            )
        )
    info_table = safe_collect(pl.concat(res, how="diagonal"))

    available_prefixes = tuple(
        filter(
            lambda x: any(col.startswith(x) for col in info_table.columns),
            index_prefixes,
        )
    )
    return info_table.select(
        "table",
        "rows",
        pl.concat_str(
            cs.starts_with(available_prefixes[0]), separator=", ", ignore_nulls=True
        ).alias(available_prefixes[0]),
        *[
            pl.concat_str(
                cs.starts_with(prefix), separator=", ", ignore_nulls=True
            ).alias(prefix)
            for prefix in available_prefixes[1:]
        ],
        "cols",
    )


def _new_repr(
    multi_table_name: str,
    dfs: dict[str, pl.DataFrame],
    index_prefixes=("idx", "fidx"),
    cardinality: bool = True,
    sep: str = ".",
    singleton_strategy: SingletonStrategy = "1",
    short_numbers: bool = True,
):
    header = table_title(multi_table_name)
    body_table = __index_repr_multi(
        dfs,
        index_prefixes=index_prefixes,
        sep=sep,
        singleton_strategy=singleton_strategy,
        cardinality=cardinality,
    )
    if short_numbers:
        body_table = body_table.with_columns(
            pl.col("rows").map_elements(human_format_safe, return_dtype=pl.String)
        )
    return "\n".join([header, repr(body_table)])
