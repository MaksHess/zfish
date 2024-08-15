from itertools import islice
from typing import TypeAlias, TypeGuard, TypeVar

import polars as pl
import polars.selectors as cs
from polars.type_aliases import SelectorType
from typing_extensions import Self

from zfish.multi_table.schemas import IDX_SEL

AnyFrame: TypeAlias = pl.LazyFrame | pl.DataFrame


class TablesBaseMixin:
    """Base class for a dataclasses with `pl.LazyFrame` or `pl.DataFrame` as fields."""

    @classmethod
    @property
    def _table_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__.keys())

    @property
    def _tables(self) -> dict[str, AnyFrame]:
        tables: dict[str, AnyFrame] = {}
        potential_tables = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k, table in potential_tables.items():
            if isinstance(table, pl.DataFrame) or isinstance(table, pl.LazyFrame):
                tables = {**tables, k: table}
            else:
                tables = {**tables, **table._tables}
        return tables

    def pipe(self, func, *args, **kwargs) -> Self:
        """Pipe a function of func(t: TableBase) -> TableBase"""
        return func(self, *args, **kwargs)

    def pipe_tables(
        self,
        func,
        *args,
        exclude_tables: tuple[str, ...] = (),
        include_tables: tuple[str, ...] | None = None,
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
                except pl.ColumnNotFoundError as e:
                    out_tables[table_name] = table

        return self.__class__(**out_tables)
        # return self.__class__(
        #     **{k: v.pipe(func, *args, **kwargs) for k, v in self._tables.items()}
        # )

    def reduce(self, func, init=None):
        for (tn0, t0), (tn1, t1) in zip(
            self._tables.items(),
            islice(self._tables.items(), 1, len(self._tables.keys())),
        ):
            if init is None:
                init = func(t0, t1)
            else:
                init = func(init, t1)
        return init

    def select(self, *sel: SelectorType, **named_sel: SelectorType) -> Self:
        return self.__class__(**{k: v.select(sel) for k, v in self._tables.items()})

    # TODO: pl.any_horizontal by default?
    def filter(self, predicate: pl.Expr):
        out_tables = {}
        for table_name, table in self._tables.items():
            try:
                out_tables[table_name] = table.filter(predicate)
            except (pl.ColumnNotFoundError, pl.ComputeError) as e:
                out_tables[table_name] = table

        return self.__class__(**out_tables)

    @property
    def idxs(self):
        return self.select(IDX_SEL)

    # def __repr__(self, attrs=("name", "shape", "idx_name")):
    def __repr__(self, hidden: tuple[str, ...] = ()):
        first_table_name = list(self.__dataclass_fields__)[0]

        # Header
        header_info = f"< {self.__class__.__name__} >"
        # Body
        print_info = []
        for table_name, table in self._tables.items():
            if table_name in hidden:
                continue
            idx_columns = table.select(IDX_SEL).columns
            if table_name == first_table_name:
                column_order = idx_columns
            else:
                new_columns = [e for e in idx_columns if e not in column_order]
                column_order += new_columns
            out_cols = (
                table_name,
                *tuple(map(human_format_safe, safe_shape(table))),
                sorted(idx_columns, key=lambda x: safe_index(column_order, x)),
            )
            print_info.append(out_cols)

        with pl.Config(
            tbl_formatting="UTF8_FULL_CONDENSED",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_rows=20,
        ) as cfg:
            # a_fmt = str(a)
            print_statement = alternative_repr(header_info, print_info)
        return print_statement


def safe_shape(df: AnyFrame, exclude_index=True) -> tuple[int | str, str]:
    return (
        getattr(df, "height", "?"),
        df.width - (len(cs.expand_selector(df, IDX_SEL)) if exclude_index else 0),
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


def alternative_repr(header_info, print_info) -> str:
    header = table_title(header_info)

    a = (
        # pl.DataFrame(list(map(lambda x: x.split(maxsplit=3), dat.split("\n")))[:-1])
        # .transpose()
        pl.DataFrame(print_info, orient="row")
        # .with_columns(pl.col("column_3").str.strip_chars("[]").str.split(","))
        .explode("column_3")
        .with_columns(
            pl.col("column_3")
            .str.strip_chars()
            .str.split(".")
            .list.to_struct()
            .struct.rename_fields(["idx type", "dim"])
        )
        .select(
            pl.col("column_0").alias("table name"),
            pl.col("column_1").alias("rows"),
            pl.col("column_2").alias("cols"),
            pl.col("column_3").struct.field("idx type"),
            pl.col("column_3").struct.field("dim"),
        )
        .group_by(["table name", "rows", "cols", "idx type"], maintain_order=True)
        .agg(pl.col("dim"))
    )  # .pipe(debug)
    body_table = (
        a.filter(pl.col("idx type") == "idx")
        .join(
            a.filter(pl.col("idx type") == "fidx"),
            on=["table name", "rows", "cols"],
            how="full",
            coalesce=True,
        )
        .drop("idx type", "idx type_right")
        .rename({"dim": "idx", "dim_right": "fidx"})
        .with_columns(
            pl.col("idx").list.join(", "), pl.col("fidx").fill_null([]).list.join(", ")
        )
    )  # .pipe(debug)
    return "\n".join([header, repr(body_table)])
    return "\n".join([header, repr(body_table)])
    return "\n".join([header, repr(body_table)])
