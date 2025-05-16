# %%
from pathlib import Path
from typing import (
    Callable,
    Literal,
    Mapping,
    NewType,
    Sequence,
    TypeAlias,
    TypeVar,
)

import pandera.polars as pa
import polars as pl
from polars._typing import SelectorType

# from zfish.multi_table.table_base_mixin import MFrameBaseMixin
# from zfish.multi_table.tables_io import LazyMFrameBase, MFrameBase

__all__ = [
    "SelectorType",
    "AnyFrame",
    "AnyFrameT",
    "FrameOrLazy",
    "id_",
]


T = TypeVar("T_")


def id_(df: T) -> T:
    return df


ColName = NewType("ColName", str)
FrameName = NewType("FrameName", str)
MFrameName = NewType("MFrameName", str)

QualColName = NewType("QualColName", str)
QualFrameName = NewType("QualFrameName", str)
QualMFrameName = NewType("QualMFrameName", str)
QualNameSeparator = Literal["."]

ColID = NewType("ColID", tuple[MFrameName, FrameName, ColName])
FrameID = NewType("FrameID", tuple[MFrameName, FrameName])
MFrameID = NewType("MFrameID", tuple[MFrameName])

Frame: TypeAlias = pl.DataFrame
LazyFrame: TypeAlias = pl.LazyFrame
AnyFrame: TypeAlias = pl.DataFrame | pl.LazyFrame
AnyFrameT = TypeVar("AnyFrameT", pl.DataFrame, pl.LazyFrame)
FrameSchema: TypeAlias = pa.DataFrameSchema
AnyFrameOrSchemaT = TypeVar(
    "AnyFrameOrSchemaT", pl.DataFrame, pl.LazyFrame, FrameSchema
)


FrameParseFuncT: TypeAlias = Callable[[AnyFrameT], AnyFrameT]
FrameParseFunc = FrameParseFuncT  # TODO: deprecate

# AnyMFrame: TypeAlias = LazyMFrameBase | MFrameBase
# AnyMFrameT = TypeVar("AnyMFrame", MFrameBase, LazyMFrameBase)

MFrameSchema: TypeAlias = dict[str, FrameSchema]
# MFrameParseFunc = Callable[AnyMFrameT, AnyMFrameT]

StrMapDict: TypeAlias = Mapping[str, str]
StrMapFunc: TypeAlias = Callable[[str], str]
StrMap: TypeAlias = Callable[[str], str] | Mapping[str, str]


IntoPaths: TypeAlias = str | Path | Sequence[str | Path]
IntoPath: TypeAlias = str | Path
IntoFrame: TypeAlias = AnyFrameOrSchemaT | IntoPath


FrameOrLazy: TypeAlias = AnyFrame  # Deprecate
