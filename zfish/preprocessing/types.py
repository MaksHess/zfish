from typing import TypeAlias, TypeVar

import polars as pl

AnyFrameT = TypeVar("AnyFrameT", pl.DataFrame, pl.LazyFrame )
FrameOrLazy: TypeAlias = pl.DataFrame | pl.LazyFrame