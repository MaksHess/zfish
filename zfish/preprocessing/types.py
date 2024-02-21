from typing import TypeVar

import polars as pl

AnyFrameT = TypeVar("AnyFrameT", pl.DataFrame, pl.LazyFrame )