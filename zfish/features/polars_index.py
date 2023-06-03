# %%
import polars as pl

INDEX_PATTERN = r"^_?(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"  # 'snake_case` and `nonumbers`, `_can_start` !lowercase
ACTIVE_INDEX_PATTERN = r"^(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"
INACTIVE_INDEX_PATTERN = r"^_(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"

from typing import Sequence


@pl.api.register_dataframe_namespace("idx")
class IndexAccessor:
    def __init__(
        self,
        df: pl.DataFrame,
        active_pattern: str = ACTIVE_INDEX_PATTERN,
        inactive_pattern: str = INACTIVE_INDEX_PATTERN,
    ):
        self._df = df
        self._active_pattern = active_pattern
        self._inactive_pattern = inactive_pattern
        print('done')

    @property
    def columns(self) -> list[str]:
        return self._df.lazy().select(pl.col(self._active_pattern), pl.col(self._inactive_pattern)).columns
    
    @property
    def columns_set(self) -> set[str]:
        return set(self.columns)
    
    @property
    def active(self) -> list[str]:
        return self._df.lazy().select(pl.col(self._active_pattern)).columns

    @property
    def active_set(self) -> set[str]:
        return set(self.active)
    
    @property
    def inactive(self) -> list[str]:
        return self._df.lazy().select(pl.col(self._inactive_pattern)).columns

    @property
    def inactive_set(self) -> set[str]:
        return set(self.inactive)
    
    def intersection(self, other: pl.DataFrame):
        order = tuple(self.active) + tuple(other.idx.active)
        return sorted(self.active_set.intersection(other.idx.active_set), key=order.index)
    
    def join(self, other: pl.DataFrame, how='outer', **kwargs) -> pl.DataFrame:
        assert 'on' not in kwargs, "Cannot set argument `on` in `idx.join`, "
        return self._df.join(other, on=self.intersection(other), how=how, **kwargs)
    
    def sort(self) -> pl.DataFrame:
        return self._df.select(pl.col(self.active), pl.exclude(self.columns), pl.col(self.inactive))
    
    def set_index(self, columns: str | Sequence[str]) -> pl.DataFrame:
        columns_inactive_name = [f'_{column}' if not column.startswith('_') else f'{column}' for column in columns]
        return self.reset_index().idx.set_active(columns_inactive_name)
    
    def reset_index(self) -> pl.DataFrame:
        return self._df.select(
            [
                pl.exclude(self.columns),
                pl.col([e for e in self.inactive]),
                pl.col([e for e in self.active]).prefix('_'),
            ]
        )
        
    def set_active(self, columns: str | Sequence[str]) -> pl.DataFrame:
        if isinstance(columns, str): 
            columns = [columns]
        columns = list(filter(lambda x: x.startswith('_'), columns))
        column_names = [c[1:] for c in columns]
        return self._df.select(
            [
                pl.col([e for e in self.active]),
                *[pl.col(c).alias(c_name) for c, c_name in zip(columns, column_names)],
                pl.exclude([e for e in self.columns]),
                pl.col([e for e in self.inactive if e not in columns]),
            ]
        )

    def set_inactive(self, columns: str | Sequence[str]) -> pl.DataFrame:
        if isinstance(columns, str):
            columns = [columns]
        columns = list(filter(lambda x: not x.startswith('_'), columns))
        return self._df.select(
            [
                pl.col([e for e in self.active if e not in columns]),
                pl.exclude(self.columns),
                pl.col([e for e in self.inactive]),
                pl.col([e for e in columns]).prefix('_'),
            ]
        )
        
    def __repr__(self) -> str:
        return f"{self.active=}\n{self.inactive=}"