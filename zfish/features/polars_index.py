# %%
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import attrs
import polars as pl
import polars.selectors as cs
from polars.selectors import is_selector
from polars.type_aliases import SelectorType

from zfish.roi._spatial_roi_config import SortKey

a: int = 3.2

# %%
INDEX_PATTERN = r"^_?(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"  # 'snake_case` and `nonumbers`, `_can_start` !lowercase
ACTIVE_INDEX_PATTERN = r"^(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"
INACTIVE_INDEX_PATTERN = r"^_(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"
INDEX_ORDER = SortKey().idx

OBJECT_INDEX = ['roi', 'object', 'label']
OBJECT_META = ['nuc_count', 'log2_nuc_count', 'cycle', 'well', 'site', 'age_class']
INTENSITY_INDEX = ['roi', 'object', 'label', 'channel']
INTENSITY_META = ['stain', 'acquisition', 'model', 'model_type', 'model_feature']

FEATURE_PATTERN = r"(?P<feature>(?:[A-Z][a-z0-9]+)+[A-Z]?)"  # 'PascCamelCase' or `PascCamelCaseX` !capitalized
CHANNEL_PATTERN = (
    r"(([a-zA-Z0-9-]+)\.(\d+))"  # 'stainName.32' or 'Can-contain-Numb3rs-AND-hyph3ns.0'
)
CHANNEL_SET_PATTERN = f"({CHANNEL_PATTERN})(\\|({CHANNEL_PATTERN}))+"  # 'DAPI.0|DAPI.1|DAPI.2', 'pH3.0|pH3.40'
OBJECT_PATTERN = r"(([a-zA-Z0-9]*)-(\d+))"  # `camelCase-1` and `canHaveNumbers3-14` !no `_` or `-` !lowercase


LABEL_FEATURE_PATTERN = f"^{FEATURE_PATTERN}$"
INTENSITY_FEATURE_PATTERN = f"^{CHANNEL_PATTERN}_{FEATURE_PATTERN}$"
CORR_FEATURE_PATTERN = f"^{CHANNEL_SET_PATTERN}_{FEATURE_PATTERN}$"
DIST_FEATURE_PATTERN = f"^{OBJECT_PATTERN}_{FEATURE_PATTERN}$"

@attrs.define(frozen=True)
class FeaturesSelector:
    label: SelectorType = cs.matches(LABEL_FEATURE_PATTERN)
    intensity: SelectorType = cs.matches(INTENSITY_FEATURE_PATTERN)
    corr: SelectorType = cs.matches(CORR_FEATURE_PATTERN)
    dist: SelectorType = cs.matches(DIST_FEATURE_PATTERN)

@attrs.define(frozen=True)
class MySelector:
    index: SelectorType = cs.matches(INDEX_PATTERN)
    active_index: SelectorType = cs.matches(ACTIVE_INDEX_PATTERN)
    inactive_index: SelectorType = cs.matches(INACTIVE_INDEX_PATTERN)
    object_index: SelectorType = cs.by_name(OBJECT_INDEX)
    object_meta: SelectorType = cs.by_name(OBJECT_META)
    features: FeaturesSelector = FeaturesSelector()
    def __call__(self, *args):
        return pl.col(*args)




class BaseSelector:
    pass



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

    @property
    def columns(self) -> list[str]:
        return self._df.lazy().select(pl.col(self._active_pattern), pl.col(self._inactive_pattern)).columns
    
    @property
    def columns_set(self) -> set[str]:
        return set(self.columns)
    
    @property
    def not_columns(self) -> list[str]:
        return self._df.lazy().select(pl.exclude(self.columns)).columns
    
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
        return self._df.select(pl.col(sorted(self.active, key=INDEX_ORDER)), pl.exclude(self.columns), pl.col(self.inactive))
    
    def set_index(self, columns: str | Sequence[str] | pl.Expr) -> pl.DataFrame:
        if isinstance(columns, pl.Expr):
            columns = self._df.lazy().select(columns).columns
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
# %%
