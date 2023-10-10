# %%
from collections import namedtuple
from itertools import combinations, permutations
from typing import TYPE_CHECKING, cast

import forge
import polars as pl
import polars.selectors as cs
from attrs import define
from polars import expr
from polars.utils.various import _prepare_row_count_args
from typing_extensions import Self

if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import SelectorType
    

# def copy_signature(instance, attribute, value):

#     if value >= instance.y:

#         raise ValueError("'x' has to be smaller than 'y'!")

# ALIASES = {
# }

# class_attrs = {}

# for func_name in cs.__all__:
#     func = getattr(cs, func_name)
#     if callable(func):
#         print(func_name)
#         annotations = inspect.get_annotations(func)
#         match annotations:
#             case {'return': 'SelectorType'}:
#                 class_attrs[ALIASES.get(func_name, func_name)] = field(default=func)
#             case _:
#                 print('_')
#         print()


# SelectorClass = attrs.make_class('SelectorClass', class_attrs, repr=True)


class _Selector:
    """Easy access to polars columns.
    """
    @staticmethod
    @forge.copy(cs.all)
    def all() -> "SelectorType":
        return cs.all()
    
    @staticmethod
    @forge.copy(cs.by_dtype)
    def dtype(*args, **kwargs) -> "SelectorType":
        return cs.by_dtype(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.by_name)
    def name(*args, **kwargs) -> "SelectorType":
        return cs.by_name(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.contains)
    def contains(*args, **kwargs) -> "SelectorType":
        return cs.contains(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.datetime)
    def datetime(*args, **kwargs) -> "SelectorType":
        return cs.datetime(*args, **kwargs)
    
    @staticmethod
    @forge.copy(cs.duration)
    def duration(*args, **kwargs) -> "SelectorType":
        return cs.duration(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.ends_with)
    def ends_with(*args, **kwargs) -> "SelectorType":
        return cs.ends_with(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.first)
    def first(*args, **kwargs) -> "SelectorType":
        return cs.first(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.float)
    def float(*args, **kwargs) -> "SelectorType":
        return cs.float(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.integer)
    def integer(*args, **kwargs) -> "SelectorType":
        return cs.integer(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.last)
    def last(*args, **kwargs) -> "SelectorType":
        return cs.last(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.matches)
    def matches(*args, **kwargs) -> "SelectorType":
        return cs.matches(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.numeric)
    def numeric() -> "SelectorType":
        return cs.numeric()

    @staticmethod
    @forge.copy(cs.starts_with)
    def starts_with(*args, **kwargs) -> "SelectorType":
        return cs.starts_with(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.temporal)
    def temporal() -> "SelectorType":
        return cs.temporal()

    @staticmethod
    @forge.copy(cs.string)
    def string(*args, **kwargs) -> "SelectorType":
        return cs.string(*args, **kwargs)

    @staticmethod
    def cat() -> "SelectorType":
        return cs.string(include_categorical=True)

    @staticmethod
    @forge.copy(pl.col)
    def __call__(*args, **kwargs) -> "Expr":
        return pl.col(*args, **kwargs)

col = _Selector()


INDEX_PATTERN = r"^_?(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"  # 'snake_case` and `nonumbers`, `_can_start` !lowercase
ACTIVE_INDEX_PATTERN = r"^(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"
INACTIVE_INDEX_PATTERN = r"^_(?P<index>[a-z0-9]+(?:_[a-z0-9]+)*)$"

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

@define(frozen=True)
class FeatureSelector:
    label: "SelectorType" = cs.matches(LABEL_FEATURE_PATTERN)
    intensity: "SelectorType" = cs.matches(INTENSITY_FEATURE_PATTERN)
    corr: "SelectorType" = cs.matches(CORR_FEATURE_PATTERN)
    dist: "SelectorType" = cs.matches(DIST_FEATURE_PATTERN)
    def __call__(self) -> "SelectorType":
        return cast("SelectorType", self.label | self.intensity | self.corr | self.dist)

@define(frozen=True)
class IndexSelector:
    active: "SelectorType" = cs.matches(ACTIVE_INDEX_PATTERN)
    inactive: "SelectorType" = cs.matches(INACTIVE_INDEX_PATTERN)
    object: "SelectorType" = cs.by_name(OBJECT_INDEX)
    meta: "SelectorType" = cs.by_name(OBJECT_META)
    def __call__(self) -> "SelectorType":
        return cs.matches(INDEX_PATTERN)
    
@define(frozen=True)
class ChannelSelector:
    def __call__(self, channel=CHANNEL_PATTERN, include_pairs=False) -> "SelectorType":
        if include_pairs:
            return cs.matches(f"^{channel}_.*$|^{channel}\|.*_.*$|^.*\|{channel}_.*$")
        else:
            return cs.matches(f"^{channel}_.*$")



@define(frozen=True)
class ChannelPairSelector:
    def __call__(self, channel1=CHANNEL_PATTERN, channel2=CHANNEL_PATTERN) -> "SelectorType":
        return cs.matches(
            '|'.join(
                ["^{}_.*$".format('\|'.join(pair)) for pair in permutations([channel1, channel2])]
            )
        )
    
@define(frozen=True)
class StainSelector:
    def __call__(self, stain) -> "SelectorType":
        return cs.matches(r"{stain}\.\d+_.*$".format(stain=stain))
    

@define(frozen=True)
class Selector:
    feature: FeatureSelector = FeatureSelector()
    index: IndexSelector = IndexSelector()
    channel: ChannelSelector = ChannelSelector()
    stain: StainSelector = StainSelector()
    channel_pair: ChannelPairSelector = ChannelPairSelector()
    
    @staticmethod
    @forge.copy(pl.col)
    def __call__(*args, **kwargs) -> "Expr":
        return pl.col(*args, **kwargs)
    
    @staticmethod   
    @forge.copy(cs.all)
    def all() -> "SelectorType":
        return cs.all()
    
    @staticmethod
    @forge.copy(cs.by_dtype)
    def dtype(*args, **kwargs) -> "SelectorType":
        return cs.by_dtype(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.by_name)
    def name(*args, **kwargs) -> "SelectorType":
        return cs.by_name(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.contains)
    def contains(*args, **kwargs) -> "SelectorType":
        return cs.contains(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.datetime)
    def datetime(*args, **kwargs) -> "SelectorType":
        return cs.datetime(*args, **kwargs)
    
    @staticmethod
    @forge.copy(cs.duration)
    def duration(*args, **kwargs) -> "SelectorType":
        return cs.duration(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.ends_with)
    def ends_with(*args, **kwargs) -> "SelectorType":
        return cs.ends_with(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.first)
    def first(*args, **kwargs) -> "SelectorType":
        return cs.first(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.float)
    def float(*args, **kwargs) -> "SelectorType":
        return cs.float(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.integer)
    def integer(*args, **kwargs) -> "SelectorType":
        return cs.integer(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.last)
    def last(*args, **kwargs) -> "SelectorType":
        return cs.last(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.matches)
    def matches(*args, **kwargs) -> "SelectorType":
        return cs.matches(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.numeric)
    def numeric() -> "SelectorType":
        return cs.numeric()

    @staticmethod
    @forge.copy(cs.starts_with)
    def starts_with(*args, **kwargs) -> "SelectorType":
        return cs.starts_with(*args, **kwargs)

    @staticmethod
    @forge.copy(cs.temporal)
    def temporal() -> "SelectorType":
        return cs.temporal()

    @staticmethod
    @forge.copy(cs.string)
    def string(*args, **kwargs) -> "SelectorType":
        return cs.string(*args, **kwargs)

    @staticmethod
    def cat() -> "SelectorType":
        return cs.string(include_categorical=True)


sel = Selector()
# # %%
# sel.index() | sel.feature() - cs.by_dtype(pl.Int64)
# # %%
# from dataclasses import dataclass
# from typing import Iterable, Optional, Union

# from anytree import AnyNode, NodeMixin, RenderTree
# from attrs import define, field
# from typing_extensions import Any, Self


# class MyNode(NodeMixin):
#     # name: str
#     # expr: Union["SelectorType", None] = None
#     # parent: Optional[Self] = None
#     # children: Iterable[Self] = tuple()
#     def __init__(self, name, expr=None, parent=None, children=None):
#         self.name = name
#         self.expr = expr
#         self.parent = parent
#         if children is None:
#             self.children = tuple()

#     def __call__(self):
#         return self.expr

#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.name=}, {self.expr=})"
#     # @property
#     # def child_names(self):
#     #     return tuple(child.name for child in self.children)
        
#     def __getattr__(self, attr: str) -> Any:
#         if attr.startswith('_'):
#             raise AttributeError(f'`{attr}` not found!')
#         else:
#             print('hooked')
#             for child in self.children:
#                 if child.name == attr:
#                     return child
#         raise AttributeError(f'`{attr}` not found!')
        
#         # try:
#         #     return super().__getattribute__(attr)
#         # except AttributeError:
#         #     pass
        
#         # for child in self.children:
#         #     if child.name == attr:
#         #         return child
#         #     else:
#         #         raise AttributeError(f"Attribute: {attr} not found!")


# col = MyNode(name='col', expr=None, parent=None)
# feature = MyNode(name='feature', expr=cs.matches(DIST_FEATURE_PATTERN), parent=col)
# label = MyNode(name='label', expr=cs.matches(LABEL_FEATURE_PATTERN), parent=feature)
# intensity = MyNode(name='intensity', expr=cs.matches(INTENSITY_FEATURE_PATTERN), parent=feature)
# index = MyNode(name='index', expr=cs.matches(INDEX_PATTERN), parent=col)

# col = MyNode(name='col', expr=None, parent=None)
# feature = MyNode(name='feature', expr=cs.matches(DIST_FEATURE_PATTERN), parent=col)
# label = MyNode(name='label', expr=cs.matches(LABEL_FEATURE_PATTERN), parent=feature)
# intensity = MyNode(name='intensity', expr=cs.matches(INTENSITY_FEATURE_PATTERN), parent=feature)
# index = MyNode(name='index', expr=cs.matches(INDEX_PATTERN), parent=col)

# # col = MyNode(name='col', expr=None, parent=None)
# # %%
# filt = (lambda attrs: [(v, None) for k, v in attrs if k != "expr"])

# from anytree.exporter import DictExporter, JsonExporter

# DictExporter(attriter=filt).export(col)
# # %%
# nmi = NodeMixin()
# nmi.__getattribute__('col')
# # %%
# a = AnyNode(name='a')
# b = AnyNode(name='b', parent=a)
# # %%
# col.child_names.index('index')
# %%
