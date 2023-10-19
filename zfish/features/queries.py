from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from zfish.features.feature_types import Resources

if TYPE_CHECKING:
    import polars as pl

    from zfish.roi.spatial_roi import Roi


class FeatureQuery(ABC):
    @abstractmethod
    def load_resources(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> "pl.DataFrame":
        raise NotImplementedError