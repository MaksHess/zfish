import enum
from dataclasses import dataclass
from typing import Literal


class Sentinel(enum.Enum):
    MISSING = enum.auto()


MISSING = Sentinel.MISSING


@dataclass(frozen=True, slots=True)
class RasterMeta:
    name: str
    type_: Literal["image", "label", "mask"]
    dims: tuple[Literal["x", "y", "z"], ...]
    scale: tuple[float, ...]
    level: int
    origin: tuple[float, ...]
    path: str | Sentinel

    def from_template(
        self,
        name=None,
        path=MISSING,
        type_=None,
        origin=None,
        scale=None,
        dims=None,
        level=None,
    ):
        return RasterMeta(
            name=self.name if name is None else name,
            type_=self.type_ if type_ is None else type_,
            dims=self.dims if dims is None else dims,
            scale=self.scale if scale is None else scale,
            level=self.level if level is None else level,
            origin=self.origin if origin is None else origin,
            path=self.path if path is MISSING else path,
        )
