from dataclasses import asdict, dataclass

from zfish.preprocessing.types import ColName, FrameID


@dataclass(frozen=True, slots=True)
class MetaBase:
    def from_template(self, **kwargs):
        return self.__class__(**{**self.to_dict(), **kwargs})

    def to_dict(self, shallow=False):
        if shallow:
            return {attr: getattr(self, attr) for attr in self.__dataclass_fields__}
        return asdict(self)


import itertools


@dataclass(frozen=True, slots=True)
class Key(MetaBase):
    comps: tuple[ColName, ...] = ()
    name: str = ""
    unique: bool = False
    frame: FrameID | None = None

    def is_in(self, columns) -> bool:
        return all(c in columns for c in self.comps)

    def intersect(self, columns: "tuple[str, ...] | Key"):
        return tuple(c for c in self.comps if c in columns)

    def union(self, columns: "tuple[str, ...] | Key"):
        res = []
        for c in itertools.chain.from_iterable(self.columns, columns):
            if c not in res:
                res.append(c)
        return tuple(res)

    def add_comp(self, col_name: ColName, strict: bool = False):
        if col_name in self.comps:
            if strict:
                raise ValueError(
                    f"can't add existing component {col_name!r} when `strict=True`"
                )
            return self.from_template()
        return self.from_template(comps=self.comps + (col_name,))

    def add_comps(self, col_names: tuple[ColName, ...], strict: bool = False):
        out_key = self
        for col_name in col_names:
            out_key = out_key.add_comp(col_name, strict=strict)
        return out_key

    def remove_comp(self, col_name: ColName, strict: bool = False):
        if col_name not in self.comps:
            if strict:
                raise ValueError(
                    f"can't remove non-existing component {col_name!r} when `strict=True`"
                )
            return self.from_template()
        return self.from_template(comps=tuple([e for e in self.comps if e != col_name]))

    def remove_comps(self, col_names: tuple[ColName, ...], strict: bool = False):
        out_key = self
        for col_name in col_names:
            out_key = out_key.remove_comp(col_name, strict=strict)
        return out_key

    def __iter__(self):
        yield from self.comps


@dataclass(frozen=True, slots=True)
class PKey(Key):
    name: str = ""
    unique: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class FKey(Key):
    target_frame: FrameID
    target_comps: tuple[ColName, ...] = None
    unique: bool = None

    def __post_init__(self):
        if self.target_comps is None:
            object.__setattr__(self, "target_comps", self.comps)
        if self.unique is None:
            object.__setattr__(self, "unique", False)


@dataclass(frozen=True, slots=True)
class MKey(Key):
    """Merge key"""

    unique: bool = False
    sep: str = "_"


@dataclass(frozen=True, slots=True)
class FrameMeta(MetaBase):
    # id_: FrameId
    pk: PKey
    fks: tuple[FKey, ...] = ()
    oks: tuple[Key, ...] = ()
    hidden: bool = False

    # TODO add init
    def __post_init__(self):
        if not isinstance(self.pk, PKey):
            object.__setattr__(self, "pk", PKey(**self.pk))
        object.__setattr__(
            self,
            "fks",
            tuple(FKey(**fk) if not isinstance(fk, FKey) else fk for fk in self.fks),
        )
        object.__setattr__(
            self,
            "oks",
            tuple(Key(**ok) if not isinstance(ok, Key) else ok for ok in self.oks),
        )

    def keys(self):
        return tuple([self.pk, *self.fks, *self.oks])

    def get_key_component(self, key):
        return getattr(self, key).comps

    def get_key_components(self):
        return tuple([e.comps for e in self.keys()])

    def get_key_components_dict(self):
        return {
            "pk": self.pk.comps,
            "fks": tuple(com for key in self.fks for com in key.comps),
            "oks": tuple(com for key in self.oks for com in key.comps),
        }
