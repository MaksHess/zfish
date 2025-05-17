# %%
import re
from dataclasses import dataclass
from itertools import chain, zip_longest
from typing import ClassVar, Literal, NewType, Self

from zfish.commons.types_ import MISSING

SEP = "."
RE_SEP = re.escape(SEP)

KEY_COMP_PATTERN = r"^[a-z0-9_]+(?:.[a-z0-9_]+)*$"
COL_NAME_PATTERN = r".+"
FRAME_NAME_PATTERN = r"\w{2,20}"
MFRAME_NAME_PATTERN = r"\w{2,20}"

NamedPattern = NewType("NamedPattern", str)


def _build_regex(
    patterns: tuple[NamedPattern, ...],
    seps: tuple[str, ...] | str = (RE_SEP,),
    re_compile=False,
):
    if isinstance(seps, str):
        seps = (seps,)
    else:
        seps = tuple(seps)

    n_seps_required = len(patterns) - 1
    n_seps = len(seps)

    if n_seps != n_seps_required:
        if n_seps != 1:
            raise ValueError(
                f"Either 0, 1 or {len(patterns)=}-1={n_seps_required} seps allowed, got {n_seps}"
            )
        if n_seps == 1:
            seps = n_seps_required * seps
    if seps == ():
        seps = ("",)
    # groups = tuple(f"(?P<{name}>{name_pattern})" for (name, name_pattern) in patterns)
    re_pattern = "".join(chain.from_iterable(zip_longest(patterns, seps, fillvalue="")))

    if re_compile:
        return re.compile(re_pattern)
    return re_pattern


NAME_PATTERN_MAP = {
    "mframe": MFRAME_NAME_PATTERN,
    "frame": FRAME_NAME_PATTERN,
    "col": COL_NAME_PATTERN,
}

NAMED_RE_GROUPS: tuple[str] = tuple(
    f"(?P<{name}>{name_pattern})" for name, name_pattern in NAME_PATTERN_MAP.items()
)
QNAMED_RE_GROUPS: dict[str, tuple[str, ...]] = {}
for i, name in enumerate(NAME_PATTERN_MAP):
    QNAMED_RE_GROUPS[f"{name}"] = NAMED_RE_GROUPS[: i + 1]


QNAME_REGEXES = {
    name: _build_regex(comps, re_compile=True)
    for name, comps in QNAMED_RE_GROUPS.items()
}

HIDDEN_GROUP = "(?P<hidden>_)?"
VERSION_GROUP = f"(?P<version>{RE_SEP}v\d\d?)?"

TABLE_REGEX = re.compile(
    f"{HIDDEN_GROUP}{_build_regex(QNAMED_RE_GROUPS['frame'])}{VERSION_GROUP}"
)


class IdParser:
    __QNAME_REGEX__: ClassVar[re.Pattern]
    __QNAME_SEP__: ClassVar[str] = SEP

    @classmethod
    def extract_mo_group_dict(
        cls,
        qname: str,
        strategy: Literal["full", "startswith", "endswith", "any"] = "full",
    ) -> dict[str, str]:
        if strategy == "full":
            mo = cls.regex().match(qname)
        else:
            raise NotImplementedError(f"Strategy {strategy!r} not implemented.")
        if mo is None:
            raise ValueError(f"Regex did not match {qname!r}: {cls.regex()} ")
        return mo.groupdict()

    @classmethod
    def from_filename(cls, qname: str) -> Self:
        group_dict = cls.extract_mo_group_dict(qname)
        return cls(**group_dict)

    @classmethod
    def separator(cls) -> str:
        return cls.__QNAME_SEP__

    @classmethod
    def regex(cls) -> re.Pattern:
        return cls.__QNAME_REGEX__


@dataclass(frozen=True, slots=True)
class FrameId(IdParser):
    __QNAME_REGEX__: ClassVar[re.Pattern] = TABLE_REGEX
    mframe: str
    frame: str
    version: str | None = None
    hidden: str | None = None
    _squeezed: tuple[tuple[str, str | int], ...] = ()

    @property
    def squeezed(self):
        return dict(self._squeezed)

    def to_filename(self):
        hidden = "" if self.hidden is None else self.hidden
        version = "" if self.version is None else self.version

        return "".join(
            [hidden, self.separator().join([self.mframe, self.frame]), version]
        )

    def to_qualname(self):
        return self.separator().join([self.mframe, self.frame])

    def to_localname(self):
        return self.frame

    def from_template(
        self,
        mframe=MISSING,
        frame=MISSING,
        version=MISSING,
        hidden=MISSING,
        _squeezed=MISSING,
    ):
        new_params = {
            "mframe": mframe if mframe is not MISSING else getattr(self, "mframe"),
            "frame": frame if frame is not MISSING else getattr(self, "frame"),
            "version": version if version is not MISSING else getattr(self, "version"),
            "hidden": hidden if hidden is not MISSING else getattr(self, "hidden"),
            "_squeezed": _squeezed
            if _squeezed is not MISSING
            else getattr(self, "_squeezed"),
        }

        return FrameId(**new_params)


# @dataclass(frozen=True, slots=True)
# class FrameId(IdParser):
#     __QNAME_REGEX__: ClassVar[re.Pattern] = QNAME_REGEXES["mframe"]
#     mframe: str


# @dataclass(frozen=True, slots=True)
# class MframeId(IdParser):
#     __QNAME_REGEX__: ClassVar[re.Pattern] = QNAME_REGEXES["mframe"]
#     mframe: str


# @dataclass(frozen=True, slots=True)
# class FrameIdShort(IdParser):
#     __QNAME_REGEX__: ClassVar[re.Pattern] = QNAME_REGEXES["frame"]
#     mframe: str
#     frame: str


# @dataclass(frozen=True, slots=True)
# class ColId(IdParser):
#     __QNAME_REGEX__: ClassVar[re.Pattern] = QNAME_REGEXES["col"]
#     mframe: str
#     frame: str
#     col: str
