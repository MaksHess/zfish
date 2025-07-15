# %%
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cmap as cc
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import Formatter, FuncFormatter, Locator, LogFormatterSciNotation
from scipy.special import expit, logit
from statsmodels.nonparametric.smoothers_lowess import lowess

# from zfish.plot.colormap import get_circular_palette, get_palette
# from zfish.plot.datashader import dsscatter

# hv.output(widget_location="bottom")
# pio.templates["gridon"].update( #show grids also for scene (i. e. 3d axes)
#     dict(
#         layout_scene_xaxis=dict(showgrid=True),
#         layout_scene_yaxis=dict(showgrid=True),
#         layout_scene_zaxis=dict(showgrid=True),
#     )
# )
# pio.templates.default = "simple_white+gridon"

BASE = r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\base.mplstyle"

UMAP = r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\umap.mplstyle"


def make_lowess(df=None, x=None, y=None, frac=0.1, delta=0, return_sorted=False):
    if df is not None:
        if x is None or y is None:
            assert df.width == 2
            y, x = df[df.columns[0]], df[df.columns[1]]
        else:
            y, x = df[y], df[x]
    return lowess(y, x, frac=frac, delta=delta, return_sorted=return_sorted)


class MinorSymLogLocator(Locator):
    """
    Blatantly stolen from https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, dummy_locator=3):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically.

        This parameter does not seem to have an effect?
        Ok it needs to be passed to adhere to the interface.
        """
        self.linthresh = dummy_locator

    def __call__(self):
        "Return the locations of the ticks"
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []
        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            "Cannot get tick locations for a %s type." % type(self)
        )


def _set_major_tick_function_formatters(
    ax, axis: Literal["xaxis", "yaxis", "both"] = "both"
):
    if axis == "both":
        axes = ("xaxis", "yaxis")
    else:
        axes = (axis,)

    old_formatters = [getattr(ax, axis).get_major_formatter() for axis in axes]

    new_formatters = []
    replace_axes = []
    for axis, old_formatter in zip(axes, old_formatters):
        if isinstance(old_formatter, LogFormatterSciNotation):
            new_formatters.append(
                lambda x, pos=None: old_formatter(x, pos)
                .replace("mathdefault", "mathregular")
                .replace("-", "\u2010")
            )
            replace_axes.append(axis)
    for axis, fmt in zip(replace_axes, new_formatters):
        getattr(ax, axis).set_major_formatter(fmt)
    # ax.yaxis.set_major_formatter(new_formatter)
    return ax


def _set_minor_tick_symlog_locator(ax):
    for axis_name in ("xaxis", "yaxis"):
        axis = getattr(ax, axis_name)
        if axis.get_scale() == "symlog":
            axis.set_minor_locator(MinorSymLogLocator(dummy_locator=1))
    return ax


from typing import Sequence, TypeVar

T = TypeVar("T")


def _safe_flatten(array_or_sequence: Sequence[T] | np.ndarray) -> tuple[T]:
    if isinstance(array_or_sequence, np.ndarray):
        return tuple(array_or_sequence.flatten())
    else:
        return tuple(array_or_sequence)


def _fix_axes_ticks(ax):
    _set_minor_tick_symlog_locator(_set_major_tick_function_formatters(ax))


def fix_axes_ticks(ax_or_axs):
    import matplotlib as mpl

    if isinstance(ax_or_axs, mpl.axes.Axes):
        _fix_axes_ticks(ax_or_axs)

    else:
        for ax in _safe_flatten(ax_or_axs):
            _fix_axes_ticks(ax)


class Colormap(cc.Colormap):
    def to_mpl(self, N=256, gamma=1.0):
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(name=self.name, colors=self.lut())


def apply_colormap(s: pl.Series, cmap=Colormap("Spectral")) -> pl.Series:
    unique_values = s.unique(maintain_order=True).to_list()
    if isinstance(cmap, cc.Colormap):
        color_map = {v: cmap(unique_values.index(v)).hex for v in unique_values}
    else:
        color_map = {v: elem for v, elem in zip(unique_values, cmap)}
    return s.replace_strict(color_map, return_dtype=pl.String)


def ccp_colormap(old_default=False):
    if old_default:
        return Colormap("cet_c9_r")
    else:
        return Colormap("cet_c9").shifted(0.5)


def division_cycle_colormap():
    spectral = Colormap("Spectral")
    div_hex5_ = [spectral(i).hex for i in [0.0, 0.25, 0.4, 0.75, 1.0]]
    div_hex5 = (
        [div_hex5_[0]] + darken_hex_list(div_hex5_[1:-1], stops=1.5) + [div_hex5_[-1]]
    )
    return Colormap(div_hex5, name="Spektral")


def celltype_colormap():
    # return Colormap(Colormap("Set1")(i) for i in range(3))
    return Colormap(["#a2142f", "#edb120", "#0072bd"])


def z_model_colormap():
    return ["#4d4d4dff", "#b15600ff", "#8d0000ff"]


def z_model_name_map():
    return {
        "NoCorrection": "No Correction",
        "ExpPos__OneStep": "1D z-model",
        "ExpPos__TwoStep": "2D z-model",
    }


def spatial_elements_colormap_dark():
    return [
        "#6A662C",
        "#2B4B5F",
        "#0F410F",
        "#4F2C6A",
        "#A74B00",  # z
        "#626262",
    ]


def spatial_elements_colormap_light():
    return [
        "#C8C379",  # c
        "#7AA9C4",  # o
        "#67A667",  # lbl_obj
        "#B4A0BE",  # m
        "#CF8548",  # iz
        "#CCCCCC",  # roi
    ]


def pooled_cycle_name_map():
    return {"9": "<=9", **dict(zip(map(str, range(10, 14)), map(str, range(10, 14))))}


@dataclass(frozen=True)
class PlotPaths:
    fig1: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\paper_manuscript\figures\Figure 1\Plots"
    )
    fig2: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\paper_manuscript\figures\Figure 2\Plots"
    )
    fig3: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\paper_manuscript\figures\Figure 3\Plots"
    )
    fig3: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\paper_manuscript\figures\Figure 4\Plots"
    )


@dataclass(frozen=True)
class ThesisPaths:
    results: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Results"
    )
    ccp: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Results\CCP"
    )
    transcription: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Results\Transcription"
    )
    classifier: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Results\Classifier"
    )
    feature_extraction: Path = Path(
        r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Results\FeatureExtraction"
    )


paths = PlotPaths()
tpaths = ThesisPaths()


@dataclass(frozen=True)
class NameMaps:
    z_models: dict[str, str] = field(default_factory=z_model_name_map)
    pooled_cycles: dict[str, str] = field(default_factory=pooled_cycle_name_map)


nmaps = NameMaps()


def darken_color(color: cc.Color, stops=1):
    color = cc.Color(color)
    STEP_SIZE = 0.5  # step size in logit-space.
    return cc.Color(
        color.hsl._replace(l=expit(logit(color.hsl.l) - (stops * STEP_SIZE))).to_rgba()
    )


def lighten_color(color: cc.Color, stops=1):
    color = cc.Color(color)
    STEP_SIZE = 0.5  # step size in logit-space.
    return cc.Color(
        color.hsl._replace(l=expit(logit(color.hsl.l) + (stops * STEP_SIZE))).to_rgba()
    )


def darken_hex_list(colors: list[str], stops=1):
    return [darken_color(e, stops=stops).hex for e in colors]


def lighten_hex_list(colors: list[str], stops=1):
    return [lighten_color(e, stops=stops).hex for e in colors]


# @dataclass(frozen=True)
@dataclass(frozen=True)
class Cmaps:
    ccp_hex: list[str] = field(init=False)
    division_cycle_hex_4: list[str] = field(init=False)
    division_cycle_hex_5: list[str] = field(init=False)
    division_cycle_hex_6: list[str] = field(init=False)
    division_cycle_hex_7: list[str] = field(init=False)
    celltype_hex: list[str] = field(init=False)
    z_models_hex: list[str] = field(default_factory=z_model_colormap)
    spatial_element_light: list[str] = field(
        default_factory=spatial_elements_colormap_light
    )
    spatial_element_dark: list[str] = field(
        default_factory=spatial_elements_colormap_dark
    )

    ccp: cc.Colormap = field(default_factory=ccp_colormap)
    division_cycle: cc.Colormap = field(default_factory=division_cycle_colormap)
    celltype: cc.Colormap = field(default_factory=celltype_colormap)

    def __post_init__(self):
        # self.ccp_hex = [self.division_cycle(i).hex for i in np.linspace(0, 1, 7)][:6]
        # self.division_cycle_hex = [self.division_cycle(i).hex for i in np.linspace(0, 1, 7)]
        # self.celltype_hex = [self.division_cycle(i).hex for i in np.linspace(0, 1, 3)]
        super().__setattr__(
            "ccp_hex", [self.ccp(i).hex for i in np.linspace(0, 1, 7)][:6]
        )
        super().__setattr__(
            "division_cycle_hex_4",
            [self.division_cycle(i).hex for i in np.linspace(0, 1, 4)],
        )

        div_hex5 = [cc.Color(e).hex for e in self.division_cycle(np.linspace(0, 1, 5))]
        stops_7 = [0.0, 0.12, 0.24, 0.5, 0.7, 0.9, 1.0]
        stops_5 = [0.12, *stops_7[3:]]

        div_hex5 = [cc.Color(e).hex for e in self.division_cycle(np.array(stops_5))]
        div_hex7 = [cc.Color(e).hex for e in self.division_cycle(np.array(stops_7))]
        super().__setattr__(
            "division_cycle_hex_5",
            div_hex5,
        )
        super().__setattr__(
            "division_cycle_hex_6",
            [self.division_cycle(i).hex for i in np.linspace(0, 1, 6)],
        )
        super().__setattr__(
            "division_cycle_hex_7",
            div_hex7,
        )
        super().__setattr__(
            "celltype_hex", [self.celltype(i).hex for i in np.linspace(0, 1, 3)]
        )


cmaps = Cmaps()

# %%
