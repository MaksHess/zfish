# %%
import logging
from dataclasses import dataclass
from itertools import zip_longest
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import polars as pl
from polars.type_aliases import SelectorType

from zfish.multi_table.schema_migration import update_meta_inplace
from zfish.plot.features import iqr_range
from zfish.preprocessing.types import AnyFrameT

logger = logging.getLogger(__name__)


def identity(x):
    return x


TRANSFORMS = {
    "log10": np.log10,
    "log2": np.log2,
    "log": np.log,
    "identity": identity,
}

DEBRIS_OUTLIER_RANGES = [
    {
        "feature": "debris_proba",
        "upper": 0.3,
    }
]

SEGMENTATION_OUTLIER_RANGES = [
    {
        "feature": "EquivalentSphericalRadius",  # um
        "lower": 4,
        "upper": 8,
    },
]
INTENSITY_OUTLIER_RANGES = [
    {
        "feature": "PCNA.0_Mean",
        "lower": 1.0,
        "upper": 10.0,
    },
    {
        "feature": "PCNA.0_Skewness",
        "upper": 3.0,
    },
    {
        "feature": "PCNA.0_Kurtosis",
        "upper": 5.0,
    },
    {
        "feature": "DAPI.1_Mean",
        "lower": 1.0,
        "upper": 8.0,
    },
    {
        "feature": "DAPI.1_Skewness",
        "lower": -2.0,
        "upper": 3.0,
    },
    {
        "feature": "DAPI.1_Kurtosis",
        "upper": 5.0,
    },
    {
        "feature": "pH3.1_Mean",
        "lower": 1.0,
        "upper": 8.0,
    },
    {
        "feature": "pH3.1_Variance",
        "lower": 0.0,
        "upper": 14.0,
    },
    {
        "feature": "pH3.1_Kurtosis",
        "upper": 5.0,
    },
]

ALIGNMENT_OUTLIER_RANGES = [
    {
        "feature": "DAPI.0|DAPI.1_PearsonR",
        "lower": 0.75,
    },
    {
        "feature": "DAPI.2|DAPI.1_PearsonR",
        "lower": 0.75,
    },
    {
        "feature": "DAPI.3|DAPI.1_PearsonR",
        "lower": 0.75,
    },
]

MDL_OUTLIER_IQRS = [
    {
        "feature": "nucleiRaw3_Pol-II-S2P.0_Mean",
        "transform": "log",
    },
    {
        "feature": "nucleiRaw3_Pol-II-S5P.2_Mean",
        "transform": "log",
    },
    {
        "feature": "nucleiRaw3_FLAG.0_Mean",
        "transform": "log",
    },
    {
        "feature": "nucleiRaw3_PCNA.0_Mean",
        "transform": "log",
    },
    {
        "feature": "nucleiRaw3_DAPI.1_Mean",
        "transform": "log",
        "iqr_mult": 2.5,
    },
    {
        "feature": "nucleiRaw3_H2B.2_Mean",
        "transform": "log",
        "iqr_mult": 2.5,
    },
    {
        "feature": "nucleiRaw3_Nanog.3_Mean",
        "transform": "log",
        "iqr_mult": 2.5,
    },
    {
        "feature": "nucleiRaw3_H3K27Ac.1_Mean",
        "transform": "log",
        "iqr_mult": 2.5,
    },
]

DEFAULT_OUTLIER_RANGES = SEGMENTATION_OUTLIER_RANGES + INTENSITY_OUTLIER_RANGES
NEW_OUTLIER_RANGES = (
    DEBRIS_OUTLIER_RANGES + SEGMENTATION_OUTLIER_RANGES + INTENSITY_OUTLIER_RANGES
)


@runtime_checkable
class Outlier(Protocol):
    feature: str
    transform: Literal["log10", "log2", "log", "identity"]

    def fit(self, df: pl.DataFrame) -> "Outlier": ...

    @property
    def lower(self) -> str | None: ...

    @property
    def upper(self) -> str | None: ...

    def indicate_lower_outlier(self, df: pl.DataFrame) -> pl.Series:
        self.fit(df)
        # col_expression = pl.col(self.feature).map_elements(
        #     TRANSFORMS[self.transform], return_dtype=pl.Float64
        # )
        col_expression = TRANSFORMS[self.transform](pl.col(self.feature))
        if self.lower is None:
            return df.select(col_expression.is_null()).to_series()
        return df.select(col_expression.lt(self.lower)).to_series()

    def indicate_upper_outlier(self, df: pl.DataFrame) -> pl.Series:
        self.fit(df)
        # col_expression = pl.col(self.feature).map_elements(
        #     TRANSFORMS[self.transform], return_dtype=pl.Float64
        # )
        col_expression = TRANSFORMS[self.transform](pl.col(self.feature))
        if self.upper is None:
            return df.select(col_expression.is_null()).to_series()
        return df.select(col_expression.gt(self.upper)).to_series()

    def indicate_outlier(self, df: pl.DataFrame) -> pl.Series:
        return (
            self.indicate_lower_outlier(df) | self.indicate_upper_outlier(df)
        ).to_frame()

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class RngOutlier(Outlier):
    feature: str
    lower: float | None = None
    upper: float | None = None
    transform: Literal["log10", "log2", "log", "identity"] = "identity"

    def fit(self, df):
        return self


@dataclass
class QuantOutlier(Outlier):
    feature: str
    q_lower: float = 0.0
    q_upper: float = 1.0
    transform: Literal["log10", "log2", "log", "identity"] = "identity"

    def fit(self, df):
        self._lower = df[self.feature].quantile(self.q_lower)
        self._upper = df[self.feature].quantile(self.q_upper)

    @property
    def lower(self) -> float:
        if not hasattr(self, "_lower"):
            raise AttributeError("Call fit() first")
        return self._lower

    @property
    def upper(self) -> float:
        if not hasattr(self, "_upper"):
            raise AttributeError("Call fit() first")
        return self._upper


@dataclass
class IQROutlier(Outlier):
    feature: str
    q_lower: float | None = 0.1
    q_upper: float | None = 0.9
    iqr_mult: float = 2.0
    transform: Literal["log10", "log2", "log", "identity"] = "identity"

    def fit(self, df):
        if not self.transform == "identity":
            # raise NotImplementedError(
            #     "IQROutlier on transformed feature columnes not yet implemented!"
            # )
            df = df.with_columns(
                pl.col(self.feature).log()
            )  # Base does not matter for IQR range
        if self.q_lower is None and self.q_upper is None:
            self._lower = df.select(pl.col(self.feature).min())[self.feature].item()
            self._upper = df.select(pl.col(self.feature).max())[self.feature].item()

        elif self.q_lower is None:
            self._lower = df.select(pl.col(self.feature).min())[self.feature].item()
            upper = df.select(pl.col(self.feature).quantile(self.q_upper))[
                self.feature
            ].item()
            iqr = upper - self._lower
            self._upper = upper + (self.iqr_mult - 1) * iqr

        elif self.q_upper is None:
            self._upper = df.select(pl.col(self.feature).max())[self.feature].item()
            lower = df.select(pl.col(self.feature).quantile(self.q_lower))[
                self.feature
            ].item()
            iqr = self._upper - lower
            self._lower = lower - (self.iqr_mult - 1) * iqr

        else:
            lower = df.select(pl.col(self.feature).quantile(self.q_lower))[
                self.feature
            ].item()
            upper = df.select(pl.col(self.feature).quantile(self.q_upper))[
                self.feature
            ].item()
            iqr = upper - lower
            self._lower = lower - (self.iqr_mult - 1) / 2 * iqr
            self._upper = upper + (self.iqr_mult - 1) / 2 * iqr

        # if not self.transform == "identity":
        #     self._lower = np.exp(self._lower)
        #     self._upper = np.exp(self._upper)

    @property
    def lower(self) -> str | None:
        if not hasattr(self, "_lower"):
            raise AttributeError("Call fit() first")
        return self._lower

    @property
    def upper(self) -> str | None:
        if not hasattr(self, "_upper"):
            raise AttributeError("Call fit() first")
        return self._upper


DEBRIS_OUTLIERS = tuple(RngOutlier(**rng) for rng in DEBRIS_OUTLIER_RANGES)
SEGMENTATION_OUTLIERS = tuple(RngOutlier(**rng) for rng in SEGMENTATION_OUTLIER_RANGES)
INTENSITY_OUTLIERS = tuple(RngOutlier(**rng) for rng in INTENSITY_OUTLIER_RANGES)
ALIGNMENT_OUTLIERS = tuple(RngOutlier(**rng) for rng in ALIGNMENT_OUTLIER_RANGES)

LABEL_OUTLIER_NAMES = [e.feature for e in SEGMENTATION_OUTLIERS + DEBRIS_OUTLIERS]
ALIGNMENT_OUTLIER_NAMES = [e.feature for e in ALIGNMENT_OUTLIERS]
INTENSITY_OUTLIER_NAMES = [e.feature for e in INTENSITY_OUTLIERS]

MDL_INTENSITY_OUTLIERS = [IQROutlier(**f) for f in MDL_OUTLIER_IQRS]


def mark_outliers_individual(
    df: AnyFrameT,
    outliers: list[Outlier],
    verbose: bool = True,
    plot_results: bool = False,
    add_any_all: bool = True,
) -> pl.DataFrame:
    is_outlier = pl.concat(
        [outlier.indicate_outlier(df) for outlier in outliers],
        how="horizontal",
    )
    is_outlier = is_outlier.with_columns(
        pl.all_horizontal(pl.all()).alias("All Outliers"),
        pl.any_horizontal(pl.all()).alias("Any Outliers"),
    )
    if verbose:
        is_below = pl.DataFrame(
            [outlier.indicate_lower_outlier(df) for outlier in outliers],
        ).with_columns(
            pl.all_horizontal(pl.all()).alias("All Outliers"),
            pl.any_horizontal(pl.all()).alias("Any Outliers"),
        )
        is_above = pl.DataFrame(
            [outlier.indicate_upper_outlier(df) for outlier in outliers],
        ).with_columns(
            pl.all_horizontal(pl.all()).alias("All Outliers"),
            pl.any_horizontal(pl.all()).alias("Any Outliers"),
        )
        outlier_summary(is_outlier, is_below, is_above)
    else:
        is_below = is_above = None

    if plot_results:
        _results_plot(df, outliers)

    if add_any_all:
        return is_outlier
    else:
        return is_outlier.select(pl.exclude("All Outliers", "Any Outliers"))


def mark_outliers(
    df: AnyFrameT,
    outliers: list[Outlier],
    verbose=True,
    how: Literal["any", "all"] = "any",
) -> pl.Series:
    is_outlier = mark_outliers_individual(
        df, outliers=outliers, verbose=verbose, add_any_all=True
    )
    if how == "any":
        return is_outlier.select(pl.col("Any Outliers")).to_series()
    elif how == "all":
        return is_outlier.select(pl.col("All Outliers")).to_series()


def outlier_summary(
    df_outliers: pl.DataFrame, is_below=None, is_above=None
) -> pl.DataFrame:
    if hasattr(df_outliers, "to_frame"):
        df_outliers = df_outliers.to_frame()
    n_total = df_outliers.height
    for c in df_outliers.columns:
        n_outliers = df_outliers[c].sum()
        logger.info(f"{c}")
        logger.info(
            f"{n_outliers:>8,}/{n_total:<8,}={n_outliers/n_total:>7.2%} outliers or null"
        )
        if is_below is not None:
            n_below = is_below[c].sum()
            logger.info(f"{n_below:>8,}/{n_total:<8,}={n_below/n_total:>7.2%} below")
        if is_above is not None:
            n_above = is_above[c].sum()
            logger.info(f"{n_above:>8,}/{n_total:<8,}={n_above/n_total:>7.2%} above")


def drop_outliers(df, outliers: Sequence[Outlier], verbose=True, plot_results=False):
    s_outliers = mark_outliers(df, outliers, verbose=verbose)
    if verbose:
        logger.info(f"{' Removing outliers ':=^50}")
        outlier_summary(s_outliers)
    df_out = df.filter(~s_outliers)
    if plot_results:
        results_plot(df, df_out, outliers)
    return df_out


def remove_outliers(df, outliers: Sequence[Outlier], verbose=True, plot_results=False):
    return drop_outliers(
        df=df, outliers=outliers, verbose=verbose, plot_results=plot_results
    )


def clip_outliers(df, outliers: Sequence[Outlier], verbose=True, plot_results=False):
    if verbose:
        df_outliers = mark_outliers(df, outliers, verbose=verbose)
        logger.info(f"{' Clipping values to outlier ranges ':=^50}")
        outlier_summary(df_outliers)
    df_out = df.clone()
    for outlier in outliers:
        outlier.fit(df)
        if outlier.lower is not None:
            df_out = df_out.with_columns(
                pl.col(outlier.feature).clip_min(outlier.lower)
            )
        if outlier.upper is not None:
            df_out = df_out.with_columns(
                pl.col(outlier.feature).clip_max(outlier.upper)
            )
    if plot_results:
        results_plot(df, df_out, outliers)
    return df_out


def results_plot(df, df_out, outliers: Sequence[Outlier], ncols=4, figsize=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    last_axes = False
    nrows = int(np.ceil(len(outliers) / ncols))
    figsize = figsize or (ncols * 4, nrows * 3)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, dpi=300)
    old_ax = axs[0]
    for outlier, ax in zip_longest(outliers, axs.flatten()):
        if outlier is None:
            ax.remove()
            if last_axes is False:
                last_axes is True
                old_ax.legend()
                continue
        sns.kdeplot(df[outlier.feature], ax=ax, label="before")
        sns.kdeplot(df_out[outlier.feature], ax=ax, label="after")
        ax.set_xlim(*iqr_range(df[outlier.feature], q_lower=0.1, q_upper=0.9, r=3))
        ax.set_title(outlier.feature)
        if outlier.lower is not None:
            ax.axvline(x=outlier.lower, color="k", linestyle="dotted")
        if outlier.upper is not None:
            ax.axvline(x=outlier.upper, color="k", linestyle="dashed")

        old_ax = ax


def _results_plot(df, outliers: Sequence[Outlier], ncols=4, figsize=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    last_axes = False
    nrows = int(np.ceil(len(outliers) / ncols))
    figsize = figsize or (ncols * 4, nrows * 3)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, dpi=300)
    old_ax = axs[0]
    for outlier, ax in zip_longest(outliers, axs.flatten()):
        if outlier is None:
            ax.remove()
            if last_axes is False:
                last_axes is True
                old_ax.legend()
                continue
        x = df.select(TRANSFORMS[outlier.transform](pl.col(outlier.feature)))[
            outlier.feature
        ]
        sns.kdeplot(x, ax=ax, label="before")
        ax.set_xlim(*iqr_range(x, q_lower=0.1, q_upper=0.9, r=3))
        ax.set_title(outlier.feature)
        if outlier.lower is not None:
            ax.axvline(x=outlier.lower, color="k", linestyle="dotted")
        if outlier.upper is not None:
            ax.axvline(x=outlier.upper, color="k", linestyle="dashed")

        old_ax = ax


def compute_nuclei_outliers(f, suffix="", return_all=False):
    import polars.selectors as cs

    from zfish.multi_table.tables_io import IDX_SEL, join, safe_collect, to_wide
    from zfish.preprocessing.transform import (
        LOG_TRANSFORM_SELECTOR,
        apply_log_transform,
    )

    OUTLIERS = (
        SEGMENTATION_OUTLIERS
        + ALIGNMENT_OUTLIERS
        + DEBRIS_OUTLIERS
        + INTENSITY_OUTLIERS
    )
    INTENSITY_TALL_FEATURE_COLS = [e.feature.split("_")[-1] for e in INTENSITY_OUTLIERS]
    CHANNEL_COLS = [e.feature.split("_")[0] for e in INTENSITY_OUTLIERS]
    OUTLIER_COLS = [e.feature for e in OUTLIERS]

    fw_nucs = (
        f.filter(pl.col("o") == "nucleiRaw3")
        .filter(pl.col("c").is_in(CHANNEL_COLS))
        .pipe_tables(
            lambda x: x.select(
                IDX_SEL, pl.col(pl.Series(INTENSITY_TALL_FEATURE_COLS).unique())
            ),
            include_tables=("intensity",),
        )
        .pipe_tables(
            to_wide,
            fidx_sep="|",
            include_tables=("label", "intensity", "correlation", "classifier"),
        )
    ).pipe(update_meta_inplace)

    df_debris_outliers = fw_nucs.classifier.pipe(
        lambda x: x.select(
            IDX_SEL,
            pl.col("debris_probas")
            .list.get(0)
            .struct.field("proba")
            .alias("debris_proba"),
        ),
    )
    df_label_outliers = fw_nucs.label.select(
        IDX_SEL, pl.col("EquivalentSphericalRadius")
    )
    df_alignment_outliers = fw_nucs.correlation.select(
        IDX_SEL, cs.ends_with("PearsonR")
    )
    df_intensity_outliers = fw_nucs.intensity.pipe(
        apply_log_transform, LOG_TRANSFORM_SELECTOR
    )
    print(df_debris_outliers.columns)
    print(df_alignment_outliers.columns)
    print(df_intensity_outliers.columns)
    print(df_label_outliers.columns)

    df_outliers = join(
        df_label_outliers,
        df_debris_outliers,
        df_alignment_outliers,
        df_intensity_outliers,
    )

    df_nuc_outliers = df_outliers.select(IDX_SEL).with_columns(
        df_outliers.pipe(mark_outliers_individual, OUTLIERS, add_any_all=False).select(
            pl.col(OUTLIER_COLS).name.suffix(suffix), pl.exclude(OUTLIER_COLS)
        )
    )
    if return_all:
        return df_nuc_outliers
    return df_nuc_outliers.filter(pl.any_horizontal(cs.by_dtype(pl.Boolean)))


def main():
    df = pl.DataFrame({"a": np.arange(1, 11, dtype=np.float32)})
    outlier = RngOutlier(feature="a", lower=3, upper=6, transform="identity")
    log_outlier = RngOutlier(
        feature="a", lower=np.log(3), upper=np.log(6), transform="log"
    )
    log10_outlier = RngOutlier(
        feature="a", lower=np.log10(3), upper=np.log10(6), transform="log10"
    )
    log2_outlier = RngOutlier(
        feature="a", lower=np.log2(3), upper=np.log2(6), transform="log2"
    )

    assert np.all(outlier.indicate_outlier(df) == log_outlier.indicate_outlier(df))
    assert np.all(outlier.indicate_outlier(df) == log10_outlier.indicate_outlier(df))
    assert np.all(outlier.indicate_outlier(df) == log2_outlier.indicate_outlier(df))


# def mark_range_outlier(
#     df: AnyFrameT,
#     feature: str,
#     lower: float | None = None,
#     upper: float | None = None,
# ) -> pl.DataFrame:
#     if lower is None and upper is None:
#         name = f"{feature}_range_outlier"
#         filter_expr = pl.col(feature).is_null()
#     elif lower is None:
#         name = f"{feature}_u_range_outlier"
#         filter_expr = ~pl.col(feature).le(upper)
#     elif upper is None:
#         name = f"{feature}_l_range_outlier"
#         filter_expr = ~pl.col(feature).ge(lower)
#     else:
#         name = f"{feature}_lu_range_outlier"
#         filter_expr = ~pl.col(feature).is_between(lower, upper)

#     is_outlier = df.select(pl.when(filter_expr).then(True).otherwise(False).alias(name))
#     return is_outlier


# def mark_range_outliers(
#     df: AnyFrameT, outlier_ranges: tuple[dict[str, Any]]
# ) -> pl.DataFrame:
#     return pl.concat(
#         [mark_range_outlier(df, **rng) for rng in outlier_ranges], how="horizontal"
#     ).with_columns(
#         pl.all_horizontal(pl.all()).alias("all_range_outlier"),
#         pl.any_horizontal(pl.all()).alias("any_range_outlier"),
#     )


# def discard_bounds(
#     df: AnyFrameT,
#     feature: str,
#     lower: float | None = None,
#     upper: float | None = None,
#     verbose: bool = True,
#     plot_results: bool = True,
#     ax=None,
# ) -> AnyFrameT:
#     if lower is None and upper is None:
#         out = df
#     elif lower is None:
#         x = [upper]
#         linestyles = ["dashed"]
#         out = df.filter(pl.col(feature).lt(upper))
#     elif upper is None:
#         x = [lower]
#         linestyles = ["dotted"]
#         out = df.filter(pl.col(feature).gt(lower))
#     else:
#         x = [lower, upper]
#         linestyles = ["dotted", "dashed"]
#         out = df.filter(pl.col(feature).is_between(lower, upper))

#     if verbose or plot_results:
#         if isinstance(df, pl.LazyFrame):
#             df: pl.DataFrame = df.collect()
#             out: pl.DataFrame = out.collect()
#             return_lazy = True
#         else:
#             return_lazy = False

#     if verbose:
#         n_obs_before = df.height
#         n_obs_after = out.height
#         logger.info(
#             f"{n_obs_after}/{n_obs_before} ({n_obs_after/n_obs_before:2.2%}) remaining \
# objs after {feature!r}"
#         )

#     if plot_results:
#         import matplotlib.pyplot as plt
#         import seaborn as sns

#         from zfish.features.polars_selector import sel

#         if ax is None:
#             fig, ax = plt.subplots(figsize=(4, 3))
#         ax = sns.kdeplot(
#             df.select(
#                 [
#                     sel.index,
#                     pl.col(feature),
#                 ]
#             ).to_pandas(),
#             x=feature,
#             ax=ax,
#         )

#         y_lims = ax.get_ylim()
#         plt.vlines(
#             x=x,
#             ymin=y_lims[0],
#             ymax=y_lims[1],
#             colors=["k"],
#             linestyles=linestyles,
#         )

#     if return_lazy:
#         return out.lazy()
#     else:
#         return out


# class OutlierRange(TypedDict):
#     feature: str
#     lower: float | None
#     upper: float | None


# def remove_outliers(
#     df: AnyFrameT, outliers: list[OutlierRange], verbose=True, plot_results=True
# ) -> AnyFrameT:
#     if verbose:
#         if isinstance(df, pl.LazyFrame):
#             n_obs_initial = df.collect().height
#         else:
#             n_obs_initial = df.height

#     for outlier in outliers:
#         df = partial(
#             discard_bounds, **outlier, verbose=verbose, plot_results=plot_results
#         )(df)

#     if verbose:
#         if isinstance(df, pl.LazyFrame):
#             n_obs_final = df.collect().height
#         else:
#             n_obs_final = df.height
#         logger.info(
#             f"{n_obs_final}/{n_obs_initial} ({n_obs_final/n_obs_initial:2.2%}) remaining \
# objs in the end."
#         )

#     return df


# # %%
