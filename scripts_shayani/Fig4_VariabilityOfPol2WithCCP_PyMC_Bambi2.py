# %%
import pickle
import time
from dataclasses import dataclass, field
from functools import partial, reduce
from operator import add, or_
from pathlib import Path
from typing import Literal

import arviz as az
import bambi as bmb
import cmap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr

# from great_tables import GT
from matplotlib.ticker import FuncFormatter
from pymc import HalfCauchy, HalfNormal, Model, Normal, math, sample
from scipy.optimize import curve_fit
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklego.meta import GroupedTransformer

from zfish.multi_table.tables_io import scan_features, scan_resources, to_wide
from zfish.plot.colormap import darken, get_circular_palette, get_palette
from zfish.plot.plot_commons import BASE, cmaps, tpaths


# %%
IDATA_KWARGS = {"log_likelihood": False}

DEFAULT_FIT_PARAMS = {
    "nuts_sampler": "nutpie",
    "cores": None,
    "idata_kwargs": IDATA_KWARGS,
}

SCALERS = {
    "standard": StandardScaler(),
    "robust99": RobustScaler(quantile_range=(1.0, 99.0)),
    "robust95": RobustScaler(quantile_range=(5.0, 95.0)),
    "robust90": RobustScaler(quantile_range=(10.0, 90.0)),
    "robust75": RobustScaler(quantile_range=(25.0, 75.0)),
}
FULL_PALETTE = [
    "#0C75B4",
    "#3F3AA9",
    "#743598",
    "#A45585",
    "#C48274",
    "#D1AF68",
    "#CAE261",
    "#A7F073",
    "#77D591",
    "#55BAA2",
]


SHORT_NAMES = {
    # Index
    "roi": "roi",
    "o": "o",
    "label": "label",
    # Target
    "nuc_Pol-II-S2P.0_Mean": "Pol2.S2P",
    # Transcription
    "nuc_Pol-II-S5P.2_Mean": "Pol2.S5P",
    # Cell Cycle Raw
    "nuc_PCNA.0_Mean": "PCNA",
    "nuc_DAPI.1_Mean": "DAPI",
    # Morphology
    "nuc_PhysicalSize": "n_Volume",
    "nuc/cyto_PhysicalSizeRatio": "n.cy_VolumeRatio",
    "nuc_Roundness": "n_Roundness",
    "nuc_Flatness": "n_Flatness",
    # ZGA
    "nuc_FLAG.0_Mean": "Pou5",
    "nuc_Nanog.3_Mean": "Nanog",
    "nuc_H3K27Ac.1_Mean": "H3K27Ac",
    "nuc_H2B.2_Mean": "H2B",
    # CCP
    "NormalizedCCP": "CCP",
    # Control
    "nuc_bCatenin.1_Mean": "bCat",
    "Noise": "Noise",
    # Groups
    "id.emb": "id_emb",
    "cycle": "cycle",
    "cycle_pooled": "cycle_p",
    "ClassCCP": "ccp_class",
    "celltype_pred": "celltype",
}

LINEAR_PREDICTORS = list(SHORT_NAMES.values())[4:17]

FEATURE_SETS = {
    "ccp": ("CCP",),
    "ccp-spline": ("bs(CCP, 3, intercept=False)",),
    "ccp-asympt": (),  # No linear predictors
    "ccp-power": (),  # No linear predictors
    "ccp-feats": (
        "PCNA",
        "DAPI",
        "n_Volume",
        "n.cy_VolumeRatio",
        "n_Roundness",
        "n_Flatness",
    ),
    "zga": ("Pou5", "Nanog", "H3K27Ac", "H2B"),
    "s5p": ("Pol2.S5P",),
    "ctrl": ("bCat", "Noise"),
}

MODEL_FORMULA_FSETS = [
    # ("ccp",),
    # ("ccp-spline",),
    ("ccp-asympt",),
    ("ccp-power",),
    # ("ccp-feats",),
    # ("ccp-feats", "ccp-spline"),
    # ("ccp-feats", "ccp-asympt"),
    # ("ccp-feats", "ccp-power"),
    # ("zga",),
    # ("zga", "ccp"),
    # ("zga", "ccp-spline"),
    ("zga", "ccp-asympt"),
    ("zga", "ccp-power"),
    # ("zga", "ccp-feats"),
    # ("zga", "ccp-feats", "ccp-spline"),
    ("zga", "ccp-feats", "ccp-asympt"),
    ("zga", "ccp-feats", "ccp-power"),
    # ("s5p",),
    # ("s5p", "zga", "ccp"),
    # ("s5p", "zga", "ccp-spline"),
    ("s5p", "ccp-asympt"),
    ("s5p", "ccp-power"),
    ("s5p", "zga", "ccp-asympt"),
    ("s5p", "zga", "ccp-power"),
    # ("s5p", "zga", "ccp-feats"),
    # ("s5p", "zga", "ccp-feats", "ccp-spline"),
    ("s5p", "zga", "ccp-feats", "ccp-asympt"),
    ("s5p", "zga", "ccp-feats", "ccp-power"),
    # ("ctrl",),
    # ("ctrl", "ccp"),
    # ("ctrl", "ccp-spline"),
    # ("ctrl", "ccp-asympt"),
    # ("ctrl", "ccp-feats"),
    # ("ctrl", "ccp-feats", "ccp-asympt"),
    # ("ctrl", "ccp-feats", "ccp-spline"),
    # ("ctrl", "zga"),
    # ("ctrl", "zga", "ccp"),
    # ("ctrl", "zga", "ccp-spline"),
    # ("ctrl", "zga", "ccp-asympt"),
    # ("ctrl", "zga", "ccp-feats"),
    # ("ctrl", "zga", "ccp-feats", "ccp-asympt"),
    # ("ctrl", "zga", "ccp-feats", "ccp-spline"),
    # ("ctrl", "s5p"),
    # ("ctrl", "s5p", "zga", "ccp"),
    # ("ctrl", "s5p", "zga", "ccp-asympt"),
    # ("ctrl", "s5p", "zga", "ccp-spline"),
    # ("ctrl", "s5p", "zga", "ccp-feats"),
    # ("ctrl", "s5p", "zga", "ccp-feats", "ccp-asympt"),
    # ("ctrl", "s5p", "zga", "ccp-feats", "ccp-spline"),
]

MODEL_FORMULA_PRED = {
    "_".join(k): tuple(reduce(add, [FEATURE_SETS[j] for j in k], ()))
    for k in MODEL_FORMULA_FSETS
}

TARGET = "Pol2.S2P"


ALL_CLASSES = [
    "M ana",
    "S early",
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S late",
    "M meta",
]

# INTERPHASE_CLASSES = ["S early", *[f"S{i}" for i in range(6)]]
INTERPHASE_CLASSES = [
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    # "S5",
]

EXCLUDE_PATTERNS = ["~^mu$", "~Intercept", "~^.*_log__$"]


RETRIES = 5
PLOT_TOP_N_MODELS = [1, 2, 4, 6]
CYCLES = [8, 9, 10, 11, 12]
SAMPLES = 10000
SCALER = "standard"

OUT_PATH = Path(
    rf"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Results\Transcription\ModelResults4\samples={SAMPLES}_scaler={SCALER}"
)
OUT_PATH.mkdir(exist_ok=True, parents=True)

CCP_QUICKSAVE = Path(
    r"C:\Users\hessm\Documents\Programming\Python\thesis\Data\for_model_clp_pow(10, 0).parquet"
)

# %%
def main():
    import os

    import appdirs

    print(os.environ.get("NUMBA_CACHE_DIR"))
    print(appdirs.user_cache_dir())
    plt.style.use(BASE)
    matplotlib.rcParams["axes.unicode_minus"] = False

    mdls = fit_all_models(
        cycles=CYCLES,
        models={
            # k.replace("asympt", "power"): MODEL_FORMULA_PRED[k]
            k: MODEL_FORMULA_PRED[k]
            for k in list(MODEL_FORMULA_PRED.keys())
        },
        fit_models=None,
        load_existing_idata=True,
    )
    loos = compute_loos(mdls)
    df_all, df_s5p, df_no_s5p = compute_comparisions(loos)

    for n in PLOT_TOP_N_MODELS:
        all_cycles = np.unique([k[0] for k in mdls])
        for cy in all_cycles:
            fig, ax = plt.subplots(figsize=(2.6, n / 2 + 0.66))
            az.plot_forest(
                [
                    linear_predictors_from_beta(mdls[(cy, k)].idata)
                    for k in (
                        list(df_s5p[cy].index[:n]) + list(df_no_s5p[cy].index[:n])
                    )
                    if k not in ["ccp-asympt", "ccp-power"]
                ][::-1],
                combined=True,
                var_names=LINEAR_PREDICTORS[:-1],
                model_names=[
                    k
                    for k in list(df_s5p[cy].index[:n]) + list(df_no_s5p[cy].index[:n])
                    if k not in ["ccp-asympt", "ccp-power"]
                ][::-1],
                filter_vars="like",
                ax=ax,
                colors=[
                    cmap.Color(e).hex
                    for e in np.concatenate(
                        [
                            cmap.Colormap("Reds")(np.linspace(0.8, 0.3, n)),
                            cmap.Colormap("Greens")(np.linspace(0.8, 0.3, n)),
                        ]
                    )
                ][::-1],
                legend=True,
            )
            ax.axvline(0, color="k", linestyle=":")
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 1.06),
                title=f"Interphase Model Cycle {cy}",
                reverse=True,
                ncols=2,
            )
            fig.savefig(
                OUT_PATH / f"forestplot_cy{cy}_top{n}.png",
                # bbox_inches="tight",
                dpi=300,
            )
            fig, axs = plt.subplots(figsize=(2.6, n / 12 * 2 + 1), nrows=2, sharex=True)
            az.plot_compare(df_s5p[cy].head(n), ax=axs[0], textsize=5.0)
            axs[0].set_xlabel("")
            axs[0].set_ylabel("")
            axs[0].set_title(f"Interphase Models Cycle {cy}\nhigher is better")

            # fig, ax = plt.subplots(figsize=(2.6, n / 16 + 1))
            az.plot_compare(df_no_s5p[cy].head(n), ax=axs[1], textsize=5.0)
            axs[1].set_title("")
            axs[1].set_ylabel("")
            for ax in axs:
                new_ticklabels = []
                for ticklabel in ax.yaxis.get_ticklabels():
                    old_text = ticklabel.get_text()
                    ticklabel.set_text(
                        ", ".join(old_text.split("_"))
                        + f"  r2={mdls[(cy, old_text)].idata.attrs['r2']:0.2f}"
                    )
                    new_ticklabels.append(ticklabel)
                ax.yaxis.set_ticklabels(new_ticklabels)

            fig.savefig(
                OUT_PATH / f"compare_cy{cy}_top{n}.png",
                # bbox_inches="tight",
                dpi=300,
            )

# %%
def frequentist_asymreg(x, y, p0=(2.5, 0, 5)):
    def asymreg(X, a, b, c):
        return a - (a - b) * np.exp(-c * X)

    params, _ = curve_fit(
        asymreg,
        x,
        y,
        loss="linear",
        method="trf",
        p0=np.array(p0),
    )

    x_ = np.linspace(x.min(), x.max(), 100)
    y_ = asymreg(x_, *params)

    return params, (x_, y_)


def get_preprocessing_pipeline(
    scaler="standard",
    passthrough=(
        "celltype_pred",
        "roi",
        "o",
        "label",
        "id.emb",
        "cycle",
        "cycle_pooled",
        "ClassCCP",
        "Intensity_outliers",
        "Noise",
    ),
):
    preprocessing_pip = make_pipeline(
        make_column_transformer(
            (
                Pipeline([("log", FunctionTransformer(np.log1p))]).set_output(
                    transform="pandas"
                ),
                ["nuc_Pol-II-S2P.0_Mean"],
            ),
            (
                Pipeline(
                    [("log", FunctionTransformer(np.log1p)), ("scl", SCALERS[scaler])]
                ).set_output(transform="pandas"),
                [
                    "nuc_PCNA.0_Mean",
                    "nuc_FLAG.0_Mean",
                    "nuc_H3K27Ac.1_Mean",
                    "nuc_H2B.2_Mean",
                    "nuc_Nanog.3_Mean",
                    "nuc_Pol-II-S5P.2_Mean",
                    "nuc_pH3.1_Mean",
                    "nuc_DAPI.1_Mean",
                    "nuc_bCatenin.1_Mean",
                    "nuc/cyto_PhysicalSizeRatio",
                ],
            ),
            (
                SCALERS[scaler],
                [
                    "nuc_PhysicalSize",
                    "cyto_PhysicalSize",
                    "nuc_Flatness",
                    "nuc_Roundness",
                ],
            ),
            (
                GroupedTransformer(
                    transformer=MinMaxScaler(feature_range=(0, 3)),
                    groups=["cycle_pooled"],
                ),
                ["cycle_pooled", "NormalizedCCP"],
            ),
            (
                "passthrough",
                list(passthrough),
            ),
            verbose_feature_names_out=False,
        ),
    ).set_output(transform="pandas")
    return preprocessing_pip


def iqr_outlier(column, ql=0.25, qu=0.75, iqr_multiplier=1.5, log=True):
    if log:
        q_lower = pl.col(column).log(10).quantile(ql)
        q_upper = pl.col(column).log(10).quantile(qu)
    else:
        q_lower = pl.col(column).quantile(ql)
        q_upper = pl.col(column).quantile(qu)
    iqr = q_upper - q_lower
    offset = iqr * iqr_multiplier
    lower_bound = q_lower - offset
    upper_bound = q_upper + offset
    if log:
        return pl.col(column).log(10).is_between(lower_bound, upper_bound)
    else:
        return pl.col(column).is_between(lower_bound, upper_bound)


def load_data(path=CCP_QUICKSAVE, scaler="standard"):
    preprocessing_pip = get_preprocessing_pipeline(scaler=scaler)
    rng = np.random.default_rng(42)

    df = (
        (
            pl.read_parquet(path)
            .filter(pl.col("NormalizedCCP").is_not_null())
            .with_columns(
                pl.when(
                    iqr_outlier("nuc_Pol-II-S2P.0_Mean", iqr_multiplier=3).over(
                        ["cycle", "ClassCCP"]
                    )
                )
                .then(pl.col("ClassCCP"))
                .otherwise(pl.lit("outlier"))
                .alias("ClassCCP_outliers")
            )
            .with_columns(
                pl.all_horizontal(
                    [
                        iqr_outlier(f, iqr_multiplier=3).over(["cycle"])
                        for f in [
                            "nuc_H2B.2_Mean",
                            "nuc_Nanog.3_Mean",
                            "nuc_H3K27Ac.1_Mean",
                        ]
                    ]
                ).alias("Intensity_outliers")
            )
            .with_columns(
                pl.col("NormalizedCCP")
                .mean()
                .over(["cycle", "ClassCCP"])
                .alias("ClassCCP_NormalizedCCP")
            )
            .with_columns(pl.col("roi").to_physical().cast(pl.String).alias("id.emb"))
        )
        .with_columns(pl.col("celltype_pred").cast(pl.String))
        .filter(pl.all_horizontal(pl.col("nuc_Nanog.3_Mean").is_not_null()))
    )

    df = df.with_columns(pl.Series("Noise", rng.normal(size=len(df))))

    df.write_parquet(Path(path).parent / f"{Path(path).stem}_outliers.parquet")

    df_s = (
        df.filter(pl.col("ClassCCP_outliers").is_in(INTERPHASE_CLASSES))
        .pipe(lambda x: pl.DataFrame(preprocessing_pip.fit_transform(x.to_pandas())))
        .filter(pl.col("Intensity_outliers"))
    )
    return df, df_s.select(list(SHORT_NAMES)).rename(SHORT_NAMES)


@dataclass
class PMCBModel:
    name: str
    features_in: tuple[str, ...] | tuple[tuple[str, ...], ...]
    idata: az.InferenceData = field(repr=False)
    model: pm.Model | None = field(repr=False, default=None)
    bambi_model: bmb.Model | None = field(repr=False, default=None)
    x_asympt: pl.Series | None = field(repr=False, default=None)
    _type: Literal["idata", "bambi", "pymc"] = field(init=False)
    target: str = TARGET
    persist_file: Path | None = None

    def __post_init__(self):
        if self.model is None and self.bambi_model is None:
            self._type = "idata"
        elif self.model is not None:
            if isinstance(self.model, pm.Model):
                self._type = "pymc"
            else:
                if not isinstance(self.model, bmb.Model):
                    raise ValueError(f"Unknown model type: {self.model!r}")
                self._type = "bambi"
                self.bambi_model = self.model
                self.model = self.bambi_model.backend.model
        elif isinstance(self.model, bmb.Model):
            self._type = "bambi"
            self.bambi_model = self.model
            self.model = self.bambi_model.backend.model
        else:
            raise ValueError(f"Unknown model type: {self.model!r}")


def pymc_diagnostics(
    mdl_idata: PMCBModel | az.InferenceData,
    save_path=None,
    overwrite_plots: bool = False,
) -> None:
    # Diagnostics
    if isinstance(mdl_idata, az.InferenceData):
        idata = mdl_idata
        if save_path is not None:
            print("can not save stuff from idata only.")
            save_path = None

    else:
        idata = mdl_idata.idata

    if save_path is not None:
        save_path.mkdir(exist_ok=True)

    # R2 (not really valid!)
    # docs: 'R² for Bayesian regression models. Only valid for linear models.'
    y_true = idata.observed_data[TARGET].values
    y_pred = idata.posterior_predictive.stack(sample=("chain", "draw"))[TARGET].values.T
    r2 = az.r2_score(y_true, y_pred)

    idata.attrs["r2"] = r2["r2"]
    idata.attrs["r2_std"] = r2["r2_std"]

    if save_path is not None:
        base_name = f"{mdl_idata.name}_{mdl_idata._type}"
        summary_path = Path(save_path) / f"{base_name}__summary.json"
        trace_path = save_path / f"{base_name}__trace.png"
        pp_vs_obs_path = save_path / f"{base_name}__pp-vs-obs.png"
        if (
            summary_path.exists()
            and trace_path.exists()
            and pp_vs_obs_path.exists()
            and not overwrite_plots
        ):
            print("Files already exist.")
            return

    title_string_r = f"{r2['r2']:.3f} ({r2['r2_std']:.3f})"
    title_string = f"{mdl_idata.name}: {title_string_r}"

    summary = az.summary(idata, var_names=EXCLUDE_PATTERNS, filter_vars="regex")
    print(summary)
    if save_path is not None:
        pl.DataFrame(summary).write_json(summary_path)

    if "posterior_predictive" in idata:
        az.plot_ppc(idata, num_pp_samples=100)
        ax = plt.gca()
        ax.set_title(title_string)

        if save_path is not None:
            ax.get_figure().savefig(pp_vs_obs_path)

    axs = az.plot_trace(idata, var_names=EXCLUDE_PATTERNS, filter_vars="regex")
    if save_path is not None:
        plt.gca().get_figure().savefig(trace_path)

    return axs


def _pymc_asymtotic_diagnostics(mdl: PMCBModel) -> None:
    # PP Check
    post = mdl.idata.posterior
    x_ = mdl.x_asympt

    mu_pp = post["asymptote"] - (post["asymptote"] - post["intercept"]) * np.exp(
        -post["rate"] * xr.DataArray(x_, dims=[f"{TARGET}_id"])
    )

    fig, ax = plt.subplots()
    # Data
    ax.scatter(
        x_, mdl.idata.observed_data[TARGET], s=0.1, alpha=0.2, label="Data", color="C0"
    )
    # Posterior Mean
    ax.scatter(
        x_,
        mu_pp.mean(("chain", "draw")),
        label="Mean Bayesian",
        color="C1",
        alpha=0.6,
        s=0.1,
        zorder=10,
    )
    # HDI Posterior Mean
    az.plot_hdi(x_, mu_pp, color="C1")
    # HDI Posterior Predictive
    az.plot_hdi(x_, mdl.idata.posterior_predictive[TARGET], color="C0")
    # Name and "R2"
    if "r2" in mdl.idata.attrs:
        r2_string = (
            f": r2 = {mdl.idata.attrs['r2']:.2f} ({mdl.idata.attrs['r2_str']:.2f})"
        )
    else:
        r2_string = ""
    ax.set_title(f"{mdl.name}{r2_string}")
    plt.legend()
    plt.show()


def _run_asymtotic_and_linear_model(
    X: pl.DataFrame,
    y: pl.Series,
    asymt_predictor_name: str = "CCP",
    linear_predictor_names: tuple[str, ...] = (),
    model_name: str = "ccp-asympt",
    censored: tuple[float | None, float | None] | None = None,
    persist_path: Path | None = None,
    load_existing: bool = True,
) -> PMCBModel:
    x_ = X[asymt_predictor_name]
    X_ = X[:, linear_predictor_names]

    if X_.is_empty():
        coords = None
    else:
        coords = {"linear_pred": X_.columns}

    with pm.Model(coords=coords) as model:
        # # Data
        # X_linear = pm.Data("X_linear", value=X_)
        # x_nonlinear = pm.Data("x_nonlinear", value=x_)
        # y_data = pm.Data(y.name, value=y)

        # Prior Error
        sigma = HalfCauchy("sigma", beta=2)

        # Prior Asymtotic Predictor
        asympt = Normal("asymptote", 5, sigma=2)
        interc = Normal("intercept", 0, sigma=1)
        rate = pm.Gamma("rate", alpha=10, beta=2.5)

        # Former Likelihood
        asympt_target = asympt - (asympt - interc) * math.exp(-rate * np.asarray(x_))

        if coords is None:
            mu = asympt_target
        else:
            # Priors Linear Predictors
            beta = pm.Normal("beta", 0.0, 1.0, dims="linear_pred")
            mu = asympt_target + pt.dot(np.asarray(X_), beta)

        if censored is None:
            # Likelihood
            _ = Normal(
                TARGET,
                mu=mu,
                sigma=sigma,
                observed=np.asarray(y),
            )

        else:
            # Latent Likelihood
            target_latent = Normal.dist(
                mu=mu,
                sigma=sigma,
            )

            _ = pm.Censored(
                TARGET,
                target_latent,
                lower=censored[0],
                upper=censored[1],
                observed=np.asarray(y),
            )

        if persist_path is not None:
            persist_file = (
                persist_path
                / f"{model_name}{'' if censored is None else f'__c{censored}'}__idata.nc"
            )
        else:
            persist_file = None

        if persist_file is None:
            idata = pm.sample(
                3000,
                **DEFAULT_FIT_PARAMS,
            )

        else:
            if persist_file.exists() and load_existing:
                print(f"load existing idata: {persist_file}")
                idata = az.from_netcdf(persist_file)
            else:
                idata = pm.sample(
                    3000,
                    **DEFAULT_FIT_PARAMS,
                )
                print(f"persisting idata: {persist_file}")
                idata.to_netcdf(persist_file)
        _ = pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    return PMCBModel(
        name=model_name,
        model=model,
        idata=idata,
        x_asympt=x_,
        features_in=tuple([asymt_predictor_name] + list(linear_predictor_names)),
        persist_file=persist_file,
    )


def _run_power_and_linear_model(
    X: pl.DataFrame,
    y: pl.Series,
    power_predictor_name: str = "CCP",
    linear_predictor_names: tuple[str, ...] = (),
    model_name: str = "ccp-power",
    censored: tuple[float | None, float | None] | None = None,
    persist_path: Path | None = None,
    load_existing: bool = True,
) -> PMCBModel:
    x_ = X[power_predictor_name]
    X_ = X[:, linear_predictor_names]

    if X_.is_empty():
        coords = None
    else:
        coords = {"linear_pred": X_.columns}

    with pm.Model(coords=coords) as model:
        # Prior Error
        sigma = HalfCauchy("sigma", beta=2)

        # Prior Asymtotic Predictor
        a = Normal("a", 0, sigma=2)
        b = Normal("b", 0, sigma=3)

        # Former Likelihood
        asympt_target = a * np.asarray(x_) ** b

        if coords is None:
            mu = asympt_target
        else:
            # Priors Linear Predictors
            beta = pm.Normal("beta", 0.0, 1.0, dims="linear_pred")
            mu = asympt_target + pt.dot(np.asarray(X_), beta)

        if censored is None:
            # Likelihood
            _ = Normal(
                TARGET,
                mu=mu,
                sigma=sigma,
                observed=np.asarray(y),
            )

        else:
            # Latent Likelihood
            target_latent = Normal.dist(
                mu=mu,
                sigma=sigma,
            )

            _ = pm.Censored(
                TARGET,
                target_latent,
                lower=censored[0],
                upper=censored[1],
                observed=np.asarray(y),
            )

        if persist_path is not None:
            persist_file = (
                persist_path
                / f"{model_name}{'' if censored is None else f'__c{censored}'}__idata.nc"
            )
        else:
            persist_file is None

        if persist_file is None:
            idata = pm.sample(
                3000,
                **DEFAULT_FIT_PARAMS,
            )
        else:
            if persist_file.exists() and load_existing:
                print(f"load existing idata: {persist_file}")
                idata = az.from_netcdf(persist_file)
            else:
                idata = pm.sample(
                    3000,
                    **DEFAULT_FIT_PARAMS,
                )
                print(f"persisting idata: {persist_file}")
                idata.to_netcdf(persist_file)

        _ = pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    return PMCBModel(
        name=model_name,
        model=model,
        idata=idata,
        x_asympt=x_,
        features_in=tuple([power_predictor_name] + list(linear_predictor_names)),
        persist_file=persist_file,
    )


def run_pymc_model(
    X: pl.DataFrame,
    y: pl.Series,
    model_name: str,
    censored=None,
    nonlinear_predictor_name="CCP",
    linear_predictor_names=(),
    persist_path=None,
    load_existing=True,
):
    if model_name.endswith("power"):
        return _run_power_and_linear_model(
            X=X,
            y=y,
            power_predictor_name=nonlinear_predictor_name,
            linear_predictor_names=linear_predictor_names,
            model_name=model_name,
            censored=censored,
            persist_path=persist_path,
            load_existing=load_existing,
        )
    elif model_name.endswith("asympt"):
        return _run_asymtotic_and_linear_model(
            X=X,
            y=y,
            asymt_predictor_name=nonlinear_predictor_name,
            linear_predictor_names=linear_predictor_names,
            model_name=model_name,
            censored=censored,
            persist_path=persist_path,
            load_existing=load_existing,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_bambi_model(
    X: pl.DataFrame,
    y: pl.Series,
    model_name: str,
    linear_predictor_names: tuple[str, ...],
    persist_path=None,
    load_existing=True,
):
    model_formula = f"{y.name} ~ {' + '.join(linear_predictor_names)}"
    model_columns = _formula_columns(model_formula)

    df = X.with_columns(y).select(model_columns).to_pandas()

    mdl = bmb.Model(model_formula, data=df)

    if persist_path is not None:
        persist_file = persist_path / f"{model_name}__idata.nc"
    else:
        persist_file is None

    if persist_file is None:
        idata = mdl.fit(
            **{"nuts_sampler": "nutpie", "cores": None, "idata_kwars": IDATA_KWARGS}
        )
        if "__obs__" in idata.posterior.dims:
            idata.posterior = idata.posterior.drop_dims("__obs__")
        if "__obs__" in idata.warmup_posterior.dims:
            idata.warmup_posterior = idata.warmup_posterior.drop_dims("__obs__")
    else:
        if persist_file.exists() and load_existing:
            print(f"load existing idata: {persist_file}")
            idata = az.from_netcdf(persist_file)
            mdl.build()
        else:
            idata = mdl.fit(
                **{"nuts_sampler": "nutpie", "cores": None, "idata_kwars": IDATA_KWARGS}
            )
            if "__obs__" in idata.posterior.dims:
                idata.posterior = idata.posterior.drop_dims("__obs__")
            if "__obs__" in idata.warmup_posterior.dims:
                idata.warmup_posterior = idata.warmup_posterior.drop_dims("__obs__")
            print(f"persisting idata: {persist_file}")
            idata.to_netcdf(persist_file)

    mdl.predict(idata, kind="response")

    return PMCBModel(
        name=model_name,
        idata=idata,
        model=mdl,
        features_in=list(X.columns),
        persist_file=persist_file,
    )


def iknots(n, min_=0, max_=1, scale="log"):
    if scale == "log":
        raw = np.logspace(0, 1, n - 1)
        rescaled = (raw - raw.min()) / np.ptp(raw) * np.ptp([min_, max_]) + min_
        return rescaled[1:-1]
    elif scale == "linear":
        return np.linspace(min_, max_, n - 1)[1:-1]


def fit_model(
    X: pl.DataFrame,
    y: pl.Series,
    model_name: str,
    nonlinear_predictor_names: tuple[str, ...],
    linear_predictor_names: tuple[str, ...],
    n_retry=RETRIES,
):
    for i in range(n_retry):
        try:
            start = time.time_ns()
            if len(nonlinear_predictor_names) == 0:
                model_ = run_bambi_model(
                    X,
                    y,
                    model_name=model_name,
                    linear_predictor_names=linear_predictor_names,
                )
            else:
                if len(nonlinear_predictor_names) != 1:
                    raise ValueError(
                        f"Only single nonlinear predictor allowed, got {len(nonlinear_predictor_names)}"
                    )
                model_ = run_pymc_model(
                    X,
                    y,
                    model_name=model_name,
                    linear_predictor_names=linear_predictor_names,
                    nonlinear_predictor_name=nonlinear_predictor_names[0],
                )
            end = time.time_ns()
            break
        except RuntimeError:
            print(f"attemnt {i} failed...")
    print(f"elapsed time: {(end - start) / 1e9}")
    return model_


def _formula_target_common_group(
    formula: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    import formulae as fm

    if formula is None:
        raise ValueError("need to provide one of `formula` and `model`.")

    model_descr = fm.model_description(formula)

    targets = model_descr.response.var_names
    features = model_descr.var_names - {""} - targets
    group_features = reduce(
        or_, [e.var_names for e in (model_descr).group_terms], set()
    )
    common_features = features - group_features

    return tuple(targets), tuple(common_features), tuple(group_features)


def _formula_columns(
    formula: str | None = None,
) -> tuple[str, ...]:
    return tuple(reduce(add, _formula_target_common_group(formula)))


def _fit_bambi(formula, df, n_retry=RETRIES, fit_kwargs=None):
    if fit_kwargs is None:
        fit_kwargs = DEFAULT_FIT_PARAMS
    else:
        fit_kwargs = {**DEFAULT_FIT_PARAMS, **fit_kwargs}
        print(fit_kwargs)

    target_common_group = _formula_target_common_group(formula)
    required_columns = tuple(reduce(add, target_common_group))

    if isinstance(df, pl.DataFrame):
        df_ = df.select(required_columns).to_pandas()
    else:
        df_ = df[required_columns]

    mdl = bmb.Model(formula, data=df_)
    for i in range(n_retry):
        try:
            start = time.time_ns()
            idata = mdl.fit(**fit_kwargs)
            # idata = pm.compute_log_likelihood(idata, model=mdl.backend.model)
            end = time.time_ns()
            break
        except RuntimeError as e:
            print(f"attemnt {i} failed...")
            print(e)
    print(f"elapsed time: {(end - start) / 1e9}")

    targets = target_common_group[0]
    if len(targets) == 1:
        target = targets[0]
    else:
        import warnings

        warnings.warn(f"Unexpected multi target with ({len(targets)}) columns.")
        target = targets

    # idata.posterior = idata.posterior.drop_vars("mu")
    mdl.predict(idata, kind="response")
    return PMCBModel(
        name=formula,
        idata=idata,
        model=mdl,
        features_in=tuple(reduce(add, target_common_group[1:])),
        target=target,
    )


def plot_frequentist_asymreg(X, y, hue=None, ax=None):
    params, (x_freq, y_freq) = frequentist_asymreg(X["CCP"], y)
    ax = sns.scatterplot(
        x=X["CCP"],
        y=y,
        hue=X["celltype"],
        s=2,
        alpha=0.3,
    )
    plt.plot(x_freq, y_freq, color="k")
    plt.show()
    return ax


def fit_all_models(
    cycles=CYCLES,
    samples=SAMPLES,
    scaler=SCALER,
    base_path: Path | None = OUT_PATH,
    old_scores_fn: str | None = "all.parquet",
    models: dict | None = None,
    fit_models: dict | None = None,
    id_cols: tuple[str, ...] = ("cycle", "model"),
    id_types: tuple[pl.DataType, ...] = (pl.Int64, pl.String),
    feature_path: Path = CCP_QUICKSAVE,
    load_existing_idata: bool = True,
):
    if base_path is None:
        base_path = Path()
    out_path = base_path
    out_path.mkdir(exist_ok=True, parents=True)

    if old_scores_fn is not None:
        old_scores_path = out_path / old_scores_fn
        if old_scores_path.exists():
            old_scores = pl.read_parquet()
        else:
            old_scores = pl.DataFrame(schema=zip(id_cols, id_types))
    else:
        old_scores = pl.DataFrame(schema=zip(id_cols, id_types))

    if models is None:
        models = {}
    if fit_models is None:
        fit_models = {}

    df_, df_s_ = load_data(path=feature_path, scaler=scaler)
    df_s = df_s_

    for cycle in cycles:
        cycle_name = f"cy{cycle:02d}"
        out_path_single = out_path / cycle_name
        out_path_single.mkdir(exist_ok=True)

        print(cycle_name)

        plot_ccp_features(df_, cycle=cycle, output_path=base_path)

        df_cy_s_ = df_s.filter(pl.col("cycle") == cycle)

        if len(df_cy_s_) > samples:
            df_cy_s = df_cy_s_.sample(samples, seed=42)
        else:
            df_cy_s = df_cy_s_

        y = df_cy_s[TARGET]
        X = df_cy_s.drop(TARGET)

        import time

        for model_name, linear_predictor_names in models.items():
            if (cycle, model_name) in fit_models:
                continue
            if (cycle, model_name) in old_scores[["cycle", "model"]].rows():
                continue

            # print(f"Bambi({cycle}): {model_name:<14}", end="")
            print(f"Fit: {model_name:<14} ({cycle})")

            for i in range(RETRIES):
                try:
                    start = time.time_ns()
                    if model_name.endswith("asympt") or model_name.endswith("power"):
                        model_ = run_pymc_model(
                            X,
                            y,
                            model_name=model_name,
                            linear_predictor_names=linear_predictor_names,
                            persist_path=out_path_single,
                            load_existing=load_existing_idata,
                        )
                    else:
                        model_ = run_bambi_model(
                            X,
                            y,
                            model_name=model_name,
                            linear_predictor_names=linear_predictor_names,
                            persist_path=out_path_single,
                            load_existing=load_existing_idata,
                        )
                    end = time.time_ns()
                    break
                except RuntimeError:
                    print(f"attemnt {i} failed...")

            fit_models[(cycle, model_name)] = model_
            print(f"{(end - start) / 1e9:>10.1f} s")
            pymc_diagnostics(model_, save_path=out_path_single)

    return fit_models


def compute_loos(mdls):
    loos = {}
    for (cycle, model_name), mdl in mdls.items():
        out_loo = OUT_PATH / f"cy{cycle:02d}" / f"{model_name}__loo.pickle"
        if out_loo.exists():
            with open(out_loo, "rb") as f_loo:
                loo = pickle.load(f_loo)
        else:
            if "log_likelihood" not in mdl.idata:
                with mdl.model:
                    pm.compute_log_likelihood(mdl.idata)
            loo = az.loo(mdl.idata, pointwise=True)
            loo.to_pickle(out_loo)
        loos[(cycle, model_name)] = loo
    return loos


def compute_comparisions(loos, with_and_without_s5p=True):
    df_all = {}
    df_s5p = {}
    df_no_s5p = {}
    for cycle in np.unique([k[0] for k in loos]):
        loos_cycle = {k: loo for k, loo in loos.items() if k[0] == cycle}
        df_compare_all = az.compare({k[1]: v for k, v in loos_cycle.items()})
        df_all[cycle] = df_compare_all
        if with_and_without_s5p:
            df_compare_s5p = az.compare(
                {k[1]: v for k, v in loos_cycle.items() if "s5p" in k[1]}
            )
            df_s5p[cycle] = df_compare_s5p
            df_compare_no_s5p = az.compare(
                {k[1]: v for k, v in loos_cycle.items() if "s5p" not in k[1]}
            )
            df_no_s5p[cycle] = df_compare_no_s5p
    return df_all, df_s5p, df_no_s5p


# def compute_comparisions(loos):
#     df_all = {}
#     df_s5p = {}
#     df_no_s5p = {}
#     for cycle in np.unique([k[0] for k in loos]):
#         loos_cycle = {k: loo for k, loo in loos.items() if k[0] == cycle}
#         df_compare_all = az.compare({k[1]: v for k, v in loos_cycle.items()})
#         df_all[cycle] = df_compare_all
#         df_compare_s5p = az.compare(
#             {k[1]: v for k, v in loos_cycle.items() if "s5p" in k[1]}
#         )
#         df_s5p[cycle] = df_compare_s5p
#         df_compare_no_s5p = az.compare(
#             {k[1]: v for k, v in loos_cycle.items() if "s5p" not in k[1]}
#         )
#         df_no_s5p[cycle] = df_compare_no_s5p
#     return df_all, df_s5p, df_no_s5p


def linear_predictors_from_beta(
    idata: az.InferenceData,
    coord_name="linear_pred",
    var_name="beta",
    linear_predictors=LINEAR_PREDICTORS,
):
    idata_out = idata.copy()
    if var_name not in idata.posterior:
        idata_out.posterior = idata.posterior[
            [e for e in linear_predictors if e in idata.posterior]
        ]
    else:
        idata_out.posterior = idata.posterior.drop_dims(coord_name).merge(
            idata.posterior[var_name].to_dataset(coord_name)
        )
        idata_out.posterior = idata_out.posterior[
            [e for e in linear_predictors if e in idata_out.posterior]
        ]
    return idata_out


def plot_knots(knots, ax):
    for knot in knots:
        ax.axvline(knot, color="0.1", alpha=0.4)
    return ax


def plot_spline_basis(bambi_model, x: pl.Series, ax=None, subsample=500):
    basis = np.array(
        bambi_model.components["mu"].design.common[
            [e for e in bambi_model.formula.main.split(" ~ ") if "bs(" in e][0]
        ]
    )
    # return basis
    df = (
        pd.DataFrame(basis)
        .assign(**{x.name: np.asarray(x)})
        .melt(x.name, var_name="basis_idx", value_name="value")
    )
    if ax is None:
        ax = plt.gca()

    for idx in df.basis_idx.unique():
        d_ = df[df.basis_idx == idx]
        # return d_
        if subsample and len(d_) > subsample:
            d = d_.sample(subsample).sort_values(x.name)
        else:
            d = d_.sort_values(x.name)
        ax.plot(d[x.name], d["value"])

    return ax


def power_formatter(x, pos):
    "The two args are the value and tick position"
    power_str = "10^{{{x}}}".format(x=x)
    return f"$\mathdefault{'{{{}}}'.format(power_str)}$"
    # return "$\mathdefault{a^b}$"


formatter = FuncFormatter(power_formatter)


def plot_ccp_features(
    df,
    cycle=11,
    feats=("nuc_Pol-II-S5P.2_Mean", "nuc_Pol-II-S2P.0_Mean"),
    output_path=None,
    subsample=10_000,
):
    if output_path is None:
        output_path = Path()
    interphase_color = "#b12405ff"

    FULL_PALETTE = [
        "#0C75B4",
        "#3F3AA9",
        "#743598",
        "#A45585",
        "#C48274",
        "#D1AF68",
        "#CAE261",
        "#A7F073",
        "#77D591",
        "#55BAA2",
    ]

    BG_PALETTE = [
        color if cycle_class in INTERPHASE_CLASSES else "#a3a3a3"
        for cycle_class, color in zip(ALL_CLASSES, FULL_PALETTE)
    ]

    FEATS = feats

    FEAT_LABELS = {
        "nuc_Pol-II-S2P.0_Mean": "Pol-II-S2P $\mathregular{Mean_{nuc}}$ [AU]",
        "nuc_Pol-II-S5P.2_Mean": "Pol-II-S5P $\mathregular{Mean_{nuc}}$ [AU]",
    }
    Y_TICKS = {
        "nuc_Pol-II-S2P.0_Mean": [0, 1, 2, 3],
        "nuc_Pol-II-S5P.2_Mean": [0, 1, 2],
    }

    df_cy10 = df.filter(pl.col("cycle") == cycle)
    if subsample is not None and df_cy10.height > subsample:
        df_cy10 = df_cy10.sample(subsample)
    # df_cy10_s = df_cy10.filter(
    #     pl.col("ClassCCP_outliers").is_in([f"S{i}" for i in range(6)])
    # ).with_columns(pl.col("celltype_pred").cast(pl.String))

    panel_size = (2.0, 1.2)
    n_cols = 1
    n_rows = 2
    fig_size = panel_size[0] * n_cols, panel_size[1] * n_rows

    with plt.style.context(BASE):
        fig, axss = plt.subplots(
            nrows=1 * n_rows,
            ncols=2 * n_cols,
            sharey=True,
            sharex="col",
            width_ratios=[2, 6],
            figsize=fig_size,
            dpi=300,
        )

        feat = FEATS[0]

        axs = axss[0]
        ax = axs[1]
        plt.sca(ax)
        sns.scatterplot(
            x=df_cy10["NormalizedCCP"],
            y=df_cy10[feat].log(10),
            hue=df_cy10["ClassCCP_outliers"],
            hue_order=[
                "M ana",
                "S early",
                *[f"S{i}" for i in range(6)],
                "S late",
                "M meta",
                "outlier",
            ],
            legend=False,
            alpha=0.3,
            s=0.5,
            palette=BG_PALETTE,
        )

        sns.boxplot(
            x=df_cy10["ClassCCP_NormalizedCCP"],
            y=df_cy10[feat].log(10),
            width=0.2,
            fill=False,
            fliersize=0,
            color="0.1",
            native_scale=True,
            linewidth=0.5,
        )

        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["0", "π", "2π"])
        ax.grid(visible=False)
        axs[0].set_ylabel(FEAT_LABELS[feat])
        ax.set_xlabel(f"Cell cycle phase (division cycle {cycle})")

        sns.violinplot(
            x=np.zeros(len(df_cy10)),
            y=df_cy10[feat].log(10),
            width=0.8,
            fill=False,
            ax=axs[0],
            linewidth=0.5,
            color="0.5",
            inner=None,
            cut=0,
            split=False,
        )
        # sns.stripplot(y=df_cy10['nuc_Pol-II-S2P.0_Mean'].log(10), ax=jax.marg_y, color='0.5', size=0.5, alpha=0.2, jitter=0.3)
        sns.boxplot(
            x=np.zeros(len(df_cy10)),
            y=df_cy10[feat].log(10),
            width=0.1,
            fill=False,
            ax=axs[0],
            linewidth=0.5,
            color="0.1",
            zorder=10,
        )

        df_interphase = df_cy10.filter(pl.col("ClassCCP").is_in(INTERPHASE_CLASSES))
        sns.violinplot(
            x=np.ones(len(df_interphase)),
            y=df_interphase[feat].log(10),
            width=0.8,
            fill=False,
            ax=axs[0],
            linewidth=0.5,
            color=interphase_color,
            inner=None,
            cut=0,
            split=False,
        )
        sns.boxplot(
            x=np.ones(len(df_interphase)),
            y=df_interphase[feat].log(10),
            width=0.1,
            fill=False,
            ax=axs[0],
            linewidth=0.5,
            color="0.1",
            zorder=10,
            fliersize=0,
        )
        axs[0].grid(visible=False)
        axs[0].set_xticklabels([])
        axs[0].set_yticks(Y_TICKS[feat])
        axs[0].yaxis.set_major_formatter(formatter)

        feat = FEATS[1]

        axs = axss[1]
        ax = axs[1]
        plt.sca(ax)
        sns.scatterplot(
            x=df_cy10["NormalizedCCP"],
            y=df_cy10[feat].log(10),
            hue=df_cy10["ClassCCP_outliers"],
            hue_order=[
                "M ana",
                "S early",
                *[f"S{i}" for i in range(6)],
                "S late",
                "M meta",
                "outlier",
            ],
            legend=False,
            alpha=0.3,
            s=0.5,
            palette=BG_PALETTE,
        )

        sns.boxplot(
            x=df_cy10["ClassCCP_NormalizedCCP"],
            y=df_cy10[feat].log(10),
            width=0.2,
            fill=False,
            fliersize=0,
            color="0.1",
            native_scale=True,
            linewidth=0.5,
        )

        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["0", "π", "2π"])
        ax.grid(visible=False)
        axs[0].set_ylabel(FEAT_LABELS[feat])
        ax.set_xlabel(f"Cell cycle phase (division cycle {cycle})")

        sns.violinplot(
            x=np.zeros(len(df_cy10)),
            y=df_cy10[feat].log(10),
            width=0.8,
            fill=False,
            ax=axs[0],
            linewidth=0.5,
            color="0.5",
            inner=None,
            cut=0,
            split=False,
        )
        # sns.stripplot(y=df_cy10['nuc_Pol-II-S2P.0_Mean'].log(10), ax=jax.marg_y, color='0.5', size=0.5, alpha=0.2, jitter=0.3)
        sns.boxplot(
            x=np.zeros(len(df_cy10)),
            y=df_cy10[feat].log(10),
            width=0.1,
            fill=False,
            ax=axs[0],
            linewidth=0.5,
            color="0.1",
            zorder=10,
        )

        df_interphase = df_cy10.filter(pl.col("ClassCCP").is_in(INTERPHASE_CLASSES))
        sns.violinplot(
            x=np.ones(len(df_interphase)),
            y=df_interphase[feat].log(10),
            width=0.8,
            fill=False,
            ax=axs[0],
            linewidth=0.5,
            color=interphase_color,
            inner=None,
            cut=0,
            split=False,
        )
        sns.boxplot(
            x=np.ones(len(df_interphase)),
            y=df_interphase[feat].log(10),
            width=0.1,
            fill=False,
            ax=axs[0],
            linewidth=0.5,
            color="0.1",
            zorder=10,
            fliersize=0,
        )
        axs[0].grid(visible=False)
        axs[0].set_xticklabels([])
        axs[0].set_yticks(Y_TICKS[feat])
        axs[0].yaxis.set_major_formatter(formatter)

        # plt.tight_layout(w_pad=0, rect=(0.25, 0.25, 0.9, 0.9))
        plt.tight_layout(w_pad=0)
        # plt.show()

        if output_path is not None:
            fig.savefig(
                Path(output_path)
                / f"pol2_ccp_dynamics_cy{cycle}_f{'_'.join(FEATS)}.png"
            )


if __name__ == "__main__":
    main()
