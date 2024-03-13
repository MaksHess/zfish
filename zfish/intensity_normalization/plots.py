# %%
from functools import reduce
from operator import add, mul

import holoviews as hv
import numpy as np
import plotly.express as px
import plotly.io as pio
import polars as pl
import polars.selectors as cs
from holoviews import dim, opts
from KDEpy import FFTKDE
from numpy.typing import ArrayLike
from scipy.interpolate import interpn

from zfish.features.polars_utils import unnest_all_structs
from zfish.intensity_normalization.models import (
    Exp,
    ExpFitLinear,
    ExpNoOffset,
    Linear,
    LogLinear,
    Model,
    apply_model_to_channel,
    fit_model_to_df,
    fit_model_to_wide_df,
)
from zfish.visualize.plot_utils import data_range

hv.extension("plotly")
hv.output(widget_location="bottom")
pio.templates["gridon"].update( #show grids also for scene (i. e. 3d axes)
    dict(
        layout_scene_xaxis=dict(showgrid=True),
        layout_scene_yaxis=dict(showgrid=True),
        layout_scene_zaxis=dict(showgrid=True),
    )
)
pio.templates.default = "simple_white+gridon"


# %%
def main():
    # %%
    df_raw = pl.read_parquet(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\intensity_normalization\sample_data.parquet"
    )
    df_fit = df_raw.sample(100_000)

    feature_types = {
        "one_step": ("Centroid.z",),
        "two_step": ("MediumPath", "EmbryoPath"),
    }
    # model_types = {'linear': Linear(loss='huber'), 'log_linear': LogLinear(loss="huber"), 'exp': Exp(loss='huber')}
    model_types = {
        "linear": Linear(loss="huber"),
        "log_linear": LogLinear(loss="huber"),
        "exp": Exp(loss="huber"),
    }

    models = {}
    for feature_type, features in feature_types.items():
        for i, (model_type, model) in enumerate(model_types.items()):
            model_name = f"{feature_type}__{model_type}"
            model_fit = fit_model_to_wide_df(df_fit, model, features)
            models[model_name] = model_fit

    figure_model_comp = plot_models(models, df_fit)


# %%
def plot_models(
    models: dict[str, Model],
    df: pl.DataFrame,
    target_column: str,  # 'Mean', 'DAPI.1_Mean' make sure to filter dataframe if tall!
    meta_columns: tuple[str, ...] = tuple(),
    n_samples: int | None = 10_000,
):
    """Visualize Illumination correction models side by side. Attention, there's a bug somwhere in the
    interaction of holoviews and plotly, interactive panel (Curve) buggy."""

    LINE_COLORS = [
        "#001c7f",
        "#b1400d",
        "#12711c",
        "#8c0800",
        "#591e71",
        "#592f0d",
        "#a23582",
        "#3c3c3c",
        "#b8850a",
        "#006374",
    ]

    SURFACE_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    BACKGROUND_SCATTER_COLOR = "#949494"

    # check all models are fitted
    assert all(
        mdl._feature_names is not None for mdl in models.values()
    ), "At least one of the models provided is not fitted."

    one_step_models = {
        name: mdl for name, mdl in models.items() if len(mdl._feature_names) == 1
    }
    two_step_models = {
        name: mdl for name, mdl in models.items() if len(mdl._feature_names) == 2
    }

    one_step_columns: tuple[str] = tuple(
        list(one_step_models.values())[0]._feature_names
    )
    two_step_columns: tuple[str, str] = tuple(
        list(two_step_models.values())[0]._feature_names
    )

    # check all models have consistent feature names
    assert all(
        tuple(mdl._feature_names) == one_step_columns
        for mdl in one_step_models.values()
    ), "Inconsistent feature names in one-step model"
    assert all(
        tuple(mdl._feature_names) == two_step_columns
        for mdl in two_step_models.values()
    ), "Inconsistent feature names in two-step model"

    model_colors = {
        **{model_name: LINE_COLORS[i] for i, model_name in enumerate(one_step_models)},
        **{
            model_name: SURFACE_COLORS[i]
            for i, model_name in enumerate(two_step_models)
        },
    }
    plot_columns = meta_columns + one_step_columns + two_step_columns + (target_column,)
    if n_samples is None:
        df_plot = df.select(plot_columns)
    else:
        df_plot = df.select(plot_columns).sample(n_samples)

    one_step_samples = df_plot.select(one_step_columns).to_numpy()
    two_step_samples = df_plot.select(two_step_columns).to_numpy()
    target = df_plot[target_column].to_numpy()
    one_step_grid = np.linspace(
        one_step_samples.min(axis=0), one_step_samples.max(axis=0)
    )
    two_step_grid_x, two_step_grid_y = np.linspace(
        two_step_samples.min(axis=0), two_step_samples.max(axis=0)
    ).T

    one_step_models = {}
    one_step_models_corr = {}
    two_step_models = {}
    two_step_models_corr = {}

    for model_name, model in models.items():
        assert model._feature_names is not None, "Model is not fitted."
        feature_type, model_type = model_name.split("__")

        n_features = len(model._feature_names)

        if feature_type == "one_step":
            X = one_step_samples
            y = target
            y_corr = model.correct(X, y)
            df_plot = df_plot.with_columns(pl.Series(model_name, y_corr))

            y_grid = model.predict(one_step_grid)
            y_grid_corr = model.correct(one_step_grid, y_grid)

            one_step_models[model_name] = hv.Curve(
                np.concatenate([one_step_grid, y_grid.reshape(-1, 1)], axis=1),
                *one_step_columns,
                target_column,
                # group='Visible',
                # label=model_colors[model_name]
            ).opts(color=model_colors[model_name])
            one_step_models[f"two_step__{model_type}"] = hv.Curve(
                # np.concatenate([one_step_grid[:3], y_grid.reshape(-1, 1)[:3]], axis=1),
                [[0, 0], [1, 1]],
                *one_step_columns,
                target_column,
                # group='Invisible',
                # label=BACKGROUND_SCATTER_COLOR
            ).opts(color=BACKGROUND_SCATTER_COLOR)

            one_step_models_corr[model_name] = hv.Curve(
                np.concatenate([one_step_grid, y_grid_corr.reshape(-1, 1)], axis=1),
                # [[0, 0], [0, 0]],
                *one_step_columns,
                target_column,
                # group='Visible',
            ).opts(color=model_colors[model_name])
            one_step_models_corr[f"two_step__{model_type}"] = hv.Curve(
                # np.concatenate([one_step_grid, y_grid_corr.reshape(-1, 1)], axis=1),
                [[0, 0], [1, 1]],
                *one_step_columns,
                target_column,
                # group='Invisible',
            ).opts(color=BACKGROUND_SCATTER_COLOR)

        else:
            X = two_step_samples
            y = target
            y_corr = model.correct(X, y)
            df_plot = df_plot.with_columns(pl.Series(model_name, y_corr))

            two_step_grid_xy = np.meshgrid(two_step_grid_x, two_step_grid_y)
            two_step_grid_z = model.grid_predict(two_step_grid_xy)
            two_step_grid_z_corr = model.grid_correct(two_step_grid_xy, two_step_grid_z)

            two_step_models[model_name] = hv.Surface(
                (two_step_grid_x, two_step_grid_y, two_step_grid_z),
                kdims=list(two_step_columns),
                vdims=target_column,
            ).opts(
                visible=True,
                cmap=[model_colors[model_name], model_colors[model_name]],
            )
            two_step_models[f"one_step__{model_type}"] = hv.Surface(
                (two_step_grid_x, two_step_grid_y, two_step_grid_z_corr),
                kdims=list(two_step_columns),
                vdims=target_column,
            ).opts(
                visible=False, cmap=[BACKGROUND_SCATTER_COLOR, BACKGROUND_SCATTER_COLOR]
            )

            two_step_models_corr[model_name] = hv.Surface(
                (two_step_grid_x, two_step_grid_y, two_step_grid_z),
                kdims=list(two_step_columns),
                vdims=target_column,
            ).opts(
                visible=True,
                cmap=[model_colors[model_name], model_colors[model_name]],
            )
            two_step_models_corr[f"one_step__{model_type}"] = hv.Surface(
                (two_step_grid_x, two_step_grid_y, two_step_grid_z_corr),
                kdims=list(two_step_columns),
                vdims=target_column,
            ).opts(
                visible=False, cmap=[BACKGROUND_SCATTER_COLOR, BACKGROUND_SCATTER_COLOR]
            )

    full_dim = hv.Dimension(
        "Model", values=list(models.keys()), default=list(models.keys())[0]
    )

    # 2D Scatterplots corrected
    dset_scatter = hv.Dataset(
        df_plot.to_pandas(), kdims=[*one_step_columns, *two_step_columns]
    )
    df_scatter = df_plot.to_pandas()
    # return df_scatter
    ylim = data_range(
        np.asarray(
            dset_scatter.data.drop([*one_step_columns, *two_step_columns], axis=1)
        ).flatten(),
        interval_range_perc=0.99,
        interval_range_padding=0.2,
    )

    # scatter_corrs = {
    #     name: hv.Scatter(dset_scatter, one_step_columns[0], name).opts(
    #         color=model_colors[name]
    #     )
    #     for name in models.keys()
    # }
    scatter_corrs = {
        name: hv.Scatter(
            np.asarray(df_scatter[[one_step_columns[0], name]]),
            one_step_columns[0],
            target_column,
        ).opts(color=model_colors[name])
        for name in models.keys()
    }

    ###
    scatter_corr_map_2d = hv.HoloMap(
        scatter_corrs,
        kdims=[full_dim],
    ).opts(opts.Scatter(ylim=ylim))
    # 2D Scatter background
    bg_scatter_raw = hv.Scatter(dset_scatter, one_step_columns[0], target_column)

    one_step_models_map = hv.HoloMap(
        {k: v for k, v in one_step_models.items()}, kdims=[full_dim]
    )

    ###
    scatter_raw_map_2d = bg_scatter_raw.opts(
        opts.Scatter(color=BACKGROUND_SCATTER_COLOR, alpha=0.7)
    ) * one_step_models_map.opts(opts.Curve(line_width=3, show_legend=False, ylim=ylim))

    # 3D Scatterplots corrected
    scatter_corrs_3d = {
        name: hv.Scatter3D(
            dset_scatter, kdims=[*two_step_columns, name], vdims=[]
        ).opts(color=model_colors[name])
        for name in models.keys()
    }
    ###
    scatter_corr_map_3d = hv.HoloMap(
        scatter_corrs_3d,
        kdims=[full_dim],
    ).opts(opts.Scatter3D(size=2, zlim=ylim))

    # 3D Models & scatter background
    bg_scatter_raw_3d = hv.Scatter3D(
        dset_scatter, kdims=[*two_step_columns, target_column], vdims=[]
    )
    two_step_models_map = hv.HoloMap(two_step_models, kdims=[full_dim])

    ###
    scatter_raw_map_3d = bg_scatter_raw_3d.opts(
        opts.Scatter3D(size=2, color=BACKGROUND_SCATTER_COLOR)
    ) * two_step_models_map.opts(opts.Surface(zlim=ylim))

    # Distribution plots

    distribution_raw = hv.Distribution(
        dset_scatter.data.clip(*ylim), target_column, label="no correction"
    ).opts(color=BACKGROUND_SCATTER_COLOR)

    distributions = {
        name: hv.Distribution(
            np.asarray(dset_scatter.data[name].clip(*ylim)), target_column
        ).opts(color=model_colors[name])
        for name in models.keys()
    }

    distributions_hmap = hv.HoloMap(distributions, kdims=[full_dim], label="corrected")

    distributions_ndoverlay = hv.NdOverlay(distributions_hmap)

    distributions_hmap_with_background = distribution_raw * distributions_hmap

    distributions_all_with_legend = distribution_raw * distributions_ndoverlay.opts(
        show_legend=True
    )

    figure = (
        hv.Layout(
            [
                scatter_raw_map_3d,
                scatter_raw_map_2d,
                distributions_all_with_legend,
                scatter_corr_map_3d,
                scatter_corr_map_2d,
                distributions_hmap_with_background,
            ]
        )
        .cols(3)
        .opts(
            opts.Scatter3D(zlim=ylim, size=1, alpha=0.5, width=500, height=350),
            opts.Scatter(ylim=ylim, size=2, alpha=0.5, width=500, height=350),
            opts.Distribution(
                invert_axes=True, yaxis="bare", cut=0, width=150, height=350
            ),
        )
    )
    return figure


def zero_one_normalize(X):
    return (X - X.min(axis=0)) / np.ptp(X, axis=0)


def estimate_density(
    X: ArrayLike,
    n_samples: int | tuple[int, ...] | None = None,
    bw=0.1,
    data_range_params: tuple[tuple[float, float] | None, ...] | None = None,
):
    DEFAULT_RANGE = (1.0, 0.05)
    X_ = np.asarray(X)
    if n_samples is None:
        if X_.shape[1] > 2:
            n_samples = 128
        else:
            n_samples = 1024
    if isinstance(n_samples, int):
        samples_tuple = (n_samples,) * X_.shape[1]
    else:
        samples_tuple = tuple(n_samples)
    if data_range_params is None:
        normalization_ranges = tuple(
            data_range(X_[:, i], DEFAULT_RANGE[0], DEFAULT_RANGE[1])
            for i in range(X_.shape[1])
        )
    else:
        range_params = [DEFAULT_RANGE if p is None else p for p in data_range_params]
        normalization_ranges = tuple(
            data_range(X_[:, i], params[0], params[1])
            for i, params in enumerate(range_params)
        )

    X_scl = zero_one_normalize(X_.clip(*np.array(normalization_ranges).T))
    kde = FFTKDE(bw=bw).fit(np.asarray(X_scl))
    grid, values = kde.evaluate(n_samples)
    points = []
    for minimum, maximum, n in zip(grid.min(axis=0), grid.max(axis=0), samples_tuple):
        points.append(np.linspace(minimum, maximum, n))
    dd = interpn(points, values.reshape((n_samples,) * X_scl.shape[1]), X_scl)
    return dd


def plot_model_(
    mdl: Model,
    df: pl.DataFrame,
    color: str | None = None,
    scatter_alpha: float = 0.2,
    surface_alpha: float = 0.9,
    samples: str | None = None,
    two_step_names: tuple[str, ...] = (
        "MediumPath",
        "EmbryoPath",
    ),
    single_step_names: tuple[str, ...] = ("Centroid.z",),
):
    DISTRIBUTION_OFF_COLOR = "rgb(150, 150, 150)"
    DISTRIBUTION_ON_COLOR = "#001489"
    SURFACE_COLOR = "rgb(45, 41, 38)"
    dark_colors = px.colors.qualitative.Set1
    if mdl._params is None or mdl._feature_names is None or mdl._target_name is None:
        raise RuntimeError("Model is not fitted!")

    n_features = len(mdl._feature_names)

    if n_features == 1:
        single_step_names = tuple(mdl._feature_names)
    elif n_features == 2:
        two_step_names = tuple(mdl._feature_names)
    else:
        raise ValueError(
            f"Invalid number of features ({n_features}), should be 1 or 2."
        )

    plot_columns = (*two_step_names, *single_step_names, mdl._target_name)
    if color is not None:
        plot_columns += (color,)
        color_2d = color
        color_3d = color
        color_2d_corr = color
        color_3d_corr = color

    if samples is None:
        df_plot = df.select(plot_columns).to_pandas()
    else:
        df_plot = df.select(plot_columns).sample(samples).to_pandas()

    X = df_plot.loc[:, mdl._feature_names]
    y = df_plot[mdl._target_name]
    y_corr = mdl.correct(X, y)

    if n_features == 2:
        dx, dy = np.linspace(np.zeros(n_features), X.max(axis=0)).T
        XX = np.meshgrid(dx, dy)
        yy = mdl.grid_predict(XX)
        yy_corr = mdl.grid_correct(XX, yy)
    else:
        dx = np.linspace([0], [X.max()])
        yy = mdl.predict(dx)
        yy_corr = mdl.correct(dx, yy)

    df_plot = df_plot.assign(**{f"{mdl._target_name}_corr": y_corr})

    if color is None:
        density_2d = estimate_density(
            np.asarray(df_plot.loc[:, [*single_step_names, mdl._target_name]]),
            data_range_params=(None, (0.99, 0.2)),
        )
        density_3d = estimate_density(
            np.asarray(df_plot.loc[:, [*two_step_names, mdl._target_name]]),
            data_range_params=(None, None, (0.99, 0.2)),
        )
        density_2d_corr = estimate_density(
            np.asarray(df_plot.loc[:, [*single_step_names]].assign(y_corr=y_corr)),
            data_range_params=(None, (0.99, 0.2)),
        )
        density_3d_corr = estimate_density(
            np.asarray(df_plot.loc[:, [*two_step_names]].assign(y_corr=y_corr)),
            data_range_params=(None, None, (0.99, 0.2)),
        )
        df_plot = df_plot.assign(
            density_2d=density_2d,
            density_3d=density_3d,
            density_2d_corr=density_2d_corr,
            density_3d_corr=density_3d_corr,
        )
        color_2d = "density_2d"
        color_3d = "density_3d"
        color_2d_corr = "density_2d_corr"
        color_3d_corr = "density_3d_corr"

    plot_data = hv.Dataset(df_plot)
    kdim = hv.Dimension("Intensity", default="raw", values=["raw", "corr"])

    # 3D Scatterplots
    scatter_raw = hv.Scatter3D(
        plot_data, kdims=[*two_step_names, y.name], vdims=color_3d
    ).opts(size=2, alpha=scatter_alpha, color=color_3d)
    scatter_corr = hv.Scatter3D(
        plot_data,
        kdims=[*two_step_names, f"{mdl._target_name}_corr"],
        vdims=color_3d_corr,
    ).opts(size=2, alpha=scatter_alpha, color=color_3d_corr)

    if n_features == 2:
        surface_raw = hv.Surface((dx, dy, yy)).opts(
            alpha=surface_alpha,
            xlabel=two_step_names[0],
            ylabel=two_step_names[1],
            zlabel=y.name,
            cmap=[SURFACE_COLOR, SURFACE_COLOR],
        )
        surface_corr = hv.Surface((dx, dy, yy_corr)).opts(
            alpha=surface_alpha,
            xlabel=two_step_names[0],
            ylabel=two_step_names[1],
            zlabel=y.name,
            cmap=[SURFACE_COLOR, SURFACE_COLOR],
        )

        overlay_raw = surface_raw * scatter_raw
        overlay_corr = surface_corr * scatter_corr
    else:
        overlay_raw = scatter_raw
        overlay_corr = scatter_corr

    holomap_3d = hv.HoloMap({"raw": overlay_raw, "corr": overlay_corr}, kdims=[kdim])

    # 2D Scatterplots
    scatter_2d_raw = hv.Points(
        plot_data, kdims=[*single_step_names, f"{mdl._target_name}"], vdims=color_2d
    ).opts(
        size=2,
        color=color_2d,
    )
    scatter_2d_corr = hv.Points(
        plot_data,
        kdims=[*single_step_names, f"{mdl._target_name}_corr"],
        vdims=color_2d_corr,
    ).opts(
        size=2,
        color=color_2d_corr,
    )

    if n_features == 2:
        overlay_2d_raw = scatter_2d_raw
        overlay_2d_corr = scatter_2d_corr
    else:
        line_raw = hv.Curve((dx, yy.flatten())).opts(
            xlabel=single_step_names[0], ylabel=y.name, color=SURFACE_COLOR
        )
        line_corr = hv.Curve((dx, yy_corr.flatten())).opts(
            xlabel=single_step_names[0],
            ylabel=y.name,
            color=SURFACE_COLOR,
        )

        overlay_2d_raw = scatter_2d_raw * line_raw
        overlay_2d_corr = scatter_2d_corr * line_corr

    holomap_2d = hv.HoloMap(
        {"raw": overlay_2d_raw, "corr": overlay_2d_corr}, kdims=[kdim]
    )

    if color is None:
        holomap_3d = holomap_3d.opts(
            opts.Scatter3D(cmap="viridis"),
        )
        holomap_2d = holomap_2d.opts(opts.Points(cmap="viridis"))
    else:
        color_order = np.unique(plot_data[color])
        category_map = dict(zip(color_order, range(len(color_order))))

        holomap_3d = holomap_3d.opts(
            opts.Scatter3D(
                color=dim(color).categorize(category_map),
                cmap=dark_colors[: len(category_map)],
            ),
            opts.Surface(cmap=[SURFACE_COLOR, SURFACE_COLOR]),
        )

        holomap_2d = holomap_2d.opts(
            opts.Points(
                color=dim(color).categorize(category_map),
                cmap=dark_colors[: len(category_map)],
            )
        )

    y_range = data_range(
        np.concatenate([y, y_corr]), scale="linear", interval_range_padding=0.2
    )
    y_range_log = (max(y_range[0], y.min()), y_range[1])

    distribution_raw = hv.Distribution(
        np.asarray(y_corr).clip(*y_range_log), "DAPI.1_Mean"
    ).opts(color=DISTRIBUTION_OFF_COLOR) * hv.Distribution(
        np.asarray(y).clip(*y_range_log), "DAPI.1_Mean"
    ).opts(
        color=DISTRIBUTION_ON_COLOR
    )

    distribution_corr = hv.Distribution(
        np.asarray(y).clip(*y_range_log), "DAPI.1_Mean_corr"
    ).opts(color=DISTRIBUTION_OFF_COLOR) * hv.Distribution(
        np.asarray(y_corr).clip(*y_range_log), "DAPI.1_Mean_corr"
    ).opts(
        color=DISTRIBUTION_ON_COLOR
    )

    holomap_distribution = hv.HoloMap(
        {
            "raw": distribution_raw,
            "corr": distribution_corr,
        },
        kdims=[kdim],
    ).opts(
        opts.Distribution(invert_axes=True, yaxis="bare", cut=0, width=100, height=400)
    )
    figure = holomap_3d + holomap_2d + holomap_distribution
    return figure.opts(opts.Points(ylim=y_range), opts.Scatter3D(zlim=y_range)).opts(
        title=str(mdl)
    )


# %%
if __name__ == "__main__":
    main()

# %%
