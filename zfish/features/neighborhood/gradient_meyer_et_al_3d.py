# %%
import numpy as np
import polars as pl
from scipy import linalg
from scipy.interpolate import RegularGridInterpolator


def gradient_meyer_et_al(sigma, V):
    """Compute gradient from difference to neighbors as in https://link.springer.com/content/pdf/10.1023/A:1011026732182.pdf
    (Meyer et al. 2001) for points p_1..p_m neighboring point p.

    Parameters
    ----------
    sigma : np.array m x 1
        ΔS_i / |v_i| for i 1..m
        where ΔS_i is the (signed) difference in S(p_i) - S(p)
        and |v_i| is the length of the vector pointing form p to p_i

    V : np.array m x ndim
        unit vector components pointing from p to p_i for i 1..m
        e. g. for the 2D case:
        np.array([
            [vhat_{x, 1}, vhat_{y, 1}],
            [vhat_{x, 2}, vhat_{y, 2}],
            ...
            [vhat_{x, m}, vhat_{y, m}],
        ])

    Returns
    -------
    np.array ndim x 1
        gradient approximations.
    """
    return np.linalg.inv(V.T @ V) @ V.T @ sigma


def circdiff(x, y):
    a = (np.atleast_1d(x) - np.atleast_1d(y)) % (2 * np.pi)
    b = (np.atleast_1d(y) - np.atleast_1d(x)) % (2 * np.pi)
    return np.where(a < b, -a, b).squeeze()


def gradient_at_index(p_idx, points_blue, adj_blue, S_blue):
    pn_idxs = np.where(adj_blue[p_idx])

    delta_S = (S_blue[pn_idxs] - S_blue[p_idx]).reshape(-1, 1)

    V_full = points_blue[pn_idxs] - points_blue[p_idx]
    V_norm = np.linalg.norm(V_full, axis=1, keepdims=True)
    V = V_full / V_norm
    sigma = delta_S / V_norm
    return gradient_meyer_et_al(sigma, V)


def gradients(points, adj, S):
    grad = []
    for i in range(points.shape[0]):
        grad.append(gradient_at_index(i, points, adj, S))
    return np.array(grad).squeeze()


def circular_gradient_at_index(p_idx, points_blue, adj_blue, S_blue):
    pn_idxs = np.where(adj_blue[p_idx])

    # delta_S = circdiff(S_blue[pn_idxs] - S_blue[p_idx]).reshape(-1, 1)
    delta_S = circdiff(S_blue[p_idx], S_blue[pn_idxs]).reshape(-1, 1)
    # delta_S = circdiff(S_blue[pn_idxs], S_blue[p_idx]).reshape(-1, 1)

    V_full = points_blue[pn_idxs] - points_blue[p_idx]
    V_norm = np.linalg.norm(V_full, axis=1, keepdims=True)
    V = V_full / V_norm
    sigma = delta_S / V_norm
    return gradient_meyer_et_al(sigma, V)


def circular_gradients(points, adj, S):
    grad = []
    for i in range(points.shape[0]):
        grad.append(circular_gradient_at_index(i, points, adj, S))
    return np.array(grad).squeeze()


def interpolate_field_with_discontinuity(
    points, noise_level=0.5, apply_mod=True, return_true_gradient=True
):
    rng = np.random.default_rng(42)

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    z = np.linspace(0, 1, 100)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    noise = rng.normal(size=xx.shape) * noise_level

    f = 1.5
    SS = (2 * xx**2 + np.sin(f * 2 * np.pi * yy) + zz) + noise
    if apply_mod:
        SS = SS % (2 * np.pi)

    SS_interp = RegularGridInterpolator((x, y, z), SS)

    n = points.shape[0]
    S = SS_interp(points)

    if return_true_gradient:
        dx, dy, dz = sympy_differentiate_expression()
        dxx = dx(xx)
        dyy = dy(yy)
        # dzz = dz(zz)
        S_gradx = RegularGridInterpolator((x, y, z), dxx)(points)
        S_grady = RegularGridInterpolator((x, y, z), dyy)(points)
        S_gradz = RegularGridInterpolator((x, y, z), np.ones_like(dxx))(points)

        return S, np.stack([S_gradz, S_grady, S_gradx]).T
    else:
        return S


def sympy_differentiate_expression():
    import sympy

    x, y, z = sympy.symbols("x y z")
    expr = 2 * x**2 + sympy.sin(1.5 * 2 * np.pi * y) + z
    dx = expr.diff(x)
    dy = expr.diff(y)
    dz = expr.diff(z)

    return [sympy.lambdify(e, de) for e, de in zip([x, y, z], [dx, dy, dz])]


def get_points(upper_limit=(1.0, 1.0, 1.0), radius=0.08):
    import poisson_disc as po

    points = po.Bridson_sampling(dims=np.array(upper_limit), radius=radius)
    return points


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector, axis=1, keepdims=True)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    return np.arccos(
        np.clip(np.sum(unit_vector(v1) * unit_vector(v2), axis=1), -1.0, 1.0)
    )


def length_difference(v1, v2):
    return np.linalg.norm(v1, axis=1) - np.linalg.norm(v2, axis=1)


def compute_errors(grad_true, grad):
    angle_error = angle_between(grad_true, grad)
    length_error = np.abs(length_difference(grad_true, grad))
    length_error_perc = length_error / np.linalg.norm(grad_true, axis=1)
    abs_diff = np.linalg.norm(grad_true - grad, axis=1)
    cosine = np.sum(grad_true * grad, axis=1) / (
        linalg.norm(grad_true, axis=1) * linalg.norm(grad, axis=1)
    )
    return pl.concat(
        [
            pl.DataFrame(grad_true, ["grad_true.x", "grad_true.y", "grad_true.z"]),
            pl.DataFrame(grad, ["grad.x", "grad.y", "grad.z"]),
            pl.DataFrame(
                {
                    "angle_error": angle_error,
                    "length_error": length_error,
                    "length_error_perc": length_error_perc,
                    "abs_diff": abs_diff,
                    "cosine": cosine,
                }
            ),
        ],
        how="horizontal",
    )


def main():
    """Run the gradient computation on randomly sampled points in a 3D space."""
    import napari
    import polars as pl

    from zfish.features.neighborhood import aggregation_functions as agg_funcs
    from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject
    from zfish.visualize.napari_utils import napari_gradients

    points = get_points()

    NOISE_LEVELS = [0.0, 0.1, 0.2, 0.4]
    PRE_SMOOTHING = {"radius": [0.1, 0.15, 0.2], "knn": [17, 50, 125], "delaunay": [1]}
    AGG_FUNC = agg_funcs.CircMean

    GRADIENT_NEIGHBORHOODS = {
        "radius": [0.14, 0.16, 0.18, 0.2, 0.25],
        "knn": [17, 29, 47, 69, 95, 125, 223],
        "delaunay": [1, 2, 3],
    }

    results = {}
    results["grad"] = {}
    results["circ"] = {}
    results["ref"] = {}

    nqo = NeighborhoodQueryObject.from_numpy(points)
    for noise_level in NOISE_LEVELS:
        print(f"starting level {noise_level}...")
        S, grad_true = interpolate_field_with_discontinuity(
            points, noise_level=noise_level, apply_mod=True, return_true_gradient=True
        )

        S_clean = interpolate_field_with_discontinuity(
            points,
            noise_level=noise_level,
            apply_mod=False,
            return_true_gradient=False,
        )

        nqo_smoothing = nqo
        for k, v in PRE_SMOOTHING.items():
            nqo_smoothing = getattr(nqo_smoothing, k)(v)

        nqo_grad = nqo
        for k, v in GRADIENT_NEIGHBORHOODS.items():
            nqo_grad = getattr(nqo_grad, k)(v)

        for q_grad, nhd_grad in nqo_grad.query_neighborhoods:
            S_smooth = nqo_smoothing.aggregate(AGG_FUNC, pl.DataFrame({"S": S}))
            for s_smooth in [None] + S_smooth.columns:
                if s_smooth is None:
                    current_s = S
                else:
                    current_s = np.asarray(
                        S_smooth.select(pl.col(s_smooth).fill_nan(0))
                    )
                nhd_adj = nhd_grad.todense()
                grad = gradients(points, nhd_adj, current_s)
                grad_circ = circular_gradients(points, nhd_adj, current_s)
                grad_ref = gradients(
                    points,
                    nhd_adj,
                    S_clean,
                )
                results["grad"][(noise_level, s_smooth, str(q_grad))] = compute_errors(
                    grad_true, grad
                )
                results["circ"][(noise_level, s_smooth, str(q_grad))] = compute_errors(
                    grad_true, grad_circ
                )
                results["ref"][(noise_level, s_smooth, str(q_grad))] = compute_errors(
                    grad_true, grad_ref
                )

    all_ = []
    for grad, r in results.items():
        for k, df in r.items():
            all_.append(
                df.with_columns(
                    pl.lit(e).alias(e_name)
                    for e, e_name in zip(k, ["noise_level", "pre_smooth", "grad_nhd"])
                ).with_columns(
                    pl.lit(grad).alias("strategy"), pl.col("pre_smooth").cast(pl.String)
                )
            )

    viewer = napari.Viewer()
    viewer.add_image(np.zeros((10, 10, 10)), scale=(0.1, 0.1, 0.1))
    viewer.add_points(
        **{
            "data": points,
            "name": f"noise_{noise_level}",
            "features": {"s": S, "sin(s)": np.sin(S), "cos(s)": np.cos(S)},
            "face_color": "sin(s)",
            "edge_width": 0.0,
            "size": 0.01,
            # "face_contrast_limits": (0, 2 * np.pi),
        }
    )

    viewer.add_vectors(
        **napari_gradients(
            pl.DataFrame(
                np.concatenate([points, grad, grad_circ], axis=1),
                schema=[
                    "Centroid-z",
                    "Centroid-y",
                    "Centroid-x",
                    "Gradient-z",
                    "Gradient-y",
                    "Gradient-x",
                    "CircGradient-z",
                    "CircGradient-y",
                    "CircGradient-x",
                ],
            ),
            gradient_column="^Gradient.*$",
            length=0.05,
            edge_width=0.01,
            name="Gradient",
            edge_contrast_limits=(0, 12),
        ),
    )
    viewer.add_vectors(
        **napari_gradients(
            pl.DataFrame(
                np.concatenate([points, grad, grad_circ], axis=1),
                schema=[
                    "Centroid-z",
                    "Centroid-y",
                    "Centroid-x",
                    "Gradient-z",
                    "Gradient-y",
                    "Gradient-x",
                    "CircGradient-z",
                    "CircGradient-y",
                    "CircGradient-x",
                ],
            ),
            gradient_column="^CircGradient.*$",
            length=0.05,
            edge_width=0.01,
            edge_contrast_limits=(0, 12),
            name="CircGradient",
        )
    )

    viewer.add_vectors(
        **napari_gradients(
            pl.DataFrame(
                np.concatenate([points, grad_true], axis=1),
                schema=[
                    "Centroid-z",
                    "Centroid-y",
                    "Centroid-x",
                    "TrueGradient-z",
                    "TrueGradient-y",
                    "TrueGradient-x",
                    # "CircGradient-z",
                    # "CircGradient-y",
                    # "CircGradient-x",
                ],
            ),
            gradient_column="^TrueGradient.*$",
            length=0.05,
            edge_width=0.01,
            edge_contrast_limits=(0, 12),
            name="TrueGradient",
        )
    )


# viewer.add_vectors(data=bbxs, length=1, vector_style='line', edge_width=32.5)
# viewer.add_vectors(
#     **napari_gradients(
#         df_all.with_columns(
#             pl.DataFrame(grad * -1, orient="row", schema=["grad.z", "grad.y", "grad.x"])
#         ),
#         length=30,
#         edge_width=4,
#         translate_group="roi",
#         translate_sort="nuc__Count",
#         translate_n_rows=10,
#     )
# )

# %%
# from zfish.image.coordinates import get_column_grid


# # %%

# # %%
# nqo = NeighborhoodQueryObject.from_numpy(points)

# (
#     adj_del,
#     adj_knn5,
#     adj_knn8,
#     adj_rad20,
#     adj_rad30,
# ) = list(nqo.delaunay().knn(k=5).knn(k=8).radius([20, 30]).neighborhoods)

# # %%
# grad_del = gradients(points, adj_del.todense(), S)
# grad_knn5 = gradients(points, adj_knn5.todense(), S)
# grad_knn8 = gradients(points, adj_knn8.todense(), S)
# grad_rad20 = gradients(points, adj_rad20.todense(), S)
# grad_rad30 = gradients(points, adj_rad30.todense(), S)

# gradx_true = SSdx_interp(points)
# grady_true = SSdy_interp(points)
# gradz_true = SSdz_interp(points)

# grad_true = np.vstack([gradx_true, grady_true, gradz_true]).T


# # %%
# errors_del = compute_errors(grad_true, grad_del).with_columns(
#     pl.DataFrame(points, ["points.x", "points.y", "points.z"])
# )
# errors_knn5 = compute_errors(grad_true, grad_knn5).with_columns(
#     pl.DataFrame(points, ["points.x", "points.y", "points.z"])
# )
# errors_knn8 = compute_errors(grad_true, grad_knn8).with_columns(
#     pl.DataFrame(points, ["points.x", "points.y", "points.z"])
# )
# errors_rad20 = compute_errors(grad_true, grad_rad20).with_columns(
#     pl.DataFrame(points, ["points.x", "points.y", "points.z"])
# )
# errors_rad30 = compute_errors(grad_true, grad_rad30).with_columns(
#     pl.DataFrame(points, ["points.x", "points.y", "points.z"])
# )

# # %%
# import polars.selectors as cs

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# for i, errors in enumerate([errors_del, errors_knn5, errors_knn8]):
#     q = ax.quiver(
#         *errors.select(cs.starts_with("points")).to_numpy().T,
#         *errors.select(cs.starts_with("grad.")).to_numpy().T,
#         normalize=True,
#         length=0.05,
#         colors=f"C{i}",
#     )


# # %%
# from typing import Any, Sequence

# import pandas as pd
# from numpy.typing import NDArray


# def napari_gradients(
#     points: pl.DataFrame | NDArray,
#     arrows: pl.DataFrame | NDArray,
#     color: pl.Series | NDArray | None = None,
#     normalize_arrows=False,
#     autoscale_arrows=True,
#     **kwargs,
# ) -> dict[str, Any]:
#     assert (
#         points.shape == arrows.shape
#     ), f"`points`: {points.shape} and `arrows`: {arrows.shape} must have the same shape!"
#     assert (
#         points.shape[1]
#         in [
#             2,
#             3,
#         ]
#     ), f"Only 2D and 3D vector fields can be plotted! Yours have dimension: {points.shape[1]}"

#     fn_kwargs = {}

#     if normalize_arrows:
#         arrows = np.asarray(arrows) / np.linalg.norm(arrows, axis=1, keepdims=True)

#     vecs = np.stack([points, arrows], axis=1)
#     fn_kwargs["data"] = vecs

#     if autoscale_arrows:
#         max_1d_volume_extent = (
#             points.select(pl.max_horizontal(cs.starts_with("points")))
#             - points.select(pl.min_horizontal(cs.starts_with("points")))
#         ).max()
#         edge_width = max_1d_volume_extent / 100
#         max_length = np.quantile(np.linalg.norm(arrows, axis=1), 0.9)
#         length = 5 * edge_width / max_length
#         fn_kwargs["length"] = length.item()
#         fn_kwargs["edge_width"] = edge_width.item()

#     if color is not None:
#         feature_name = getattr(color, "name", "feature")
#         fn_kwargs["features"] = pd.DataFrame({feature_name: color})
#         fn_kwargs["edge_color"] = feature_name

#     return {
#         **fn_kwargs,
#         **kwargs,
#     }


# # %%
# fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
# for errors, name in zip(
#     [errors_del, errors_knn5, errors_knn8, errors_rad20, errors_rad30],
#     ["delaunay", "knn5", "knn8", "rad20", "rad30"],
# ):
#     sns.kdeplot(errors["abs_diff"], label=name, cumulative=True)
# plt.legend()

# fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
# for errors, name in zip(
#     [errors_del, errors_knn5, errors_knn8, errors_rad20, errors_rad30],
#     ["delaunay", "knn5", "knn8", "rad20", "rad30"],
# ):
#     sns.kdeplot(errors["angle_error"], label=name, cumulative=True)
# plt.legend()
# # %%
# viewer = napari.Viewer()
# viewer.add_image(SS, scale=(0.01, 0.01, 0.01), colormap="inferno")
# for errors, name in zip(
#     [errors_del, errors_knn5, errors_knn8, errors_rad20, errors_rad30],
#     ["delaunay", "knn5", "knn8", "rad20", "rad30"],
# ):
#     viewer.add_vectors(
#         **napari_gradients(
#             errors.select(cs.starts_with("points")),
#             errors.select(cs.starts_with("grad.")),
#             normalize_arrows=False,
#             color=errors["angle_error"],
#             edge_contrast_limits=(0, 10),
#             name=name,
#         )
#     )
# # viewer.add_vectors(**napari_gradients(errors.select(cs.starts_with('points')), errors.select(cs.starts_with('grad_true.')), normalize_arrows=False), edge_color='g')
# # %%
# napari_gradients(
#     errors.select(cs.starts_with("points")),
#     errors.select(cs.starts_with("grad.")),
#     normalize_arrows=False,
#     color=errors["angle_error"],
#     edge_contrast_limits=(0, 10),
#     name=name,
# )


# # %%
# from zfish.roi.spatial_roi import Roi
# from zfish.roi.visualize import imshow_roi

# # %%
# fn = r"C:\Users\hessm\Documents\zfish_local\imgs\B02_px+0198_py-2152.h5"

# lazy_roi = Roi.from_file(fn, level=1)

# # %%
# roi = lazy_roi.sel(c=["DAPI.0", "DAPI.1"]).drop_dim("l").compute()
# # %%
# imshow_roi(roi)
# # %%
# import matplotlib.pyplot as plt
# import napari
# import numpy as np
# import seaborn as sns
# from scipy.interpolate import RegularGridInterpolator
# from sklearn.metrics import f1_score

# # %%
# x = np.linspace(0, 1, 100)
# y = np.linspace(0, 1, 100)
# z = np.linspace(0, 1, 100)
# xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

# f = 1.5
# SS = 3 * xx**2 + np.sin(f * 2 * np.pi * yy) + zz

# SS_interp = RegularGridInterpolator((x, y, z), SS)

# SSdx = 6 * xx
# SSdy = f * 2 * np.pi * np.cos(f * 2 * np.pi * yy)
# SSdz = np.ones_like(zz)

# SSdx_interp = RegularGridInterpolator((x, y, z), SSdx)
# SSdy_interp = RegularGridInterpolator((x, y, z), SSdy)
# SSdz_interp = RegularGridInterpolator((x, y, z), SSdz)
# # %%
# quiver_subsample = 10

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# # plt.imshow(zz, origin='lower')

# ax.set_aspect("equal")
# ax.quiver(
#     xx[
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#     ],
#     yy[
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#     ],
#     zz[
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#     ],
#     SSdx[
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#     ],
#     SSdy[
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#     ],
#     SSdz[
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#         quiver_subsample // 2 :: quiver_subsample,
#     ],
#     length=0.01,
# )
# # %%
# import poisson_disc as po

# rng = np.random.default_rng(42)

# points = po.Bridson_sampling(dims=np.array([1.0, 1.0, 1.0]), radius=0.05)
# n = points.shape[0]
# S = SS_interp(points)

# # %%
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# ax.set_aspect("equal")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=S)
# # %%
# # from zfish.features.neighborhood.neighborhood_aggregation import aggregate_column_dense_parallel, aggregate_table_dense_parallel, aggregate_weighted_table_dense_parallel
# import polars as pl
# from sklearn.neighbors import NearestNeighbors

# from zfish.features.neighborhood import aggregation_functions as agg_funcs
# from zfish.features.neighborhood.neighborhoods import (
#     # AdjType,
#     NeighborhoodQueryObject,
# )

# # %%
# funcs = [agg_funcs.Mean, agg_funcs.Median, agg_funcs.Count, agg_funcs.Neighbors, agg_funcs.Quantile(0.1), agg_funcs.Quantile(0.5)]
# # funcs = [agg_funcs.Mean, agg_funcs.Median, agg_funcs.Count, agg_funcs.Quantile(0.1), agg_funcs.Quantile(0.5)]

# df = pl.DataFrame({'feature': S})
# # %%
# res = (
#     NeighborhoodQueryObject.from_numpy(points, adj_type=AdjType.DENSE_NAN)
#     .delaunay(n_steps=[1])
#     .knn(k=[5, 8])
#     .radius(r=0.1)
#     .aggregate(funcs, df)
# )
# # %%
# res2 = (
#     NeighborhoodQueryObject.from_numpy(points, adj_type=AdjType.SPARSE_CSR)
#     .delaunay(n_steps=[1])
#     .knn(k=[5, 8])
#     .radius(r=0.1)
#     .aggregate(funcs, df)
# )
# # %%
# res
# # %%
# %%timeit
# res = (
#     NeighborhoodQueryObject.from_numpy(points, adj_type=AdjType.DENSE_NAN)
#     .delaunay(n_steps=[1, 2])
#     .knn(k=[5, 8])
#     .radius(r=[0.1, 0.2, 0.8])
#     .aggregate(funcs, df)
# )

# # %%
# %%timeit
# res2 = (
#     NeighborhoodQueryObject.from_numpy(points, adj_type=AdjType.SPARSE_CSR)
#     .delaunay([1, 2])
#     .knn(k=[5, 8])
#     .radius(r=[0.1, 0.2, 0.8])
#     .aggregate(funcs, df)
# )

# # %%
# for nhd in nq.neighborhoods:
#     print(nhd)
# # %%
# for nhd in nq2.neighborhoods:
#     print(nhd)
# # %%


# # %%


# res = nq.aggregate(funcs, df)
# # %%
# res2 = nq2.aggregate(funcs, df)
# # %%
# res
# # %%
# from polars.testing import assert_frame_equal

# assert_frame_equal(res, res2)
# # %%

# # %%
# nq.aggregate(pl.Series('ones', np.ones(len(points))), agg_funcs.Median)

# # %%
# adjs = [e.adj for e in NeighborhoodQueryObject.from_numpy(points, adj_type=AdjType.SPARSE_CSR).radius([0.0, 0.1, 0.5, 2.0]).neighborhoods]
# adjs_coo = [sparse.coo_array(adj) for adj in adjs]
# adjs_nan = [csgraph.csgraph_to_dense(adj, null_value=np.nan) for adj in adjs]
# np.random.seed(42)
# f = np.random.randn(adj.shape[0], 1)
# f2 = np.random.randn(adj.shape[0], 3)
# # %%
# for i in range(4):
#     adj = adjs[i]
#     adj_nan = adjs_nan[i]
#     print(repr(adj))

#     %timeit aggregate_csr(adj, f, agg_funcs.Mean)
#     %timeit aggregate_csr_single(adj, f, agg_funcs.Mean)
#     %timeit aggregate_column_dense_parallel(adj_nan, f, agg_funcs.Mean)
#     %timeit aggregate_table_dense_parallel(adj_nan, f, agg_funcs.Mean)

# # %%
# for i in range(4):
#     adj = adjs[i]
#     adj_nan = adjs_nan[i]
#     print(repr(adj))

#     %timeit aggregate_csr(adj, f2, agg_funcs.Mean)
#     # %timeit aggregate_csr_single(adj, f, agg_funcs.Mean)
#     # %timeit aggregate_column_dense_parallel(adj_nan, f, agg_funcs.Mean)
#     %timeit aggregate_table_dense_parallel(adj_nan, f2, agg_funcs.Mean)
# # %%
# i = 3
# adj = adjs[i]
# adj_nan = adjs_nan[i]
# %timeit aggregate_csr(adj, f, agg_funcs.Mean)
# %timeit aggregate_weighted_csr(adj, f, agg_funcs.Mean)
# %timeit aggregate_table_dense_parallel(adj_nan, f, agg_funcs.Mean)
# %timeit aggregate_weighted_table_dense_parallel(adj_nan, f, agg_funcs.Mean)
# # %timeit aggregate_csr_single(adj, f, agg_funcs.Mean)
# # %%
# np.allclose(aggregate_csr(adj, f, agg_funcs.Mean), aggregate_csr_single(adj, f, agg_funcs.Mean))
# # %%
# # aggregate_csr(adj, f, agg_funcs.Mean)
# np.allclose(aggregate_csr2(adj, f, agg_funcs.Mean), aggregate_csr(adj, f, agg_funcs.Mean))
# np.allclose(aggregate_column_dense_parallel(adj_nan, f, agg_funcs.Mean), aggregate_csr(adj, f, agg_funcs.Mean))
# # %%
# # %%timeit
# aggregate_table_dense_parallel(adjs_nan[3], f, agg_funcs.Sum)
# # %%
# # %%timeit
# sns.histplot(aggregate_weighted_table_dense_parallel(adjs_nan[1], f, agg_funcs.Sum).flatten())

# # %%
# %%timeit
# adj * sparse.csr_array(f.T)
# # %%
# %%timeit
# tt = adj @ f
# # %%
# %%timeit
# aggregate_rows_csr(tt, agg_funcs.Sum)

# # %%
# adjacency_matrix = adj_nan

# adjacency_array = adjacency_matrix[0, :]
# adjacency_mask = np.isnan(adjacency_array)

# adjacent_features = adjacency_array[adjacency_mask]
# adjacent_features
# # %%
# %%timeit
# aggregate_rows(adj3_nan, agg_funcs.Sum)

# # %%
# %%timeit
# aggregate_rows_csr(adj3, agg_funcs.Sum)
# # %%
# from scipy import sparse
# from scipy.sparse import csgraph

# adj_raw = np.arange(16).reshape((4, 4))
# adj_raw[2, :] = 0
# adj_csr = sparse.csr_array(adj_raw)
# adj_nan = csgraph.csgraph_to_dense(adj_csr, null_value=np.nan)

# # %%
# %%timeit
# aggregate_rows(adj_nan, agg_funcs.CircMean)
# # %%
# %%timeit
# aggregate_rows_csr(adj_csr, agg_funcs.CircMean)
# # %%
# %%timeit
# (adj3 * f1).todense()
# # %%
# %%timeit
# (adj3.todense() * f1)
# # %%
# nq = (
#     NeighborhoodQueryObject.from_numpy(points, adj_type=AdjType.SPARSE_CSR)
#     # .delaunay([1, 2])
#     # .knn(k=[5, 8])
#     .radius(r=[0.20, 0.30, 0.9])
# )

# adj = next(nq.neighborhoods).adj
# adj2 = next(nq.neighborhoods).adj
# adj3 = next(nq.neighborhoods).adj
# # %%
# from scipy.sparse.csgraph import csgraph_to_dense
# # %%
# adj_nan = csgraph_to_dense(adj, null_value=np.nan)
# adj2_nan = csgraph_to_dense(adj2, null_value=np.nan)
# adj3_nan = csgraph_to_dense(adj3, null_value=np.nan)
# # %%
# %%timeit
# aggregate_table_dense_parallel(adj3_nan, f1, agg_funcs.Sum)
# # %%
# adj2 * adj3
# # %%
# f1 = np.random.randn(adj2.shape[0], 1)
# f2 = f1.T
# # %%

# csr_array(adj2 * f2).sum(axis=1)
# # %%
# adj2 = csr_matrix_power(adj, 2)
# adj2
# # %%
# adj2.data = np.ones_like(adj2.data)
# # %%
# adj2.setdiag(0)
# adj2.eliminate_zeros()
# adj2
# # %%
# from zfish.roi.spatial_roi import Roi
# # %%
# fn = r"M:\marvwy\20220721_ZE4i2_aligned\imgs\B05_px-0493_py-1526.h5"

# lazy_roi = Roi.from_file(fn, level=1)
# # %%
# nucs = lazy_roi.sel(l='nucleiRaw3').drop_dim('c').compute()
# # %%
# chull = convex_hull_image(nucs_np)
# # %%
# viewer = napari.Viewer()
# viewer.add_labels(nucs_np)
# viewer.add_labels(chull)
# # %%
# np.linalg.matrix_power(adj, 1)
# # %%
# %%timeit
# np.linalg.matrix_power(adj, 5)
# # %%
# %%timeit
# csr_matrix_power(adj_csr, 5)
# # %%
# nn = NearestNeighbors().fit(points)

# next(get_knn_neighborhood(nn, k=5, include_self=False, adj_type=AdjType.SPARSE_CSR)).adj.todense()

# next(get_knn_neighborhood_(nn, k=5, include_self=False)).adj


# # %%
# adj_del = delaunay_adjacency(points)
# adj_knn5 = next(
#     get_knn_neighborhood(
#         NearestNeighbors().fit(points), k=5, include_self=False, not_adjacent_value=0
#     )
# ).adj
# adj_knn8 = next(
#     get_knn_neighborhood(
#         NearestNeighbors().fit(points), k=8, include_self=False, not_adjacent_value=0
#     )
# ).adj
# adj_knn8 = next(
#     get_knn_neighborhood(
#         NearestNeighbors().fit(points), k=8, include_self=False, not_adjacent_value=0
#     )
# ).adj
# adj_rad20 = next(
#     get_radius_neighborhood(
#         NearestNeighbors().fit(points), r=0.20, include_self=False, not_adjacent_value=0
#     )
# ).adj
# adj_rad30 = next(
#     get_radius_neighborhood(
#         NearestNeighbors().fit(points), r=0.30, include_self=False, not_adjacent_value=0
#     )
# ).adj
