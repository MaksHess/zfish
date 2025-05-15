# %%
from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import easing_functions as ease_func
import h5py
import itk
import numpy as np
import pandas as pd
import polars as pl
import pyvista as pv
import vtk
from pyvista import PolyData, examples
from scipy.spatial import ConvexHull
from zfish.abbott_legacy import h5_files as h5
from zfish.abbott_legacy.conversions import to_itk, to_numpy
from zfish.abbott_legacy.h5_files import h5_summary
from zfish.io.draw import angle_between, rot_safe
from zfish.plot.plot_commons import cmaps

WRITE_PATH = r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Plots"

EASING_FUNCTIONS = {
    "linear": ease_func.LinearInOut,
    "sine": ease_func.SineEaseInOut,
    "quad": ease_func.QuadEaseInOut,
    "quadratic": ease_func.QuarticEaseInOut,
    "cubic": ease_func.CubicEaseInOut,
    "quintic": ease_func.QuinticEaseInOut,
    "circular": ease_func.CircularEaseInOut,
    "exponential": ease_func.ExponentialEaseInOut,
    "elastic": ease_func.ElasticEaseInOut,
    "bounce": ease_func.BounceEaseInOut,
    "back": ease_func.BackEaseInOut,
}

EasingFunction = Literal[EASING_FUNCTIONS.keys()]
pv.set_jupyter_backend("html")
pv.global_theme.notebook = False


def main():
    example_stream_plot()
    # import polars as pl

    # pv.set_jupyter_backend("html")
    # pv.global_theme.notebook = False

    # df = pl.read_parquet(
    #     r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\linear_models_in_r_features.parquet"
    # )
    # df = pl.read_csv(
    #     r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\neighborhood\gradient_RADIUS-50s_CircMean__NormalizedCcp.csv"
    # )
    # print(df)

    # df_one = df.filter(pl.col("roi") == "F04_px+2084_py-0112")

    # p = stream_plot(
    #     df_one,
    #     stream_source_n_points=1000,
    #     point_size=15,
    #     convex_hull=False,
    # )
    # p.show()


def to_vtk(img):
    return itk.vtk_image_from_image(to_itk(img))


def discrete_marching_cubes(
    lbl_img: Union[h5py.Dataset, itk.Image], smooth_iter=0, calculate_neighbors=False
):
    labels = np.unique(to_numpy(lbl_img))[1:]
    lbl_img = to_vtk(to_itk(lbl_img))

    discrete = vtk.vtkDiscreteMarchingCubes()
    discrete.SetInputData(lbl_img)
    discrete.SetComputeAdjacentScalars(calculate_neighbors)
    for i, l in enumerate(labels):
        discrete.SetValue(i, l)
    discrete.Update()
    return pv.wrap(discrete.GetOutput()).smooth(smooth_iter)


def smooth_discrete_marching_cubes(
    lbl_img: Union[h5py.Dataset, itk.Image],
    smoothing_iterations=0,
    calculate_neighbors=False,
    use_flyingedges=True,
):
    labels = np.unique(to_numpy(lbl_img))[1:]
    lbl_img = to_vtk(to_itk(lbl_img))

    if use_flyingedges:
        discrete = vtk.vtkDiscreteFlyingEdges3D()
    else:
        discrete = vtk.vtkDiscreteMarchingCubes()
        discrete.SetComputeAdjacentScalars(calculate_neighbors)

    discrete.SetInputData(lbl_img)
    for i, l in enumerate(labels):
        discrete.SetValue(i, l)

    pass_band = 0.001
    feature_angle = 120.0

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(discrete.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return pv.wrap(smoother.GetOutput())


def discrete_marching_cubes_vtk(
    lbl_img: Union[h5py.Dataset, itk.Image], smooth_iter=0, calculate_neighbors=False
):
    labels = np.unique(to_numpy(lbl_img))[1:]
    lbl_img = to_vtk(to_itk(lbl_img))

    discrete = vtk.vtkDiscreteMarchingCubes()
    discrete.SetInputData(lbl_img)
    discrete.SetComputeAdjacentScalars(calculate_neighbors)
    for i, l in enumerate(labels):
        discrete.SetValue(i, l)
    discrete.Update()
    return discrete.GetOutput()


MESH_COMPONENTS = ("points", "cells")


def attach_scalars_to_mesh(
    mesh,
    df,
    mesh_label_array="Scalars",
    df_label_column="label",
    features=None,
    target="points",
    inplace=True,
):
    if not inplace:
        mesh = mesh.copy()
    if target not in MESH_COMPONENTS:
        raise ValueError(f"Target must be one of {MESH_COMPONENTS}")
    data = mesh.point_data if target == MESH_COMPONENTS[0] else mesh.cell_data
    if mesh_label_array not in data:
        raise ValueError(
            f"Mesh {target[:-1]}_array does not contain: {mesh_label_array!r}"
        )
    if df_label_column not in "label":
        raise ValueError(f"df does not contain column: {df_label_column!r}")
    if features is None:
        features = [e for e in df.columns if e != df_label_column]
    else:
        if not all([e in df for e in features]):
            raise ValueError(f"df doesn not contain all of: {features}")

    for feature in features:
        f_map = (
            df.select(df_label_column, feature)
            .to_pandas()
            .set_index(df_label_column)[feature]
            # .to_dict()
        )
        data[feature] = vec_translate(data[mesh_label_array], f_map)
    return mesh


def vec_translate(a, d):
    return np.vectorize(d.__getitem__)(a)


def _generate_orbital_path(
    viewup: tuple[int, int, int] = (0, 0, -1),
    center: tuple[int, int, int] = (0, 0, 0),
    extent: float = 1.0,
    shift: float = 0.0,
    factor: float = 3.0,
    plotter: pv.Plotter = None,
):
    """
    Copied from pyvista.Plotter.generate_orbital_path, implemented as a static function. Behaves
    like the function if `plotter` is passed.
    https://github.com/pyvista/pyvista/blob/release/0.44/pyvista/plotting/plotter.py#L6091-L6138
    """
    if plotter is None:
        assert viewup is not None and center is not None and extent is not None, (
            "Either provide `plotter` or `viewup`, `center` and `extent`"
        )
    if viewup is None:
        viewup = plotter._theme.camera.viewup
    center = np.array(plotter.center)
    bnds = np.array(plotter.bounds)
    radius = (bnds[1] - bnds[0]) * factor
    y = (bnds[3] - bnds[2]) * factor
    if y > radius:
        radius = y
    center += np.array(viewup) * shift
    return center, radius, viewup


def generate_orbital_path(
    n_points: int = 20,
    viewup: tuple[int, int, int] = (0, 0, -1),
    center: tuple[int, int, int] = (0, 0, 0),
    extent: float = 1.0,
    shift: float = 0.0,
    factor: float = 3.0,
    plotter: pv.Plotter = None,
):
    center, radius, viewup = _generate_orbital_path(
        viewup=viewup,
        center=center,
        extent=extent,
        shift=shift,
        factor=factor,
        plotter=plotter,
    )
    return pv.Polygon(center=center, radius=radius, normal=viewup, n_sides=n_points)


def generate_arc_path(
    n_points: int = 20,
    arc_degree: float = 10,
    arc_shift_degree: float = 0,
    viewup: tuple[int, int, int] = None,
    center: tuple[int, int, int] = None,
    extent: float = None,
    shift: float = 0.0,
    factor: float = 3.0,
    plotter: pv.Plotter = None,
    easing_function: EasingFunction = "sine",
):
    if plotter is not None:
        camera_viewup = plotter._theme.camera.viewup
        if viewup is None:
            viewup = plotter._theme.camera.viewup
        if center is None:
            center = plotter.center
        if extent is None:
            bnds = plotter.bounds
            dx = bnds[1] - bnds[0]
            dy = bnds[3] - bnds[2]
            extent = max(dx, dy)
    else:
        assert viewup is not None and center is not None and extent is not None, (
            "Either provide `plotter` or `viewup`, `center` and `extent`"
        )
        camera_viewup = pv.global_theme.camera.viewup
    # print(extent, factor)
    center = np.array(center)
    viewup = np.array(viewup)
    radius = np.array(extent) * factor
    # center, radius, viewup = _generate_orbital_path(
    #     viewup=viewup,
    #     center=center,
    #     extent=extent,
    #     shift=shift,
    #     factor=factor,
    #     plotter=plotter,
    # )

    center += viewup * shift

    n_points_ = n_points // 2

    arc_rad = arc_degree / 360 * 2 * np.pi

    arc_shift_rad = arc_shift_degree / 360 * 2 * np.pi

    t_ = np.linspace(0, arc_rad, n_points_, endpoint=False)

    ff = EASING_FUNCTIONS[easing_function](start=0, end=t_.max())
    t = np.array(list(map(ff, t_ / t_.max())))
    t = t + arc_shift_rad

    pts_x = radius * np.cos(t)
    pts_y = radius * np.sin(t)
    pts_z = np.zeros(len(t))
    points_ = np.stack([pts_x, pts_y, pts_z]).T

    rotation = rot_safe(camera_viewup, viewup)
    # return rotation
    points = rotation.apply(points_) + center

    d = pv.MultipleLines(np.concatenate([points, points[::-1]]))
    d["t"] = d["Texture Coordinates"][:, 0]
    d["pos"] = np.concatenate([t_, t_[::-1]])
    d.field_data["diameter"] = 2 * radius

    return d


def polyhull(points):
    hull = ConvexHull(points)
    faces = np.column_stack(
        (3 * np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)
    ).flatten()
    poly = pv.PolyData(hull.points, faces)
    return poly


def compute_norm(vector_components):
    expr = pl.sum_horizontal(
        *[pl.col(component) ** 2 for component in vector_components]
    )
    return expr


def _get_streamplot_components(
    df,
    centroid_columns=("Centroid-z", "Centroid-y", "Centroid-x"),
    vector_components=("CcpGradient-z", "CcpGradient-y", "CcpGradient-x"),
    vector_scale: str | None = "LogCcpNorm",
    vector_scale_factor: float = 1.0,
    stream_terminal_speed=0.5,
    stream_source_n_points=3000,
    **streamline_kwargs,
):
    point_cloud = pv.PolyData(df.select(centroid_columns).to_numpy())
    point_cloud["vectors"] = df.select(vector_components).to_numpy()
    if vector_scale is None:
        vector_norms = df.select(
            compute_norm(vector_components).log().alias("LogCcpNorm")
        ).to_numpy()
        vector_scale = "logCcpNorm"
    else:
        vector_norms = df.select(vector_scale).to_numpy()
    point_cloud["norm"] = vector_norms

    geom = pv.Arrow()  # This could be any dataset

    grid = pv.ImageData()
    grid.origin = (0, 0, 0)
    grid.spacing = (10, 10, 10)
    grid.dimensions = (30, 65, 65)

    interp_raw = grid.interpolate(point_cloud, radius=50, strategy="mask_points")

    hull = polyhull(point_cloud.points)
    interp = grid.interpolate(
        interp_raw.select_enclosed_points(hull).threshold(
            0.5, scalars="SelectedPoints"
        ),
        strategy="mask_points",
    )
    quiver_points = point_cloud.glyph(
        orient="vectors", scale="norm", factor=vector_scale_factor, geom=geom
    )
    # quiver_interp = interp.glyph(
    #     orient="vectors", scale="norm", factor=vector_scale_factor, geom=geom
    # )
    quiver_interp = None

    z_min, z_max, y_min, y_max, x_min, x_max = point_cloud.bounds
    max_extent = max([z_max - z_min, y_max - y_min, x_max - x_min])
    radius = max_extent / 2
    default_kwargs = {
        "source_radius": radius,
        "terminal_speed": stream_terminal_speed,
        "n_points": stream_source_n_points,
    }
    streamline_kwargs["return_source"] = True
    # return interp, interp_raw, hull, grid, point_cloud
    stream, src = interp.streamlines(
        "vectors",
        **{
            **default_kwargs,
            **streamline_kwargs,
        },
    )
    return {
        "points": point_cloud,
        "hull": hull,
        "grid": grid,
        "interp_raw": interp_raw,
        "interp": interp,
        "quiver_points": quiver_points,
        "quiver_interp": quiver_interp,
        "stream": stream,
        "stream_source": src,
    }


def stream_plot(
    df,
    centroid_columns=("centroid.z", "centroid.y", "centroid.x"),
    vector_components=("grad.z", "grad.y", "grad.x"),
    vector_scale: str | None = "LogCcpNorm",
    points=True,
    point_hue: str | None = "NormalizedCCP",
    point_hue_norm: tuple[float, float] = (0.0, 2 * np.pi),
    point_palette=cmaps.ccp.to_mpl(),
    point_size=None,
    quiver_points=False,
    quiver_interp=False,
    convex_hull=False,
    stream=True,
    stream_hue: str | None = "IntegrationTime",
    stream_hue_norm: tuple[float, float] | None = None,
    stream_source=False,
    vector_scale_factor: float = 1.0,
    stream_terminal_speed: float = 0.001,
    stream_source_n_points: int = 1000,
    stream_render_lines_as_tubes: bool = False,
    stream_line_width: float | None = None,
    stream_opacity: float | None = 0.7,
    stream_color: float | None = "black",
    return_components: bool = False,
    po: pv.Plotter | None = None,
    **streamline_kwargs,
):
    plot_components = _get_streamplot_components(
        df,
        centroid_columns=centroid_columns,
        vector_components=vector_components,
        vector_scale=vector_scale,
        vector_scale_factor=vector_scale_factor,
        stream_terminal_speed=stream_terminal_speed,
        stream_source_n_points=stream_source_n_points,
        **streamline_kwargs,
    )
    if po is None:
        po = pv.Plotter(
            line_smoothing=True, point_smoothing=True, polygon_smoothing=True
        )

    if points:
        ps = plot_components["points"]
        if point_hue is not None:
            ps[point_hue] = df.select(point_hue).to_numpy()
            ps.set_active_scalars(point_hue)
        po.add_mesh(
            ps,
            cmap=point_palette,
            render_points_as_spheres=True,
            point_size=point_size,
            clim=point_hue_norm,
        )
    if convex_hull:
        po.add_mesh(plot_components["hull"], style="wireframe")
    if quiver_points:
        po.add_mesh(plot_components["quiver_points"])
    if quiver_interp:
        po.add_mesh(plot_components["quiver_interp"])
    if stream:
        po.add_mesh(
            plot_components["stream"],
            scalars=stream_hue,
            color=stream_color,
            opacity=stream_opacity,
            clim=stream_hue_norm,
            line_width=stream_line_width,
            render_lines_as_tubes=stream_render_lines_as_tubes,
        )
    if stream_source:
        po.add_mesh(plot_components["stream_source"])
    # po.enable_anti_aliasing()
    if return_components:
        return plot_components, po
    return po


def get_points(
    df,
    centroid_columns=("centroid.z", "centroid.y", "centroid.x"),
    features=(),
    feature_norms=(),
):
    points = pv.PolyData(df.select(centroid_columns).to_numpy())
    for feature in features:
        points[feature] = df.select(feature).to_numpy()

    for feature_norm in feature_norms:
        points.user_dict["contrast_limits"] = dict(zip(features, feature_norms))
    return points


def plot_points(
    points: pv.PolyData,
    hue=None,
    hue_norm=None,
    palette="turbo",
    size=None,
    po: pv.Plotter | None = None,
    **kwargs,
):
    if po is None:
        po = pv.Plotter()

    if hue is not None:
        points.set_active_scalars(hue)

    po.add_mesh(
        points,
        cmap=palette,
        render_points_as_spheres=True,
        point_size=size,
        clim=hue_norm,
        **kwargs,
    )
    return po


def write_gif(df, features, out_file, hue_norm=None, fps=10):
    points = get_points(df, features=features)
    points
    dir(points)
    default_cpos = [
        (-1095.103384830345, 1144.4129339671388, -366.5885879128631),
        (137.70453514739228, 358.02896801799864, 295.4597435897436),
        (-0.6399956399907925, -0.5638948033048347, 0.5219465792574736),
    ]
    po = pv.Plotter(
        off_screen=True,
        notebook=False,
        point_smoothing=True,
        window_size=[1024, 768],
        image_scale=1,
    )

    po = plot_points(points, hue_norm=hue_norm)
    # po.add_mesh(
    #     points,
    #     render_points_as_spheres=True,
    #     point_size=8,
    #     smooth_shading=True,
    #     split_sharp_edges=True,
    # )
    po.camera_position = default_cpos
    po.open_gif(out_file, subrectangles=True, fps=fps)

    for f in features:
        po.mesh.set_active_scalars(f)

        po.write_frame()

    return po


def example_write_gif(write_path=WRITE_PATH):
    from zfish.multi_table.schemas_v2 import sel
    from zfish.multi_table.tables_io import scan_resources_and_features

    rl, fl = scan_resources_and_features()

    r = rl.collect()

    out_file_name = "test.gif"
    out_file = Path(write_path) / out_file_name
    features = ("PhysicalSize", "Roundness", "Elongation", "Flatness", "FeretDiameter")

    df = (
        r.label_objects.filter(pl.col("o") == "nucleiRaw3")
        .join(
            fl.label.select(sel.idx, pl.col(features)).collect(),
            on=["roi", "o", "label"],
        )
        .filter(pl.col("roi") == pl.col("roi").get(8))
    )
    write_gif(df, features=features, out_file=out_file)


def example_stream_plot(write_path=WRITE_PATH):
    import polars as pl

    pv.set_jupyter_backend("trame")
    pv.global_theme.notebook = False
    # df = pl.read_parquet(
    #     r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\linear_models_in_r_features.parquet"
    # )
    df = pl.read_csv(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\neighborhood\gradient_RADIUS-50s_CircMean__NormalizedCcp.csv"
    )

    df_one = df.filter(pl.col("roi") == "F03_px+2058_py+0095")

    comps, p = stream_plot(
        df_one,
        points=True,
        stream_source_n_points=1000,
        point_size=12,
        convex_hull=False,
        centroid_columns=("Centroid-z", "Centroid-y", "Centroid-x"),
        vector_components=("CcpGradient-z", "CcpGradient-y", "CcpGradient-x"),
        point_hue="RADIUS-50s_CircMean__NormalizedCcp",
        # point_hue_norm=(0, 2*np.pi),
        stream_hue="IntegrationTime",
        stream_hue_norm=(-400, 400),
        stream_render_lines_as_tubes=True,
        stream_line_width=4,
        return_components=True,
    )
    p.parallel_projection = True

    if write_path is not None:
        p.export_html(Path(write_path) / "ccp_gradient_stream.html")

    p.show()


def example_checkbox(write_path=WRITE_PATH):
    mesh = pv.Sphere()

    p = pv.Plotter()
    actor = p.add_mesh(mesh)

    def toggle_vis(flag) -> None:
        actor.SetVisibility(flag)

    p.add_checkbox_button_widget(toggle_vis, value=True)
    if write_path is not None:
        p.export_html(Path(write_path) / "checkbox_test.html")
    p.show()


import time
from threading import Thread

from pyvista.core.utilities.helpers import is_pyvista_dataset


def _shift_cpos_down(shift, plotter):
    position, focus, viewup = plotter.camera_position
    new_position = np.array(position) + shift * np.array(viewup)
    new_focus = np.array(focus) + shift * np.array(viewup)
    plotter.set_position(new_position, render=False)
    plotter.set_focus(new_focus, render=False)


def _shift_cpos_left(shift, plotter):
    plotter.camera.roll += 90
    position, focus, viewright = plotter.camera_position
    plotter.camera.roll -= 90
    new_position = np.array(position) + shift * np.array(viewright)
    new_focus = np.array(focus) + shift * np.array(viewright)
    plotter.set_position(new_position, render=False)
    plotter.set_focus(new_focus, render=False)


from typing import Any


def orbit_on_path(
    plotter,
    path=None,
    focus=None,
    step=0.5,
    viewup=None,
    write_frames=False,
    threaded=False,
    progress_bar=False,
    zoom=1.0,
    shift_right=0.0,
    shift_up=0.0,
    out_static_files: dict[int, str | Path] | None = None,
    callbacks: tuple[dict[str,]] = (),
):
    """Orbit on the given path focusing on the focus point.

    Parameters
    ----------
    path : pyvista.PolyData
        Path of orbital points. The order in the points is the order of
        travel.

    focus : sequence[float], optional
        The point of focus the camera. For example ``(0.0, 0.0, 0.0)``.

    step : float, default: 0.5
        The timestep between flying to each camera position. Ignored when
        ``plotter.off_screen = True``.

    viewup : sequence[float], optional
        The normal to the orbital plane.

    write_frames : bool, default: False
        Assume a file is open and write a frame on each camera
        view during the orbit.

    threaded : bool, default: False
        Run this as a background thread.  Generally used within a
        GUI (i.e. PyQt).

    progress_bar : bool, default: False
        Show the progress bar when proceeding through the path.
        This can be helpful to show progress when generating
        movies with ``off_screen=True``.

    zoom : float, default: 1.0
        zoom applied at every position.

    out_static_files : dict[int, str|Path] | None, default: None
        store individual frames as screenshots.

    Examples
    --------
    Plot an orbit around the earth.  Save the gif as a temporary file.

    >>> from pathlib import Path
    >>> from tempfile import mkdtemp
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.load_globe()
    >>> texture = examples.load_globe_texture()
    >>> filename = Path(mkdtemp()) / 'orbit.gif'
    >>> plotter = pv.Plotter(window_size=[300, 300])
    >>> _ = plotter.add_mesh(
    ...     mesh, texture=texture, smooth_shading=True
    ... )
    >>> plotter.open_gif(filename)
    >>> viewup = [0, 0, 1]
    >>> orbit = plotter.generate_orbital_path(
    ...     factor=2.0, n_points=24, shift=0.0, viewup=viewup
    ... )
    >>> plotter.orbit_on_path(
    ...     orbit, write_frames=True, viewup=viewup, step=0.02
    ... )

    See :ref:`orbiting_example` for a full example using this method.

    """
    if focus is None:
        focus = plotter.center
    if viewup is None:
        viewup = plotter._theme.camera.viewup
    if path is None:
        path = plotter.generate_orbital_path(viewup=viewup)
    if not is_pyvista_dataset(path):
        path = pv.PolyData(path)
    points = path.points

    if out_static_files is None:
        out_static_files = {}

    # Make sure the whole scene is visible
    if "diameter" in path.field_data:
        plotter.camera.thickness = path.field_data["diameter"]
    else:
        plotter.camera.thickness = path.length

    if progress_bar:
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            raise ImportError("Please install `tqdm` to use ``progress_bar=True``")

    def orbit():
        """Define the internal thread for running the orbit."""
        points_seq = tqdm(points) if progress_bar else points

        for i, point in enumerate(points_seq):
            tstart = time.time()  # include the render time in the step time
            plotter.set_position(point, render=False)
            plotter.set_focus(focus, render=False)
            plotter.set_viewup(viewup, render=False)
            _shift_cpos_down(-shift_up, plotter)
            _shift_cpos_left(-shift_right, plotter)
            plotter.camera.zoom(zoom)
            plotter.renderer.ResetCameraClippingRange()
            print(callbacks)
            if len(callbacks) > 0:
                if len(callbacks) != 1:
                    callback = callbacks[i]
                else:
                    callbacks_ = callbacks[1]
                    for k, callback in callbacks_.items():
                        attrs = k.split(".")
                        func = plotter
                        for attr in attrs:
                            func = getattr(func, attr)
                        func(**callback)
            if write_frames:
                plotter.write_frame()
            else:
                plotter.render()
            if i in out_static_files:
                plotter.screenshot(out_static_files[i])
            sleep_time = step - (time.time() - tstart)
            if sleep_time > 0 and not plotter.off_screen:
                time.sleep(sleep_time)
        if write_frames:
            plotter.mwriter.close()

    if threaded:
        thread = Thread(target=orbit)
        thread.start()
    else:
        orbit()


def arc_orbit_mov(
    plotter,
    n_points,
    arc_degree,
    easing_function,
    out_filename,
    new_plotter=True,
    arc_shift_degree=0,
    shift=None,
    factor=None,
    viewup=None,
    quality=5,
    framerate=24,
    out_folder=WRITE_PATH,
    zoom=1.0,
    shift_up=0.0,
    shift_right=0.0,
    focus=None,
    static_screenshots: tuple[int | float, ...] = (
        0,
        0.5,
    ),  # int for index, float [0.0, 1.0] for percentage along path
    preview=False,
):
    if shift is None:
        shift = plotter.bounds[-1] - plotter.bounds[-2]
    if factor is None:
        factor = 3.0

    arc_orbit = generate_arc_path(
        center=focus,
        n_points=n_points,
        arc_degree=arc_degree,
        arc_shift_degree=arc_shift_degree,
        viewup=viewup,
        shift=shift,
        factor=factor,
        plotter=plotter,
        easing_function=easing_function,
    )
    if out_folder is None:
        out_folder = Path()
    out_file = Path(out_folder) / out_filename

    if len(static_screenshots) > 0:
        all_idxs = np.arange(n_points)
        static_screenshot_idxs = [
            round(e * (n_points - 1)) if isinstance(e, float) else e
            for e in static_screenshots
        ]
        idxs = sorted([all_idxs[e] for e in static_screenshot_idxs])
        out_static_files = {
            i: out_file.parent / f"{out_file.stem}_static{i:03d}" for i in idxs
        }

    if new_plotter:
        _plotter = pv.Plotter(off_screen=True)
        for name, actor in plotter.actors.items():
            _plotter.add_actor(actor)
    else:
        _plotter = plotter

    _plotter.show(auto_close=False)

    _plotter.open_movie(out_file, quality=quality, framerate=framerate)

    orbit_on_path(
        _plotter,
        path=arc_orbit,
        write_frames=True,
        progress_bar=True,
        viewup=viewup,
        zoom=zoom,
        shift_right=shift_right,
        shift_up=shift_up,
        focus=focus,
        out_static_files=out_static_files,
    )
    _plotter.close()


def example_arc_orbit_mov(write_path=WRITE_PATH):
    n_points = 100
    arc_degree = 50
    easing_function = "sine"
    quality = 5
    viewup = (0, 1, 0)
    zoom = 1.2
    shift_right = -10
    shift_up = 10

    mesh = pv.examples.load_ant()

    p = pv.Plotter()
    p.camera.up = viewup
    _ = p.add_mesh(mesh, color="firebrick")

    arc_orbit_mov(
        p,
        n_points=n_points,
        arc_degree=arc_degree,
        easing_function=easing_function,
        out_filename="ant_arc_orbit.mov",
        quality=quality,
        out_folder=write_path,
        viewup=viewup,
        zoom=zoom,
        shift_right=shift_right,
        shift_up=shift_up,
    )
    return p


# example_arc_orbit_mov()
