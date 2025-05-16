# %%
import polars as pl
import polars.selectors as cs

CCP_GRAD_QUICKSAVE = (
    r"C:\Users\hessm\Documents\Programming\Python\thesis\Data\ccp_with_grad.parquet"
)

df_grad_ = pl.read_parquet(CCP_GRAD_QUICKSAVE)  # .rename({"grad.y.": "grad.y"})
# %%
import seaborn as sns

sns.scatterplot(
    df_grad_.to_pandas(),
    x="log2_nuc__Count_corr",
    y="log2_nuc__Count",
    # hue="Ccp__Mean",
)

# %% Gradient plot grid
n_cols = 6
import napari
import numpy as np
from scipy import linalg

from zfish.plot.plot_commons import cmaps
from zfish.visualize.napari_utils import napari_centroids, napari_gradients

pre_feature = "NormalizedCCP"
# sort_by = ["cycle", "Ccp__Mean"]
# sort_by = "order"
length_scaler = 0.5
size_scaler = 0.5

dynamic_size = (
    (1 / df_grad_["log2_nuc__Count_corr"])
    / (1 / df_grad_["log2_nuc__Count_corr"].max())
    * 15
    * size_scaler
)


viewer = napari.Viewer()
viewer.add_points(
    **napari_centroids(
        df_grad_,
        name="ccp",
        centroid_column="centroid",
        features=cs.by_name("NormalizedCCP"),
        size=dynamic_size,
        translate_group=None,
        translate_sort=None,
        translate_n_rows=n_cols,
        face_contrast_limits=(0, 2 * np.pi),
        face_colormap=cmaps.ccp.to_mpl(),
    ),
)

viewer.add_points(
    **napari_centroids(
        df_grad_,
        name="div",
        centroid_column="centroid",
        features=cs.by_name("div"),
        size=dynamic_size,
        translate_group=None,
        translate_sort=None,
        translate_n_rows=n_cols,
        face_contrast_limits=(-0.0001, 0.0001),
        face_colormap="PiYG",
    ),
)

viewer.add_vectors(
    **{
        **napari_gradients(
            df_grad_.select(cs.starts_with("centroid"), cs.starts_with("grad")),
            name="grad",
            length=40,
            edge_width=4,
            translate_group=None,
        ),
        **{"edge_color": "#DDDDDDFF", "features": None},
    }
)

# %%
import pyvista as pv

from zfish.visualize.pyvista_utils import (
    _get_streamplot_components,
    compute_norm,
    stream_plot,
)

# %%
STATIC_PATH = r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Plots\ccp_gradient_stream_8-11_static.png"
DYNAMIC_PATH = r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Plots\ccp_gradient_stream_8-11.html"


rois = [
    "B02_px+0198_py-2152",
    "F05_px-0015_py-1358",
    "E04_px+2742_py-0060",
    "F03_px+0159_py+2187",
]

subplots_shape = (2, 2)

subplots = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
]

size_scale_stat = 2.5
size_scale = 1.5

n_lines_stat = 2000

sizes = [
    12,
    10,
    8,
    6,
]
default_cpos = [
    (-1302.5324833545992, 482.93383621972845, 50.71494986009872),
    (114.42073220492563, 342.67707889727745, 345.55821403064544),
    (-0.12186830802510701, -0.985650006753266, 0.11679974180786815),
]
pv.global_theme.font.family = "arial"
po_stat = pv.Plotter(
    line_smoothing=True,
    point_smoothing=True,
    polygon_smoothing=True,
    shape=subplots_shape,
    window_size=(800, 800),
    image_scale=4,
    off_screen=True,
)
po_dyn = pv.Plotter(
    line_smoothing=True,
    point_smoothing=True,
    polygon_smoothing=True,
    shape=subplots_shape,
    window_size=(800, 800),
    image_scale=4,
    off_screen=False,
)

po_stat.parallel_projection = True
po_dyn.parallel_projection = True


for i, roi in enumerate(rois):
    df_one = (
        df_grad_.with_columns(
            (compute_norm(("grad.z", "grad.y", "grad.x"))).alias("norm")
        )
        .with_columns((pl.col("norm").log()).alias("log_norm"))
        .filter(pl.col("roi") == roi)
    )

    po_stat.subplot(*subplots[i])
    po_dyn.subplot(*subplots[i])

    # po.reset_camera(clipping_range=True)  # Optional, to fix clipping
    comps, p_stat = stream_plot(
        df_one,
        points=True,
        # stream_source_n_points=500,
        stream_source_n_points=n_lines_stat,
        point_size=sizes[i] * size_scale_stat,
        convex_hull=False,
        vector_scale="log_norm",
        centroid_columns=("nuc_Centroid-z", "nuc_Centroid-y", "nuc_Centroid-x"),
        vector_components=("grad.z", "grad.y", "grad.x"),
        point_hue="NormalizedCCP",
        # point_hue_norm=(0, 2*np.pi),
        # stream_hue="IntegrationTime",
        stream_hue=None,
        stream_opacity=0.3,
        # stream_hue_norm=(-400, 400),
        stream_render_lines_as_tubes=False,
        stream_line_width=3,
        return_components=True,
        stream=True,
        quiver_points=False,
        po=po_stat,
    )
    comps, p_dyn = stream_plot(
        df_one,
        points=True,
        # stream_source_n_points=500,
        stream_source_n_points=1000,
        point_size=sizes[i] * size_scale,
        convex_hull=False,
        vector_scale="log_norm",
        centroid_columns=("nuc_Centroid-z", "nuc_Centroid-y", "nuc_Centroid-x"),
        vector_components=("grad.z", "grad.y", "grad.x"),
        point_hue="NormalizedCCP",
        # point_hue_norm=(0, 2*np.pi),
        # stream_hue="IntegrationTime",
        stream_hue=None,
        stream_opacity=0.3,
        # stream_hue_norm=(-400, 400),
        stream_render_lines_as_tubes=False,
        stream_line_width=3,
        return_components=True,
        stream=True,
        quiver_points=False,
        po=po_dyn,
    )
    for po in [po_stat, po_dyn]:
        po.camera_position = [
            (-1188.8848612793934, 468.43133859485954, 54.90920175263625),
            (109.05490893125534, 339.9550971984863, 324.98784255981445),
            (-0.12186830802510701, -0.985650006753266, 0.11679974180786815),
        ]
        po.camera_set = True  # prevent auto-reset of camera

for po in [po_stat, po_dyn]:
    po.link_views()

zoom_factor = 1.3
po_stat.reset_camera()
po_stat.camera.zoom(zoom_factor)
# for i in range(len(rois)):
#     po.subplot(*subplots[i])
#     po.camera.zoom(zoom_factor)


out_cpos = po_stat.show(
    return_cpos=True,
)
po_stat.screenshot(STATIC_PATH)


po_dyn.export_html(DYNAMIC_PATH)
po_dyn.reset_camera()
po_dyn.show()
