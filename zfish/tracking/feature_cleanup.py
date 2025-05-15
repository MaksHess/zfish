# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from zfish.dev.umap_polar_space import apply_transformer_to_dataframe
from zfish.features.polars_preprocessing import (
    DEFAULT_OUTLIER_RANGES,
    LOG_TRANSFORM_PATTERNS,
    apply_log_transform,
    get_metadata,
    mark_range_outlier,
    mark_range_outliers,
    remove_outliers,
)
from zfish.plot.features import plot_features_by_embryo

# %%
fn_features = Path(r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s1.parquet")
fn_features_out = fn_features.parent / f"{fn_features.stem}_clean.parquet"
fn_labels = (
    r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s1_segmentation.zarr"
)
fn_images = r"E:\sshami\Visiscope\20231026H1A488_compressed\20231026H1A1_s1.zarr"
# %%
df_raw = pl.read_parquet(fn_features)
df_trans = df_raw.pipe(apply_log_transform)

df_index = df_trans.select(cs.by_name(["t", "z", "y", "x", "img_label"]))
df_features = df_trans.select(
    cs.all() - cs.by_name(df_index.columns) - cs.matches("PrincipalAxes")
)
# %%
df_index.group_by("t", maintain_order=True).count().to_pandas().plot(x="t", y="count")
# %%
# plot_features_by_embryo(df_features, q_lower=0, q_upper=1, q_range=1.2)
# plot_features_by_embryo(df_features, q_lower=0.1, q_upper=0.9, q_range=2)
# %%
outlier_ranges = [
    {"feature": "Elongation", "upper": 6.0},
    {"feature": "Flatness", "upper": 6.0},
    {"feature": "EquivalentSphericalRadius", "lower": 3},
    # {"feature": "EquivalentSphericalRadius", "upper": 10},
    # {"feature": "EquivalentEllipsoidDiameter.b", "upper": 20},
    # {"feature": "EquivalentEllipsoidDiameter.c", "upper": 50},
    {"feature": "Roundness", "upper": 1.01},
]

outliers = mark_range_outliers(df_features, outlier_ranges)

# df_clean = df_raw.filter(~outliers['any_range_outlier'])
# df_clean.write_parquet(fn_features_out)
# df_clean.group_by("t", maintain_order=True).count().to_pandas().plot(x="t", y="count")
# %% Validate outlier selection
from zfish.tracking.io import load_image, load_labels

# %%
img = load_image(fn_images, level=2, scale=(1.0, 2.6, 2.6))
# %%
# lbls = load_labels(fn_labels, n_max=20)

# %%
import napari

from zfish.visualize.imshow import imshow_spatial_image

# %%
viewer = napari.Viewer()
imshow_spatial_image(img, viewer, contrast_limits=(80, 500))
# imshow_spatial_image(lbls, viewer)
viewer.add_points(
    df_index.select(["t", "z", "y", "x"]),
    properties={'_'.join(feat.split('_')[:-2]): outliers[feat].cast(pl.Float32) for feat in outliers.columns},
    face_color="any",
    # properties={feat: df_features[feat].cast(pl.Float32) for feat in df_features.columns},
    # face_color="EquivalentSphericalRadius",
    # face_contrast_limits=(1.0, 11.0),
    size=3,
)

# %%