# %%
from pathlib import Path

import napari
import polars as pl

from zfish.features.ccp.classifiers.clf_io import get_annotations, load_classifier

pl.enable_string_cache(True)

from zfish.features.polars_preprocessing import get_metadata, get_metadata_old
from zfish.features.polars_utils import read_table, unnest_all_structs
from zfish.preprocessing.sliding_window_samples import stratified_sample
from zfish.roi.spatial_roi import Roi, RoiMap
from zfish.roi.visualize import imshow_roi

# %%
fld_features = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_tcorr\Exp(features=MediumPath;EmbryoPath, loss=huber, pos_offset=True)"
)
fld_imgs = Path(r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\imgs")
fn_clf = Path(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\intensity_normalization\nucleiRaw3_classifier_celltype.clf"
)
df_nucs = (
    read_table(fld_features, "nucleiRaw3")
    .pipe(unnest_all_structs)
    .sort(["roi", "object", "label"])
)
df_meta = get_metadata(df_nucs)
# %%
if fn_clf.exists():
    clf = load_classifier(fn_clf)
    df_ann = get_annotations(clf)
    print(f"Total annotations: {len(df_ann)}")
    print(
        df_ann.join(df_meta, on="roi", how="left")
        .group_by(pl.col(["cycle", "roi"]))
        .len()
        .sort(["cycle", "roi"])
        .head(10)
    )
# %%

df_sample = (
    df_meta.sort("roi").filter(stratified_sample("cycle", n=5, seed=42)).sort("cycle")
)
df_sample
# %%
sample_embryos = df_sample["roi"].to_list()
sample_embryos


# %%
# old indices: 5, 6, 11, 12, 13, 16, 17, 18
# IDX = 10, 15, control_wells
# BEAUTY = (18, roi='E04_px+2742_py-0060', cycle=10)
# BEAUTY_ISH = (20, roi='E03_px+1457_py-1849', cycle=11)
# WEIRDO = (19, 'E06_px+2071_py-0054')
# RIPPED = ('C02_px-1778_py+0437', 'E02_px-2372_py-0247')

IDX = 13
sample_embryo = sample_embryos[IDX]
roi_lazy = Roi.from_file(
    next(fld_imgs.glob(f"{sample_embryo}.h5")),
    level=1,
    features_root=fld_features / sample_embryo,
)
# %%
roi = roi_lazy.sel(c=["DAPI.1", "PCNA.0", "bCatenin.1"], l="nucleiRaw3").compute()
roi.images.attrs["napari_channel_kwargs"] = {
    "DAPI.1": {
        "contrast_limits": (0, 1000),
    },
    "PCNA.0": {
        "contrast_limits": (0, 4000),
        "colormap": "red",
    },
    "bCatenin.1": {
        "contrast_limits": (30, 1000),
        "colormap": "green",
    },
}
# %%
viewer = imshow_roi(roi, roi_name_as_prefix=True)


# %%
import hvplot.polars

pl.DataFrame(
    {
        "feature": clf._classifier.feature_names_in_,
        "importance": clf._classifier.feature_importances_,
    }
).sort("importance").plot.barh("feature", "importance", height=1000)
