# %%
from pathlib import Path

import napari
import polars as pl

pl.enable_string_cache(True)

from zfish.features.polars_preprocessing import get_metadata, get_metadata_old
from zfish.features.polars_utils import unnest_all_structs
from zfish.preprocessing.sliding_window_samples import stratified_sample
from zfish.roi.spatial_roi import Roi, RoiMap
from zfish.roi.visualize import imshow_roi

# %%
fld = Path(r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\imgs")
# %%
fld_features = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_tcorr\Linear(features=MediumPath;EmbryoPath, loss=huber)"
)
df_full_allmodels = pl.scan_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\intensity_normalization\nuc_features_all_models.parquet"
)
df_full = df_full_allmodels.filter(
    pl.col("model") == "Linear(loss='huber', features=['MediumPath', 'EmbryoPath'])"
)
df_nuc_lazy = df_full.filter(pl.col("object") == "nucleiRaw3")
df_nuc = (
    df_nuc_lazy.collect()
    .sort(["roi", "object", "label"])
    .pipe(unnest_all_structs, sep=".")
)
# %%
df_meta = get_metadata_old(df_nuc)
df_sample = df_meta.filter(stratified_sample("cycle", n=5))
# %%
sample_embryos = [
    "D03_px-2069_py-0635",
    "B05_px-2211_py-0661",
    "C07_px-2146_py-1132",
    "G03_px-0093_py-0054",
    "F05_px-1746_py-0396",
    "D05_px-1249_py-2249",
    "C04_px+0314_py-2230",
    "G03_px+0101_py-2178",
    "B05_px-0493_py-1526",
    "F05_px+1412_py-0002",
    "D05_px+2284_py+0204",
    "E05_px-0106_py+1980",
    "E02_px-2372_py-0247",
    "C02_px+1825_py-1320",
    "E03_px+1457_py-1849",
    "F05_px+2458_py+0579",
    "C04_px+0198_py-0661",
    "G03_px-1410_py+1399",
    "G05_px-1287_py-1132",
    "E04_px-1823_py-0854",
    "C06_px-2276_py-0867",
    "B07_px-1882_py-0583",
    "G03_px-2334_py-0719",
    "E06_px-1759_py-0725",
    "E05_px-1798_py-0764",
]

processed = [
    "F05_px+1412_py-0002",
    "C06_px-2276_py-0867",
    "E05_px-1798_py-0764",
    "C04_px+0198_py-0661",
    "G05_px-1287_py-1132",
    "G03_px-2334_py-0719",
    "B07_px-1882_py-0583",
    "E06_px-1759_py-0725",
    "E05_px-0106_py+1980",
    "G03_px+0101_py-2178",
    "D05_px+2284_py+0204",
    "B05_px-0493_py-1526",
]

# new embryos cy 9, 11, 12
new_embryos = [
    "E02_px-0015_py-2120",  # 9
    "D02_px-0015_py-1539",  # 9
    "G07_px-2146_py+0321",  # 11
    "D05_px-1933_py+0611",  # 11
    "G03_px-2340_py+0308",  # 11
    "D05_px+2239_py+1419",  # 12
    "B07_px+1573_py+1871",
    "C02_px-1623_py+1916",
]
# %%

sample_embryo = new_embryos[5]
roi_lazy = Roi.from_file(
    next(fld.glob(f"{sample_embryo}.h5")),
    level=1,
    features_root=fld_features / sample_embryo,
)
# %%
# roi = roi_lazy.sel(c=['DAPI.1'], l='nucleiRaw3').compute()
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
        "contrast_limits": (30, 1200),
        "colormap": "green",
    },
}
# roi = roi_lazy.sel(c=['DAPI.1', 'PCNA.0']).drop_dim('l').compute()
# %%
#TODO: make sure annotations are loaded properly!
viewer = imshow_roi(roi, roi_name_as_prefix=True)

# %%
from zfish.features.ccp.classifiers.clf_io import get_annotations, load_classifier

# %%
clf = load_classifier(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\ccp\classifiers\nucleiRaw3_classifier3.clf")
df_ann = get_annotations(clf)
df_ann.write_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\ccp\classifiers\nucleiRaw3_classifier3_annotations.parquet")
# %%
df_ann.join(df_meta, on=['roi'])['cycle'].value_counts()