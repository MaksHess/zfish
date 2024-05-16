# %%
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_utils import read_table
from zfish.preprocessing.sliding_window_samples import stratified_sample
from zfish.roi.spatial_roi import (
    Roi,
    apply_t_decay_factors,
    apply_z_decay_models_to_roi,
    read_models,
)

# %%
df_nuc = read_table(r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_tcorr\ExpFitLinear(loss='huber', features=['MediumPath', 'EmbryoPath'])", _object='nucleiRaw3')

# %%
df_meta = get_metadata(df_nuc)
df_samples = df_meta.filter(stratified_sample(by='cycle', n=2, seed=15)).sort('log2_nucleiRaw3_Count').filter(pl.col('cycle')<13)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(df_meta.sort('log2_nucleiRaw3_Count')['log2_nucleiRaw3_Count'])
# %%
img_fld = Path(r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\imgs")
channels = [
    'DAPI.1',
    # 'Pol-II-S2P.0',
    # 'Pol-II-S5P.2',
    # 'H3K27Ac.1',
]
level = 1

def load_rois(roi_ids, img_fld, channels, level, t_decay_factors=None, z_decay_models=None):
    res = {}
    for roi_id in tqdm(roi_ids):
        roi_fn = img_fld / f"{roi_id}.h5"
        roi = Roi.from_file(roi_fn, level=level).sel(c=channels)
        if t_decay_factors is not None:
            df_t_decay = pl.read_parquet(t_decay_factors)
            roi = apply_t_decay_factors(roi, df_t_decay)
        if z_decay_models is not None:
            models = read_models(z_decay_models)
            roi = apply_z_decay_models_to_roi(models, roi, two_step_label='embryoRaw')
        roi_channels = roi.drop_dim('l').compute().images
        # roi_channels = roi_channels.where(roi_channels < 0, 0).astype(np.uint16)
        res[roi_id] = roi_channels
    return res
# %%
tdfs = r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\models\t_decay\correction_factors.parquet"
zdms = r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\models\z_decay\Linear(loss='huber', features=['MediumPath', 'EmbryoPath'])"
zdms = r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\models\z_decay\ExpNoOffset(loss='huber', features=['MediumPath', 'EmbryoPath'])"
zdms = r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\models\z_decay\Exp(loss='huber', features=['MediumPath', 'EmbryoPath'], p_offset=False)"

# imgs = load_rois(df_samples['roi'].to_list()[:3], img_fld, channels, level, t_decay_factors=tdfs, z_decay_models=None)
# %%
import numpy as np

for k in imgs:
    imgs[k] = imgs[k].astype(np.uint16)
# %%
from functools import partial

from zfish.visualize.grids import arrange_on_grid, rect_grid

canvas = arrange_on_grid(imgs, index_function=partial(rect_grid, aspect_ratio=3))
# %%
import napari

from zfish.roi.visualize import imshow_roi

viewer = napari.Viewer()
viewer.add_image(canvas, channel_axis=1, name=['DAPI', 'Pol-II-S2P', 'Pol-II-S5P'])

# %%
fn_roi = img_fld / f"{df_samples['roi'].to_list()[-1]}.h5"

lazy_roi = Roi.from_file(fn_roi, level=1)
# %%
models = read_models(zdms)
# %%
lazy_roi_corr = apply_z_decay_models_to_roi(models, lazy_roi.drop_sel(c=['bCatenin.1', 'ALYREF.2', 'Pol-II-S5P.2', 'Nanog.3']), two_step_label='embryoRaw')
# %%
roi = lazy_roi_corr.sel(c='DAPI.1').drop_dim('l').compute()
# %%
imshow_roi(roi)
# %%
img = roi.images
img = img.clip(np.iinfo(np.uint16).min, np.iinfo(np.uint16).max).astype(np.uint16)
# %%
from zfish.visualize.imshow import imshow_spatial_image
imshow_spatial_image(img)