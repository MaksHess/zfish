# %%
from zfish.features.correlation import get_colocalization_features
from zfish.roi.spatial_roi import Roi

# %%
fn = r"M:\marvwy\20220721_ZE4i2_aligned\imgs\B05_px+1341_py-1526.h5"

roi_lazy = Roi.from_file(fn, level=1)
# %%
roi = roi_lazy.sel(l='nucleiRaw3', c=['DAPI.0', 'DAPI.1', 'DAPI.2', 'DAPI.3']).compute()
# %%
df_polars = get_colocalization_features(roi.labels, roi.sel(c='DAPI.0').images, roi.sel(c='DAPI.1').images)
df_pandas = df_polars.to_pandas()
# %%
import numpy as np
from itertools import product
channels = np.asarray(roi.images.c)

for c1, c2 in product(channels, channels):
    print(c1, c2)
# %%
roi.images