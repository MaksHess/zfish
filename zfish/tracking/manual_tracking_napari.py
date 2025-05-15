# %%
from pathlib import Path

import napari

from zfish.tracking import io

# %%
fn = Path(r"Z:\hmax\Visiscope\20240425H1A488_compressed\20240425_H1A488_s2_segmentation.zarr")
# %%
label_gen = io.label_generator(fn)
# %%b 
lbls = list(label_gen)
# %%
import napari

# %%
from zfish.io.datasource import hierarchical_labels

# %%
lbls = [hierarchical_labels() for _ in range(10)]
# %%
cells = [lbl.sel(c='cell') for lbl in lbls]
# %%
import xarray as xr

cells = xr.concat(cells, 't')
# %%
viewer = napari.view_labels(cells)  
# %%
import numpy as np
rng = np.random.default_rng(seed=43)

scores = np.array(list(range(1, 11)) + [12, 15])
total = np.sum(scores)
n = 100_000

a_scores = []
b_scores = []
diffs = []
for _ in range(n):
    a_score = rng.choice(scores, 6, replace=False).sum()
    b_score = total - a_score
    a_scores.append(a_score)
    b_scores.append(b_score)
    diffs.append(a_score - b_score)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(diffs)
# %%
import xarray as xr
nucs = xr.concat(lbls, dim='t')
# %%
napari.view_labels(nucs)