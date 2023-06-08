# %%
from pathlib import Path

import polars as pl
from zfish.features.intensity import get_distribution_features
from zfish.features.label import (
    get_label_features,
    get_position_and_orientation_features,
)

from spatial_roi import Roi, RoiMap

# pl.enable_string_cache(True)

# %%
fld = Path(r"M:\marvwy\20220721_ZE4i2_aligned\imgs")
fns = list(fld.glob("*.h5"))
# %%
data = RoiMap.from_files(fns[1:3], level=1)
# %%
structures = ['embryoRaw', 'cells', 'nucleiRaw3']
channels = ['DAPI-0', 'DAPI-1']
# %%
structures = ['cells', 'nucleiRaw3']
channels = ['DAPI-0']
# %%
for name, lazy_roi in data.items():
    print(name)
    # roi = lazy_roi.sel(c=channels, l=structures)
    # print(roi.paths)
    roi = lazy_roi.sel(c=channels, l=structures).compute()
    tables = {}
    for structure in structures:
        print(structure)
        lbl = roi.sel(l=structure).labels
        tables[structure] = get_label_features(lbl)
        for channel in channels:
            print(channel)
            img = roi.sel(c=channel).images
            df = get_intensity_features(lbl, img)
            tables[structure] = tables[structure].join(df, on='label')
    roi.tables = tables
    print("writing tables...")
    roi.write_tables()

# %%
data = RoiMap.from_files(fns[:4], level=1)
for name, roi in data.items():
    print(repr(roi))
    # %%
data.sel(roi='B02_px+0385_py-0060').first().tables['embryoRaw'].select(pl.exclude("^DAPI-2.*$"))