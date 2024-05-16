# %%
from zfish.features.io import scan_tables
from zfish.features.polars_selector import sel
from zfish.features.polars_utils import unnest_all_structs
from pathlib import Path
# %%
fld_features = Path(r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_tcorr_v3_consolidated\z_model=LogLinear__TwoStep")

dfs = scan_tables(fld_features)
# %%
dfs