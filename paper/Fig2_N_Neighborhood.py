# %%
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.polars  # noqa
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from holoviews import opts

import zfish.features.neighborhood.aggregation_functions as agg_funcs
from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject
from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_selector import sel
from zfish.features.polars_utils import read_table, unnest_all_structs
from zfish.preprocessing.sliding_window_samples import stratified_sample

hv.extension("bokeh", "matplotlib")
# %%
# %matplotlib inline
# %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
# %%
STYLE_SHEET = (
    r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\mystyle.mplstyle"
)
df_debris = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers\predictions\debris_pred.parquet"
)
df_celltype = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers\predictions\celltype_pred.parquet"
)
df_nuc_raw = (
    read_table(
        r"C:\Users\hessm\Documents\zfish_local\features_tcorr\Linear(loss='huber', features=['MediumPath', 'EmbryoPath'])",
        _object="nucleiRaw3",
    )
    .join(df_celltype, on=["roi", "object", "label"], how="left")
    .join(df_debris, on=["roi", "object", "label"], how="left")
)
df_emb_raw = read_table(
    r"C:\Users\hessm\Documents\zfish_local\features_tcorr\Linear(loss='huber', features=['MediumPath', 'EmbryoPath'])",
    _object="embryoRaw",
)
# %%


df_nuc = df_nuc_raw.filter(pl.col("debris_pred") == False)
# %%
CONTROL_WELLS = ["B07", "C07", "D07", "E07"]

# sample_embryos = [
#     "B02_px+0385_py-0060",
#     "C04_px-0002_py+2310",
#     "C04_px-2146_py+0224",
#     "C06_px-2276_py-0867",
#     "E04_px-1823_py-0854",
#     "E04_px-2424_py+0579",
#     "E05_px+2168_py+1386",
#     "E05_px-1798_py-0764",
#     "E06_px-0118_py-2462",
#     "E06_px-1759_py-0725",
#     "E07_px-0280_py-1306",
#     "F05_px+0166_py+1787",
#     "F05_px+1948_py-1778",
#     "F05_px-0015_py-1358",
#     "F05_px-1746_py-0396",
#     "G03_px+2426_py-0008",
#     "G03_px-0093_py-0054",
#     "G05_px-0370_py+1942",
# ]

df_meta = get_metadata(df_nuc, control_wells=CONTROL_WELLS)
df_emb = df_meta.join(
    df_emb_raw, left_on=["roi", "parent.embryoRaw"], right_on=["roi", "label"]
)
sample_embryos = (
    df_meta.filter(pl.col("control_well_for_acquisition") > 1)
    .filter(stratified_sample(by="cycle", n=3, seed=42))
    .filter(pl.col("cycle") < 13)
)["roi"].to_list()

sample_embryos_sml = (
    df_meta.filter(pl.col("control_well_for_acquisition") > 1)
    .filter(stratified_sample(by="cycle", n=3, seed=42))
    .filter(pl.col("cycle") < 13)
    .filter(stratified_sample(by='cycle', n=2, seed=42))
)["roi"].to_list()

df_plot = df_nuc.join(df_meta, on=["roi"])
df_one = df_plot.filter(pl.col("roi") == df_plot["roi"][0])
df_ten = df_plot.filter(pl.col("roi").is_in(df_meta["roi"][:10]))
df_sample = df_plot.filter(pl.col("roi").is_in(sample_embryos))
df_sample_sml = df_plot.filter(pl.col('roi').is_in(sample_embryos_sml))
# %%
import numpy as np

nq = NeighborhoodQueryObject.from_dataframe(df_one)
# %%
nq.radius(r=np.arange(10, 200, 10)).knn(k=[2, 10, 40]).delaunay().touch().aggregate(agg_funcs.Count)
# %%
qq = nq.radius([10, 20, 30], self_loops=[True, False]).radius([10, 20, 30], distance=True, self_loops=[True, False]).queries
# %%
str(qq[-1])
# %%
from zfish.features.neighborhood.neighborhoods import radius, radius_kernel
# %%
nq.radius([10, 20, 30]).queries