# %%
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
# %%
fn = r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s1.parquet"
fn = r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s2.parquet"
df = pl.read_parquet(fn)
df2 = pl.read_parquet(fn)
# %%
sns.displot(df['PhysicalSize'])
sns.displot(df['Roundness'])
# %%
df.filter(pl.col('PhysicalSize')>500)