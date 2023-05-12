# %%
from pathlib import Path

from zfish.roi._roi_formatting import SPATIAL_DIMS
from zfish.roi.spatial_roi import EXPERIMENT_INDEX, Roi, RoiMap

# %%
fns = list(Path(r"M:\marvwy\20220721_ZE4i2_aligned\imgs").glob('*.h5'))
# %%
dater = RoiMap.from_files(
    fns, 
    level=1, 
    )

# %%
import polars as pl

from zfish.features.polars_utils import join


def expand_roi(df: pl.DataFrame):
    WELL, WX, WY = 'well', 'wx', 'wy'
    assert 'roi' in df
    index_cols = sorted(set(EXPERIMENT_INDEX).intersection(df.columns), key=EXPERIMENT_INDEX.index)
    return (
        df.select(
        [
            pl.col(index_cols),
            pl.col('roi').str.split('_').arr.first().alias(WELL), 
            pl.col('roi').str.extract(r"px([-+]\d+)_").cast(int).alias(WX), 
            pl.col('roi').str.extract(r"py([-+]\d+)").cast(int).alias(WY),
            pl.exclude(index_cols),
        ]
        )
    )


tbl = (
dater.table('nucleiRaw3')
.select([
    pl.col(EXPERIMENT_INDEX),
    pl.col('roi').str.split('_').arr.first().alias('well'), 
    pl.col('roi').str.extract(r"px([-+]\d+)_").cast(int).alias('wx'), 
    pl.col('roi').str.extract(r"py([-+]\d+)").cast(int).alias('wy'),
    pl.col('BoundingBox'),
    pl.col('Centroid'),
    ])

)

tbl_well = tbl.filter(pl.col('well') == 'B02')
roi_names = tbl_well.select(pl.col('roi').unique()).to_numpy().squeeze()
l_names = tbl_well.select(pl.col('object').unique()).to_numpy().squeeze()
dater_well = dater.sel(roi=roi_names, l=l_names, c='DAPI-1')

def translate_roi(roi: Roi, translation: dict[str, float]):
    return Roi(data={
        Roi._LABELS_KEY: roi.labels.assign_coords({k: t + roi.labels[k] for k, t in translation.items()}),
        Roi._IMAGES_KEY: roi.images.assign_coords({k: t + roi.images[k] for k, t in translation.items()}),
        },
        tables=roi.tables,
        name=roi.name,
        )

dater_well_shifted = RoiMap(rois={roi_name: translate_roi(dater_well.sel(roi=roi_name), {'x': tx * 0.325, 'y': ty * 0.325}) for roi_name, tx, ty in dater_well.table().pipe(expand_roi).groupby('roi').agg(pl.col('wx', 'wy').first()).rows()}, name=f"{dater_well.name}_shifted")
# %%
from zfish.roi.visualize import imshow_map

dater_well_shifted = dater_well_shifted.compute()

# %%
imshow_map(dater_well_shifted)