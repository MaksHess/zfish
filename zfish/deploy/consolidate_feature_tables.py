# %%
import argparse
from pathlib import Path

import polars as pl

from zfish.features.polars_utils import COLUMN_CASTS, read_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    args = parser.parse_args()
    
    IDX = args.idx
    root = Path(r"data/active/hmax/MARVWY_RESTORED/20220721_ZE4i2_aligned1/features_tcorr_v3")
    out_root = root.parent / f"{root.name}_consolidated"
    # out_root_delta = root.parent / f"{root.name}_delta"
    # OUTPUT_FORMAT: Literal['parquet', 'delta'] = 'parquet' # DELTA WAS SLOWER FOR FULL READS ONLY FASTER FOR PARTIAL READS.


    # %%

    # for fld in root_glob('*'):
    fld = sorted(list(root.glob('*')))[IDX]
    fns = list(fld.rglob('*.parquet'))
    objects = pl.Series([fn.stem for fn in fns]).unique(maintain_order=True)
    # %%
    for obj in objects:
        print(f'reading table {obj!r}...')
        df = read_table(fld, _object=obj)
        print('casting...')
        roi_type = pl.Enum(df['roi'].unique().sort())
        obj_type = pl.Enum(objects.sort())

        out_path = out_root / fld.name / f"{obj}.parquet"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        
        print(f'writing {out_path.name!r}...')
        df.cast({**COLUMN_CASTS, **{'roi': roi_type, 'object': obj_type}}).write_parquet(out_path)

