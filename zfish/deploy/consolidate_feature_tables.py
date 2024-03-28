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
    
    
    root = Path(r"/data/active/hmax/MARVWY_RESTORED/20220721_ZE4i2_aligned1/features_tcorr_v3")
    print(f'consolidating {root.name}')

    # OUTPUT_FORMAT: Literal['parquet', 'delta'] = 'parquet' # DELTA WAS SLOWER FOR FULL READS ONLY FASTER FOR PARTIAL READS.

    flds = sorted(list(root.glob('*')))
    fld = flds[IDX%len(flds)] # go through the filenames twice
    
    if IDX < len(flds):
        out_root = root.parent / f"{root.name}_consolidated"
    else:
        out_root = root.parent / f"{root.name}_delta"
    
    print(f'output fld: {out_root.name}')
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
        
        if IDX < len(flds):
            print(f'writing parquet {out_path.name!r}...')
            df.cast({**COLUMN_CASTS, **{'roi': roi_type, 'object': obj_type}}).write_parquet(out_path)
        else:
            print(f'writing delta {out_path.name!r}...')
            import re
            columns = df.columns
            safe_column_map = {e: re.sub('\\.', '-', e) for e in columns}
            df.cast({**COLUMN_CASTS, **{'roi': roi_type, 'object': obj_type}}).rename(safe_column_map).write_delta(out_path)

if __name__ == '__main__':
    main()

    