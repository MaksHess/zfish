# %%
import argparse
import sys
from pathlib import Path

from zfish.tracking import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("folder", type=Path)
    parser.add_argument("-s", "--feature_suffix", type=str, default="")
    parser.add_argument('-i', "--compute_intensity_features", action='store_true')

    args = parser.parse_args()

    fns = sorted(list(args.folder.glob("*[0-9].zarr")))
    fns_seg = sorted(list(args.folder.glob("*[0-9]_segmentation.zarr")))
    fn = fns[args.idx]
    fn_seg = fns_seg[args.idx]
    
    out_fn = fn.parent / (f"{fn.stem}{args.feature_suffix}.parquet")
    
    lbls_gen = io.label_generator(
        fn_seg, scale=(1.0, 0.65, 0.65), dims=("z", "y", "x"), l_coords=("nuclei",)
    )
    if args.compute_intensity_features:
        img_gen = io.image_generator(
            fn,
            level=0,
            scale=(1.0, 0.65, 0.65),
            dims=("c", "z", "y", "x"),
            c_coords=("H1A",),
        )

    if args.compute_intensity_features:
        io.extract_features(lbls_gen, img_gen, out_path=out_fn)
    else:
        io.extract_label_features(lbls_gen, out_path=out_fn)
        
if __name__ == '__main__':
    main()