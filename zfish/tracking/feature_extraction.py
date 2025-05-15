# %%
import sys
from pathlib import Path

from zfish.tracking import io

site = 2

root = Path(r"Z:\hmax\Visiscope\20240425H1A488_compressed")

fn_lbls = root / f"20240425_H1A488_s{site}_segmentation.zarr"

fn_img = root / f"20240425_H1A488_s{site}.zarr"
out_fn = root / f"20240425_H1A488_s{site}.parquet"
print(fn_img)
print(out_fn)
lbls_gen = io.label_generator(
    fn_lbls, scale=(1.0, 0.65, 0.65), dims=("z", "y", "x"), l_coords=("nuclei",)
)
img_gen = io.image_generator(
    fn_img,
    level=0,
    scale=(1.0, 0.65, 0.65),
    dims=("c", "z", "y", "x"),
    c_coords=("H1A",),
)

df_lbls = io.extract_features(lbls_gen, img_gen, out_path=out_fn)

# %%
site = 1

root = Path(r"Z:\hmax\Visiscope\20240425H1A488_compressed")

fn_lbls = root / f"20240425_H1A488_s{site}_segmentation.zarr"

fn_img = root / f"20240425_H1A488_s{site}.zarr"
out_fn = root / f"20240425_H1A488_s{site}.parquet"
print(fn_img)
print(out_fn)
lbls_gen = io.label_generator(
    fn_lbls, scale=(1.0, 0.65, 0.65), dims=("z", "y", "x"), l_coords=("nuclei",)
)
img_gen = io.image_generator(
    fn_img,
    level=0,
    scale=(1.0, 0.65, 0.65),
    dims=("c", "z", "y", "x"),
    c_coords=("H1A",),
)

df_lbls = io.extract_features(lbls_gen, img_gen, out_path=out_fn)
# %%
import zarr

fn = r"Z:\hmax\Visiscope\20240425H1A488_compressed\20240425_H1A488_s2.zarr"

movie = zarr.open(fn, "r")
# %%
frame = movie[2][:60, ...].squeeze()
# %%
import napari

napari.view_image(frame, scale=(1.0, 1.0, 1.3, 1.3))
# %%
