# %%
from pathlib import Path

import imageio

from zfish.tracking.io import load_image

# %%
fn = Path(r"E:\sshami\Visiscope\20231026H1A488_compressed\20231026H1A1_s1.zarr")
fn_out = Path(r"M:\marvwy\VisiScope\20231026H1A488_tiff")  / f"{fn.stem}-t50.tiff"
# %%
img = load_image(fn, level=0, n_max=60)
# %%
imageio.mvolwrite(fn_out, ims=img.data[:50], format='tiff', bigtiff=True)