# %%
from zfish.features.neighborhood.neighborhoods import (
    NeighborhoodQueryObject,
    delaunay_adjacency,
    get_neighborhood_from_adjacency,
    mask_clipped_delaunay_adjacency,
)
from zfish.roi.spatial_roi import Roi
from zfish.roi.visualize import imshow_roi
from zfish.visualize.imshow import imshow_spatial_image
from zfish.visualize.napari import napari_adjacency

# %%
fn = r"M:\marvwy\20220721_ZE4i2_aligned\imgs\B05_px-1565_py+1716.h5"

lazy_roi = Roi.from_file(fn, level=1)
roi = lazy_roi.sel(l=['nucleiRaw3', 'embryoRaw']).drop_dim('c').compute()

# %%
lbl = roi.sel(l='nucleiRaw3').labels
mask = roi.sel(l='embryoRaw').labels
# %%
nq = NeighborhoodQueryObject.from_labelimage(lbl)
# %%
da = delaunay_adjacency(nq.points)
# %%
dam2 = mask_clipped_delaunay_adjacency(nq.points, mask, n_samples=100, percentage_in=0.2)
dam5 = mask_clipped_delaunay_adjacency(nq.points, mask, n_samples=100, percentage_in=0.5)
dam7 = mask_clipped_delaunay_adjacency(nq.points, mask, n_samples=100, percentage_in=0.7)
# %%
import napari
from scipy import sparse

viewer = napari.Viewer()
imshow_roi(roi, viewer)
# viewer.add_vectors(**napari_adjacency(nq.points, da, alpha=0.5))
# viewer.add_vectors(**napari_adjacency(nq.points, dam2, alpha=0.5, edge_color='green'))
viewer.add_vectors(**napari_adjacency(nq.points, sparse.triu(dam5), alpha=0.2, highlight_idx=20))
# viewer.add_vectors(**napari_adjacency(nq.points, dam7, alpha=0.5, edge_color='green'))