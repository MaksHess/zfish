# %%
import json
import logging
from dataclasses import asdict
from pathlib import Path

import btrack
import napari
import numpy as np
import polars as pl
from btrack import config
from btrack.utils import tracks_to_napari

from zfish.tracking.io import extract_features, load_features, load_image, load_labels

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# %%
fn_labels = Path(r"Z:\hmax\RESTORED\marvwy\Visiscope\20231026H1A488_compressed\20231026H1A1_s1_segmentation.zarr")
# fn_image = Path(r"E:\sshami\Visiscope\20231026H1A488_compressed\20231026H1A1_s1.zarr")
lbls = load_labels(fn_labels, n_max=10)
# %%
img = load_image(fn_image, level=2, scale=(1.0, 2.6, 2.6), n_max=10)

# %%
from zfish.visualize.imshow import imshow_spatial_image

viewer = napari.Viewer()
# viewer.add_labels(np.asarray(lbls), scale=(1, 1.0, 0.65, 0.65))
imshow_spatial_image(lbls, viewer)
imshow_spatial_image(img, viewer)
# %%
lbls
# %%
img
# %%
# fn_image = r"M:\marvwy\VisiScope\20230329_compressed\20230329-H1-GFP2_s4.zarr"
LOAD_IMAGE = False
IMAGE_LEVEL = 3
IMAGE_SCALE = (1.0, 2.6, 2.6)
LOAD_LABELS = False
EXTRACT_FEATURES = False
LOAD_FEATURES = True
# fn_features = Path(r"M:\marvwy\VisiScope\20231026H1A488_compressed\20231026H1A1_s1_lowres.parquet")
fn_features = Path(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\tracking\local_data\e2_s1_lowres.parquet"
)

# fn_labels = Path(fn_image).parent / f"{Path(fn_image).stem}_segmentation.zarr"
# fn_features = Path(fn_image).parent / f"{Path(fn_image).stem}.parquet"


# %% Load resources
if LOAD_IMAGE:
    image = load_image(fn_image, level=IMAGE_LEVEL, scale=IMAGE_SCALE)
else:
    image = None

if LOAD_LABELS:
    lbls = load_labels(fn_labels)
else:
    lbls = None

if EXTRACT_FEATURES:
    assert lbls is not None, "Need to load labels for feature extraction."
    df = extract_features(lbls, image, write_to=fn_features)

if LOAD_FEATURES:
    df = load_features(fn_features)

df = df.with_columns(
    pl.lit(1).alias("Constant")
)  # add constant feature for visualization
objs = btrack.io.objects_from_array(df.to_numpy(), default_keys=df.columns)
# FEATURES = df.columns
FEATURES = ["Roundness", "PhysicalSize", "H1A_Median", "H1A_StandardDeviation"]

# %% Helper functions for selection
from functools import reduce
from operator import add


def get_tree_by_ids(ids, tracks):
    trees = [get_tree_by_id(id_, tracks) for id_ in ids]
    return reduce(add, trees, [])


def get_tree_by_id(id_, tracks):
    root = get_track_by_id(id_, tracks)
    return extract_tree_recursive(root, tracks)


def get_track_by_id(id_, tracks):
    for track in tracks:
        if track["ID"] == id_:
            return track


def extract_tree_recursive(node, tracks, aggregator=None):
    if aggregator is None:
        aggregator = []
        aggregator.append(node)

    for child_id in node.children:
        print(child_id)
        child = get_track_by_id(child_id, tracks)
        aggregator.append(child)
        extract_tree_recursive(child, tracks, aggregator=aggregator)
    return aggregator


def tree_to_napari(tree):
    return dict(zip(["data", "properties", "graph"], tracks_to_napari(tree)))


def napari_tree_to_bbx(tree_napari, xyz_buffer=20):
    _, tmin, zmin, ymin, xmin = tree_napari["data"].min(axis=0)
    _, tmax, zmax, ymax, xmax = tree_napari["data"].max(axis=0)
    return {
        "t": slice(tmin, tmax),
        "z": slice(zmin - xyz_buffer, zmax + xyz_buffer),
        "y": slice(ymin - xyz_buffer, ymax + xyz_buffer),
        "x": slice(xmin - xyz_buffer, xmax + xyz_buffer),
    }


def napari_tree_to_point_selector(tree_napari, xyz_buffer=20):
    _, tmin, zmin, ymin, xmin = tree_napari["data"].min(axis=0)
    _, tmax, zmax, ymax, xmax = tree_napari["data"].max(axis=0)
    return (
        pl.col("t").is_between(tmin, tmax)
        & pl.col("z").is_between(zmin - xyz_buffer, zmax + xyz_buffer)
        & pl.col("y").is_between(ymin - xyz_buffer, ymax + xyz_buffer)
        & pl.col("x").is_between(xmin - xyz_buffer, xmax + xyz_buffer)
    )

def split_events_from_graph(graph, tracks):
    split_events = []
    split_points = []
    for k, vs in graph.items():
        for v in vs:
            tt = get_track_by_id(k, tracks)
            tf = get_track_by_id(v, tracks)
            p_start = np.array([tf.t[-1], tf.z[-1], tf.y[-1], tf.x[-1]])
            p_end = np.array([tt.t[0], tt.z[0], tt.y[0], tt.x[0]])
            vec = np.array([p_start, p_end-p_start])
            split_events.append(vec)
            split_points.append(p_start)
    return split_points, split_events
        

# %%
pp, vv = split_events_from_graph(graph, tracks)
# %%
viewer = napari.Viewer()
viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=3)
viewer.add_points(pp, size=7, face_color='red')
viewer.add_vectors(vv, edge_width=3)
# %% Load tracks
import json
from pathlib import Path

from btrack.io import HDF5FileHandler

from zfish.tracking.tracking import Parameters

files = Path(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\tracking\local_data\tracking_results\run_0"
)

trackss = []
params = []
for fn in sorted(files.glob("*.h5"), key=lambda x: int(x.stem.split("_")[-2])):
    print(fn)
    fn_params = fn.parent / f"{'_'.join(fn.stem.split('_')[:-1] + ['params'])}.json"

    with HDF5FileHandler(fn, "r") as f:
        tracks = f.tracks
        trackss.append(tracks)
    with open(fn_params) as fh:
        params_dict = json.load(fh)
        params.append(
                {
                    k: v
                    for k, v in params_dict.items()
                    if k not in ["tracks_out", "params_out"]
                }
        )

names = ['; '.join(f"{k}={v}" for k, v in param.items() if k in param['_repr_params']) for param in params]
# names = [
#     f"d_th={params.dist_thresh}; t_th={params.time_thresh}; u_meth={params.update_method}"
#     for params in configs
# ]
# %% Visualize all tracks
import napari


def color_cycle():
    while True:
        yield "green"
        yield "blue"
        yield "red"
        # yield "red"
        # yield "blue"
        # yield "green"


cycle = color_cycle()

viewer = napari.Viewer()
viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=3)
for name, tracks in zip(names[:], trackss[:]):
    # for track in trackss:
    print(name)
    data, properties, graph = tracks_to_napari(tracks)
    viewer.add_tracks(
        data,
        properties=properties,
        graph=graph,
        name=name,
        visible=False,
        colormap=next(cycle),
        color_by="Roundness",
        tail_length=4,
    )
# %% Visualize individual tracks
roots = [track.root for track in tracks if track.root != track.ID]

root_ids = pl.Series("root_id", roots).value_counts()["root_id"].to_list()

root_trees = get_tree_by_ids(root_ids, tracks)

viewer = napari.Viewer()
viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=3)
viewer.add_tracks(**tree_to_napari(root_trees))

# %%
track = tracks[0]
dir(track)
# %%
from copy import deepcopy
from typing import Any


def remove_tracks(tracks_napari: dict[str, Any], remove: tuple[int]):
    data, features = remove_nodes_from_data_and_features(
        tracks_napari["data"], tracks_napari["properties"], remove
    )
    tracks_out = {
        "data": data,
        "properties": features,
        "graph": remove_nodes_from_graph(tracks_napari["graph"], remove),
    }
    return tracks_out


def remove_nodes_from_data_and_features(data, features, drop=tuple()):
    selector = np.where(~np.isin(data[:, 0], drop))[0]
    data_out = data[selector, :]
    features_out = {k: v[selector] for k, v in features.items()}
    return data_out, features_out


def remove_nodes_from_graph(graph, drop=tuple()):
    graph = {k: v for k, v in deepcopy(graph).items() if k not in drop}
    for lst in graph.values():
        for d in drop:
            if d in lst:
                lst.remove(d)
    return graph


# %%
from pathlib import Path

from btrack.io import HDF5FileHandler
from btrack.utils import tracks_to_napari

fn = r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\tracking\quick.h5"

with HDF5FileHandler(fn, "r") as f:
    tracks = f.tracks


# tracks_napari = dict(zip(['data', 'properties', 'graph'], ))
# tracks_sml = remove_tracks(tracks_napari, list(np.arange(500, 1000)))
data, properties, graph = tracks_to_napari(tracks)

# %%


import napari

viewer = napari.Viewer()
# viewer.add_points(df.select(["t", "z", "y", "x"]).to_numpy(), size=3)
viewer.add_tracks(**tracks_napari)
# viewer.add_labels(lbls, scale=(1.0, 0.65, 0.65))

# %%
t2 = dict(zip(["data", "properties", "graph"], tracks_to_napari(tracks[0])))
# %%
len(graph)
# %%
viewer = napari.Viewer()
viewer.add_points(df_crp.select(["t", "z", "y", "x"]).to_numpy(), size=3)
viewer.add_tracks(data=t2["data"], properties=t2["properties"])
# %%


# %%
viewer = napari.Viewer()
# viewer.add_points(df_crp.select(['t', 'z', 'y', 'x']).to_numpy(), size=3)

viewer.add_tracks(
    **tree_to_napari(tracks),
    name="Tracks",
    blending="translucent",
    visible=True,
)
# viewer.add_points(df_crp.select(['t', 'z', 'y', 'x']).to_numpy(), size=60, opacity=0.1)


# %%
from zfish.visualize.imshow import imshow_spatial_image

root = get_track_by_id(159, tracks)
tree = extract_tree_recursive(root, tracks)
tree_napari = dict(zip(["data", "properties", "graph"], tracks_to_napari(tree)))


viewer = napari.Viewer()
imshow_spatial_image(image.sel(**napari_tree_to_bbx(tree_napari)), viewer)
viewer.add_points(
    df.select(["t", "z", "y", "x"]).filter(napari_tree_to_point_selector(tree_napari)),
    size=3,
)
viewer.add_tracks(**tree_napari)
