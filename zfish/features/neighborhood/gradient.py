# %%
import napari
import networkx as nx
import numpy as np
import polars as pl
from scipy import spatial
from sklearn.neighbors import KDTree

from zfish.features.neighborhood.neighborhoods import (
    Nhd,
    knn_adjacency_matrix,
    radius_adjacency_matrix,
)
from zfish.features.polars_utils import nest_structs, unnest_all_structs
from zfish.visualize.napari import napari_adjacency, napari_centroids

# df = pl.read_csv(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\ccp\Ccp_cycle7-8-9.csv")
df = pl.read_csv(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\ccp\Ccp_cycle10.csv")
df_one = df.filter(pl.col('roi')=='B07_px+2788_py+0295')
# df_one = df.filter(pl.col('roi')=='E03_px+2148_py+0598')
# %%
df.groupby('roi').count().sort(by='count')
# %%
ccp_column = 'NormalizedCcpUmap'
site_dfs = [e for _, e in df.pipe(nest_structs).groupby('roi', maintain_order=True)]
nhds = [Nhd.from_dataframe(d) for d in site_dfs]
aggs = [nhd.radius([10, 20, 30, 40, 50], include_self=True) for nhd in nhds]
agg_dfs = [agg.circmean(ccp_column) for agg in aggs]
full_dfs = [pl.concat([a, b], how='horizontal') for a, b in zip(site_dfs, agg_dfs)]
full_df = pl.concat(full_dfs).pipe(unnest_all_structs)
# df_one = full_df.filter(pl.col('roi')=='E03_px+2148_py+0598')
# %%
columns = full_df.select(f'^.*{ccp_column}.*$').columns
columns = [
    ccp_column,
    f'RADIUS-30s_CircMean__{ccp_column}',
    f'RADIUS-50s_CircMean__{ccp_column}',
    ]
# points = napari_centroids(full_df, centroid_column='CentroidTrans', features=columns[1])

# %%
from typing import Any, Sequence


def circdiff(x, y):
    a = (np.atleast_1d(x) - np.atleast_1d(y)) % (2*np.pi)
    b = (np.atleast_1d(y) - np.atleast_1d(x)) % (2*np.pi)
    return np.where(a<b, -a, b).squeeze()


def compute_cellcycle_gradient(df_one, column: str = 'NormalizedCcp', r=50):
    points = df_one[sorted(df_one.select('^Centroid-[xyz]$').columns, reverse=True)].to_numpy()
    tree = KDTree(points)
    adjacency_matrix = radius_adjacency_matrix(tree, r=r, include_self=False, not_adjacent_value=0)
    vectors = (np.expand_dims(points, 0) - np.expand_dims(points, 1))
    feature = df_one[column].to_numpy()
    feature_diff = circdiff(np.expand_dims(feature, 0), np.expand_dims(feature, 1))
    adjacent_vectors = vectors * np.expand_dims(adjacency_matrix, -1)
    grad = (adjacent_vectors * np.expand_dims(feature_diff, -1)).sum(axis=0)
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_n = grad / norm
    return grad_n, norm

def compute_cellcycle_gradient_knn(df_one, column: str = 'NormalizedCcp', k=10):
    points = df_one[sorted(df_one.select('^Centroid-[xyz]$').columns, reverse=True)].to_numpy()
    tree = KDTree(points)
    adjacency_matrix = knn_adjacency_matrix(tree, k=k, include_self=False, not_adjacent_value=0)
    vectors = (np.expand_dims(points, 0) - np.expand_dims(points, 1))
    feature = df_one[column].to_numpy()
    feature_diff = circdiff(np.expand_dims(feature, 0), np.expand_dims(feature, 1))
    adjacent_vectors = vectors * np.expand_dims(adjacency_matrix, -1)
    grad = (adjacent_vectors * np.expand_dims(feature_diff, -1)).sum(axis=0)
    norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_n = grad / norm
    return grad_n, norm

def compute_cellcycle_gradient_all(df: pl.DataFrame, groupby: str = 'roi', column: str = 'NormalizedCcp', r=50):
    res = []
    for _, d in df.groupby(groupby):
        grad, norm = compute_cellcycle_gradient(d, column=column, r=r)
        grad_z = grad[:, 0]
        grad_y = grad[:, 1]
        grad_x = grad[:, 2]
        res.append(d.select(pl.col('^Centroid.*$'), pl.col(column)).with_columns([
            pl.Series('CcpGradient-z', grad_z.flatten()), 
            pl.Series('CcpGradient-y', grad_y.flatten()), 
            pl.Series('CcpGradient-x', grad_x.flatten()), 
            pl.Series('CcpNorm', norm.flatten()),
            ])
            )
    return pl.concat(res)

def compute_cellcycle_gradient_knn_all(df: pl.DataFrame, groupby: str = 'roi', column: str = 'NormalizedCcp', k=10):
    res = []
    for _, d in df.groupby(groupby):
        grad, norm = compute_cellcycle_gradient_knn(d, column=column, k=k)
        grad_z = grad[:, 0]
        grad_y = grad[:, 1]
        grad_x = grad[:, 2]
        res.append(d.select(pl.col('^Centroid.*$'), pl.col(column)).with_columns([
            pl.Series('CcpGradient-z', grad_z.flatten()), 
            pl.Series('CcpGradient-y', grad_y.flatten()), 
            pl.Series('CcpGradient-x', grad_x.flatten()), 
            pl.Series('CcpNorm', norm.flatten()),
            ])
            )
    return pl.concat(res)

def napari_gradients(
        df, 
        centroid_column: str | Sequence[str] = '^Centroid-.*$', 
        gradient_column: str | Sequence[str] = '^.*Gradient.*$',
        normalize: bool = True,
        color_norm: bool = True,
        translate_column: str | None = None,
        translate_yx: tuple[float, float] = (700.0, 700.0),
        **kwargs,
        ) -> dict[str, Any]:
    more_args = {}
    if normalize or color_norm:
        df = df.with_columns(pl.concat_list(gradient_column).list.eval(pl.all().pow(2).sum().sqrt()).explode().alias('Norm'))
    if normalize:
        df = df.with_columns(pl.col(gradient_column) / pl.col('Norm'))
    if color_norm:
        more_args['features'] = df.select(pl.col('Norm')).to_pandas()
        more_args['edge_color'] = 'Norm'

    centroid_columns = sorted(df.select(pl.col(centroid_column)).columns, reverse=True)
    gradient_columns = sorted(df.select(pl.col(gradient_column)).columns, reverse=True)
    start = df.select(centroid_columns)
    direction = df.select(gradient_columns)
    assert start.shape[1] in [2, 3]
    assert direction.shape == start.shape
    vecs = np.stack([start, direction], axis=1)
    return {
    **{
        'data': vecs,
        'length': 10
    },
    **more_args,
    **kwargs,
    }
# %%
# res = compute_cellcycle_gradient_all(full_df, column='NormalizedCcp')
# select_embryos= full_df['roi'].unique(maintain_order=True).to_numpy()[[5, 12, 15, 19, 31, 32, 36]]
select_embryos= full_df['roi'].unique(maintain_order=True).to_numpy()[[0, 1, 2, 7, 11, 13, 15, 18, 22, 25]]
# select_df = full_df.filter(pl.col('roi').is_in(list(select_embryos)))

vecs = []
ress = []
for column in columns:
    print(column)
    res = compute_cellcycle_gradient_all(full_df, column=column, r=70)
    res = res.with_columns(pl.col('CcpNorm').log1p().alias('LogCcpNorm'))
    res_out = pl.concat([full_df.select(['roi', 'object', 'label']), res], how='horizontal').fill_nan(None).drop_nulls()
    ress.append(res_out)

    res = compute_cellcycle_gradient_knn_all(full_df, column=column, k=10)
    res = res.with_columns(pl.col('CcpNorm').log1p().alias('LogCcpNorm'))
    res_out = pl.concat([full_df.select(['roi', 'object', 'label']), res], how='horizontal').fill_nan(None).drop_nulls()
    ress.append(res_out)
    # res_out.write_csv(f"gradient_{column}.csv")
    # start = res.select(['CentroidTrans-z', 'CentroidTrans-y', 'CentroidTrans-x']).to_numpy()
    # direction = res.select(['CcpGradient-z', 'CcpGradient-y', 'CcpGradient-x']).to_numpy()
    # vec = {
    #     'data': np.nan_to_num(np.stack([start, direction], axis=1)), 
    #     'length': 20, 
    #     'edge_color': 'LogCcpNorm', 
    #     'features': res[['LogCcpNorm']].to_pandas(),
    #     'edge_width': 5,
    #     }
    # vecs.append(vec)

# %%
import napari

ccp_column_p = '^.*NormalizedCcpUmap$'
viewer = napari.Viewer()


@viewer.bind_key('c')
def visible_off(viewer):
    for layer in viewer.layers:
        layer.visible = False

@viewer.bind_key('v')
def visible_on(viewer):
    for layer in viewer.layers:
        layer.visible = True




@viewer.bind_key('n')
def next_layer(viewer, state=[0]):
        active_layer = state[0]
        try:
            viewer.layers[active_layer-2].visible = False
            viewer.layers[active_layer-1].visible = False
            viewer.layers[active_layer].visible = True
            viewer.layers[active_layer+1].visible = True
            state[0] = active_layer + 2
        except IndexError:
            state[0] =  0
        

# viewer.bind_key('n', next_pair)
    

for res in ress:
    ccp_column = res.select(ccp_column_p).columns[0]
    viewer.add_points(**napari_centroids(res, 
                                        centroid_column='^CentroidTrans-[xyz]$', 
                                        features=[ccp_column], 
                                        size=10, 
                                        edge_color=ccp_column,
                                        face_colormap='turbo',
                                        edge_colormap='turbo')
                                        )
    viewer.add_vectors(**napari_gradients(res, centroid_column='^CentroidTrans-[xyz]$', length=20, edge_width=4, opacity=1.0, color_norm=True))

# %%
select_embryos= full_df['roi'].unique(maintain_order=True).to_numpy()[[13]][0]
r = ress[-1]
r_one = r.filter(pl.col('roi')==select_embryos)

ccp_column_p = '^.*NormalizedCcpUmap$'
viewer = napari.Viewer()
ccp_column = r_one.select(ccp_column_p).columns[0]
viewer.add_points(**napari_centroids(r_one), 
                                    centroid_column='^CentroidTrans-[xyz]$', 
                                    features=[ccp_column], 
                                    size=10, 
                                    edge_color=ccp_column,
                                    face_colormap='turbo',
                                    edge_colormap='turbo')
                                    
viewer.add_vectors(**napari_gradients(r_one, centroid_column='^CentroidTrans-[xyz]$', length=20, edge_width=4, opacity=1.0, color_norm=True))
# %%
viewer = napari.Viewer()
viewer.add_points(**napari_centroids(df_pl, 
                                    centroid_column='^CentroidTrans-[xyz]$', 
                                    features=[ccp_column], 
                                    size=10, 
                                    edge_color=ccp_column,
                                    face_colormap='turbo',
                                    edge_colormap='turbo')
                                    )
# %%
one = out.filter(pl.col('roi').str.starts_with('F04_px+208'))
two = out.filter(pl.col('roi').str.starts_with('G02_px+243'))
viewer = napari.Viewer()
viewer.add_points(**napari_centroids(one, 
                                    centroid_column='^Centroid-[xyz]$', 
                                    features=[ccp_column], 
                                    size=10, 
                                    edge_color=ccp_column,
                                    face_colormap='turbo',
                                    edge_colormap='turbo')
                                    )
viewer.add_points(**napari_centroids(one, 
                                    centroid_column='^CentroidTrans-[xyz]$', 
                                    features=[ccp_column], 
                                    size=10, 
                                    edge_color=ccp_column,
                                    face_colormap='turbo',
                                    edge_colormap='turbo')
                                    )
# print(trans)
# # return trans
# out = trans.join(df, on='roi', how='outer').with_columns([
#     (pl.col('Centroid-x') + pl.col('trans_x')).alias('CentroidTrans-x'),
#     (pl.col('Centroid-y') + pl.col('trans_y')).alias('CentroidTrans-y'),
#     pl.col('Centroid-z').alias('CentroidTrans-z'),
# ]).drop(['trans_x', 'trans_y'])
# %%
def translate_on_grid(df, groupby='roi', dx=700.0, dy=700.0, n_rows=None):
    rois = df[groupby].unique(maintain_order=True)
    n_embryos = len(rois)
    print(n_embryos)
    if n_rows is None:
        n_rows = int(np.ceil(np.sqrt(n_embryos)))
    n_cols = int(np.ceil(n_embryos / n_rows))

    trans_x_idx = pl.Series('trans_x', [i for i in range(n_rows) for _ in range(n_cols)][:n_embryos])
    trans_y_idx = pl.Series('trans_y', [i for _ in range(n_rows) for i in range(n_cols)][:n_embryos])
    print(trans_x_idx)
    print(trans_y_idx)

    trans_x = trans_x_idx * dx
    trans_y = trans_y_idx * dy
    trans = pl.concat([rois.to_frame(), trans_x.to_frame(), trans_y.to_frame()], how='horizontal')
    print(trans)
    # return trans
    return trans.join(df, on='roi', how='outer').with_columns([
        (pl.col('Centroid-x') + pl.col('trans_x')).alias('CentroidTrans-x'),
        (pl.col('Centroid-y') + pl.col('trans_y')).alias('CentroidTrans-y'),
        pl.col('Centroid-z').alias('CentroidTrans-z'),
    ]).drop(['trans_x', 'trans_y'])
    # return (
    # df.clone().join(trans, on='roi')
    # .with_columns([
    #     (pl.col('Centroid-x') + pl.col('trans_x')).alias('CentroidTrans-x'),
    #     (pl.col('Centroid-y') + pl.col('trans_y')).alias('CentroidTrans-y'),
    #     pl.col('Centroid-z').alias('CentroidTrans-z'),
    # ]).drop(['trans_x', 'trans_y'])
    # )
# %%
full_df = pl.read_csv(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\neighborhood\gradient_cy7-8-9_RADIUS-50s_CircMean__NormalizedCcpPalantir.csv")
select_embryos= full_df['roi'].unique(maintain_order=True).to_numpy()[[5, 12, 15, 19, 31, 32, 36]]
select_df = full_df.filter(pl.col('roi').is_in(list(select_embryos)))
# %%
import napari

ccp_column = 'NormalizedCcpUmap'
viewer = napari.Viewer()
viewer.add_points(**napari_centroids(translate_on_grid(df, n_rows=7), 
                                     centroid_column='CentroidTrans', 
                                     features=[ccp_column], 
                                     size=10, 
                                     edge_color=ccp_column,
                                     face_colormap='turbo',
                                     edge_colormap='turbo')
                                     )
# viewer.add_vectors(**napari_gradients(translate_on_grid(select_df), centroid_column='^CentroidTrans-[xyz]$', length=10, edge_width=2, color_norm=True))
# %%
viewer = napari.Viewer()
viewer.add_points(**napari_centroids(full_df, 
                                     centroid_column='CentroidTrans', 
                                     features=[ccp_column], 
                                     size=10, 
                                     edge_color=ccp_column,
                                     face_colormap='turbo',
                                     edge_colormap='turbo')
                                     )
for vec in vecs:
    viewer.add_vectors(**vec)
# viewer.add_vectors(**vecs[1])
# viewer.add_vectors(**vecs[2])
# viewer.add_vectors(**vecs[3])
# viewer.add_vectors(**vecs[4])
# %%
from pathlib import Path

import imageio
from PIL import Image

fld = Path(r"C:\Users\hessm\OneDrive\Documents\PhD\Courses\SysBio2023_AdvancedComputationalBiology")
fn = r"C:\Users\hessm\OneDrive\Documents\PhD\Courses\SysBio2023_AdvancedComputationalBiology\CCP_cycle10_single_gradient-{0}.png"
frames = [imageio.imread(fn.format(i)) for i in list(range(6)) + [5, 5, 5, 5, 5] + list(reversed(range(5)))]
frames = [Image.fromarray(frame) for frame in frames]
frames[0].save(fld / "array.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
# %%
viewer = napari.Viewer()
viewer.add_points(**napari_centroids(res, 
                                     centroid_column='CentroidTrans', 
                                     features=[column], 
                                     face_colormap='turbo', 
                                     size=5, 
                                     face_contrast_limits=(0, 2*np.pi)))
start = res.select(['CentroidTrans-z', 'CentroidTrans-y', 'CentroidTrans-x']).to_numpy()
direction = res.select(['CcpGradient-z', 'CcpGradient-y', 'CcpGradient-x']).to_numpy()
vec = {'data': np.stack([start, direction], axis=1), 'length': 10}
viewer.add_vectors(**vec)

# feature = points[:, 1]
# feature = feature / feature.max()

# feature_diff = np.expand_dims(feature, 0) - np.expand_dims(feature, 1)
# %%
# %%
# %%

# %%
viewer = napari.Viewer()
viewer.add_points(points, features={'feature': feature}, face_color='feature', size=5, face_contrast_limits=(0, 2*np.pi), face_colormap='turbo')
viewer.add_vectors(grad_vecs, features={'norm': norm.squeeze()}, edge_color='norm', edge_colormap='viridis', length=10)

# %%

# %%