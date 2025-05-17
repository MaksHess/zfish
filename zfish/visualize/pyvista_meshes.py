from typing import TYPE_CHECKING

import itk
import np
import pyvist as pv
import vtk

from zfish.abbott_legacy.conversions import to_itk, to_numpy

if TYPE_CHECKING:
    import h5py
    import polars as pl
    import Union


def to_vtk(img):
    return itk.vtk_image_from_image(to_itk(img))


def discrete_marching_cubes(
    lbl_img: Union[h5py.Dataset, itk.Image], smooth_iter=0, calculate_neighbors=False
):
    labels = np.unique(to_numpy(lbl_img))[1:]
    lbl_img = to_vtk(to_itk(lbl_img))

    discrete = vtk.vtkDiscreteMarchingCubes()
    discrete.SetInputData(lbl_img)
    discrete.SetComputeAdjacentScalars(calculate_neighbors)
    for i, l in enumerate(labels):
        discrete.SetValue(i, l)
    discrete.Update()
    return pv.wrap(discrete.GetOutput()).smooth(smooth_iter)


def smooth_discrete_marching_cubes(
    lbl_img: Union[h5py.Dataset, itk.Image],
    smoothing_iterations=0,
    calculate_neighbors=False,
    use_flyingedges=True,
):
    labels = np.unique(to_numpy(lbl_img))[1:]
    lbl_img = to_vtk(to_itk(lbl_img))

    if use_flyingedges:
        discrete = vtk.vtkDiscreteFlyingEdges3D()
    else:
        discrete = vtk.vtkDiscreteMarchingCubes()
        discrete.SetComputeAdjacentScalars(calculate_neighbors)

    discrete.SetInputData(lbl_img)
    for i, l in enumerate(labels):
        discrete.SetValue(i, l)

    pass_band = 0.001
    feature_angle = 120.0

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(discrete.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return pv.wrap(smoother.GetOutput())


def discrete_marching_cubes_vtk(
    lbl_img: Union[h5py.Dataset, itk.Image], smooth_iter=0, calculate_neighbors=False
):
    labels = np.unique(to_numpy(lbl_img))[1:]
    lbl_img = to_vtk(to_itk(lbl_img))

    discrete = vtk.vtkDiscreteMarchingCubes()
    discrete.SetInputData(lbl_img)
    discrete.SetComputeAdjacentScalars(calculate_neighbors)
    for i, l in enumerate(labels):
        discrete.SetValue(i, l)
    discrete.Update()
    return discrete.GetOutput()


MESH_COMPONENTS = ("points", "cells")


def attach_scalars_to_mesh(
    mesh: pv.PolyData,
    df: pl.DataFrame,
    mesh_label_array: str = "Scalars",
    df_label_column: str = "label",
    features: tuple[str, ...] | None = None,
    target: Literal["points", "cells"] = "points",
    inplace: bool = True,
):
    if not inplace:
        mesh = mesh.copy()
    if target not in MESH_COMPONENTS:
        raise ValueError(f"Target must be one of {MESH_COMPONENTS}")
    data = mesh.point_data if target == MESH_COMPONENTS[0] else mesh.cell_data
    if mesh_label_array not in data:
        raise ValueError(
            f"Mesh {target[:-1]}_array does not contain: {mesh_label_array!r}"
        )
    if df_label_column not in df.columns:
        raise ValueError(f"df does not contain column: {df_label_column!r}")
    if features is None:
        features = [e for e in df.columns if e != df_label_column]
    else:
        if not all([e in df for e in features]):
            raise ValueError(f"df doesn not contain all of: {features}")

    for feature in features:
        f_map = (
            df.select(df_label_column, feature)
            .to_pandas()
            .set_index(df_label_column)[feature]
            # .to_dict()
        )
        data[feature] = vec_translate(data[mesh_label_array], f_map)
    return mesh


def vec_translate(a, d):
    return np.vectorize(d.__getitem__)(a)
