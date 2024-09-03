# %%
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

if TYPE_CHECKING:
    from polars.type_aliases import SelectorType


# Base coordinate names for fast changes:
IDX = "idx"
WELL = "well"
ROW = "row"
COL = "col"
ROI = "roi"
OBJ = "o"
CH = "c"
STAIN = "stain"
ACQUIS = "acquisition"
MS = "m"
LABEL = "label"
PARENT = "parent"
CHILD = "child"
DIST_TRANSF = "dtf"
NHD = "nhood"

I = IDX  # noqa: E741
SI = "."

PARENT_IDX = (
    f"{I}{SI}{OBJ}{SI}{PARENT}",
    f"{I}{SI}{ROI}",
    f"{I}{SI}{LABEL}{SI}{PARENT}",
)
CHILD_IDX = (
    f"{I}{SI}{OBJ}{SI}{CHILD}",
    f"{I}{SI}{ROI}",
    f"{I}{SI}{LABEL}{SI}{CHILD}",
)

LABEL_OBJECT_IDX = (f"{IDX}{SI}{OBJ}", f"{IDX}{SI}{ROI}", f"{IDX}{SI}{LABEL}")


CAT_TYPE = pl.Categorical(ordering="lexical")

IDX_SEL = cs.starts_with(f"{IDX}{SI}")
LABEL_OBJECT_SEL = cs.by_name(LABEL_OBJECT_IDX)
SEL_IDX = IDX_SEL
SEL_LABEL_OBJECT = LABEL_OBJECT_SEL
SEL_O = cs.starts_with(f"{IDX}{SI}{OBJ}")
SEL_ROI = cs.starts_with(f"{IDX}{SI}{ROI}")
SEL_LABEL = cs.starts_with(f"{IDX}{SI}{LABEL}")
SEL_PARENT = cs.by_name(PARENT_IDX)
SEL_CHILD = cs.by_name(CHILD_IDX)


@dataclass
class PolarsSelector:
    idx: "SelectorType" = SEL_IDX
    roi: "SelectorType" = SEL_ROI
    object_type: "SelectorType" = SEL_O
    label: "SelectorType" = SEL_LABEL
    label_object: "SelectorType" = SEL_LABEL_OBJECT
    parent: "SelectorType" = SEL_PARENT
    child: "SelectorType" = SEL_CHILD
    empty: "SelectorType" = ~cs.all()


sel = PolarsSelector()


def build_lazy_tables(schema):
    tables = {}
    for name, table_schema in schema.items():
        tables[name] = pl.LazyFrame(schema=table_schema)
    return tables


def build_schema():
    schema = {}

    schema["resources.wells"] = {
        f"{I}{SI}{WELL}": CAT_TYPE,  # PK
        "row": CAT_TYPE,
        "col": CAT_TYPE,
        "is_control_well": pl.Boolean,
        "control_well_for_acquisition": pl.UInt16,
        "translate.plate.x": pl.Float64,
        "translate.plate.y": pl.Float64,
    }

    schema["resources.rois"] = {
        f"{I}{SI}{ROI}": CAT_TYPE,  # PK
        "well": CAT_TYPE,  # FK(wells)
        "site": CAT_TYPE,
        f"{ROI}.path": pl.String,
        "translate.well.x": pl.Float64,
        "translate.well.y": pl.Float64,
    }

    schema["resources.channels"] = {
        f"{I}{SI}{CH}": CAT_TYPE,  # PK
        "stain": CAT_TYPE,
        "acquisition": pl.UInt8,
        "wavelength": pl.UInt16,
        "contrast_limits": pl.List(
            pl.Float64
        ),  # CANNOT USE ARRAY HERE, breaks upon read/write!!!!
    }

    schema["resources.object_types"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,  # PK
        "hierarchy_level": pl.UInt8,
        "parents": pl.List(CAT_TYPE),
    }

    schema["resources.multiscale_levels"] = {
        f"{I}{SI}{MS}": pl.UInt8,  # PK
        "scale.z": pl.Float64,
        "scale.y": pl.Float64,
        "scale.x": pl.Float64,
    }

    schema["resources.images"] = {
        f"{I}{SI}{MS}": pl.UInt8,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{CH}": CAT_TYPE,
        # "t": pl.Datetime | pl.UInt32,
        f"{CH}.path": CAT_TYPE,
        # PK (m, roi, c, [t])
        # FK (m -> ms.m), (roi -> rois.roi), (c -> cs.c) ...
    }

    schema["resources.xy_model_corrs"] = {
        f"{I}{SI}wavelength": pl.UInt16,
        "path.xy_model_corr": CAT_TYPE,
    }

    # TODO: remove 'z_model.' for metadata columns?
    schema["resources.z_models"] = {
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}z_model": CAT_TYPE,
        "z_model.full_name": CAT_TYPE,
        "z_model.type": CAT_TYPE,
        f"z_model.{OBJ}": CAT_TYPE,
        "z_model.dtf": CAT_TYPE,
        "z_model.ndim": pl.UInt8,
        "z_model.path": pl.String,
        # "z_model.params": pl.Struct(
        #     {
        #         "features": pl.List(pl.String),
        #         "loss": pl.Enum(["huber", "linear"]),
        #         "pos_offset": pl.Boolean,
        #     }
        # ),
    }

    schema["resources.t_models"] = {
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}t_model": CAT_TYPE,
        "t_model.type": CAT_TYPE,
        "t_model.correction_factor": pl.Float64,
        "delta_t_min": pl.Float64,
    }

    schema["resources.t_model_corrs"] = {
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}t_model": CAT_TYPE,
        "t_model.type": CAT_TYPE,
        "correction_factor": pl.Float64,
        "delta_t_min": pl.Float64,
    }

    schema["resources.label_images"] = {
        f"{I}{SI}{MS}": pl.UInt8,
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        # "t": pl.Datetime | pl.UInt32,
        f"{OBJ}.path": pl.String,
        "resample_scale_factors": pl.List(pl.Float64),
        # PK (m, o, roi, [t])
        # FK as one would expect
    }

    schema["resources.label_objects"] = {
        f"{I}{SI}{MS}": pl.UInt8,
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        "bbx.z.lower": pl.Int32,
        "bbx.z.upper": pl.Int32,
        "bbx.y.lower": pl.Int32,
        "bbx.y.upper": pl.Int32,
        "bbx.x.lower": pl.Int32,
        "bbx.x.upper": pl.Int32,
        "centroid.z": pl.Float64,
        "centroid.y": pl.Float64,
        "centroid.x": pl.Float64,
        # UNIQUE (o, roi, label) -> more explicit, better PK?
    }

    schema["resources.hierarchy"] = {
        # child and parent share roi
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{OBJ}{SI}{PARENT}": CAT_TYPE,
        f"{I}{SI}{LABEL}{SI}{PARENT}": pl.UInt32,
        f"{I}{SI}{OBJ}{SI}{CHILD}": CAT_TYPE,
        f"{I}{SI}{LABEL}{SI}{CHILD}": pl.UInt32,
    }

    schema["resources.classifiers"] = {
        f"{I}{SI}clf_name": CAT_TYPE,
        f"{I}{SI}{OBJ}": CAT_TYPE,
        "classes": pl.List(CAT_TYPE),
        "annotation.total_Count": pl.Int32,
        "annotation.classes_Count": pl.List(
            pl.Struct({"annotation": CAT_TYPE, "count": pl.Int32})
        ),
        "pred.total_Count": pl.Int32,
        "pred.classes_Count": pl.List(
            pl.Struct({"annotation": CAT_TYPE, "count": pl.Int32})
        ),
    }

    schema["features.label"] = {
        # "id": pl.UInt32,
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        "PhysicalSize": pl.Float64,
        "Elongation": pl.Float64,
        "Flatness": pl.Float64,
        "Roundness": pl.Float64,
        "FeretDiameter": pl.Float64,
        "Perimeter": pl.Float64,
        "EquivalentSphericalPerimeter": pl.Float64,
        "EquivalentSphericalRadius": pl.Float64,
        "PerimeterOnBorder": pl.Float64,
        "PerimeterOnBorderRatio": pl.Float64,
        "EquivalentEllipsoidDiameter-a": pl.Float64,
        "EquivalentEllipsoidDiameter-b": pl.Float64,
        "EquivalentEllipsoidDiameter-c": pl.Float64,
        "OrientedBoundingBoxSize-a": pl.Float64,
        "OrientedBoundingBoxSize-b": pl.Float64,
        "OrientedBoundingBoxSize-c": pl.Float64,
        "PrincipalAxes-a-x": pl.Float64,
        "PrincipalAxes-a-y": pl.Float64,
        "PrincipalAxes-a-z": pl.Float64,
        "PrincipalAxes-b-x": pl.Float64,
        "PrincipalAxes-b-y": pl.Float64,
        "PrincipalAxes-b-z": pl.Float64,
        "PrincipalAxes-c-x": pl.Float64,
        "PrincipalAxes-c-y": pl.Float64,
        "PrincipalAxes-c-z": pl.Float64,
        "OrientedBoundingBoxDirection-a-x": pl.Float64,
        "OrientedBoundingBoxDirection-a-y": pl.Float64,
        "OrientedBoundingBoxDirection-a-z": pl.Float64,
        "OrientedBoundingBoxDirection-b-x": pl.Float64,
        "OrientedBoundingBoxDirection-b-y": pl.Float64,
        "OrientedBoundingBoxDirection-b-z": pl.Float64,
        "OrientedBoundingBoxDirection-c-x": pl.Float64,
        "OrientedBoundingBoxDirection-c-y": pl.Float64,
        "OrientedBoundingBoxDirection-c-z": pl.Float64,
        "OrientedBoundingBoxVertices-a-x": pl.Float64,
        "OrientedBoundingBoxVertices-a-y": pl.Float64,
        "OrientedBoundingBoxVertices-a-z": pl.Float64,
        "OrientedBoundingBoxVertices-b-x": pl.Float64,
        "OrientedBoundingBoxVertices-b-y": pl.Float64,
        "OrientedBoundingBoxVertices-b-z": pl.Float64,
        "OrientedBoundingBoxVertices-c-x": pl.Float64,
        "OrientedBoundingBoxVertices-c-y": pl.Float64,
        "OrientedBoundingBoxVertices-c-z": pl.Float64,
        "Centroid-x": pl.Float64,
        "Centroid-y": pl.Float64,
        "Centroid-z": pl.Float64,
        "OrientedBoundingBoxOrigin-x": pl.Float64,
        "OrientedBoundingBoxOrigin-y": pl.Float64,
        "OrientedBoundingBoxOrigin-z": pl.Float64,
        "BoundingBox-x-lower": pl.Int32,
        "BoundingBox-x-upper": pl.Int32,
        "BoundingBox-y-lower": pl.Int32,
        "BoundingBox-y-upper": pl.Int32,
        "BoundingBox-z-lower": pl.Int32,
        "BoundingBox-z-upper": pl.Int32,
        # PK (id)
        # UNIQUE (o, roi, label) -> alternative PK
        # FK (id -> label_objects.id), ...
    }

    schema["features.intensity"] = {
        # "id": pl.UInt32,
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        "Mean": pl.Float64,
        "Median": pl.Float64,
        "Minimum": pl.Float64,
        "Maximum": pl.Float64,
        "Sum": pl.Float64,
        "Variance": pl.Float64,
        "StandardDeviation": pl.Float64,
        "Skewness": pl.Float64,
        "Kurtosis": pl.Float64,
        "WeightedElongation": pl.Float64,
        "WeightedFlatness": pl.Float64,
        "CenterOfGravity-x": pl.Float64,
        "CenterOfGravity-y": pl.Float64,
        "CenterOfGravity-z": pl.Float64,
        "WeightedPrincipalAxes-a-x": pl.Float64,
        "WeightedPrincipalAxes-a-y": pl.Float64,
        "WeightedPrincipalAxes-a-z": pl.Float64,
        "WeightedPrincipalAxes-b-x": pl.Float64,
        "WeightedPrincipalAxes-b-y": pl.Float64,
        "WeightedPrincipalAxes-b-z": pl.Float64,
        "WeightedPrincipalAxes-c-x": pl.Float64,
        "WeightedPrincipalAxes-c-y": pl.Float64,
        "WeightedPrincipalAxes-c-z": pl.Float64,
        "WeightedPrincipalMoments-a": pl.Float64,
        "WeightedPrincipalMoments-b": pl.Float64,
        "WeightedPrincipalMoments-c": pl.Float64,
        "MaximumIndex-x": pl.Int32,
        "MaximumIndex-y": pl.Int32,
        "MaximumIndex-z": pl.Int32,
        "MinimumIndex-x": pl.Int32,
        "MinimumIndex-y": pl.Int32,
        "MinimumIndex-z": pl.Int32,
        # PK (id, c) each id (labelobject) can have multiple rows
        # for measurements in different channel
        # FK (id -> label_objects.id), (c -> cs.c), ...
    }

    schema["features.correlation"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}{CH}{SI}0": CAT_TYPE,
        f"{I}{SI}{CH}{SI}1": CAT_TYPE,
        "PearsonR": pl.Float64,
        "SpearmanR": pl.Float64,
        "KendallTau": pl.Float64,
    }

    schema["features.distance"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}{OBJ}{SI}to": CAT_TYPE,
        f"{I}{SI}{LABEL}{SI}to": pl.UInt32,
        f"{I}{SI}{DIST_TRANSF}": CAT_TYPE,
        "Centroid": pl.Float64,
        "Maximum": pl.Float64,
        "Minimum": pl.Float64,
        "MaximumIndex-z": pl.Int32,
        "MaximumIndex-y": pl.Int32,
        "MaximumIndex-x": pl.Int32,
        "MinimumIndex-z": pl.Int32,
        "MinimumIndex-y": pl.Int32,
        "MinimumIndex-x": pl.Int32,
    }

    schema["features.density_count"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}{NHD}": pl.String,
        "Count": pl.UInt32,
    }

    schema["features.density_distance"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}{NHD}": pl.String,
        "Max": pl.Float64,
        "Mean": pl.Float64,
    }

    schema["features.classifier"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}clf_name": CAT_TYPE,
        "annotation": CAT_TYPE,
        "pred": CAT_TYPE,
        "probas": pl.List(pl.Struct({"annotation": CAT_TYPE, "proba": pl.Float64})),
    }
    return schema


EMPTY_TABLES = build_lazy_tables(build_schema())

# %%
