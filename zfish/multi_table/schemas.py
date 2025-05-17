# %%
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import polars as pl
import polars.selectors as cs

if TYPE_CHECKING:
    from zfish.preprocessing.types import SelectorType


TABLE_NAME_REGEX = re.compile(
    r"^(?P<hidden>_)?(?P<multi_table>[A-Z]\w*)\.(?P<table>\w*)(\.v(?P<version>\d\d?))?$"
)


# Base coordinate names for fast changes:
IDX = "idx"
FIDX = "fidx"
SIDX = "sidx"
MIDX = "midx"
I = IDX  # noqa: E741

DIM = "dim"
D = DIM


SI = "."
I_ = f"{I}{SI}"
FI_ = f"{FIDX}{SI}"
D_ = f"{DIM}{SI}"

# Coordinate systems (CS)
ROOT = "root"
DATASET = "dset"
PLATE = "plate"
WELL = "well"
ROW = "row"
COL = "col"
SITE = "site"  # microscopy site, ~ fixed size
ROI = "roi"
LABEL_OBJ = "label_object"

REGION = "region"  # -> CS + bounds


T = "t"
Z = "z"
Y = "y"
X = "x"
MS = "m"
IT = "it"
IZ = "iz"
IY = "iy"
IX = "ix"
CH = "c"
OBJ = "o"
LABEL = "label"


FTYPE = "ftype"
FEATURE = "f"
STAIN = "stain"
ACQUIS = "acquisition"
PARENT = "parent"
CHILD = "child"
DIST_TRANSF = "dtf"
NHD = "nhood"

_LABEL_OBJ = f":{LABEL_OBJ}"

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


CAT_TYPE: TypeAlias = pl.Categorical(ordering="lexical")

IDX_SEL = cs.starts_with(f"{IDX}{SI}")
FIDX_SEL = cs.starts_with(f"{FIDX}{SI}")
SIDX_SEL = cs.starts_with(f"{SIDX}{SI}")

LABEL_OBJECT_SEL = cs.by_name(LABEL_OBJECT_IDX)
SEL_IDX = IDX_SEL
SEL_LABEL_OBJECT = LABEL_OBJECT_SEL
SEL_O = cs.starts_with(f"{IDX}{SI}{OBJ}")
SEL_ROI = cs.starts_with(f"{IDX}{SI}{ROI}")
SEL_LABEL = cs.starts_with(f"{IDX}{SI}{LABEL}")
SEL_PARENT = cs.by_name(PARENT_IDX)
SEL_CHILD = cs.by_name(CHILD_IDX)


@dataclass(frozen=True, slots=True)
class PolarsSelector:
    idx: "SelectorType" = SEL_IDX
    roi: "SelectorType" = SEL_ROI
    dim: "SelectorType" = cs.starts_with(DIM)
    object_type: "SelectorType" = SEL_O
    label: "SelectorType" = SEL_LABEL
    label_object: "SelectorType" = SEL_LABEL_OBJECT
    parent: "SelectorType" = SEL_PARENT
    child: "SelectorType" = SEL_CHILD
    empty: "SelectorType" = ~cs.all()


sel = PolarsSelector()


def build_lazy_tables(schema, include_hidden=True):
    tables = {}
    for name, table_schema in schema.items():
        if not include_hidden:
            if name.startswith("_"):
                continue
        tables[name] = pl.LazyFrame(schema=table_schema)
    return tables


D = "dim"
D_ = f"{D}{SI}"

R = "arr"
R_ = f"{R}{SI}"

I = "idx"
I_ = f"{I}{SI}"


SPATIAL_DIM = pl.Enum(
    [
        "t",
        "z",
        "y",
        "x",
    ]
)
MULTISCALE_DIM = pl.Enum(["m"])
SAMPLED_DIM = pl.Enum(
    [
        "it",
        "iz",
        "iy",
        "ix",
    ]
)
CHANNEL_DIM = pl.Enum(["c", "intensity", "z_model", "t_model"])
OBJECT_TYPE_DIM = pl.Enum(["o", "label", "binary", "dtf"])

BASE_DIM = (
    SPATIAL_DIM.union(SAMPLED_DIM)
    .union(MULTISCALE_DIM)
    .union(CHANNEL_DIM)
    .union(OBJECT_TYPE_DIM)
)

COORD_DIM = pl.Enum(
    [
        "plate",
        "well",
        "roi",
        "label_object",
    ]
)

RASTER_DIM = pl.Enum(
    [
        "image",
        "label_image",
        "binary_mask",
        "distance_transform",
    ]
)

PROCESSING_DIM = pl.Enum(
    [
        "pyramid_creation",
        "registration",
        "segmentation",
        "hierarchy",
        "decay_model",
        "clf_model",
        "ccp_model",
        "reg_model",
        "nhood",
        "feature",
    ]
)

COMPOUND_DIM = COORD_DIM.union(RASTER_DIM).union(PROCESSING_DIM)

ALL_DIM_ENUM = BASE_DIM.union(COMPOUND_DIM)

DIM_TYPE_ENUM = pl.Enum(
    [
        "nominal",
        "ordinal",
        "interval",
        "ratio",
    ]
)


def build_schema(include_hidden=True):
    schema = {}

    schema["_Dims.base"] = {
        f"{D}": ALL_DIM_ENUM,
        "type": DIM_TYPE_ENUM,
        "long_name": pl.String,
        "unit": pl.String,
    }

    schema["_Dims.compound"] = {
        f"{D}": COMPOUND_DIM,
        "type": pl.Enum(["nominal", "ordinal", "hierarchical"]),
        f"{D}.components": pl.List(BASE_DIM),
        "type.components": pl.List(pl.Enum(["nominal", "ordinal"])),
    }

    schema["_Dim.resources.raster"] = {
        f"{D}": pl.Enum(["o", "c", "roi", "well", "plate", "dsets"]),
        "long_name": pl.String,
        "type.base": pl.Enum(["nominal", "ordinal"]),
        "type.sub": pl.Enum(["hierarchical"]),
        "cardinality": pl.Int32,
        "values": pl.List(pl.String),
    }
    schema["_Dim.resources.model"] = {
        f"{D}": pl.Enum(
            ["t_model", "xy_model", "z_model", "clf_model", "ccp_model", "reg_model"]
        )
    }
    schema["_Dim.resources.compound"] = {
        f"{D}": pl.Enum(
            ["label_object", "label_image", "hierarchy", "image", "features"]
        ),
        "long_name": pl.String,
        "type.base": pl.Enum(["compound"]),
    }
    schema["_Dim.continuous"] = {
        f"{D}": pl.Enum(["x", "y", "z", "t"]),
        "long_name": pl.String,
        "type.base": pl.Enum(["interval", "ratio"]),
        "type.sub": pl.Enum(["spatial", "temporal"]),
        "bound.upper": pl.Float64,
        "bound.lower": pl.Float64,
        "unit": pl.String,
    }
    schema["_Dim.icontinuous"] = {
        f"{D}": pl.Enum(["ix", "iy", "iz", "it", "m"]),
        "long_name": pl.String,
        "type.base": pl.Enum(["ordinal"]),
        "type.sub": pl.Enum(["ispatial", "itemporal", "multiscale"]),
        "slice.lower": pl.Int32,
        "slice.upper": pl.Int32,
        "unit": pl.String,
    }

    schema["_Mark.subets"] = {
        "idx.subset": CAT_TYPE,
        "type": pl.Enum(["outliers", "stratum", "class_prediction", "cv_folds"]),
        "feature": pl.String,
    }

    schema["_Mark.outliers"] = {
        "idx.outlier": CAT_TYPE,
    }

    schema["_Resources.roots.v2"] = {
        f"{I}{SI}{ROOT}": CAT_TYPE,
        "t.start": pl.Datetime,
        "path": pl.String,
        "name": pl.String,
        "description": pl.String,
    }

    schema["Resources.plates.v2"] = {
        f"{I}{SI}{PLATE}": CAT_TYPE,
        "path": pl.String,
        "description": pl.String,
    }

    schema["Resources.wells"] = {
        f"{I}{SI}{WELL}": CAT_TYPE,  # PK
        "row": CAT_TYPE,
        "col": CAT_TYPE,
        "is_control_well": pl.Boolean,
        "control_well_for_acquisition": pl.UInt16,
        "translate.plate.x": pl.Float64,
        "translate.plate.y": pl.Float64,
    }

    schema["Resources.wells.v2"] = {
        f"{I}{SI}{WELL}": CAT_TYPE,  # PK
        f"{FI_}{PLATE}": CAT_TYPE,  # FK
        "row": CAT_TYPE,
        "col": CAT_TYPE,
        "is_control_well": pl.Boolean,
        "control_well_for_acquisition": pl.UInt16,
        "plate.x.translate": pl.Float64,
        "plate.y.translate": pl.Float64,
        "plate.z.translate": pl.Float64,
    }

    schema["Resources.rois"] = {
        f"{I}{SI}{ROI}": CAT_TYPE,  # PK
        "well": CAT_TYPE,  # FK(wells)
        "site": CAT_TYPE,  # FK(wells)
        f"{ROI}.path": pl.String,
        # "well": CAT_TYPE,
        "translate.well.x": pl.Float64,
        "translate.well.y": pl.Float64,
        "translate.well.z": pl.Float64,
    }

    schema["Resources.rois.v2"] = {
        f"{I}{SI}{ROI}": CAT_TYPE,  # PK
        "domain.z.lower": pl.Float64,
        "domain.z.upper": pl.Float64,
        "domain.y.lower": pl.Float64,
        "domain.y.upper": pl.Float64,
        "domain.x.lower": pl.Float64,
        "domain.x.upper": pl.Float64,
        f"{FI_}well": CAT_TYPE,  # FK(wells)
        # "well": CAT_TYPE,
        "well.x.translate": pl.Float64,
        "well.y.translate": pl.Float64,
        "well.z.translate": pl.Float64,
        "path": pl.String,
    }

    schema["_Resources.acquisition"] = {
        f"{I}{SI}{ACQUIS}": pl.UInt8,  # PK
        f"{ROI}.path": pl.String,
    }

    schema["_Resources.coordinate_systems"] = {
        f"{I_}{ROOT}": CAT_TYPE,
        f"{I_}{PLATE}": CAT_TYPE,
        f"{I_}{WELL}": CAT_TYPE,
        f"{I_}{ROI}{_LABEL_OBJ}": CAT_TYPE,
        f"{I_}{OBJ}{_LABEL_OBJ}": CAT_TYPE,
        f"{I_}{SI}{LABEL}{_LABEL_OBJ}": pl.UInt32,
        f"{D_}{Z}.lower": pl.Float64,
        f"{D_}{Z}.upper": pl.Float64,
        f"{D_}{Y}.lower": pl.Float64,
        f"{D_}{Y}.upper": pl.Float64,
        f"{D_}{X}.lower": pl.Float64,
        f"{D_}{X}.upper": pl.Float64,
    }

    {
        "idx.root": pl.Categorical(ordering="lexical"),
        "idx.plate": pl.Categorical(ordering="lexical"),
        "idx.well": pl.Categorical(ordering="lexical"),
        "idx.roi:label_object": CAT_TYPE,
        "idx.o:label_object": CAT_TYPE,
        "idx.lbl:label_object": CAT_TYPE,
        "dim.z.lower": pl.Float64,
        "dim.z.upper": pl.Float64,
        "dim.y.lower": pl.Float64,
        "dim.y.upper": pl.Float64,
        "dim.x.lower": pl.Float64,
        "dim.x.upper": pl.Float64,
    }

    schema["Resources.channels"] = {
        f"{I}{SI}{CH}": CAT_TYPE,  # PK
        "stain": CAT_TYPE,
        "acquisition": pl.UInt8,
        "wavelength": pl.UInt16,
        "contrast_limits": pl.List(
            pl.Float64
        ),  # CANNOT USE ARRAY HERE, breaks upon read/write!!!!
    }

    schema["Resources.object_types"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,  # PK
        "hierarchy_level": pl.UInt8,
        "parents": pl.List(CAT_TYPE),
    }

    schema["Resources.multiscale_levels"] = {
        f"{I}{SI}{MS}": pl.UInt8,  # PK
        "scale.z": pl.Float64,
        "scale.y": pl.Float64,
        "scale.x": pl.Float64,
    }

    schema["Resources.images"] = {
        f"{I}{SI}{MS}": pl.UInt8,
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        # "t": pl.Datetime | pl.UInt32,
        f"{ROI}.path": CAT_TYPE,
        # PK (m, roi, c, [t])
        # FK (m -> ms.m), (roi -> rois.roi), (c -> cs.c) ...
    }

    schema["_Resources.xy_model_corrs"] = {
        f"{I}{SI}wavelength": pl.UInt16,
        "path": CAT_TYPE,
    }

    schema["_Params.decay_model"] = {
        "idx.model": CAT_TYPE,
        "type": pl.Enum(["xy", "z", "t"]),
        "ndim": pl.Enum(["1D", "2D", "3D"]),
        "loss": pl.Enum(["linear", "huber"]),
        "features.intensity": pl.Enum(["Mean", "Median", "Sum"]),
        "coords.names": pl.List(pl.String),
        "Resources.dft": pl.List(pl.String),
        # 2D  and 3D only
        "fidx.o": CAT_TYPE,
        "fidx.labels": pl.List(pl.UInt32),  # len == 0 -> 1D ...
    }

    # TODO: remove 'z_model.' for metadata columns?
    schema["Resources.z_models"] = {
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}model": CAT_TYPE,
        "path": pl.String,
        # "z_model.params": pl.Struct(
        #     {
        #         "features": pl.List(pl.String),
        #         "loss": pl.Enum(["huber", "linear"]),
        #         "pos_offset": pl.Boolean,
        #     }
        # ),
    }

    schema["Resources.t_models"] = {
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}model": CAT_TYPE,
        "correction_factor": pl.Float64,
        "delta_t_min": pl.Float64,
    }

    schema["Resources.t_model_corrs"] = {
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}model": CAT_TYPE,
        "correction_factor": pl.Float64,
        "delta_t_min": pl.Float64,
    }

    schema["Resources.label_images"] = {
        f"{I}{SI}{MS}": pl.UInt8,
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        # "t": pl.Datetime | pl.UInt32,
        f"{OBJ}.path": pl.String,
        "resample_scale_factors": pl.List(pl.Float64),
        # PK (m, o, roi, [t])
        # FK as one would expect
    }

    schema["Resources.label_objects"] = {
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

    schema["Resources.label_objects.v2"] = {
        "idx.roi": CAT_TYPE,
        "idx.o": CAT_TYPE,
        "idx.roi.o.lbl": pl.UInt32,
        "lbl_centroid.z": pl.Float64,
        "lbl_centroid.y": pl.Float64,
        "lbl_centroid.x": pl.Float64,
        "domain.z.lower": pl.Float64,
        "domain.z.upper": pl.Float64,
        "domain.y.lower": pl.Float64,
        "domain.y.upper": pl.Float64,
        "domain.x.lower": pl.Float64,
        "domain.x.upper": pl.UInt8,
        "idx.m": pl.UInt8,
        "slice.m.iz.lower": pl.Int32,
        "slice.m.iz.upper": pl.Int32,
        "slice.m.iy.lower": pl.Int32,
        "slice.m.iy.upper": pl.Int32,
        "slice.m.ix.lower": pl.Int32,
        "slice.m.ix.upper": pl.Int32,
    }

    schema["Resources.hierarchy"] = {
        # child and parent share roi
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{OBJ}{SI}{PARENT}": CAT_TYPE,
        f"{I}{SI}{LABEL}{SI}{PARENT}": pl.UInt32,
        f"{I}{SI}{OBJ}{SI}{CHILD}": CAT_TYPE,
        f"{I}{SI}{LABEL}{SI}{CHILD}": pl.UInt32,
    }

    schema["Resources.classifiers"] = {
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

    schema["_Resources.neighborhood.types"] = {
        f"{I_}{NHD}": pl.String,
    }

    schema["Features.label"] = {
        # "id": pl.UInt32,
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        # Shape
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
        # Orientation
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

    schema["Features.intensity"] = {
        # "id": pl.UInt32,
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{CH}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        # Scalar
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
        # Position
        "CenterOfGravity-x": pl.Float64,
        "CenterOfGravity-y": pl.Float64,
        "CenterOfGravity-z": pl.Float64,
        # Orientation
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

    schema["Features.correlation"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}{CH}{SI}0": CAT_TYPE,
        f"{I}{SI}{CH}{SI}1": CAT_TYPE,
        "PearsonR": pl.Float64,
        "SpearmanR": pl.Float64,
        "KendallTau": pl.Float64,
    }

    schema["Features.distance"] = {
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

    schema["Features.density_count"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}{NHD}": pl.String,
        "Count": pl.UInt32,
    }

    schema["Features.density_distance"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}{NHD}": pl.String,
        "Max": pl.Float64,
        "Mean": pl.Float64,
    }

    schema["Features.classifier"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}clf_name": CAT_TYPE,
        "annotation": CAT_TYPE,
        "pred": CAT_TYPE,
        "probas": pl.List(pl.Struct({"annotation": CAT_TYPE, "proba": pl.Float64})),
    }

    schema["_Features.ccp"] = {
        f"{I}{SI}{OBJ}": CAT_TYPE,
        f"{I}{SI}{ROI}": CAT_TYPE,
        f"{I}{SI}{LABEL}": pl.UInt32,
        f"{I}{SI}run_id": pl.String,
        f"{I}{SI}id": pl.Int32,
        "CCP": pl.Float64,
        "NormalizedCCP": pl.Float64,
    }
    if not include_hidden:
        return {k: v for k, v in schema.items() if not k.startswith("_")}
    return schema


SCHEMA_HIDDEN = build_schema()
SCHEMA = build_schema(include_hidden=False)
LAZY_TABLES_EMPTY_HIDDEN = build_lazy_tables(SCHEMA_HIDDEN)
LAZY_TABLES_EMPTY = build_lazy_tables(SCHEMA)
LAZY_SCHEMA_EMPTY = LAZY_TABLES_EMPTY
