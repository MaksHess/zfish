# %%
import re
from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from zfish.multi_table.schemas_id_v2 import FrameId
from zfish.preprocessing.types import ColName, QualColName, QualFrameName

if TYPE_CHECKING:
    from zfish.preprocessing.types import SelectorType
from functools import reduce
from operator import or_

from zfish.features.constants import (
    ColocalizationFeature,
    DistanceFeature,
    DistanceFunction,
    IntensityFeature,
    LabelFeature,
)

TABLE_NAME_REGEX = re.compile(
    r"^(?P<hidden>_)?(?P<multi_table>[A-Z]\w*)\.(?P<table>\w*)(\.v(?P<version>\d\d?))?$"
)
VALID_CHARS = "\w"

PK_ANNOTATION = "<pk></pk>"
FK_ANNOATION = "<fk>{col_to}</fk>"
FK_REGEX = re.compile(r"<fk>(?P<fk_column>[^/]*)</fk>")
PK_REGEX = re.compile(r"<pk></pk>")


PK = PK_ANNOTATION


def FK(col_to: str) -> str:
    """Add foreign key annotation."""
    return f"{FK_ANNOATION.format(col_to=col_to)}"


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
SITE = "site"  # microscopy site, ~ fixed size
ROI = "roi"
LABEL_OBJ = "label_object"

BOUNDS = "bnd"
IBOUNDS = "ibnd"

BOUNDS_ = f"{BOUNDS}{SI}"
IBOUNDS_ = f"{BOUNDS}{SI}"

REGION = "region"  # -> CS + bounds

BOUNDING_BOX = "bbx"

# Position
T = "t"
Z = "z"
Y = "y"
X = "x"

# Index
IT = "it"
IZ = "iz"
IY = "iy"
IX = "ix"

# Difference
DT = "dt"
DZ = "dz"
DY = "dy"
DX = "dx"

MS = "m"

CH = "c"
OBJ = "o"
FTYPE = "ftype"
FEATURE = "f"
Z_MODEL = "z_model"
T_MODEL = "t_model"
DIST_TRANSF = "dtf"
NHD = "nhood"
CLF = "clf_name"

STAIN = "stain"
ACQUIS = "acquisition"
ROW = "row"
COL = "col"
LABEL = "label"
LBL_OBJ = "lbl_obj"

PARENT = "parent"
PARENT_DIST = "to"
CHILD = "child"


_LABEL_OBJ = f":{LABEL_OBJ}"


SEL_ROI = cs.matches(f"^{ROI}$")
SEL_OBJ = cs.matches(f"^{OBJ}(.to)?(.{PARENT})?(.{CHILD})?$")
SEL_LABEL = cs.matches(f"^{LABEL}(.to)?(.{PARENT})?(.{CHILD})?$")
SEL_CH = cs.matches(f"^{CH}(.\d)?$")
SEL_MS = cs.matches(f"^{MS}")

LABEL_OBJECT_IDX = (f"{ROI}", f"{OBJ}", f"{LABEL}")
PARENT_IDX = (f"{ROI}", f"{OBJ}.{PARENT}", f"{LABEL}.{PARENT}")
CHILD_IDX = (f"{ROI}", f"{OBJ}.{CHILD}", f"{LABEL}.{CHILD}")


SEL_INDEX = reduce(
    or_,
    [
        *[cs.matches(f"^{e}$") for e in (ROOT, PLATE, WELL, ROI)],
        SEL_OBJ,
        SEL_LABEL,
        SEL_CH,
        *[
            cs.matches(f"^{e}$")
            for e in [
                MS,
                "z_model",
                "t_model",
                CLF,
                DIST_TRANSF,
                NHD,
            ]
        ],
    ],
)


@dataclass(frozen=True, slots=True)
class PolarsSelector:
    idx: "SelectorType" = SEL_INDEX
    roi: "SelectorType" = SEL_ROI
    multiscale_level: "SelectorType" = SEL_MS
    object_type: "SelectorType" = SEL_OBJ
    label: "SelectorType" = cs.matches(f"^{LABEL}$")
    label_object: "SelectorType" = reduce(
        or_, [SEL_ROI, cs.matches(f"^{OBJ}$"), cs.matches(f"^{LABEL}$")]
    )
    parent: "SelectorType" = reduce(
        or_,
        [SEL_ROI, cs.matches(f"^{OBJ}.{PARENT}$"), cs.matches(f"^{LABEL}.{PARENT}$")],
    )
    child: "SelectorType" = reduce(
        or_,
        [SEL_ROI, cs.matches(f"^{OBJ}.{CHILD}$"), cs.matches(f"^{LABEL}.{CHILD}$")],
    )
    scale: "SelectorType" = cs.matches("(m.)?{SCALE}")
    spatial: "SelectorType" = cs.matches("[-\.][zyx]")
    bbx: "SelectorType" = cs.matches(f"{BOUNDING_BOX}.i?[zyx]")
    centroid: "SelectorType" = cs.matches(f"[Cc]entroid[\.-][zyx]")
    umap: "SelectorType" = cs.matches("umap[_\.-]\d+")
    ccp: "SelectorType" = cs.matches("CCP")
    nccp: "SelectorType" = cs.matches("NormalizedC[Cc][Pp]")
    empty: "SelectorType" = ~cs.all()


sel = PolarsSelector()
IDX_SEL = sel.idx
LABEL_OBJECT_SEL = sel.label_object


def build_lazy_tables(
    schema: dict[str, pl.Schema | pa.DataFrameSchema], include_hidden=True
) -> dict[str, pl.LazyFrame]:
    tables = {}
    for name, table_schema in schema.items():
        if not include_hidden:
            if name.startswith("_"):
                continue
        if isinstance(table_schema, pa.DataFrameSchema):
            pl_table_schema = {k: v.dtype.type for k, v in table_schema.columns.items()}
            table = pl.LazyFrame(schema=pl_table_schema)
            table_out = table_schema.validate(table)
        else:
            table_out = pl.LazyFrame(schema=table_schema)

        tables[name] = table_out

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
CHANNEL_DIM = pl.Enum(["c", "xy_model", "z_model", "t_model"])
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
        "spatial_object",
    ]
)

RASTER_DIM = pl.Enum(
    [
        "image",
        "label_image",
        "binary_mask",
        "dtf",
    ]
)

TYPE_DIM = pl.Enum(
    [
        "spatial.z",
        "spatial.y",
        "spatial.x",
        "temporal.t",
        "channel.c",
        "intensity.val",
        "object_type.o",
        "object_instance.o.lbl",
        "coord_sys.plate",
        "coord_sys.well:plate",
        "coord_sys.roi:well",
        "coord_sys.roi:o",
        "coord_sys.roi:o:lbl",
    ]
)

PROCESSING_DIM = pl.Enum(
    [
        "raster.resample",
        "raster.pyramid",
        "raster.registration",
        "raster.segmentation",
        "raster.aggregation",
        "model.bias",
        "model.clf",
        "model.cpp",
        "model.reg",
        "object.raster.aggregate",
        "object.hierarchy.aggregate",
        "object.nhd.aggregate",
        "object.feature.embed",
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


FK_MS = f"{FK(f'Resources.multiscale_levels.{MS}')}"
FK_CH = f"{FK(f'Resources.channels.{CH}')}"
FK_OBJ = f"{FK(f'Resources.object_types.{OBJ}')}"
FK_ROI = f"{FK(f'Resources.rois.{ROI}')}"
FK_ROOT = f"{FK(f'Resources.roots.{ROOT}')}"
FK_PLATE = f"{FK(f'Resources.plates.{PLATE}')}"
FK_WELL = f"{FK(f'Resources.wells.{WELL}')}"

FK_ROI_LABEL = f"{FK(f'Resources.label_objects.{ROI}')}"
FK_O_LABEL = f"{FK(f'Resources.label_objects.{OBJ}')}"
FK_LABEL = f"{FK(f'Resources.label_objects.{LABEL}')}"

CAT_TYPE = pl.Categorical("lexical")


SCHEMA_REGIONS = (
    {
        "Regions": {
            "type": pa.Column(
                pl.String,
                checks=pa.Check.isin(
                    ["root", "experiment", "plate", "well", "roi", "o", "label_object"]
                ),
            ),
            "parent": CAT_TYPE,
            "name": pa.Column(CAT_TYPE, unique=True),
            "dims": pl.List(pl.Enum(["t", "z", "y", "x"])),
            "origin.z": pa.Column(pl.Float64, default=0.0),
            "origin.y": pa.Column(pl.Float64, default=0.0),
            "origin.x": pa.Column(pl.Float64, default=0.0),
            "extent.dz": pl.Float64,
            "extent.dy": pl.Float64,
            "extent.dx": pl.Float64,
            "e.dz": pa.Column(pl.List(pl.Float64), default=(1.0, 0.0, 0.0)),
            "e.dy": pa.Column(pl.List(pl.Float64), default=(0.0, 1.0, 0.0)),
            "e.dx": pa.Column(pl.List(pl.Float64), default=(0.0, 0.0, 1.0)),
            "centroid.z": pl.Float64,
            "centroid.y": pl.Float64,
            "centroid.x": pl.Float64,
            "bound.z.lower": pa.Column(pl.Float64, nullable=True),
            "bound.y.lower": pa.Column(pl.Float64, nullable=True),
            "bound.x.lower": pa.Column(pl.Float64, nullable=True),
            "bound.z.upper": pa.Column(pl.Float64, nullable=True),
            "bound.y.upper": pa.Column(pl.Float64, nullable=True),
            "bound.x.upper": pa.Column(pl.Float64, nullable=True),
        },
        "Regions.dim": {
            "name": pa.Column(CAT_TYPE, unique=True),
            "parent": CAT_TYPE,
            "dims": pl.List(pl.Enum(["t", "z", "y", "x"])),
            "origin.z": pa.Column(pl.Float64, default=0.0),
            "origin.y": pa.Column(pl.Float64, default=0.0),
            "origin.x": pa.Column(pl.Float64, default=0.0),
            "extent.dz": pl.Float64,
            "extent.dy": pl.Float64,
            "extent.dx": pl.Float64,
            "e.dz": pa.Column(pl.List(pl.Float64), default=(1.0, 0.0, 0.0)),
            "e.dy": pa.Column(pl.List(pl.Float64), default=(0.0, 1.0, 0.0)),
            "e.dx": pa.Column(pl.List(pl.Float64), default=(0.0, 0.0, 1.0)),
        },
        "Regions.sampled": {
            "type": pa.Column(
                pl.String,
                checks=pa.Check.isin(
                    ["root", "experiment", "plate", "well", "site", "lbl_obj", "bbx"]
                ),
            ),
            "m": pa.Column(pl.UInt8, default=0),
            "scale.m.dz": pl.Float64,
            "scale.m.dy": pl.Float64,
            "scale.m.dx": pl.Float64,
            "shape.m.iz": pl.Int32,
            "shape.m.iy": pl.Int32,
            "shape.m.ix": pl.Int32,
            "iorigin.m.z": pl.Float32,
            "iorigin.m.y": pl.Float32,
            "iorigin.m.x": pl.Float32,
        },
    },
)


def build_schema(include_hidden=True):
    schema = {}

    schema["Root.coordinate_system"] = {f"{PK}cs": pl.String}

    schema["Root.dims"] = {
        f"{PK}dim": pl.String,
        "comps": pl.List(pl.String),
    }

    schema["Root.coords"] = {
        f"{FK('Root.dims.dim')}{PK}dim": pl.String,
        "coord": pl.List(pl.String),
    }

    schema["Root.mframes"] = {
        f"{PK}mframe": pl.String,
    }
    schema["Root.frames"] = {
        f"{PK}mframe": pl.String,
        f"{PK}frame": pl.String,
        "keys.type_": pl.Enum(["pk", "fk", "ck", ""]),
    }

    # Individual component of a column. in case of scalar feature == column
    schema["Root.columns"] = {
        f"{PK}mframe": pl.String,
        f"{PK}frame": pl.String,
        f"{PK}col": pl.String,
        "comps": pa.Column(pl.List(pl.String), default=[]),
        "rank": pl.UInt16,
        "name": pl.String,
        "descr": pa.Column(pl.String, nullable=True),
        "unit": pa.Column(pl.String, nullable=True),
    }

    # For grouping things like compound keys or vectorial quantities.
    schema["Tables.nested_columns"] = {
        f"{PK}mframe": pl.String,
        f"{PK}frame": pl.String,
        f"{PK}cols": pl.List(pl.String),
        "v_name": pl.String,  # virtual -> created from pk
        "name": pl.String,
        "type": pl.Enum(
            ["pk", "compound_key", "nd_feature", "nested_feature", "feature_set"]
        ),
    }
    schema["Tables.fks"] = {
        f"{PK}mframe": pl.String,
        f"{PK}frame": pl.String,
        f"{PK}cols": pl.List(pl.String),
        f"{FK('Tables.fks.mframe')}target.mframe": pl.String,
        f"{FK('Tables.fks.frame')}target.frame": pl.String,
        f"{FK('Tables.fks.cols')}target.cols": pl.List(pl.String),
    }

    REGION_TABLES = ["plates", "wells", "rois", "label_objects"]
    SAMPLED_REGION_TABLES = ["rois", "label_objects"]

    schema["_Resources.plates.v2"] = {
        f"{PK}{PLATE}": CAT_TYPE,
        f"{D_}{Z}.lower": pl.Float64,
        f"{D_}{Z}.upper": pl.Float64,
        f"{D_}{Y}.lower": pl.Float64,
        f"{D_}{Y}.upper": pl.Float64,
        f"{D_}{X}.lower": pl.Float64,
        f"{D_}{X}.upper": pl.Float64,
        f"{FK_ROOT}{ROOT}": CAT_TYPE,
        "path": pl.String,
        "description": pl.String,
    }

    DIM_BASE_ENUM = pl.Enum(["roi", "m", "o", "c", "t", "z", "y", "x"])
    DIM_REGION_ENUM = pl.Enum(["plate", "well", "site", "roi"])
    DIM_OBJECT_ENUM = pl.Enum([""])
    DIM_RASTER_ENUM = pl.Enum(["image", "label_image", "label_object"])

    schema["_Resources.roots"] = {
        f"{PK}{ROOT}": pl.String,
        "t.start": pl.Datetime,
        "t.end": pl.Datetime,
        "dims.base": pl.List(DIM_BASE_ENUM),
        "dims.raster": pl.List(
            pl.Struct({"name": DIM_RASTER_ENUM, "comps": pl.List(DIM_BASE_ENUM)})
        ),
        # "images": pl.List()
        "cs": pl.List(CAT_TYPE),
        "os": pl.List(CAT_TYPE),
        "label_object": pl.List(CAT_TYPE),
        "path": pl.String,
        "description": pl.String,
    }

    schema["Resources.plates"] = {
        # f"{FK(f'Resources.roots.{ROOT}')}{ROOT}": pl.String,
        f"{PK}{PLATE}": CAT_TYPE,
        # "path": pl.String,
        "description": pl.String,
    }

    schema["Resources.wells"] = {
        f"{PK}{WELL}": CAT_TYPE,  # PK
        f"{FK_PLATE}{PLATE}": CAT_TYPE,  # FK
        # "path": pl.String,
        "row": CAT_TYPE,
        "col": CAT_TYPE,
        "is_control_well": pl.Boolean,
        "control_well_for_acquisition": pl.UInt16,
        "plate.z": pa.Column(pl.Float64, default=0),
        "plate.y": pl.Float64,
        "plate.x": pl.Float64,
    }

    schema["_Resources.wells.v2"] = {
        f"{PK}{WELL}": CAT_TYPE,  # PK
        f"{D_}{Z}.lower": pl.Float64,
        f"{D_}{Z}.upper": pl.Float64,
        f"{D_}{Y}.lower": pl.Float64,
        f"{D_}{Y}.upper": pl.Float64,
        f"{D_}{X}.lower": pl.Float64,
        f"{D_}{X}.upper": pl.Float64,
        f"{FK_PLATE}{PLATE}": CAT_TYPE,  # FK
        "row": CAT_TYPE,
        "col": CAT_TYPE,
        "is_control_well": pl.Boolean,
        "control_well_for_acquisition": pl.UInt16,
        "plate.x": pl.Float64,
        "plate.y": pl.Float64,
        "plate.z": pl.Float64,
    }

    schema["Resources.rois"] = {
        f"{PK}{ROI}": CAT_TYPE,  # PK
        f"{FK_WELL}{WELL}": CAT_TYPE,  # FK(wells)
        # "well": CAT_TYPE,
        "well.z": pa.Column(pl.Float64, default=0),
        "well.y": pl.Float64,
        "well.x": pl.Float64,
        "path": pl.String,
    }

    schema["_Resources.rois.v2"] = {
        f"{PK}{ROI}": CAT_TYPE,  # PK
        f"{D_}{Z}.lower": pl.Float64,
        f"{D_}{Z}.upper": pl.Float64,
        f"{D_}{Y}.lower": pl.Float64,
        f"{D_}{Y}.upper": pl.Float64,
        f"{D_}{X}.lower": pl.Float64,
        f"{D_}{X}.upper": pl.Float64,
        f"{FK_WELL}{WELL}": CAT_TYPE,  # FK(wells)
        # "well": CAT_TYPE,
        "well.x": pl.Float64,
        "well.y": pl.Float64,
        "well.z": pl.Float64,
        "path": pl.String,
    }

    schema["_Resources.acquisition"] = {
        f"{PK}{ACQUIS}": pl.UInt8,  # PK
        f"{ROI}.path": pl.String,
    }

    schema["Regions.coordinate_systems"] = {
        f"{PK}{ROOT}": CAT_TYPE,
        f"{PK}{PLATE}": CAT_TYPE,
        f"{PK}{WELL}": CAT_TYPE,
        f"{PK}{ROI}": CAT_TYPE,
        f"{PK}{OBJ}": CAT_TYPE,
        f"{PK}{LBL_OBJ}": pl.UInt32,
        f"{BOUNDS_}{Z}.lower": pl.Float64,
        f"{BOUNDS_}{Z}.upper": pl.Float64,
        f"{BOUNDS_}{Y}.lower": pl.Float64,
        f"{BOUNDS_}{Y}.upper": pl.Float64,
        f"{BOUNDS_}{X}.lower": pl.Float64,
        f"{BOUNDS_}{X}.upper": pl.Float64,
    }

    schema["Resources.channels"] = {
        f"{PK}{CH}": CAT_TYPE,  # PK
        "stain": CAT_TYPE,
        "acquisition": pl.UInt8,
        "wavelength": pl.UInt16,
        "contrast_limits": pa.Column(
            pl.List(pl.Float64), nullable=True
        ),  # CANNOT USE ARRAY HERE, breaks upon read/write!!!!
    }

    schema["Resources.object_types"] = {
        f"{PK}{OBJ}": CAT_TYPE,  # PK
        "hierarchy_level": pl.UInt8,
        "is_primary": pa.Column(pl.Boolean, default=True),
        "parents": pl.List(CAT_TYPE),
    }

    schema["Resources.multiscale_levels"] = {
        f"{PK}{MS}": pl.UInt8,  # PK
        "scale.iz": pl.Float64,
        "scale.iy": pl.Float64,
        "scale.ix": pl.Float64,
    }

    schema["Resources.images"] = {
        f"{FK_CH}{PK}{CH}": CAT_TYPE,
        f"{FK_MS}{PK}{MS}": pl.UInt8,
        f"{FK_ROI}{PK}{ROI}": CAT_TYPE,
        # "t": pl.Datetime | pl.UInt32,
        f"{CH}.path": CAT_TYPE,
        # PK (m, roi, c, [t])
        # FK (m -> ms.m), (roi -> rois.roi), (c -> cs.c) ...
    }

    schema["_Resources.xy_model_corrs"] = {
        f"{FK('Resources.channels.wavelength')}{PK}wavelength": pl.UInt16,
        "path": CAT_TYPE,
    }

    schema["_Params.decay_model"] = {
        f"{PK}model": CAT_TYPE,
        "cls": pl.Enum(["Linear", "LogLinear", "Exp"]),
        "type": pl.Enum(["xy", "z", "t"]),
        "ndim": pl.Enum(["1D", "2D", "3D"]),
        "loss": pl.Enum(["linear", "huber"]),
        "dims": pl.List(pl.Enum(["z", "y", "x", "t", "c", "roi"])),
        "Features.intensity": pl.Enum(["Mean", "Median", "Sum"]),
        "Resources.dtf": pl.Enum(["DistanceToBorder", "DistanceAlong-z"]),
        # 2D  and 3D only
        "Resources.o": CAT_TYPE,
        "Resources.label:s": pl.List(pl.UInt32),  # len == 0 -> 1D ...
    }

    # TODO: remove 'z_model.' for metadata columns?
    schema["Resources.z_models"] = {
        f"{PK}z_model": CAT_TYPE,
        f"{FK_CH}{PK}{CH}": CAT_TYPE,
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
        f"{PK}t_model": CAT_TYPE,
        f"{FK_CH}{PK}{CH}": CAT_TYPE,
        f"{FK_ROI}{PK}{ROI}": CAT_TYPE,
        "correction_factor": pl.Float64,
        "delta_t_min": pl.Float64,
    }

    schema["_Resources.t_model_corrs"] = {
        f"{PK}model": CAT_TYPE,
        f"{FK_CH}{PK}{CH}": CAT_TYPE,
        f"{FK_CH}{PK}{ROI}": CAT_TYPE,
        "correction_factor": pl.Float64,
        "delta_t_min": pl.Float64,
    }

    schema["Resources.label_images"] = {
        f"{FK_OBJ}{PK}{OBJ}": CAT_TYPE,
        f"{FK_MS}{PK}{MS}": pl.UInt8,
        f"{FK_ROI}{PK}{ROI}": CAT_TYPE,
        # "t": pl.Datetime | pl.UInt32,
        f"{OBJ}.path": pl.String,
        "resample_scale_factors": pa.Column(pl.List(pl.Float64), nullable=True),
        # PK (m, o, roi, [t])
        # FK as one would expect
    }

    schema["Resources.label_objects"] = {
        f"{PK}{LABEL}": pl.UInt32,
        f"{FK_OBJ}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI}{PK}{ROI}": CAT_TYPE,
        "centroid.z": pa.Column(pl.Float64, nullable=True),
        "centroid.y": pa.Column(pl.Float64, nullable=True),
        "centroid.x": pa.Column(pl.Float64, nullable=True),
        f"{FK_MS}{PK}{MS}": pl.UInt8,
        "m.iz.lower": pa.Column(pl.Int32, nullable=True),
        "m.iz.upper": pa.Column(pl.Int32, nullable=True),
        "m.iy.lower": pa.Column(pl.Int32, nullable=True),
        "m.iy.upper": pa.Column(pl.Int32, nullable=True),
        "m.ix.lower": pa.Column(pl.Int32, nullable=True),
        "m.ix.upper": pa.Column(pl.Int32, nullable=True),
    }

    schema["Resources.spatial_objects"] = {
        f"{FK_OBJ}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI}{PK}{ROI}": CAT_TYPE,
        f"{PK}{LABEL}": pl.UInt32,
        "centroid.z": pl.Float64,
        "centroid.y": pl.Float64,
        "centroid.x": pl.Float64,
        f"{BOUNDS_}{Z}.lower": pl.Float64,
        f"{BOUNDS_}{Z}.upper": pl.Float64,
        f"{BOUNDS_}{Y}.lower": pl.Float64,
        f"{BOUNDS_}{Y}.upper": pl.Float64,
        f"{BOUNDS_}{X}.lower": pl.Float64,
        f"{BOUNDS_}{X}.upper": pl.Float64,
        # f"{PK}{MS}.i": pl.UInt8,
        # "m.iz.lower": pl.Int32,
        # "m.iz.upper": pl.Int32,
        # "m.iy.lower": pl.Int32,
        # "m.iy.upper": pl.Int32,
        # "m.ix.lower": pl.Int32,
        # "m.ix.upper": pl.Int32,
        # f"slice.{PK}{MS}": pl.UInt8,
        # "slice.m.iz.lower": pl.Int32,
        # "slice.m.iz.upper": pl.Int32,
        # "slice.m.iy.lower": pl.Int32,
        # "slice.m.iy.upper": pl.Int32,
        # "slice.m.ix.lower": pl.Int32,
        # "slice.m.ix.upper": pl.Int32,
    }

    schema["Resources.hierarchy"] = {
        # child and parent share roi
        f"{PK}{ROI}": CAT_TYPE,
        f"{PK}{OBJ}{SI}{PARENT}": CAT_TYPE,
        f"{PK}{LABEL}{SI}{PARENT}": pl.UInt32,
        f"{PK}{OBJ}{SI}{CHILD}": CAT_TYPE,
        f"{PK}{LABEL}{SI}{CHILD}": pl.UInt32,
    }

    schema["Resources.classifiers"] = {
        f"{PK}clf_name": CAT_TYPE,
        f"{FK_OBJ}{OBJ}": CAT_TYPE,
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

    schema["Regions.plates"] = schema["Resources.plates"]
    schema["Regions.wells"] = schema["Resources.wells"]
    schema["Regions.rois"] = schema["Resources.rois"]
    schema["Regions.spatial_objects"] = {
        f"{FK_ROI}{PK}{ROI}": CAT_TYPE,
        f"{FK_OBJ}{PK}{OBJ}": CAT_TYPE,
        f"{PK}{LABEL}": pl.UInt32,
        "centroid.z": pa.Column(pl.Float64, nullable=True),
        "centroid.y": pa.Column(pl.Float64, nullable=True),
        "centroid.x": pa.Column(pl.Float64, nullable=True),
        f"{BOUNDS_}{Z}.lower": pl.Float64,
        f"{BOUNDS_}{Z}.upper": pl.Float64,
        f"{BOUNDS_}{Y}.lower": pl.Float64,
        f"{BOUNDS_}{Y}.upper": pl.Float64,
        f"{BOUNDS_}{X}.lower": pl.Float64,
        f"{BOUNDS_}{X}.upper": pl.Float64,
    }

    schema["Raster.images"] = schema["Resources.images"]
    schema["Raster.label_images"] = schema["Resources.label_images"]
    schema["Raster.label_objects"] = schema["Resources.label_objects"]

    schema["Models.z_models"] = schema["Resources.z_models"]
    schema["Models.t_models"] = schema["Resources.t_models"]
    # schema["Models.bg_models"] = {

    # }
    schema["Raster.label_objects"] = schema["Resources.label_objects"]

    schema["_Resources.neighborhood.types"] = {
        f"{PK}{NHD}": pl.String,
    }

    schema["Features.label"] = {
        # "id": pl.UInt32,
        f"{FK_O_LABEL}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI_LABEL}{PK}{ROI}": CAT_TYPE,
        f"{FK_LABEL}{PK}{LABEL}": pl.UInt32,
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
        "EquivalentEllipsoidDiameter-a": pl.Float64,
        "EquivalentEllipsoidDiameter-b": pl.Float64,
        "EquivalentEllipsoidDiameter-c": pl.Float64,
        # Position
        "Centroid-x": pl.Float64,
        "Centroid-y": pl.Float64,
        "Centroid-z": pl.Float64,
        # Orientation
        "PrincipalAxes-a-x": pl.Float64,
        "PrincipalAxes-a-y": pl.Float64,
        "PrincipalAxes-a-z": pl.Float64,
        "PrincipalAxes-b-x": pl.Float64,
        "PrincipalAxes-b-y": pl.Float64,
        "PrincipalAxes-b-z": pl.Float64,
        "PrincipalAxes-c-x": pl.Float64,
        "PrincipalAxes-c-y": pl.Float64,
        "PrincipalAxes-c-z": pl.Float64,
        "BoundingBox-x-lower": pl.Int32,
        "BoundingBox-x-upper": pl.Int32,
        "BoundingBox-y-lower": pl.Int32,
        "BoundingBox-y-upper": pl.Int32,
        "BoundingBox-z-lower": pl.Int32,
        "BoundingBox-z-upper": pl.Int32,
        "OrientedBoundingBoxSize-a": pl.Float64,
        "OrientedBoundingBoxSize-b": pl.Float64,
        "OrientedBoundingBoxSize-c": pl.Float64,
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
        "OrientedBoundingBoxOrigin-x": pl.Float64,
        "OrientedBoundingBoxOrigin-y": pl.Float64,
        "OrientedBoundingBoxOrigin-z": pl.Float64,
        # PK (id)
        # UNIQUE (o, roi, label) -> alternative PK
        # FK (id -> label_objects.id), ...
    }

    schema["Features.intensity"] = {
        # "id": pl.UInt32,
        f"{FK_O_LABEL}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI_LABEL}{PK}{ROI}": CAT_TYPE,
        f"{FK_LABEL}{PK}{LABEL}": pl.UInt32,
        f"{FK_CH}{PK}{CH}": CAT_TYPE,
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
        f"{FK_O_LABEL}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI_LABEL}{PK}{ROI}": CAT_TYPE,
        f"{FK_LABEL}{PK}{LABEL}": pl.UInt32,
        f"{FK_CH}{PK}{CH}{SI}0": CAT_TYPE,
        f"{FK_CH}{PK}{CH}{SI}1": CAT_TYPE,
        "PearsonR": pl.Float64,
        "SpearmanR": pl.Float64,
        "KendallTau": pl.Float64,
    }

    schema["Features.distance"] = {
        f"{FK_O_LABEL}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI_LABEL}{PK}{ROI}": CAT_TYPE,
        f"{FK_LABEL}{PK}{LABEL}": pl.UInt32,
        f"{PK}{OBJ}{SI}to": CAT_TYPE,
        f"{PK}{LABEL}{SI}to": pl.UInt32,
        f"{PK}{DIST_TRANSF}": CAT_TYPE,
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
        f"{FK_O_LABEL}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI_LABEL}{PK}{ROI}": CAT_TYPE,
        f"{FK_LABEL}{PK}{LABEL}": pl.UInt32,
        f"{PK}{NHD}": pl.String,
        "Count": pl.UInt32,
    }

    schema["Features.density_distance"] = {
        f"{FK_O_LABEL}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI_LABEL}{PK}{ROI}": CAT_TYPE,
        f"{FK_LABEL}{PK}{LABEL}": pl.UInt32,
        f"{PK}{NHD}": pl.String,
        "Max": pl.Float64,
        "Mean": pl.Float64,
    }

    schema["Features.classifier"] = {
        f"{FK_O_LABEL}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI_LABEL}{PK}{ROI}": CAT_TYPE,
        f"{FK_LABEL}{PK}{LABEL}": pl.UInt32,
        f"{PK}clf_name": CAT_TYPE,
        "annotation": CAT_TYPE,
        "pred": CAT_TYPE,
        "probas": pl.List(pl.Struct({"annotation": CAT_TYPE, "proba": pl.Float64})),
    }

    schema["_Features.ccp"] = {
        f"{FK_O_LABEL}{PK}{OBJ}": CAT_TYPE,
        f"{FK_ROI_LABEL}{PK}{ROI}": CAT_TYPE,
        f"{FK_LABEL}{PK}{LABEL}": pl.UInt32,
        f"{PK}run_id": pl.String,
        f"{PK}id": pl.Int32,
        "CCP": pl.Float64,
        "NormalizedCCP": pl.Float64,
        "DistanceToCircleCCP": pl.Float64,
        "Gradient.z": pl.Float64,
        "Gradient.y": pl.Float64,
        "Gradient.x": pl.Float64,
    }
    if not include_hidden:
        return {k: v for k, v in schema.items() if not k.startswith("_")}
    return schema


ColID = namedtuple("ColID", ("mframe", "frame", "col"))
FrameID = namedtuple("FrameID", ("mframe", "frame"))


def parse_keys_from_annotated_column_name(col: str) -> dict[str, str]:
    results = {}

    pk_matches = PK_REGEX.search(col)
    fk_col = FK_REGEX.findall(col)

    col_name: ColName = PK_REGEX.sub("", FK_REGEX.sub("", col))
    results["column_name"] = col_name

    if pk_matches:
        results["pk"] = col_name

    if len(fk_col) > 1:
        raise ValueError(f"More than one FK found {fk_col!r} in {col!r}")
    elif len(fk_col) == 1:
        fk_parts = ColID(*fk_col[0].split("."))
        results["fk"] = fk_parts

    return results


def parse_keys_from_annotated_columns(cols: tuple[str, ...]) -> pl.DataFrame:
    return pl.DataFrame([parse_keys_from_annotated_column_name(col) for col in cols])


from zfish.multi_table.schema_metadata import FKey, FrameMeta, Key, PKey


def parse_frame_schema(
    schema: pl.Schema, name: str = None, metadata: dict[str, Any] | None = None
) -> pa.DataFrameSchema:
    if metadata is None:
        metadata = {}

    parse_results = parse_keys_from_annotated_columns(list(schema))
    new_column_names = parse_results["column_name"].to_list()
    pk = tuple(parse_results["pk"].drop_nulls())
    metadata["pk"] = pk
    if "fk" in parse_results:
        fks = list(
            (row[0], ColID(*row[1]))
            for row in parse_results.filter(pl.col("fk").is_not_null())
            .select("column_name", "fk")
            .rows()
        )
        metadata["fks"] = tuple(e[0] for e in fks)
        metadata["fk_relations"] = fks

    parsed_schema = {}
    for new_column_name, (column_name, dtype) in zip(new_column_names, schema.items()):
        if isinstance(dtype, pa.Column):
            pa_column = dtype
            pa_column.name = new_column_name
        elif type(dtype) is pl.Enum:
            pa_dtype = pl.Categorical
            checks = pa.Check.isin(dtype.categories.to_list())
            pa_column = pa.Column(name=new_column_name, dtype=pa_dtype, checks=checks)
        else:
            pa_dtype = dtype
            checks = None
            pa_column = pa.Column(name=new_column_name, dtype=pa_dtype, checks=checks)
        parsed_schema[new_column_name] = pa_column

    return pa.DataFrameSchema(
        parsed_schema,
        metadata=extract_frame_meta(metadata),
        name=name,
        unique=pk,
        coerce=True,
        add_missing_columns=True,
    )


def extract_frame_meta(meta: dict[str, Any]):
    df_meta = meta
    if "fk_relations" in df_meta:
        fk_target_frames = tuple(rel[:2] for name, rel in df_meta["fk_relations"])
        fk_target_cols = tuple(rel[-1] for name, rel in df_meta["fk_relations"])
        fk_cols = tuple(name for name, rel in df_meta["fk_relations"])
        fkeys = []

        for target_frame, comps, target_comps in (
            pl.DataFrame(
                {
                    "cols": fk_cols,
                    "tframes": fk_target_frames,
                    "tcols": fk_target_cols,
                }
            )
            .group_by("tframes")
            .agg(("cols", "tcols"))
            .rows()
        ):
            fkey = FKey(
                tuple(comps),
                name=target_frame[1],
                target_frame=tuple(target_frame),
                target_comps=tuple(target_comps),
            )
            fkeys.append(fkey)
    else:
        fkeys = ()

    frame_meta = FrameMeta(
        # frame_id,
        PKey(
            df_meta["pk"],
        ),
        fks=tuple(fkeys),
        hidden=df_meta["hidden"],
        # tuple(FKey(df_meta['fk_comps']))
    )
    return frame_meta


def extract_mframe_meta(tbls):
    fmetas = []
    for frame_id, schema in tbls._schemas.items():
        frame_meta = extract_frame_meta(schema)
        fmetas.append(frame_meta)
    return fmetas


def parse_schemas(
    schemas: dict[str, pl.Schema],
) -> dict[str, tuple[pl.Schema, dict[str, str]]]:
    parsed_schema = {}
    for name, schema in schemas.items():
        # mo = TABLE_NAME_REGEX.match(name)

        # if mo is None:
        #     raise ValueError(
        #         f"Invalid table name: {name} should be of the form '[_]Multitable.frame_name[.v12]'"
        #     )
        # hidden, multi_table_name, frame_name, version_str, version_digit = mo.groups()
        # qual_name = FrameID(multi_table_name, frame_name)

        frame_id = FrameId.from_filename(name)

        metadata = {
            "hidden": (frame_id.hidden is not None),
        }
        if frame_id.version is not None:
            metadata["version"] = frame_id.version
            # metadata["version_nr"] = int(version_digit)

        parsed_schema[frame_id] = parse_frame_schema(
            schema, name=name, metadata=metadata
        )
    return parsed_schema


SCHEMA_HIDDEN = build_schema()
SCHEMA = build_schema(include_hidden=False)

PARSED_SCHEMA = parse_schemas(SCHEMA)
LAZY_TABLES_EMPTY = build_lazy_tables(PARSED_SCHEMA)
# %%

# Q_FRAME_NAME_PATTERN = (
#     rf"(?P<mfname>{MFRAME_NAME_PATTERN}){SEP}(?P<fname>{FRAME_NAME_PATTERN})"
# )
# Q_COL_NAME_PATTER = rf"{Q_FRAME_NAME_PATTERN}{SEP}(?P<colname>{COL_NAME_PATTERN})"

# COL_NAME_REGEX = re.compile(COL_NAME_PATTERN)
# FRAME_NAME_REGEX = re.compile(FRAME_NAME_PATTERN)
# MFRAME_NAME_REGEX = re.compile(MFRAME_NAME_PATTERN)
# FRAME_QNAME_REGEX = re.compile(Q_FRAME_NAME_PATTERN)
# COL_QNAME_REGEX = re.compile(Q_COL_NAME_PATTER)
# COL_COMP_REGEX = re.compile("(.*?)(-[a-c])?(-[x-z])?$")
# KEY_COMP_REGEX = re.compile("")

# NAMED_COMPONENT_PATTERN = {""}
