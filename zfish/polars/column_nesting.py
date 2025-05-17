import polars as pl
import polars.selectors as cs

from zfish.features.polars_utils import nest

pl.enable_string_cache()


def unnest_struct(df, struct_column="idx", sep=""):
    field_names = [f.name for f in df.schema[struct_column].fields]

    return df.with_columns(
        pl.col(struct_column).struct.rename_fields(
            [f"{struct_column}{sep}{field_name}" for field_name in field_names]
        )
    ).unnest(struct_column)


def unnest_structs(df, column_selector=cs.all(), sep=""):
    columns = cs.expand_selector(df, column_selector)
    for column in columns:
        df.collect_schema()[column]
        if isinstance(df.schema[column], pl.Struct):
            df = df.pipe(unnest_struct, struct_column=column, sep=sep)
    return df


def dtype_depth(dtype):
    if isinstance(dtype, pl.Field):
        dtype = dtype.dtype

    if dtype.is_nested():
        if hasattr(dtype, "fields"):
            return 1 + (max(map(dtype_depth, dtype.fields)) if dtype else 0)
        # else:
        #     warn(f'Non-struct nested dtypes like f{dtype} cannot be traversed.')
    return 0


def unnest(df, to_level=-1, sep="", column_selector=cs.all()):
    max_depth = max(map(dtype_depth, df.collect_schema().dtypes()))
    if to_level < 0:
        to_level = max_depth + to_level + 1
    for _ in range(to_level):
        df = df.pipe(unnest_structs, sep=sep, column_selector=column_selector)
    return df


def example():
    df = pl.DataFrame(
        {
            "CenterOfGravity.z": [204.95362889430325, 151.36779498254737],
            "idx.dim.o": ["cyto", "nucleiRaw3"],
            "idx.roi.f": ["E05_px-0106_py+1980", "B05_px+0056_py+2258"],
            "idx.label": [1397, 1561],
            "idx.dim.c": ["XRN2.3", "ALYREF.2"],
            "CenterOfGravity.y": [516.5710344860328, 423.1548345101487],
            "CenterOfGravity.x": [158.55340760589576, 298.4185843785503],
        }
    )
    return df
