# %%
from typing import Any, Self

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from zfish.multi_table.features_aggregate import is_singleton, squeeze_singleton_dims
from zfish.multi_table.schema_metadata import FKey, FrameMeta, Key, PKey
from zfish.multi_table.schemas_v2 import sel
from zfish.multi_table.tables_io import (
    AnyFrame,
    FrameId,
    Regions,
    make_table_dataclasses_from_schema,
    to_wide,
)


def migration_rename_label_objects_table(df: pl.DataFrame) -> pl.DataFrame:
    pass


def pa_schema_from_polars(schema_pl: pl.Schema, metadata, required_columns=()):
    schema = {}
    for col_name, dtype in schema_pl.items():
        if col_name in required_columns:
            nullable = False
        else:
            nullable = True
        schema[col_name] = pa.Column(name=col_name, dtype=dtype, nullable=True)
    return pa.DataFrameSchema(
        columns=schema, metadata=metadata, unique=metadata.pk.comps
    )


def compute_new_meta(columns_new: tuple[str, ...], meta: FrameMeta):
    pk_new = PKey(meta.pk.intersect(columns_new))
    fks_new = tuple([fk for fk in meta.fks if fk.is_in(columns_new)])
    oks_new = tuple([ok for ok in meta.oks if ok.is_in(columns_new)])
    meta_new = meta.from_template(pk=pk_new, fks=fks_new, oks=oks_new)
    return meta_new


def update_meta_inplace(mframe) -> Self:
    for k, meta in mframe._table_metas.items():
        columns_new = mframe._tables[k].collect_schema().names()
        meta_new = compute_new_meta(columns_new, meta)
        mframe._table_metas[k] = meta_new
    return mframe


def update_meta(mframe) -> Self:
    new_metas = {}
    for k, meta in mframe._table_metas.items():
        columns_new = mframe._tables[k].collect_schema().names()
        meta_new = compute_new_meta(columns_new, meta)
        new_metas[k] = meta_new

    return mframe


def _rename_columns_key(key: dict[str, Any], rename_map: dict[str, str]):
    out_key = {}
    for k_comp, v_comp in key.items():
        if "comps" in k_comp:
            out_key[k_comp] = tuple(rename_map.get(e, e) for e in v_comp)
        else:
            out_key[k_comp] = v_comp
    return out_key


def _rename_columns(meta: dict[str, Any], rename_map: dict[str, str]):
    out_meta = {}
    out_meta["pk"] = _rename_columns_key(meta["pk"], rename_map=rename_map)
    out_meta["fks"] = tuple(
        _rename_columns_key(fk, rename_map=rename_map) for fk in meta["fks"]
    )
    out_meta["oks"] = tuple(
        _rename_columns_key(fk, rename_map=rename_map) for fk in meta["oks"]
    )
    return out_meta


def rename_meta_columns_inplace(tbls, rename_map: dict[str, str]) -> Self:
    for k, meta in tbls._table_metas.items():
        columns_new = [
            rename_map.get(v, v) for v in tbls._tables[k].collect_schema().names()
        ]

        meta_new = FrameMeta(**_rename_columns(meta.to_dict(), rename_map=rename_map))
        tbls._table_metas[k] = meta_new
    return tbls


def construct_new_schema(
    df_new: AnyFrame,
    schema: pa.DataFrameSchema,
) -> tuple[AnyFrame, pa.DataFrameSchema]:
    columns_new = df_new.columns
    meta = list(schema.get_metadata().values())[0]["dataframe"]
    meta_new = compute_new_meta(columns_new, meta)
    required_columns = tuple(dict.fromkeys(e for k in meta_new.keys() for e in k.comps))
    schema_new = pa_schema_from_polars(
        df_new.collect_schema(), meta_new, required_columns=required_columns
    )
    return df_new, schema_new


def features_to_wide(
    f,
    on_sel=sel.idx - sel.label_object,
    index_sel=sel.label_object,
    values_sel=None,
    prefix=True,
    sep="_",
    fidx_sep="-",
    squeeze=False,
):
    out_tables = {}
    out_schemas = {}
    squeezed = {}
    if squeeze:
        pass
    dfs_new = f.pipe_tables(
        to_wide,
        on_sel=on_sel,
        index_sel=index_sel,
        values_sel=values_sel,
        prefix=prefix,
        sep=sep,
        fidx_sep=fidx_sep,
    )._tables
    schemas = f._schemas
    for df_new, (frame_id, schema) in zip(dfs_new.values(), schemas.items()):
        df_new, schema_new = construct_new_schema(df_new, schema)
        new_frame_id = FrameId("FeaturesWide", frame=frame_id.frame)
        schema_new.name = new_frame_id.to_filename()
        out_tables[new_frame_id] = df_new
        out_schemas[new_frame_id] = schema_new

    FeaturesWide, LazyFeaturesWide = make_table_dataclasses_from_schema(
        "FeaturesWide", out_schemas
    )

    return FeaturesWide.from_tables(out_tables)


def label_object_split(r, short_name_map=None):
    if short_name_map is None:
        short_name_map = {}

    out_tables = {}
    out_schemas = {}
    df = r.label_objects
    dfs_new = {k: v for k, v in df.group_by("o", maintain_order=True)}
    schema = r._schemas[FrameId("Resources", "label_objects")]
    order = r.object_types["o"].to_list()
    for name, df in sorted(dfs_new.items(), key=lambda x: order.index(x[0][0])):
        df_new, schema_new = construct_new_schema(df, schema)
        new_frame_id = FrameId(
            "FeaturesO", frame=", ".join([short_name_map.get(n, n) for n in name])
        )
        schema_new.name = new_frame_id.to_filename()
        out_tables[new_frame_id] = df_new
        out_schemas[new_frame_id] = schema_new

    FeaturesO, LazyFeaturesO = make_table_dataclasses_from_schema(
        "FeaturesO", out_schemas
    )

    return FeaturesO.from_tables(out_tables)


from zfish.multi_table.table_base_mixin import MFrameBaseMixin


def squeeze_mframe(mframe: MFrameBaseMixin, selector=sel.idx):
    out_frames = {}
    out_schemas = {}

    for frame_id, frame, schema in zip(
        mframe._table_ids, mframe._tables.values(), mframe._schemas.values()
    ):
        new_mframe_name = f"{frame_id.mframe}Sqz"
        new_frame, squeezed = squeeze_frame(frame, return_squeezed=True)

        new_frame_id = frame_id.from_template(
            mframe=new_mframe_name,
            _squeezed=tuple(squeezed.items()) + tuple(frame_id._squeezed),
        )
        new_frame, new_schema = construct_new_schema(new_frame, schema)
        out_frames[new_frame_id] = new_frame
        out_schemas[new_frame_id] = new_schema
        # print(schema)

    FeaturesOSqz, LazyFeaturesOSqz = make_table_dataclasses_from_schema(
        new_mframe_name, out_schemas
    )

    return FeaturesOSqz.from_tables(out_frames)
    #     for name,Fdf in sorted(dfs_new.items(), key=lambda x: order.index(x[0][0])):
    #         df_new, schema_new = construct_new_schema(df, schema)
    #         new_frame_id = FrameId(
    #             "FeaturesO", frame=", ".join([short_name_map.get(n, n) for n in name])
    #         )
    #         schema_new.name = new_frame_id.to_filename()
    #         out_tables[new_frame_id] = df_new
    #         out_schemas[new_frame_id] = schema_new

    # FeaturesO, LazyFeaturesO = make_table_dataclasses_from_schema(
    #     "FeaturesO", out_schemas
    # )

    # return FeaturesO.from_tables(out_tables)


def squeeze_frame(
    df: pl.DataFrame, return_squeezed: bool = False
) -> pl.DataFrame | tuple[pl.DataFrame, tuple[tuple[str, Any], ...]]:
    singleton_cols = df.pipe(squeeze_singleton_dims)["column"]
    squeezed = df.select(pl.col(e).unique() for e in singleton_cols)
    df_out = df.select(pl.exclude(singleton_cols))
    if return_squeezed:
        return df_out, {e.name: e.item() for e in squeezed}
    return df_out
