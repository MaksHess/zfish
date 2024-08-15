import argparse
import logging
import sys

import polars as pl
import polars.selectors as cs
from pydantic_yaml import parse_yaml_file_as

from zfish.deploy.run_new_feature_extraction import CorrelationFeatureExtractionParams
from zfish.features.colocalization import get_colocalization_features
from zfish.multi_table.feature_query_schemas import (
    _aggregate_image_paths2,
    _aggregate_label_paths2,
    _construct_correlation_queries,
    _lazy_load_images,
    _lazy_load_labels,
)
from zfish.multi_table.tables_io import read_resources

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

err_handler = logging.StreamHandler(sys.stderr)
err_handler.setLevel("DEBUG")
err_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
    )
)

out_handler = logging.StreamHandler(sys.stdout)
out_handler.setLevel("DEBUG")
# out_handler.setLevel("DEBUG")
out_handler.setFormatter(
    logging.Formatter(
        "%(name)-12s: %(levelname)-8s %(message)s",
    )
)
# logger.addHandler(err_handler)
logger.addHandler(out_handler)


def main():
    logger.info("starting feature extraction...")
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("-p", "--feature_extraction_params", type=str)
    args = parser.parse_args()
    logger.info(f"CLI arguments parsed, job_id = {args.idx}")
    # params = FeatureExtractionParams.parse_file(args.feature_extraction_params)

    logger.info("parsing parameter file.")
    params = parse_yaml_file_as(
        CorrelationFeatureExtractionParams, args.feature_extraction_params
    )

    logger.info("loading resource tables.")
    r = read_resources(params.resource_path, use_pyarrow=False)
    logger.info("constructing all queries & selecting query batch.")
    df_all_queries = _construct_correlation_queries(
        r,
        levels=params.levels,
        wells=params.wells,
        rois=params.rois,
        channel_pairs=params.channel_pairs,
        object_types=params.object_types,
        features=params.features,
        return_queries=params.process_queries,
    )

    dfs_sub_queries = list(
        df_all_queries.group_by(params.parallelize_over, maintain_order=True)
    )
    logger.info(
        f"processing query batch at index {args.idx} out of {len(dfs_sub_queries)} batches..."
    )
    query_id, df_q = dfs_sub_queries[args.idx]
    subquery_suffix = subquery_suffix = "__".join(
        f"{e}={n}" for e, n in zip(params.parallelize_over, query_id)
    )
    logger.info(f"batch name: {subquery_suffix}")
    logger.info(f"# queries:  {df_q.height}")

    logger.info("lazy loading images & labels...")
    df_image_ids = (
        pl.concat(
            [
                df_q.select(["idx.m", "idx.roi", pl.col("idx.c.0").alias("idx.c")]),
                df_q.select(["idx.m", "idx.roi", pl.col("idx.c.1").alias("idx.c")]),
            ]
        ).unique(maintain_order=True)
        # .with_columns(pl.lit(None).cast(pl.Categorical).alias("mask.o"))
    )

    df_label_image_ids = df_q.select(["idx.m", "idx.roi", "idx.o"]).unique(
        maintain_order=True
    )

    df_images = _aggregate_image_paths2(
        r, df_image_ids=df_image_ids, z_model=params.z_model, t_model=params.t_model
    )
    df_labels = _aggregate_label_paths2(r, df_label_image_ids=df_label_image_ids)

    images_lazy = _lazy_load_images(df_images)
    logger.info(f"{len(images_lazy)} images loaded (lazy).")
    labels_lazy = _lazy_load_labels(df_labels)
    logger.info(f"{len(labels_lazy)} labels loaded (lazy).")

    columns = ("idx.m", "idx.roi", "idx.o", "idx.c.0", "idx.c.1", "features")
    dfs = []
    for i, (m, roi, o, c0, c1, features) in enumerate(df_q.select(columns).iter_rows()):
        logger.info(f"{i:<4}{m:<3}{o:<12}({c0}, {c1})")
        logger.info("loading images...")
        lbl = labels_lazy[(m, roi, o)].compute()
        ch0 = images_lazy[(m, roi, c0)].compute()
        ch1 = images_lazy[(m, roi, c1)].compute()
        logger.info("extracting features...")
        df = (
            get_colocalization_features(
                lbl, ch0, ch1, features=features, resource_in_name=False
            )
            .rename({"label": "idx.label"})
            .with_columns(
                pl.lit(m).cast(pl.UInt8).alias("idx.m"),
                pl.lit(roi).cast(pl.Categorical).alias("idx.roi"),
                pl.lit(o).cast(pl.Categorical).alias("idx.o"),
                pl.lit(c0).cast(pl.Categorical).alias("idx.c.0"),
                pl.lit(c1).cast(pl.Categorical).alias("idx.c.1"),
            )
        )
        dfs.append(df)
    logger.info("concatenating tabels...")
    df_out = pl.concat(dfs).select(
        cs.by_name(["idx.roi", "idx.o", "idx.label", "idx.m", "idx.c.0", "idx.c.1"]),
        ~cs.by_name(["idx.roi", "idx.o", "idx.label", "idx.m", "idx.c.0", "idx.c.1"]),
    )

    params.output_path.mkdir(exist_ok=True)
    out_file = params.output_path / f"features.correlation__{subquery_suffix}.parquet"
    logger.info(f"writing output table ({out_file.name})")
    df_out.write_parquet(out_file)
    logger.info("done.")


if __name__ == "__main__":
    main()
