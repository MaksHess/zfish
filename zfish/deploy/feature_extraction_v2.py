# %%
import argparse
import logging
from collections import defaultdict

import polars as pl
from pydantic_yaml import parse_yaml_file_as

import zfish.features.polars_utils as pu
from zfish.features.feature_extraction_parameters_v2 import FeatureExtractionParams
from zfish.roi.spatial_roi import Roi, apply_z_decay_models_to_roi, read_models

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def main():
    logger.info("Parsing argunments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("-p", "--feature_extraction_parameters", type=str)
    args = parser.parse_args()
    
    logger.info("Parsing yaml file & selecting site...")
    params = parse_yaml_file_as(
        FeatureExtractionParams, args.feature_extraction_parameters
    )
    from pprint import pprint
    pprint(params)
    print()
    site_params = params.get_site_params_by_index(args.idx)
    pprint(site_params)
    print()
    
    logger.info("Lazy loading roi.")
    return
    lazy_roi = Roi.from_file(
        site_params.roi_path,
        level=site_params.level,
        features_root=site_params.output_path,
        lazy_tables=True,
    )
    
    logger.info("Validating resources.")
    site_params = site_params.validate_parameters_with(lazy_roi)
    
    logger.info("Selecting resources.")
    lazy_roi_resources = lazy_roi.sel(
        l=list(site_params.features.resources.label_images),
        c=list(site_params.features.resources.channels),
    )
    
    #TODO: t_decay_models & correction

    logger.info("Loading z-decay models.")
    z_decay_models = read_models(params.intensity_correction.z_decay_models)

    # Set model to None if no model is found for a channel.
    if site_params.intensity_correction.z_decay_default_models is None:
        logger.warning(
            f"Uncorrected channels: {set(site_params.features.resources.channels).difference(z_decay_models.keys())}"
        )
        z_decay_models = {
            **{k: None for k in site_params.features.resources.channels},
            **z_decay_models,
        }

    lazy_roi_resources_corr = apply_z_decay_models_to_roi(
        models=z_decay_models,
        roi=lazy_roi_resources,
        two_step_label=params.intensity_correction.z_decay_two_step_label,
    )

    tables = process_queries(
        site_params.features.queries(lazy_roi_resources_corr), lazy_roi_resources_corr
    )

    print("writing tables...")
    lazy_roi_resources_corr.tables = tables
    lazy_roi_resources_corr.write_tables()
    print()
    print("done.")


def process_queries(
    queries: list["FeatureQuery"], roi: Roi
) -> "dict[str, pl.DataFrame]":
    result = defaultdict(list)

    for query in queries:
        print(f"{query!r}")
        result[query.label_image].append(query.compute(roi))

    return {k: pu.join(v, on="label") for k, v in result.items()}


if __name__ == "__main__":
    main()
