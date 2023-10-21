# %%
import argparse
import json
import logging
import sys
from collections import defaultdict
from pprint import pformat

import polars as pl
import pydantic
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, to_yaml_str

import zfish.features.polars_utils as pu
from zfish.features.feature_extraction_parameters import FeatureExtractionParams
from zfish.features.queries import FeatureQuery
from zfish.roi.spatial_roi import Roi, apply_z_decay_models_to_roi, read_models

# logging.basicConfig(
#     level='DEBUG',
#     format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
#         )
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
    logger.info(f"{title('Starting Feature Extraction')}")
    logger.info("Parsing CLI argunments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("-p", "--feature_extraction_parameters", type=str)
    args = parser.parse_args()

    logger.info("Parsing yaml parameter file...\n")
    params = parse_yaml_file_as(
        FeatureExtractionParams, args.feature_extraction_parameters
    )
    logger.debug(f"{nested_repr(params)}\n")

    logger.info("Extracting site...")
    site_params = params.get_site_params_by_index(args.idx)
    logger.debug(f"{nested_repr(site_params)}\n")

    logger.info("Lazy loading roi.")
    lazy_roi = Roi.from_file(
        site_params.roi_path,
        level=site_params.level,
        features_root=site_params.output_path,
        lazy_tables=False,  # FIXME: Bug where LazyFrame.collect() fails with message: exceptions.ComputeError: TypeError: 'bytes' object is not callable
    )
    logger.debug(f"\n{lazy_roi}\n")

    logger.info("Validating resources.")
    site_params = site_params.validate_parameters_with(lazy_roi)

    logger.info("Selecting resources.")
    lazy_roi_resources = lazy_roi.sel(
        l=list(site_params.features.resources.label_images),
        c=list(site_params.features.resources.channels),
    )
    logger.debug(f"\n{lazy_roi_resources}\n")

    logger.info(f"{title('Intensity Correction')}")
    # TODO: t_decay_models & correction

    logger.info("Loading z-decay models.")
    z_decay_models = read_models(params.intensity_correction.z_decay_models)
    logger.debug(f"{list(map(nested_repr, z_decay_models))}")

    uncorrected_channels = set(site_params.features.resources.channels).difference(
        z_decay_models.keys()
    )

    if len(uncorrected_channels) > 0:
        logger.warning(f"Uncorrected channels: {uncorrected_channels}")

    # Set model to None if no model is found for a channel.
    z_decay_models = {
        **{k: None for k in site_params.features.resources.channels},
        **z_decay_models,
    }
    logger.debug(f"{highlight('Models')}\n{z_decay_models}")
    logger.info("Applying z correction")
    lazy_roi_resources_corr = apply_z_decay_models_to_roi(
        models=z_decay_models,
        roi=lazy_roi_resources,
        two_step_label=params.intensity_correction.z_decay_two_step_label,
    )

    logger.info(f"{title('Feature Extration')}")
    queries = site_params.features.queries(lazy_roi_resources_corr)
    logger.info(f"{nested_repr([q.__class__.__name__ for q in queries])}")
    tables = process_queries(queries, lazy_roi_resources_corr)

    logger.info("Writing tables...\n")
    lazy_roi_resources_corr.tables = tables
    lazy_roi_resources_corr.write_tables()
    logger.info(f"{title('Done')}")


def title(s: str) -> str:
    return f"{' '+s+' ':=^60}"


def highlight(s: str) -> str:
    return f"{' '+s+' ':-^40}"


def pydantic_yaml(model, default_flow_style=True) -> str:
    return f"{highlight(model.__class__.__name__)}\n{to_yaml_str(model, default_flow_style=default_flow_style, indent=2)}\n"


def pydantic_json(model) -> str:
    return f"{highlight(model.__class__.__name__)}\n{model.model_dump_json(indent=2)}\n"


def nested_repr(
    object_, indent=1, width=120, depth=None, compact=True, sort_keys=False
):
    if isinstance(object_, BaseModel):
        if pydantic.version.VERSION < '2':
            dict_ = json.loads(object_.json())
        else:
            dict_ = json.loads(object_.model_dump_json())
    else:
        dict_ = object_

    name = object_.__class__.__name__
    formatted = pformat(
        dict_,
        indent=indent,
        width=width,
        depth=depth,
        compact=compact,
        sort_dicts=sort_keys,
    )
    return f"{highlight(name)}\n{formatted}\n"


def process_queries(
    queries: list["FeatureQuery"], roi: Roi
) -> "dict[str, pl.DataFrame]":
    result = defaultdict(list)

    for query in queries:
        logger.info(f"{nested_repr(query)}")
        result[query.label_image].append(query.compute(roi))

    return {k: pu.join(v, on="label") for k, v in result.items()}


if __name__ == "__main__":
    main()


# %%
