import argparse
import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from itertools import chain, product
from pathlib import Path
from pprint import pformat
from typing import TypeAlias

import btrack
import polars as pl
from btrack.constants import BayesianUpdates

from zfish.tracking.io import load_features

logger = logging.getLogger(__name__)

Volume: TypeAlias = tuple[tuple[float, float], ...]

FEATURES = [
    "PhysicalSize",
    "Roundness",
    "H1A_Median",
    "H1A_StandardDeviation",
    "H1A_Skewness",
    "H1A_Kurtosis",
]

TRACKER_CONFIG = [
    "features",
    "update_method",
    "volume",
    "tracking_updates",
    "max_search_radius",
    "optimizer_options",
]
MOTION_CONFIG = ["max_lost"]
HYPOTHESIS_CONFIG = [
    "lambda_time",
    "lambda_dist",
    "lambda_link",
    "lambda_branch",
    "theta_dist",
    "theta_time",
    "dist_thresh",
    "time_thresh",
    "apop_thresh",
    "segmentation_miss_rate",
    "apoptosis_rate",
    "relax",
]
OTHER_CONFIG = [
    "optimize",
]


@dataclass
class Parameters:
    # Tracker
    optimize: bool = True
    features: tuple[str, ...] = tuple()
    update_method: BayesianUpdates = BayesianUpdates.APPROXIMATE
    volume: Volume = ((0, 665.6), (0, 665.6), (0, 251.0))
    tracking_updates: tuple[str, ...] = ("motion",)  # ("motion", "visual")
    max_search_radius: float = 15
    optimizer_options: dict[str, int] = None
    
    # Motion model
    max_lost: int = 1

    # Hypothesis model
    lambda_time: float = 5.0
    lambda_dist: float = 3.0
    lambda_link: float = 10.0
    lambda_branch: float = 20.0

    theta_dist: float = 20.0
    theta_time: float = 5.0
    dist_thresh: float = 40.0
    time_thresh: float = 2.0
    apop_thresh: float = 5
    segmentation_miss_rate: float = 0.1
    apoptosis_rate: float = 0.001
    relax: bool = True
    
    def __post_init__(self):
        if self.optimizer_options is None:
            self.optimizer_options = {'tm_lim': 6_000_000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("-p", "--path_to_input", type=str)
    args = parser.parse_args()

    output_folder = Path(args.path_to_input).parent / "tracking_results"
    output_folder.mkdir(exist_ok=True)

    base_name = Path(args.path_to_input).stem
    tracks_out_file = output_folder / f"{base_name}_{args.idx}_tracks.h5"
    config_out_file = output_folder / f"{base_name}_{args.idx}_config.json"

    base_config = btrack.config.load_config(
        r"/data/active/marvwy/VisiScope/20230329_compressed/napari_v2.json"
    )
    
    df = (
        load_features(args.path_to_input).select(['t', 'z', 'y', 'x'] + FEATURES).with_columns(pl.lit(1).alias('Constant'))
    )
    objs = btrack.io.objects_from_array(df.to_numpy(), default_keys=df.columns)

    parameter_generators = [
        parameter_gen(
            time_thresh=(1.0, 2.0),
            dist_thresh=(15.0, 20.0, 30.0, 45.0),
        ),
        parameter_gen(
            lambda_branch=(20.0, 40.0, 80.0),
            dist_thresh=(15.0, 30.0, 90),
        ),
    ]
    

    all_parameters = list(chain(*parameter_generators))
    parameters = all_parameters[int(args.idx)]
    
    print(all_parameters)
    print()
    print(parameters)

    for k, v in asdict(parameters).items():
        if k in TRACKER_CONFIG:
            setattr(base_config, k, v)
        elif k in MOTION_CONFIG:
            setattr(base_config.motion_model, k, v)
        elif k in HYPOTHESIS_CONFIG:
            setattr(base_config.hypothesis_model, k, v)
        elif k in OTHER_CONFIG:
            continue
        else:
            raise ValueError(f"Unknown argument {k}")
        
    with open(config_out_file, 'w') as f:
        f.write(base_config.json(indent=2))
        

    with btrack.BayesianTracker() as tracker:
        tracker.configure(base_config)
    #     # tracker.features = FEATURES
        tracker.append(objs)
    #     tracker.max_lost = 1
    #     tracker.update_method = BayesianUpdates.APPROXIMATE
    #     tracker.max_search_radius = 15
    #     tracker.volume = ((0, 300), (0, 300), (0, 251.0))
    #     tracker.track(tracking_updates=["motion"])
    #     hypoth = tracker.optimise(options={"tm_lim": 60_000 * 100})
        if parameters.optimize:
            optimized = tracker.optimise()
            
        tracker.export(tracks_out_file, obj_type='obj_type_1')

    #     tracker.export(
    #         r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\tracking\quick2.h5",
    #         obj_type="obj_type_1",
    #     )
    #     data, properties, graph = tracker.to_napari()
    #     tracks = tracker.tracks


def parameter_gen(
    optimize: tuple[bool, ...] | None = None,
    features: tuple[tuple[str, ...], ...] | None = None,
    update_method: tuple[BayesianUpdates, ...] | None = None,
    volume: tuple[Volume, ...] | None = None,
    tracking_updates: tuple[tuple[str, ...], ...] | None = None,
    max_search_radius: tuple[float, ...] | None = None,
    max_lost: tuple[int, ...] | None = None,
    lambda_time: float | None = None,
    lambda_dist: float | None = None,
    lambda_link: float | None = None,
    lambda_branch: float | None = None,
    theta_dist: float | None = None,
    theta_time: float | None = None,
    dist_thresh: float | None = None,
    time_thresh: float | None = None,
    apop_thresh: float | None = None,
    segmentation_miss_rate: float | None = None,
    apoptosis_rate: float | None = None,
    relax: bool | None = None,
):
    params_set = {k: v for k, v in locals().items() if v is not None}
    for value_pair in product(*params_set.values()):
        yield Parameters(**{k: v for k, v in zip(params_set.keys(), value_pair)})


if __name__ == '__main__':
    main()