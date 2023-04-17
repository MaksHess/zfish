# %%
import argparse

# from pathlib import Path
# from typing import Any
# import yaml
# from devtools import debug
from zfish.features.correlation import get_colocalization_features
from zfish.features.distance import get_distance_features
from zfish.features.feature_extraction_parameters import (
    ROI,
    FeatureExtractionParams,
    load_roi,
)
from zfish.features.intensity import get_intensity_features
from zfish.features.label import get_label_features

from zfish.visualize.imshow import imshow
    

# %%

params = FeatureExtractionParams.parse_file("feature_extraction.yaml")
site_params = params.get_roi_by_index(1).validate_roi()
lazy_roi = load_roi(site_params.roi_path, level=1)
roi = ROI(
    channel_images=lazy_roi.channel_images.sel(
        c=list(site_params.features.resources.channels)
    ),
    label_images=lazy_roi.label_images.sel(
        c=list(site_params.features.resources.labels)
    ),
).compute()

features = {label: dict() for label in site_params.features.resources.labels}

# %%
for label in site_params.features.resources.labels:
    print(f"starting feature extraction for {label}...")
    if label in site_params.features.label.labels:
        print("extracting label features...")
        label_image = roi.label_images.sel(c=label)
        features[label]["label"] = get_label_features(label_image)

    if label in site_params.features.intensity.labels:
        print("extracting intensity features...")
        label_image = roi.label_images.sel(c=label)
        features[label]["intensity"] = []
        for channel in site_params.features.intensity.channels:
            print(f"channel: {channel}")
            channel_image = roi.channel_images.sel(c=channel)
            features[label]["intensity"].append(
                get_intensity_features(label_image, channel_image)
            )

    if label in site_params.features.correlation.labels:
        print("extracting correlation features...")
        label_image = roi.label_images.sel(c=label)
        features[label]["correlation"] = []
        for channel1, channel2 in site_params.features.correlation.channel_pairs:
            print(f"channel pair: {(channel1, channel2)}")
            channel_image1 = roi.channel_images.sel(c=channel1)
            channel_image2 = roi.channel_images.sel(c=channel2)
            features[label]["correlation"].append(
                get_colocalization_features(label_image, channel_image1, channel_image2)
            )

    if label in site_params.features.distance.labels:
        print("extracting distance features...")
        label_image = roi.label_images.sel(c=label)
        features[label]["distance"] = []
        for label_to, label_id in site_params.features.distance.label_objects:
            print(f"label object: {(label_to, label_id)}")
            label_image_to = roi.label_images.sel(c=label_to)
            features[label]["distance"].append(
                get_distance_features(label_image, label_image_to, label_id)
            )
    print()

# %%
import spatialdata_plot
from spatialdata.datasets import blobs
# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("-p", "--feature_extraction_parameters", type=str)
    args = parser.parse_args()

    params = FeatureExtractionParams.parse_file(args.feature_extraction_parameters)
    site_params = params.get_roi_by_index(args.idx).validate_roi()

    roi = load_roi(site_params.roi_path)
    features = dict()


if __name__ == "__main__":
    main()
