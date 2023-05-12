# %%
import argparse
from collections import defaultdict
from dataclasses import dataclass

from zfish.features.correlation import get_colocalization_features
from zfish.features.distance import get_distance_features
from zfish.features.feature_extraction_parameters import FeatureExtractionParams
from zfish.features.intensity import get_intensity_features
from zfish.features.label import get_label_features
from zfish.features.polars_utils import join
from zfish.roi.spatial_roi import Roi


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("-p", "--feature_extraction_parameters", type=str)
    args = parser.parse_args()

    params = FeatureExtractionParams.parse_file(args.feature_extraction_parameters)
    site_params = params.get_roi_by_index(args.idx).validate_roi()

    lazy_roi = Roi.from_file(
        site_params.roi_path,
        level=site_params.level,
        features_root=site_params.output_path,
        lazy_tables=True,
    )
    roi = lazy_roi.sel(
        l=list(site_params.features.resources.label_images),
        c=list(site_params.features.resources.channels),
    )

    features = defaultdict(list)

    for label in site_params.features.resources.label_images:
        label_image = roi.sel(l=label).labels.compute()
        print(f"starting feature extraction for {label}...")
        if label in site_params.features.label.labels:
            print("extracting label features...")
            features[label].append(get_label_features(label_image))

        if label in site_params.features.intensity.labels:
            print("extracting intensity features...")
            for channel in site_params.features.intensity.channels:
                print(f"channel: {channel}")
                channel_image = roi.sel(c=channel).images.compute()
                features[label].append(get_intensity_features(label_image, channel_image))

        if label in site_params.features.correlation.labels:
            print("extracting correlation features...")
            for channel1, channel2 in site_params.features.correlation.channel_pairs:
                print(f"channel pair: {(channel1, channel2)}")
                channel_image1 = roi.sel(c=channel1).images.compute()
                channel_image2 = roi.sel(c=channel2).images.compute()
                features[label].append(
                    get_colocalization_features(label_image, channel_image1, channel_image2)
                )

        if label in site_params.features.distance.labels:
            print("extracting distance features...")
            for label_to, label_id in site_params.features.distance.label_objects:
                print(f"label object: {(label_to, label_id)}")
                label_image_to = roi.sel(l=label_to).labels.compute()
                features[label].append(
                    get_distance_features(label_image, label_image_to, label_id)
                )
        print()

    tables = {k: join(v, on="label") for k, v in features.items()}
    roi.tables = tables
    roi.write_tables()

if __name__ == "__main__":
    main()
