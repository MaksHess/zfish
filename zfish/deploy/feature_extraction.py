# %%
import argparse
from collections import defaultdict

from zfish.features.correlation import get_colocalization_features
from zfish.features.distance import get_distance_features
from zfish.features.feature_extraction_parameters import FeatureExtractionParams
from zfish.features.intensity import get_intensity_features
from zfish.features.label import get_label_features
from zfish.features.polars_utils import join
from zfish.roi.spatial_roi import Roi, apply_z_decay_models_to_roi, read_models

# # %%
# idx = 21
# feature_extraction_parameters = r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\deploy\feature_extraction.yaml"

# params = FeatureExtractionParams.parse_file(feature_extraction_parameters)
# site_params = params.get_roi_by_index(idx).validate_roi()

# lazy_roi = Roi.from_file(
#     site_params.roi_path,
#     level=site_params.level,
#     features_root=site_params.output_path,
#     lazy_tables=False,
# )
# lazy_roi_resources = lazy_roi.sel(
#     l=list(site_params.features.resources.label_images),
#     c=list(site_params.features.resources.channels),
# )
# z_decay_models = read_models(params.intensity_correction.z_decay_models)

# # Set model to None if no model is found for a channel.
# if site_params.intensity_correction.z_decay_default_models is None:
#     print(f"Uncorrected channels: {set(site_params.features.resources.channels).difference(z_decay_models.keys())}")
#     z_decay_models = {
#         **{k: None for k in site_params.features.resources.channels},
#         **z_decay_models,
#     }

# lazy_roi_resources_corr = apply_z_decay_models_to_roi(
#     models=z_decay_models,
#     roi=lazy_roi_resources,
#     two_step_label=params.intensity_correction.z_decay_two_step_label
# )

# # %%
# import napari

# from zfish.roi.visualize import imshow_roi

# img = lazy_roi_resources.sel(c='DAPI.1').drop_dim('l').compute()
# img_corr = lazy_roi_resources_corr.sel(c='DAPI.1').drop_dim('l').compute()

# viewer = imshow_roi(img)
# viewer = imshow_roi(img_corr, viewer)


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
    lazy_roi_resources = lazy_roi.sel(
        l=list(site_params.features.resources.label_images),
        c=list(site_params.features.resources.channels),
    )

    z_decay_models = read_models(params.intensity_correction.z_decay_models)

    # Set model to None if no model is found for a channel.
    if site_params.intensity_correction.z_decay_default_models is None:
        print(f"Uncorrected channels: {set(site_params.features.resources.channels).difference(z_decay_models.keys())}")
        z_decay_models = {
            **{k: None for k in site_params.features.resources.channels},
            **z_decay_models,
        }

    lazy_roi_resources_corr = apply_z_decay_models_to_roi(
        models=z_decay_models,
        roi=lazy_roi_resources,
        two_step_label=params.intensity_correction.z_decay_two_step_label,
    )

    features = defaultdict(list)

    for label in site_params.features.resources.label_images:
        label_image = lazy_roi_resources_corr.sel(l=label).labels.compute()
        print(f"starting feature extraction for {label}...")
        if label in site_params.features.label.labels:
            print("extracting label features...")
            features[label].append(get_label_features(label_image))

        if label in site_params.features.intensity.labels:
            print("extracting intensity features...")
            for channel in site_params.features.intensity.channels:
                print(f"channel: {channel}")
                channel_image = lazy_roi_resources_corr.sel(c=channel).images.compute()
                features[label].append(
                    get_intensity_features(label_image, channel_image)
                )

        if label in site_params.features.correlation.labels:
            print("extracting correlation features...")
            for channel1, channel2 in site_params.features.correlation.channel_pairs:
                print(f"channel pair: {(channel1, channel2)}")
                channel_image1 = lazy_roi_resources_corr.sel(
                    c=channel1
                ).images.compute()
                channel_image2 = lazy_roi_resources_corr.sel(
                    c=channel2
                ).images.compute()
                features[label].append(
                    get_colocalization_features(
                        label_image, channel_image1, channel_image2
                    )
                )

        if label in site_params.features.distance.labels:
            print("extracting distance features...")
            for label_to, label_id in site_params.features.distance.label_objects:
                print(f"label object: {(label_to, label_id)}")
                label_image_to = lazy_roi_resources_corr.sel(
                    l=label_to
                ).labels.compute()
                features[label].append(
                    get_distance_features(label_image, label_image_to, label_id)
                )
        print()

    tables = {k: join(v, on="label") for k, v in features.items()}
    lazy_roi_resources_corr.tables = tables
    lazy_roi_resources_corr.write_tables()


if __name__ == "__main__":
    main()
