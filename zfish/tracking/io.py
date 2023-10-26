import numpy as np
import polars as pl

# %% Feature extraction function
LABEL_FEATURES = [
    "PhysicalSize",
    "Elongation",
    "Flatness",
    "Roundness",
    "Perimeter",
    "EquivalentSphericalPerimeter",
    "EquivalentSphericalRadius",
    "Centroid",
    "PrincipalAxes",
    "EquivalentEllipsoidDiameter",
]
INTENSITY_FEATURES = [
    "Mean",
    "Median",
    "Kurtosis",
    "Skewness",
    "Maximum",
    "Minimum",
    "Variance",
    "StandardDeviation",
]


def extract_features(lbls, img=None, write_to=None):
    from zfish.features.label import get_label_features
    from zfish.features.polars_utils import unnest_all_structs

    dfs = []
    for i in range(len(lbls)):
        print(f"extracting label features t={i}")
        df = get_label_features(
            lbls[i],
            features=LABEL_FEATURES,
        )

        if img is not None:
            from zfish.features.intensity import get_intensity_features

            print(f"extracting intensity features t={i}")
            df_intensity = get_intensity_features(
                lbls[i], img[i], features=INTENSITY_FEATURES
            )
            df = df.join(df_intensity, on="label")

        dfs.append(
            df.pipe(unnest_all_structs)
            .rename(
                {
                    "Centroid.x": "x",
                    "Centroid.y": "y",
                    "Centroid.z": "z",
                    "label": "img_label",
                }
            )
            .with_columns(pl.lit(i).alias("t"))
        )

    df_out = pl.concat(dfs)
    if write_to is not None:
        df_out.write_parquet(write_to)
    return df_out


# %% Image, label & feature loaders
def load_labels(fn, scale=(1, 1.0, 0.65, 0.65), dims=("t", "z", "y", "x")):
    import zarr

    from zfish.image.image import to_si

    zarray = zarr.open(fn)
    n_images = len(list(zarray))

    lbls = np.empty((n_images, 251, 1024, 1024), dtype=np.uint16)
    for i in range(n_images):
        print(f"loading labelimage t={i}")
        lbls[i, :] = zarray[i][...]
    return to_si(lbls, dims=dims, scale=scale)


def load_image(
    fn,
    level=1,
    scale=(1.0, 0.65, 0.65),
    dims=("t", "c", "z", "y", "x"),
    c_coords=["H1A"],
):
    import zarr

    from zfish.image.image import to_si

    arr_img = zarr.open(fn)
    img = arr_img[level][...]
    return to_si(img, dims=dims, scale=scale, c_coords=c_coords).squeeze()


def load_features(fn):
    import polars as pl

    return pl.read_parquet(fn)
