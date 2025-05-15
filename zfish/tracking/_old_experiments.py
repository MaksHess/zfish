# %%
import zarr

from zfish.tracking.io import load_image

img = load_image(r"E:\sshami\Visiscope\20231026H1A488_compressed\20231026H1A1_s2.zarr", level=2, scale=(1.0,  2.3, 2.3))

# %%
arr = zarr.open(r"E:\sshami\Visiscope\20231026H1A488_compressed\20231026H1A1_s2.zarr")
# %%
import napari

from zfish.visualize.imshow import imshow_spatial_image

viewer = napari.Viewer()
# imshow_spatial_image(img, viewer)
viewer.add_image(img.to_numpy(), scale=(1, 1.0, 1.3, 1.3))



# %% Attempt 4.1
results = []
for params in chain(
    parameter_gen(
        max_search_radius=(15,),
        max_lost=(0, 1, 3),
        features=(
            ("PhysicalSize",),
            ("Roundness", "Flatness", "Elongation"),
            ("H1A_Median",),
            ("H1A_StandardDeviation",),
            ("H1A_Skewness",),
            ("H1A_Kurtosis",),
            ("H1A_Median", "H1A_StandardDeviation", "H1A_Skewness", "H1A_Kurtosis"),
        ),
        tracking_updates=(("motion", "visual"),),
        optimize=(True,),
    ),
):
    results.append(run_tracking(objs, params))
# %% Attempt 4
results = []
for params in chain(
    parameter_gen(
        max_search_radius=(15,),
        max_lost=(0, 1, 3),
        features=(None,),
        tracking_updates=(("motion",),),
        optimize=(True,),
    ),  
):
    results.append(run_tracking(objs, params))
# %% Attempt 3
results = []
for params in chain(
    parameter_gen(
        max_search_radius=(10, 15),
        max_lost=(1, 0),
        features=(None,),
        tracking_updates=(("motion",),),
    ),
):
    results.append(run_tracking(objs, params))
# %% Attempt 2
results = []
for params in chain(
    parameter_gen(
        max_search_radius=(10,),
        features=(
            # None,
            # ("PhysicalSize",),
            # ("Roundness", "Flatness", "Elongation"),
            ("H1A_Median",),
            ("H1A_StandardDeviation",),
            ("H1A_Skewness",),
            ("H1A_Kurtosis",),
            ("H1A_Median", "H1A_StandardDeviation", "H1A_Skewness", "H1A_Kurtosis"),
        ),
        tracking_updates=(("motion", "visual"),),
    ),
    parameter_gen(
        max_search_radius=(15,),
        features=(
            None,
            ("PhysicalSize",),
            ("Roundness", "Flatness", "Elongation"),
            ("H1A_Median",),
            ("H1A_StandardDeviation",),
            ("H1A_Skewness",),
            ("H1A_Kurtosis",),
            ("H1A_Median", "H1A_StandardDeviation", "H1A_Skewness", "H1A_Kurtosis"),
        ),
        tracking_updates=(("motion", "visual"),),
    ),
):
    results.append(run_tracking(objs, params))
# %%
old_params = []
for params in chain(
    parameter_gen(max_search_radius=(5, 10, 20, 50, 100, 200)),
    parameter_gen(
        # max_search_radius=(20, 50),
        features=(
            None,
            ("PhysicalSize",),
            ("PhysicalSize", "Roundness"),
            ("PhysicalSize", "Roundness", "Flatness"),
            (
                "PrincipalAxes.a-x",
                "PrincipalAxes.a-y",
                "PrincipalAxes.a-z",
                "PrincipalAxes.b-x",
                "PrincipalAxes.b-y",
                "PrincipalAxes.b-z",
                "PrincipalAxes.c-x",
                "PrincipalAxes.c-y",
                "PrincipalAxes.c-z",
            ),
            (
                "PrincipalAxes.b-x",
                "PrincipalAxes.c-y",
                "PrincipalAxes.a-x",
                "EquivalentSphericalPerimeter",
                "EquivalentSphericalRadius",
                "Elongation",
                "Roundness",
                "PhysicalSize",
                "EquivalentEllipsoidDiameter.a",
                "Perimeter",
                "PrincipalAxes.c-x",
                "EquivalentEllipsoidDiameter.b",
                "PrincipalAxes.b-y",
                "PrincipalAxes.c-z",
                "PrincipalAxes.b-z",
                "EquivalentEllipsoidDiameter.c",
                "PrincipalAxes.a-y",
                "PrincipalAxes.a-z",
                "Flatness",
            ),
            # tuple(df.drop("label", "x", "y", "z", "t").columns),
        ),
        tracking_updates=(
            ("motion",),
            ("motion", "visual"),
        ),
    ),
):
    old_params.append(params)
    
def name_from_params(
    param, include=("max_search_radius", "tracking_updates", "features")
):
    short_names = {
        "max_search_radius": "ms",
        "tracking_updates": "tu",
        "features": "feat",
    }
    included_params = {short_names.get(k, k): param[k] for k in include}
    return ";".join(f"{k}={v}" for k, v in included_params.items())


names = [
    f"ms={params.max_search_radius}; tu={','.join(e[0] for e in params.tracking_updates)}; feat={params.features}"
    for params in old_params
]

