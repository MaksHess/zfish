# %%
from pathlib import Path

import polars as pl
import polars.selectors as cs
from sklccp.ccp_transformers import PolarsSelectorNew
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from zfish.classifiers.clf_io import load_classifier
from zfish.classifiers.utils import (
    get_parameter_grid,
    get_pipeline,
    load_pipeline,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_grid_search,
    plot_roc_curve,
    save_pipeline,
    train_test_split,
)
from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_selector import sel
from zfish.features.polars_utils import (
    drop_null_columns,
    plot_df_nulls,
    read_table,
    unnest_all_structs,
)

# %% Load features & annotations
classifier_name = "cellcycle2c"
target = "ann4"  # ann0/1: 6 classes,  ann2: 4 classes, ann3: 3 classes, ann4: 2 classes
label_order = ['M', 'S']

# classifier_name = "cellcycle3c"
# target = "ann3"  # ann0/1: 6 classes,  ann2: 4 classes, ann3: 3 classes, ann4: 2 classes
# label_order = ['M pro/meta', 'M ana/telo', 'S']

# classifier_name = "cellcycle4c"
# target = "ann2"  # ann0/1: 6 classes,  ann2: 4 classes, ann3: 3 classes, ann4: 2 classes
# label_order = ['M pro/meta', 'M ana/telo', 'S early', 'S']

# classifier_name = 'cellcycle6c'
# target = "ann0"
# label_order = ["M pro", "M meta", "M ana", "M telo", "S early", "S"]


features = sel.label | sel.intensity | sel.density | sel.correlation| sel.distance

run_grid_search = True

# input paths
root_path = Path(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers")
fn_annotations = (
    root_path / "annotations/ann_cellcycle_clean.parquet"
)  # annotations without debris
fld_features = Path(
    r"C:\Users\hessm\Documents\zfish_local\features_tcorr\Linear(loss='huber', features=['MediumPath', 'EmbryoPath'])"
)

# output paths
out_fn_clf = root_path / f"trained_classifiers/{classifier_name}_clf.pkl"
out_fn_pred = root_path / f"predictions/{classifier_name}_pred.parquet"
out_fn_clf.parent.mkdir(exist_ok=True)
out_fn_pred.parent.mkdir(exist_ok=True)


df_ann = pl.read_parquet(fn_annotations).with_columns(pl.col("label").cast(pl.Int64))
df_nuc_raw = (
    read_table(fld_features, _object="nucleiRaw3")
    .select(sel.index, ~sel.index)
    .pipe(unnest_all_structs)
)

# CONTROL_WELLS = ["B07", "C07", "D07", "E07"]
# df_meta_raw = get_metadata(
#     df_nuc_raw, drop_objects_with_no_parents=False, control_wells=CONTROL_WELLS
# )

# %% Generate X and y (X can still contain the index, features that don't match `features` are dropped in the pipeline)
# merge annotations and features, remove columns with lots of nans
df_Xy = (
    df_ann.join(df_nuc_raw, on=["roi", "object", "label"])
    .fill_nan(None)
    .pipe(drop_null_columns, strategy="perc", allowed_percentage=0.01)
    .with_columns(sel.correlation.fill_null(0))
    .pipe(drop_null_columns, strategy="any")
)

df_Xy.pipe(plot_df_nulls)

X = df_Xy.drop(target)
y = df_Xy[target].cast(pl.Enum(label_order))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


# %% Get classification pipeline and parameter grid
if run_grid_search:
    pip = get_pipeline(features)
    params = get_parameter_grid()

# %% Run grid search and plot results
if run_grid_search:
    grid_search = GridSearchCV(pip, params, cv=4, verbose=2)

    grid_search.fit(X_train, y_train.to_pandas())

    fig_grid_search = plot_grid_search(grid_search, output_file = out_fn_clf.parent / f"{classifier_name}_plot_GridSearch.html")
fig_grid_search
# %% Select the best estimator, compute scoring metrics, plot feature importance
if run_grid_search:
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    print(f"Accuracy score: {clf.score(X_test, y_test)}")
else:
    clf = load_pipeline(out_fn_clf)
# %% Plot classifier scores & feature importance
cm_display = plot_confusion_matrix(
    clf,
    X_test,
    y_test,
    xticks_rotation="vertical",
    normalize=None,
    labels=label_order,
    output_file=out_fn_clf.parent / f"{classifier_name}_plot_ConfusionMatrix.png",
)
roc_display = plot_roc_curve(
    clf,
    X_test,
    y_test,
    output_file=out_fn_clf.parent / f"{classifier_name}_plot_RocCurve.png",
)


plot_feature_importance(clf, n_features=30, output_file=out_fn_clf.parent / f"{classifier_name}_plot_FeatureImportance.html")

# %% Apply the classifier on all objects
celltype_proba = clf.predict_proba(df_nuc_raw)
celltype_pred = clf.predict(df_nuc_raw)

# %% Save final classifier and predictions / prediction-probas
save_pipeline(clf, out_fn_clf)

df_pred = (
    df_nuc_raw.select(["roi", "object", "label"])
    .with_columns(pl.Series(f"{classifier_name}_pred", celltype_pred, dtype=pl.Enum(label_order)))
    .with_columns(
        pl.DataFrame(celltype_proba, schema=list(clf.classes_)).select(
            pl.all().name.suffix("_proba")
        )
    )
)
df_pred.write_parquet(out_fn_pred)

# # %% Sanity check plots
# CONTROL_WELLS = ["B07", "C07", "D07", "E07"]
# df_meta_raw = get_metadata(
#     df_nuc_raw, drop_objects_with_no_parents=False, control_wells=CONTROL_WELLS
# )


# to_plot_clean = pl.concat(
#     [
#         df_pred.filter(pl.col("M meta_proba") > 0.8),
#         df_pred.filter(pl.col("M ana_proba") > 0.8),
#     ]
# ).select(['roi', 'object', 'label', f'{classifier_name}_pred']).join(df_nuc_raw, on=['roi', 'object', 'label']).join(df_meta_raw, on=['roi'])


# to_plot = df_pred.filter(pl.col('cellcycle6c_pred').is_in(['M meta', 'M ana'])).select(['roi', 'object', 'label', f'{classifier_name}_pred']).join(df_nuc_raw, on=['roi', 'object', 'label']).join(df_meta_raw, on=['roi'])
# # %%
# to_plot.hvplot.box(by=f'{classifier_name}_pred', y='PhysicalSize', width=250, height=200, grid=True, ylim=(0, 2000))

# # %%
# to_plot.sort('cellcycle6c_pred').hvplot.box(by=['cycle', f'{classifier_name}_pred'], y='PhysicalSize', ylim=(0, 1500), width=800, height=300, grid=True)

# # %%

# %%
