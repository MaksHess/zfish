# %%
from pathlib import Path

import polars as pl
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV

from zfish.classifiers.utils import (
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
classifier_name = "debris"
run_grid_search = False
features = sel.label | sel.intensity | sel.density | sel.correlation | sel.distance
target = (
    "is_debris"  # ann0/1: 6 classes,  ann2: 4 classes, ann3: 3 classes, ann4: 2 classes
)

# input paths
root_path = Path(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers")
fn_annotations = (
    root_path / "annotations/ann_is_debris.parquet"
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

# %% Split into train test (stratifying over y since there's not a lot of debris annotations)
X = df_Xy.drop(target)
y = df_Xy[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

print("Training data:")
print(y_train.value_counts())
print()
print("Testing data:")
print(y_test.value_counts())

# %% Get classification pipeline and parameter grid
if run_grid_search:
    from feature_engine.selection import DropCorrelatedFeatures
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler

    pip = get_pipeline(features)
    pip.set_params(
        **{
            "classifier": RandomForestClassifier(
                class_weight="balanced", oob_score=False
            )
        }
    )
    params = [
        {
            "imputer": [KNNImputer()],
            "scaler": [StandardScaler()],
            "reducer": [PCA()],
            "reducer__n_components": [10, 20, 50],
            "classifier__n_estimators": [20, 50, 100],
        },
        {
            "reducer": [DropCorrelatedFeatures()],
            "reducer__method": ["spearman"],
            "reducer__threshold": [0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
            "classifier__n_estimators": [20, 50, 100],
        },
        {
            "reducer": ["passthrough"],
            "classifier__n_estimators": [20, 50, 100],
        },
    ]

# %% Run grid search
if run_grid_search:
    grid_search = GridSearchCV(pip, params, cv=3, verbose=2)

    grid_search.fit(X_train, y_train.to_pandas())

# %% Select the best estimator, compute scoring metrics, plot 
if run_grid_search:
    fig_grid_search = plot_grid_search(
        grid_search,
        n_best=30,
        output_file=out_fn_clf.parent / f"{classifier_name}_plot_GridSearch.html",
    )

    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    print(f"Accuracy score: {clf.score(X_test, y_test)}")
else:
    clf = load_pipeline(out_fn_clf)
    fig_grid_search = None

fig_grid_search

# %%
plt.style.use({"figure.figsize": (1.8, 1.6), "figure.constrained_layout.use": False, "font.family" : "bahnschrift"})

cm_display = plot_confusion_matrix(
    clf,
    X_test,
    y_test,
    xticks_rotation="vertical",
    normalize=None,
    output_file=out_fn_clf.parent / f"{classifier_name}_plot_ConfusionMatrix.png",
)
roc_display = plot_roc_curve(
    clf,
    X_test,
    y_test,
    output_file=out_fn_clf.parent / f"{classifier_name}_plot_RocCurve.png",
)

# %%
plot_feature_importance(clf, n_features=30, output_file=out_fn_clf.parent / f"{classifier_name}_plot_FeatureImportance.html")

# %% Apply the classifier on all objects
debris_proba = clf.predict_proba(df_nuc_raw)
debris_pred = clf.predict(df_nuc_raw)

# %% Save final classifier and predictions / prediction-probas / plots
save_pipeline(clf, out_fn_clf)

df_pred = (
    df_nuc_raw.select(["roi", "object", "label"])
    .with_columns(pl.Series(f"{classifier_name}_pred", debris_pred))
    .with_columns(pl.DataFrame(debris_proba[:, 1], schema=[f"{classifier_name}_proba"]))
)
df_pred.write_parquet(out_fn_pred)

# %% Sanity check plots
CONTROL_WELLS = ["B07", "C07", "D07", "E07"]
df_meta_raw = get_metadata(
    df_nuc_raw, drop_objects_with_no_parents=False, control_wells=CONTROL_WELLS
)

df_nuc = df_nuc_raw.join(df_pred, on=["roi", "object", "label"]).filter(
    pl.col("debris_pred").not_()
)
df_meta = get_metadata(
    df_nuc, drop_objects_with_no_parents=False, control_wells=CONTROL_WELLS
)
