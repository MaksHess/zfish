# %%
import logging
from dataclasses import dataclass
from functools import partial, reduce
from operator import or_
from typing import TYPE_CHECKING, Literal, Union

import polars as pl
import polars.selectors as cs
from polars.type_aliases import SelectorType
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from zfish.plot.features import plot_features_by_embryo
from zfish.preprocessing.types import AnyFrameT

if TYPE_CHECKING:
    from zfish.preprocessing.pipeline import CcpParams

LOG_TRANSFORM_PATTERNS = [
    cs.matches(e)
    for e in [
        "^.*_Mean$",
        "^.*_Median$",
        "^.*_Maximum$",
        "^.*_Minimum$",
        "^.*_StandardDeviation$",
        "^.*_Variance$",
    ]
]

LOG_TRANSFORM_SELECTOR = reduce(or_, LOG_TRANSFORM_PATTERNS)

Scaler = Union[StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, SplineTransformer]

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LogTransformParams:
    strategy: Literal["skip", "log1p"] = "log1p"
    columns: SelectorType = reduce(or_, LOG_TRANSFORM_PATTERNS)
    verbose: bool = True

@dataclass(frozen=True)
class ScalerParams:
    scaler_instance: Scaler | None = None
    features: SelectorType = cs.all()  # w/ resp. to FeatureSelectionParams.all_features
    over: "SelectorType | None" = None


@dataclass(frozen=True)
class MultiScalerParams:
    scalers: tuple[ScalerParams, ...] = (
        ScalerParams(PowerTransformer(), features=cs.all()),
        ScalerParams(
            SplineTransformer(n_knots=5, knots="quantile", extrapolation="periodic"),
            features=cs.by_name("Roundness"),
        ),
    )
    verbose: bool = True
    plot_results: bool = False


def handle_scalers(df, params: "CcpParams") -> pl.DataFrame:
    scaler_params = params.normalize.scalers
    selection_params = params.selection.feature_selection

    df_out = df.clone()
    for scaler_param in scaler_params.scalers:
        features = cs.expand_selector(
            df, (scaler_param.features & selection_params.features)
        )
        over = cs.expand_selector(df, scaler_param.over) if scaler_param.over else None
        name = scaler_param.scaler_instance.__class__.__name__
        if scaler_params.verbose:
            logger.info(f"{name}: {features} 'over': {over}")
            logger.info(f"{df_out.select(features).describe()}")

        if scaler_params.plot_results:
            plot_features_by_embryo(
                df.select(features),
                hue=over,
                title=f"{name} {features=}{over=}",
            )

        df_out = apply_transformer_to_dataframe(
            df_out,
            transformer=scaler_param.scaler_instance,
            columns=features,
            over=over,
        )
        print(df_out)
        features = df_out.columns

        if scaler_params.verbose:
            logger.info(f"{df_out.select(df_out.columns).describe()}")

        if scaler_params.plot_results:
            plot_features_by_embryo(
                df_out.select(features),
                hue=over,
                title=f"{name} {features=}{over=}",
            )

    return df_out


def apply_transformer_to_dataframe(
    df,
    transformer: Scaler,
    columns: "SelectorType | None" = None,
    over: str | None = None,
):
    print(f'applying transfomer {transformer}')
    if columns is not None:
        if over is None:
            df_passthrough = df.select(pl.exclude(columns))
            df_trans = df.select(pl.col(columns))
            print(df_passthrough.shape, df_trans.shape)
            print(columns)
            trans_np = transformer.fit_transform(df_trans)
            print('=='*20)
            print(transformer.get_feature_names_out())
            print(df_passthrough.shape, trans_np.shape)
            
            df_trans = pl.DataFrame(
                trans_np, schema=list(transformer.get_feature_names_out())
            )
            return pl.concat([df_trans, df_passthrough], how="horizontal")
        else:
            groups = df[over].unique(maintain_order=True)
            res = []
            for group in groups:
                df_passthrough = df.filter(pl.col(over) == group).select(
                    pl.exclude(columns)
                )
                df_trans = df.filter(pl.col(over) == group).select(pl.col(columns))
                df_trans = pl.DataFrame(
                    transformer.fit_transform(df_trans), schema=df_trans.columns
                )
                res.append(pl.concat([df_trans, df_passthrough], how="horizontal"))
            return pl.concat(res, how="vertical")
    if over is None:
        return pl.DataFrame(transformer.fit_transform(df), schema=df.columns)
    else:
        groups = df[over].unique(maintain_order=True)
        res = []
        for group in groups:
            df_trans = df.filter(pl.col(over) == group)
            df_trans = pl.DataFrame(
                transformer.fit_transform(df_trans), schema=df_trans.columns
            )
            res.append(df_trans)
        return pl.concat(res, how="vertical")

def apply_log_transform(df: AnyFrameT, column_patterns=LOG_TRANSFORM_PATTERNS):
    return df.select(
        [
            ~column_patterns,
            column_patterns.log1p(),
            # ~(cs.matches("|".join(LOG_TRANSFORM_PATTERNS))),
            # cs.matches("|".join(LOG_TRANSFORM_PATTERNS)).log1p(),
        ]
    )


