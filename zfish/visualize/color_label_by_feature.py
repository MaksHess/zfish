import matplotlib as mpl
import polars as pl


def get_colormap(
    df: pl.DataFrame,
    feature: str,
    cmap_name="inferno",
    lower_quantile: float = 0.0,
    upper_quantile: float = 1.0,
):
    assert "Label" in df, "Need `Label` column!"
    assert feature in df, f"Need `{feature}` column!"
    df_norm = (
        df.select(
            [
                pl.col("Label"),
                pl.col(feature),
                pl.col(feature).quantile(lower_quantile).alias(f"lower_quantile"),
                pl.col(feature).quantile(upper_quantile).alias(f"upper_quantile"),
            ]
        )
        .with_columns(
            [
                (pl.col("upper_quantile") - pl.col("lower_quantile")).alias(
                    "inter_quantile_range"
                ),
            ]
        )
        .select(
            [
                pl.col("Label"),
                (pl.col(feature) - pl.col("lower_quantile"))
                / pl.col("inter_quantile_range"),
            ]
        )
    )
    cmap = mpl.colormaps[cmap_name]
    return dict(zip(df_norm["Label"].to_numpy(), cmap(df_norm["Centroid"])))
