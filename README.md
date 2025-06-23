# zfish

## features

Features on active share under /data/active/hmax/shayani/features/aggregated

## scripts

/scripts_shayani

### Fig4_VariabilityOfPol2WithCCP_Predictions.ipynb
Spatial predictions of a single embryo (https://makshess.quarto.pub/thesis/#fig-transcription-prediction).

### Fig4_VariabilityOfPol2WithCCP_PyMC_Bambi.py (also ...Bambi2.py)
Models with CCP and ZGA factors a a predictors (https://makshess.quarto.pub/thesis/#fig-transcription-ccp-models).

### Fig4_VariabilityOfPol2_CCP_only.py
Models with CCP as a predictor (https://makshess.quarto.pub/thesis/#fig-transcription-zga-models).

### ccp_gradients_compute.py
Computing CCP gradients (https://makshess.quarto.pub/thesis/#fig-streamplot).
Based on _ccp_gradients.py

### ccp_gradients_plot.py
Plotting CCP gradients (https://makshess.quarto.pub/thesis/#fig-streamplot).
Based on _ccp_gradients.py

### ccp_train.py
Train a single CCP model.
-> additional dependency: [scikit-ccp](https://github.com/MaksHess/scikit-ccp)

### write_parquet_tables.py
Aggregated parquet tables (/data/active/hmax/shayani/features/aggregated) from tall tables.
