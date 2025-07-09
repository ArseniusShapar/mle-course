## DVC commands

```
dvc init

dvc stage add -n clean_data \
  -d data/raw/adult.csv \
  -o data/processed/clean.csv \
  python scripts/clean_data.py data/raw/adult.csv data/processed/clean.csv

dvc stage add -n transform_data \
  -d data/processed/clean.csv \
  -o data/processed/transformed.csv \
  python scripts/transform_data.py data/processed/clean.csv data/processed/transformed.csv

dvc stage add -n evaluate_model \
  -d data/processed/transformed.csv \
  -d models/model.joblib \
  -o metrics.json \
  -o evaluation_plots \
  --metrics-no-cache metrics.json \
  python scripts/evaluate.py data/processed/transformed.csv models/model.joblib metrics.json evaluation_plots

dvc repro

dvc push
```
