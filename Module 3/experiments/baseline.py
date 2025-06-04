import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from evaluation import build_prc, build_roc, calculate_metrics, metrics_barplot
from mlflow.models.signature import infer_signature
from preprocessing import load_data, preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

experiment_name = "experiment_5"
mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    df = load_data()
    df = preprocess(df)
    df = df.drop(["native-country"], axis=1)
    df = pd.get_dummies(df, columns=df.columns[:-1], drop_first=True)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    params = {"n_estimators": 20, "max_depth": 3, "criterion": "entropy"}
    model = RandomForestClassifier(**params)
    mlflow.set_tag("model", "Random Forest")
    mlflow.log_params(params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_metrics(calculate_metrics(y_test, y_pred))
    mlflow.log_figure(metrics_barplot(y_test, y_pred), "metrics_barplot.png")
    mlflow.log_figure(build_roc(y_test, y_pred_proba), "roc_curve.png")
    mlflow.log_figure(build_prc(y_test, y_pred_proba), "pr_curve.png")

    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:1],
        registered_model_name="random-forest-classifier",
    )
