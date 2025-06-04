import mlflow
import mlflow.sklearn
import numpy as np
from evaluation import build_prc, build_roc, calculate_metrics, metrics_barplot
from mlflow.models.signature import infer_signature
from preprocessing import load_data, preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

experiment_name = "experiment_4"
mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    df = load_data()
    df = preprocess(df)
    df = df.drop(
        ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"], axis=1
    )
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    params = {"penalty": "l2", "C": 1.0}
    model = LogisticRegression(**params)
    mlflow.set_tag("model", "Logistic Regression")
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
        registered_model_name="logistic-regression",
    )
