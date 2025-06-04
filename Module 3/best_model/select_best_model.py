import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

target_metric = "f1_score"

experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
experiments = [exp for exp in experiments if exp.name != "Default"]
runs = [client.search_runs(exp.experiment_id, max_results=1)[0] for exp in experiments]
runs.sort(key=lambda run: run.data.metrics[target_metric], reverse=True)
best_run = runs[0]

print(best_run.info)
print(f"Metric {target_metric} = {best_run.data.metrics[target_metric]}")

model_uri = f"runs:/{best_run.info.run_id}/model"
mlflow.artifacts.download_artifacts(model_uri, dst_path="./best_model")
