import mlflow
import mlflow.sklearn


def predict(X):
    model = mlflow.sklearn.load_model("./best_model/model")
    return model.predict(X)
