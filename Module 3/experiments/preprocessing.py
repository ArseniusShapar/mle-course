import openml
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data() -> pd.DataFrame:
    dataset = openml.datasets.get_dataset(1590)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    df: pd.DataFrame = X.copy()
    df["target"] = y
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("fnlwgt", axis=1)

    df = df.dropna()

    scaler = MinMaxScaler()
    df[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]] = scaler.fit_transform(
        df[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]]
    )

    df["sex"] = df["sex"].map({"Male": 0, "Female": 1})
    df["target"] = df["target"].map({"<=50K": 0, ">50K": 1})

    return df
