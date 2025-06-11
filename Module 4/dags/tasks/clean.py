import pandas as pd


def clean_data(**kwargs):
    ti = kwargs["ti"]
    df = pd.DataFrame(ti.xcom_pull(key="merged_data", task_ids="merge_data"))

    df = df.drop("fnlwgt", axis=1)
    df = df.dropna()

    categorical_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
    ]

    df[categorical_cols] = df[categorical_cols].astype("category")

    ti.xcom_push(key="cleaned_data", value=df.to_dict())
