import sys

import pandas as pd

input_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(input_path)

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

df.to_csv(output_path, index=False)
