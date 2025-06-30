import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

input_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(input_path)

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
numeric_cols = [
    "age",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

processor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,
                max_categories=10,
            ),
            categorical_cols,
        ),
        ("num", MinMaxScaler(), numeric_cols),
    ],
    remainder="passthrough",
)
processor.set_output(transform="pandas")

df = processor.fit_transform(df)
df["target"] = df["remainder__income"].map({"<=50K": 0, ">50K": 1})
df = df.drop("remainder__income", axis=1)

df.to_csv(output_path, index=False)
