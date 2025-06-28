import io
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def load_artifacts() -> tuple[CountVectorizer, LogisticRegression]:
    vectorizer = joblib.load("model/vectorizer.joblib")
    model = joblib.load("model/model.joblib")
    return vectorizer, model


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['prepocessed_text'] = df['text'].map(
        lambda x: re.sub(r'[^A-Za-z0-9\s]+', '', x)
    )

    df['prepocessed_text'] = df['prepocessed_text'].map(lambda x: re.sub(r'\s+', ' ', x.strip()))

    return df


def make_prediction(df: pd.DataFrame, vectorizer, model) -> list[int]:
    X = vectorizer.transform(df['text'])
    y_pred = model.predict(X)
    return y_pred.tolist()


def csv_to_dataframe(file) -> pd.DataFrame:
    content = file.read().decode('utf-8')
    data = io.StringIO(content)
    df = pd.read_csv(data)
    return df
