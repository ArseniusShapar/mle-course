import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def load_artifacts(vectorizer_path='./artifacts/count_vectorizer.joblib',
                   model_path='./artifacts/logistic_regression.joblib') -> tuple[CountVectorizer, LogisticRegression]:
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['prepocessed_text'] = df['text'].map(
        lambda x: re.sub(r'[^A-Za-z0-9\s]+', '', x)
    )

    df['prepocessed_text'] = df['prepocessed_text'].map(lambda x: re.sub(r'\s+', ' ', x.strip()))

    return df


def make_predict(df: pd.DataFrame, vectorizer, model) -> list[int]:
    X = vectorizer.transform(df['text'])
    y_pred = model.predict(X)
    return y_pred.tolist()
