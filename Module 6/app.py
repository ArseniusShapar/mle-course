import io

import pandas as pd
from flask import Flask, Response, request, jsonify

from utils import load_artifacts, preprocess, make_prediction

app = Flask(__name__)
vectorizer, model = load_artifacts()


@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="OK", status=200)


@app.route("/invocations", methods=["POST"])
def predict():
    content = request.data.decode('utf-8')
    df = pd.read_csv(io.StringIO(content))

    df = preprocess(df)
    y_pred = make_prediction(df, vectorizer, model)
    return jsonify({"y_pred": y_pred})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
