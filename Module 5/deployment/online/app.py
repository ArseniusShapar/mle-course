import pandas as pd
from flask import Flask, jsonify, request
from utils import load_artifacts, make_predict, preprocess

app = Flask(__name__)
vectorizer, model = load_artifacts()


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    text = data['text']

    df = pd.DataFrame({'text': [text]})

    df = preprocess(df)
    y_pred = make_predict(df, vectorizer, model)[0]
    return jsonify({'y_pred': y_pred})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
