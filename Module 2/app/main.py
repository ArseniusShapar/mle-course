from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import os
import json


model_dir = os.getenv("MODEL_DIR", "../models")
output_dir = os.getenv("OUTPUT_DIR", "../outputs")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

model = load_model(f'{model_dir}/MNIST_keras_CNN.h5', compile=False)

y_pred = np.argmax(model.predict(x_test), axis=1)

result = {
    "y_true": y_test.tolist(),
    "y_pred": y_pred.tolist()
}
with open(os.path.join(output_dir, "output.json"), "w") as f:
    json.dump(result, f)

print("Prediction saved to output.json")

