import os
import joblib
import numpy as np


def predict_new_flower(sepal_len, sepal_wid, petal_len, petal_wid):
    model_path = os.path.join("outputs", "models", "iris_model.pkl")
    scaler_path = os.path.join("outputs", "models", "scaler.pkl")

    # 1. Verification Check
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Missing files! Please run 'python main.py' first.")

    # 2. Load both saved binaries
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    species_names = ['Setosa', 'Versicolor', 'Virginica']

    # 3. Format input into a 2D array
    raw_features = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])

    # 4. CRUCIAL STEP: Scale features before predicting
    scaled_features = scaler.transform(raw_features)

    # 5. Classify using the scaled dimensions
    prediction_index = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    predicted_species = species_names[prediction_index]
    confidence = probabilities[prediction_index] * 100

    print("\n=========================================")
    print("         NEW FLOWER PREDICTION           ")
    print("=========================================")
    print(f"Input Features : Sepal [{sepal_len}x{sepal_wid}], Petal [{petal_len}x{petal_wid}]")
    print(f"Predicted Class: {predicted_species}")
    print(f"Confidence     : {confidence:.2f}%")


if __name__ == "__main__":
    # Test sample (A textbook Setosa flower)
    predict_new_flower(sepal_len=5.1, sepal_wid=3.5, petal_len=1.4, petal_wid=0.2)
