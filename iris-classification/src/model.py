import os
import joblib
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, scaler, output_dir="outputs/models"):
    """Trains a Logistic Regression model and saves both the model and scaler to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize and train model
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model binary
    model_path = os.path.join(output_dir, "iris_model.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] Model successfully saved to: {model_path}")

    # Save the scaler binary (CRUCIAL FIX)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Scaler successfully saved to: {scaler_path}")

    return model
