import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model(train_path="data/processed/train.csv", model_dir="outputs/models"):
    """Loads processed training data, trains a Linear Regression model, and saves it."""
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Processed training file missing at {train_path}. Run preprocess.py first.")

    print("Loading processed training dataset...")
    train_df = pd.read_csv(train_path)

    # Separate features and target (MEDV)
    X_train = train_df.drop(columns=['MEDV'])
    y_train = train_df['MEDV']

    print("Training classical Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model weights to disk
    model_path = os.path.join(model_dir, "linear_regression_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Success! Model weights securely saved to: {model_path}")


if __name__ == "__main__":
    train_model()
