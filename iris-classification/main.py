import os
from src.data_preprocessing import load_and_preprocess_data
from src.model import train_model
from src.evaluate import evaluate_and_plot


def run_pipeline():
    # Define primary local data path
    csv_path = os.path.join("data", "raw", "iris.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing raw data! Please place iris.csv in {os.path.dirname(csv_path)} first.")

    print("[START] Executing Iris Classification Pipeline...")

    # 1. Preprocess (Returns 6 items, including scaler)
    X_train, X_test, y_train, y_test, scaler, df_original = load_and_preprocess_data(csv_path)

    # 2. Train (CRUCIAL FIX: Pass the scaler variable here)
    model = train_model(X_train, y_train, scaler)

    # 3. Evaluate and Save Results
    evaluate_and_plot(model, X_test, y_test, df_original)

    print("[SUCCESS] Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
