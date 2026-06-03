import os
from src.data_preprocessing import load_and_preprocess_data
from src.feature_extraction import extract_features
from src.model import train_model
from src.evaluate import evaluate_model


def run_pipeline():
    csv_path = os.path.join("data", "raw", "spam.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError("Missing raw text data! Run 'python make_csv.py' first.")

    print("[START] Executing Spam Detection NLP Pipeline...")

    # 1. Preprocess
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(csv_path)

    # 2. Extract Text Features (TF-IDF Vectorization)
    X_train_vec, X_test_vec, vectorizer = extract_features(X_train, X_test)

    # 3. Train
    model = train_model(X_train_vec, y_train, vectorizer)

    # 4. Evaluate
    evaluate_model(model, X_test_vec, y_test)

    print("[SUCCESS] NLP Pipeline execution complete!")


if __name__ == "__main__":
    run_pipeline()
