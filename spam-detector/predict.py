import os
import joblib


def classify_message(custom_message):
    model_path = os.path.join("outputs", "models", "spam_model.pkl")
    vectorizer_path = os.path.join("outputs", "models", "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Pipeline files missing. Run 'python main.py' first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Preprocess text input locally
    clean_msg = custom_message.lower().strip()

    # Transform into text frequencies using the loaded vocabulary mapping
    vectorized_msg = vectorizer.transform([clean_msg])

    prediction = model.predict(vectorized_msg)[0]
    probabilities = model.predict_proba(vectorized_msg)[0]

    label_map = {0: "HAM (Normal Mail)", 1: "SPAM (Dangerous/Junk)"}
    confidence = probabilities[prediction] * 100

    print("\n=========================================")
    print("         LIVE TEXT SCANNER UNIT          ")
    print("=========================================")
    print(f"Message   : \"{custom_message}\"")
    print(f"Result    : {label_map[prediction]}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    # Test sample: Modify this sentence to test your spam filter!
    test_email = "CONGRATULATIONS! You won a cash prize click here to claim free coins"
    classify_message(test_email)
