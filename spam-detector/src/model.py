import os
import joblib
from sklearn.naive_bayes import MultinomialNB


def train_model(X_train, y_train, vectorizer, output_dir="outputs/models"):
    """Trains a Naive Bayes NLP model and serializes both model and feature processor."""
    os.makedirs(output_dir, exist_ok=True)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save both pieces of the pipeline
    joblib.dump(model, os.path.join(output_dir, "spam_model.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))

    print(f"[INFO] Spam Model and Vectorizer saved to: {output_dir}")
    return model
