from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(X_train, X_test):
    """Converts raw text arrays into numerical TF-IDF feature matrices."""
    # Use stop_words='english' to auto-drop common filler words (the, is, at, and)
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    # Fit ONLY on training vectors to avoid data leakage
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return X_train_vectorized, X_test_vectorized, vectorizer
