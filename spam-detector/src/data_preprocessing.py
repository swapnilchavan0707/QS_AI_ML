import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Basic text standardization processor."""
    if not isinstance(text, str):
        return ""
    # Lowercase text and remove leading/trailing extra spaces
    text = text.lower().strip()
    return text


def load_and_preprocess_data(csv_path):
    """Loads text raw data, processes label formats, and splits vectors."""
    df = pd.read_csv(csv_path)

    # Process text arrays
    df['clean_text'] = df['text'].apply(clean_text)

    # Convert labels: ham -> 0, spam -> 1
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['clean_text'].values
    y = df['target'].values

    # Stratified split to ensure balance across train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, df
