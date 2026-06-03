import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(csv_path):
    """Loads raw Iris data, handles splitting, and applies feature scaling."""
    # Define exact column names since raw UCI data lacks a header row
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    # Load dataset
    df = pd.read_csv(csv_path, names=column_names)

    # Strip any accidental whitespace from the species names
    df['species'] = df['species'].str.strip()

    # Map text species to integers for machine learning compatibility
    species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['target'] = df['species'].map(species_mapping)

    # Separate features (X) and targets (y)
    X = df.drop(columns=['species', 'target']).values
    y = df['target'].values

    # Split into 80% train and 20% test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (Mean=0, Variance=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df