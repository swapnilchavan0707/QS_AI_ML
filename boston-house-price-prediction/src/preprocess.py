import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_pipeline(input_path="data/raw/boston_raw.csv", output_dir="data/processed"):
    """Loads raw Boston data from local files, normalizes features, and exports split sets."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data file missing at {input_path}. Run data_loader.py first.")

    df = pd.read_csv(input_path)

    # Isolate independent features from the median home price target (MEDV)
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']

    # Segment data using an 80/20 train/test breakdown
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply Standard Z-score scaling to avoid magnitude bias
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Package matrices back into structural DataFrames for saving
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['MEDV'] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['MEDV'] = y_test.values

    # Write processed artifacts to disk
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Success! Normalized train/test datasets saved to: {output_dir}")


if __name__ == "__main__":
    preprocess_pipeline()
