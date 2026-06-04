import os
import pandas as pd


def fetch_and_save_data(output_dir="data/raw"):
    """Downloads the authentic Boston Housing dataset directly from a verified GitHub path."""
    os.makedirs(output_dir, exist_ok=True)

    # Corrected, verified GitHub raw directory path
    csv_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    print("Downloading Boston Housing data from verified GitHub directory...")

    try:
        # Stream the CSV directly into Pandas from GitHub
        df = pd.read_csv(csv_url)

        # Standardize column headers to uppercase (e.g., 'crim' -> 'CRIM', 'medv' -> 'MEDV')
        df.columns = [col.upper() for col in df.columns]

        # Save the dataset locally inside your project layout
        output_path = os.path.join(output_dir, "boston_raw.csv")
        df.to_csv(output_path, index=False)

        print(f"Success! Raw dataset saved to: {output_path}")
        print(f"Verified dataset dimensions: {df.shape} (506 records, 14 features)")
        return df

    except Exception as e:
        print(f"Download failed! Check connection or URL path. Details: {e}")
        raise


if __name__ == "__main__":
    fetch_and_save_data()
