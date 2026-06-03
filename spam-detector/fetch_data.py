import os
import pandas as pd


def download_full_dataset():
    # The complete, exact URL for the official SMS Spam dataset mirror
    target_url = "https://githubusercontent.com"

    output_dir = os.path.join("data", "raw")
    output_file = os.path.join(output_dir, "spam.csv")

    print("[INFO] Fetching full production text database from source...")

    try:
        # Load directly into pandas using tab-separation
        df = pd.read_csv(target_url, sep='\t', names=['label', 'text'])

        # Ensure the destination directories exist
        os.makedirs(output_dir, exist_ok=True)

        # Save as standard clean CSV
        df.to_csv(output_file, index=False)
        print(f"[SUCCESS] Loaded {len(df)} real messages into: {output_file}")

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    download_full_dataset()
