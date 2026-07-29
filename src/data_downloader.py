import os
import urllib.request

def download_data():
    raw_url = "https://github.com/shrikant-temburwar/Loan-Prediction-Dataset/raw/master/train.csv"
    
    # Define directories
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    dest_path = os.path.join(raw_dir, "train.csv")
    
    print(f"Downloading dataset from {raw_url}...")
    try:
        urllib.request.urlretrieve(raw_url, dest_path)
        print(f"Dataset successfully saved to {dest_path}")
        print(f"File size: {os.path.getsize(dest_path)} bytes")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    # Ensure we run from the project root directory
    download_data()
