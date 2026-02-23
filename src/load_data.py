import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetes.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print(df.head())
    print(df.describe())