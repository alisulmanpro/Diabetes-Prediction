import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetes.csv"
CLEAN_PATH = Path(__file__).resolve().parents[1] / "data" / "clean" / "diabetes_clean.csv"

def clean_data():
    df = pd.read_csv(DATA_PATH)

    # Columns where zero is invalid
    invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Replace 0 with NaN
    for col in invalid_zero_cols:
        df[col] = df[col].replace(0, pd.NA)

    # Show missing count
    print("Missing values after replacing zeros:")
    print(df.isna().sum())

    # Fill with median
    for col in invalid_zero_cols:
        df[col] = df[col].fillna(df[col].median())

    print("\nMissing values after imputation:")
    print(df.isna().sum())

    # Save cleaned data
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print("\nCleaned dataset saved.")

if __name__ == "__main__":
    clean_data()