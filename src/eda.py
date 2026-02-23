import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CLEAN_PATH = Path(__file__).resolve().parents[1] / "data" / "clean" / "diabetes_clean.csv"

def run_eda():
    df = pd.read_csv(CLEAN_PATH)

    print("Shape:", df.shape)
    print("\nClass Distribution:")
    print(df["Outcome"].value_counts())
    print("\nClass Percentage:")
    print(df["Outcome"].value_counts(normalize=True) * 100)

    # Correlation Matrix
    plt.figure()
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Matrix")
    plt.show()

    # Feature distributions by Outcome
    features = df.columns[:-1]

    for col in features:
        plt.figure()
        sns.boxplot(x="Outcome", y=col, data=df)
        plt.title(f"{col} vs Outcome")
        plt.show()

if __name__ == "__main__":
    run_eda()
