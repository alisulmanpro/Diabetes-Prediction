import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

CLEAN_PATH = Path(__file__).resolve().parents[1] / "data" / "clean" / "diabetes_clean.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_model.pkl"

def explain():
    df = pd.read_csv(CLEAN_PATH)
    x = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    model = joblib.load(MODEL_PATH)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    # Global Feature Importance
    plt.figure()
    shap.summary_plot(shap_values, x_test, show=False)
    plt.title("SHAP Summary Plot")
    plt.show()

    # Bar Plot Importance
    plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar)")
    plt.show()

    # Explain single patient
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        x_test.iloc[0],
        matplotlib=True
    )
    plt.show()

if __name__ == "__main__":
    explain()