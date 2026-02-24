# Diabetes Prediction — Machine Learning Pipeline & API

**Repository:** alisulmanpro / `diabetes-prediction` <br>
**Dataset:** Pima Indians Diabetes Database from the UCI Machine Learning Repository

A compact, production-oriented project that trains an XGBoost classifier to predict diabetes from the Pima dataset and exposes a FastAPI inference endpoint. The repo includes data cleaning, EDA, baseline and XGBoost training, threshold tuning for recall, SHAP explanations, and a minimal FastAPI service for serving predictions.

---

## Key highlights

* Data cleaning for biologically invalid zeros (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
* Baseline model (Logistic Regression) + decision-threshold tuning to prioritize **recall**
* Stronger model: **XGBoost** with class imbalance handling (`scale_pos_weight`)
* Model metrics (on test split): **ROC-AUC ≈ 0.817**, **Recall ≈ 0.80** (using tuned threshold)
* Explainability: **SHAP** global and per-prediction explanations
* Serving: **FastAPI** endpoint with Pydantic validation and Swagger docs

---

## Repository structure

```
diabetes-prediction/
├─ data/
│  ├─ raw/diabetes.csv
│  └─ clean/diabetes_clean.csv
├─ models/
│  ├─ xgb_model.pkl
│  └─ baseline_model.pkl
├─ src/
│  ├─ load_data.py
│  ├─ clean_data.py
│  ├─ eda.py
│  ├─ train_baseline.py
│  ├─ train_xgb.py
│  ├─ explain_model.py
├─ main.py
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## Quickstart — local (minimal)

1. Clone the repo:

```bash
git clone https://github.com/alisulmanpro/diabetes-prediction.git
cd diabetes-prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# mac / linux
source .venv/bin/activate
# windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Install deps:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Add dataset: place `diabetes.csv` (Pima Indians Diabetes CSV) in `data/raw/`.

5. Clean data:

```bash
python src/clean_data.py
# produces data/clean/diabetes_clean.csv
```

6. Run EDA (optional; opens plots):

```bash
python src/eda.py
```

7. Train XGBoost model:

```bash
python src/train_xgb.py
# saved to models/xgb_model.pkl
```

8. Run SHAP explainability:

```bash
python src/explain_model.py
# generates SHAP summary (global) and bar plots
```

9. Run the API locally:

```bash
uvicorn src.api:app --reload
# visit http://127.0.0.1:8000/docs for Swagger UI
```

---

## API — usage examples

**Endpoint:** `POST /predict`
**Input (JSON):**

```json
{
  "Pregnancies": 2,
  "Glucose": 150,
  "BloodPressure": 80,
  "SkinThickness": 30,
  "Insulin": 120,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Sample curl:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":150,"BloodPressure":80,"SkinThickness":30,"Insulin":120,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```

**Response:**

```json
{
  "diabetes_probability": 0.7801,
  "prediction": 1
}
```

> Note: the API uses the XGBoost model and returns probability + binary prediction using the tuned threshold (`0.40` by default). Adjust threshold in `src/api.py` if you prefer a different operating point.

---

## Important implementation notes

* **Missing / invalid zeros**: Certain features in the Pima dataset use `0` to indicate missing. The cleaning script replaces those zeros with `NaN` and imputes medians for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.
* **Scaling**: XGBoost was trained on raw features (no scaling). If you switch to a model that requires scaling (e.g., logistic regression as the deployed model), export and load the scaler alongside the model.
* **Threshold tuning**: In medical use-cases we prioritized **recall** (reduce false negatives). That is why inference uses a lower threshold than 0.5.
* **Explainability**: `src/explain_model.py` uses SHAP `TreeExplainer` to produce global and local explanations for the XGBoost model. Include these plots in your README or demo to increase trust.

---

## Reproducibility & evaluation

* The project uses a single stratified train/test split (random seed = 42). For robust evaluation, add k-fold cross-validation and report mean ± std for metrics.
* Current test metrics (from a held-out split): **ROC-AUC ≈ 0.817**, **Recall ≈ 0.80** (XGBoost, tuned threshold).

---

## Next steps / enhancements

* Add **GridSearchCV** / **Optuna** hyperparameter tuning.
* Use **cross-validation** and report confidence intervals.
* Containerize with **Docker** and add CI/CD for model retraining.
* Build a small **Streamlit** demo for interactive per-patient explanations (SHAP force plots).
* Add input validation rules and monitoring (data drift alerts, model performance tracking).

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit and push changes (`git push origin feat/your-feature`)
4. Open a pull request with a clear description & tests where relevant

---

## License

Recommend: **MIT License** — change as you see fit.

