# Engineering Graduate Salary Prediction

## Mission & Problem

Extending the *Stint* mission of connecting ALU students to real opportunities,
this project predicts an engineering graduate's **starting salary** from academic
records, standardized aptitude scores, and personality traits — so students can
see which factors actually move their employability and pay.

## Dataset

**Source:** AMEO 2015 *Engineering Graduate Salary Prediction* (Aspiring Minds
Employability Outcomes), on Kaggle:
[manishkc06/engineering-graduate-salary-prediction](https://www.kaggle.com/datasets/manishkc06/engineering-graduate-salary-prediction).
**2,998 rows × 34 columns** — academic percentages (10th/12th/GPA), 10th/12th
boards, college tier & state, AMCAT aptitude scores (English, Logical, Quant),
domain-skill scores, all Big-Five personality traits, specialization (42
categories) and gender. Target: `Salary` (INR).

### Visualizations (in the notebook)

Correlation heatmap, salary-distribution histogram, strongest-predictor scatter,
train/test loss curve, and before/after best-fit-line scatter — see
[`summative/linear_regression/multivariate.ipynb`](summative/linear_regression/multivariate.ipynb)
(figures also under `summative/linear_regression/figures/`).

## Public API

- **Swagger UI:** `https://salary-prediction-api.onrender.com/docs`
- **Prediction endpoint:** `POST https://salary-prediction-api.onrender.com/predict`
- **Retrain endpoint:** `POST https://salary-prediction-api.onrender.com/retrain` (multipart CSV upload)

> Replace the host above with your own Render URL once deployed. The free Render
> instance may take ~30–60 s to wake on the first request.

Example request body:

```json
{
  "CollegeTier": 1, "collegeGPA": 78.5, "English": 620, "Logical": 560,
  "Quant": 640, "Domain": 0.72, "ComputerProgramming": 480,
  "conscientiousness": 0.5, "agreeableness": 0.2, "extraversion": -0.3,
  "nueroticism": 0.1, "openess_to_experience": 0.4
}
```

## Video Demo

▶️ **YouTube:** _add link here_

## Models

Four regressors are trained and compared on test **MSE/RMSE** (the loss metric):
stochastic gradient descent (`SGDRegressor`), closed-form `LinearRegression`,
`DecisionTreeRegressor`, and `RandomForestRegressor`. The best model (lowest test
RMSE — **Linear Regression**) is saved to
`summative/linear_regression/models/best_model.pkl` and served by the API.

## Repository structure

```
linear_regression_model/
├── pyproject.toml                # uv-managed (Python 3.12)
├── render.yaml                   # Render blueprint for the API
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb    # Task 1: EDA, models, loss curves, save best
│   │   ├── data/Engineering_graduate_salary.csv
│   │   ├── models/               # best_model.pkl, scaler.pkl, metadata.pkl
│   │   └── figures/
│   ├── API/
│   │   ├── prediction.py         # Task 2: FastAPI (predict + retrain + CORS)
│   │   ├── requirements.txt
│   │   └── models/
│   └── FlutterApp/               # Task 3: single-page mobile predictor
```

## Running the notebook / API locally

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create env + install deps
# Notebook
uv run jupyter lab summative/linear_regression/multivariate.ipynb
# API (Swagger at http://127.0.0.1:8000/docs)
uv run uvicorn prediction:app --app-dir summative/API --reload
```

## Running the mobile app

Requires the [Flutter SDK](https://docs.flutter.dev/get-started/install).

```bash
cd summative/FlutterApp
flutter pub get
# 1) Set the API URL: edit lib/main.dart -> kApiBaseUrl = "<your Render URL>"
# 2) Launch on a connected device or emulator:
flutter run
```

Enter values in the 12 fields (each with its allowed range shown), tap
**Predict**, and the estimated annual salary appears in the result card.
Out-of-range or missing values are flagged inline and by the API.

## Deploying the API to Render

1. Push this repo to GitHub.
2. On [Render](https://render.com) → **New → Blueprint**, connect the repo; it
   reads `render.yaml` and deploys the `summative/API` service.
3. Copy the resulting URL into `kApiBaseUrl` in the Flutter app and the links above.
