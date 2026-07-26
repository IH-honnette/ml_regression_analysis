# Design: Graduate Salary Prediction — Full Deployment Pipeline

**Date:** 2026-07-24
**Course:** ALU — Summative: First Model Deployment

## Mission & Problem

Extends the *Stint* mission (connecting ALU students to real opportunities). We
build a linear-regression model that predicts an engineering graduate's starting
salary from academic records, standardized aptitude and domain-skill scores, and
personality traits — helping students see which factors actually move
employability and pay. This is **not** the house-price example and is not generic.

## Dataset

- **Source:** AMEO 2015 "Engineering Graduate Salary Prediction" (Aspiring Minds
  Employability Outcomes), mirrored publicly. Original on Kaggle:
  `manishkc06/engineering-graduate-salary-prediction`.
- **Shape:** 2,998 rows × 34 columns. Target: `Salary` (INR, 35k–4M, mean ~305k).
- **Richness:** academic percentages (10th/12th/GPA), 10th/12th boards, college
  tier & state, standardized aptitude scores (English, Logical, Quant), domain
  skill scores (ComputerProgramming, ElectronicsAndSemicon, ComputerScience, …),
  all Big-Five personality traits, specialization (42 categories), gender.

## Task 1 — Notebook (`summative/linear_regression/multivariate.ipynb`)

1. **Load & inspect** the 34-column dataset.
2. **Visualizations (interpreted):** correlation heatmap, salary distribution
   histogram, and scatter of the strongest predictor vs salary.
3. **Feature engineering:**
   - Drop identifiers: `ID`, `DOB`, `CollegeID`, `CollegeCityID`.
   - Drop sparse subject-skill columns dominated by `-1` sentinels
     (`ElectronicsAndSemicon`, `ComputerScience`, `MechanicalEngg`,
     `ElectricalEngg`, `TelecomEngg`, `CivilEngg`).
   - Convert categoricals to numeric (Gender → 0/1; the rest handled but the
     deployed model uses a curated numeric subset).
4. **Standardize** features with `StandardScaler`.
5. **Deployed feature set (curated, ~12 inputs):** `CollegeTier`, `collegeGPA`,
   `English`, `Logical`, `Quant`, `Domain`, `ComputerProgramming`,
   `conscientiousness`, `agreeableness`, `extraversion`, `nueroticism`,
   `openess_to_experience`. The notebook still shows full-dataset EDA and
   feature-engineering reasoning for the rubric.
6. **Models compared (4):**
   - `SGDRegressor` — stochastic gradient-descent linear regression (required).
   - `LinearRegression` — closed-form OLS (2nd sklearn linear implementation).
   - `DecisionTreeRegressor`.
   - `RandomForestRegressor`.
7. **Loss curves:** train vs test loss (MSE) across epochs for the SGD model via
   `partial_fit`.
8. **Best model** chosen by lowest test RMSE/MSE, saved with `joblib`
   (`models/best_model.pkl`, `models/scaler.pkl`).
9. **Scatter before/after:** actual-vs-predicted with the fitted best-fit line.
10. **Single-row prediction** demo on one test row.

## Task 2 — API (`summative/API/prediction.py`, FastAPI on Render)

- `GET /` health check.
- `POST /predict` — Pydantic `BaseModel` enforcing type + realistic range on
  every input (percentages 0–100, GPA 0–10, aptitude 0–900, personality −4..4,
  CollegeTier 1–2). Returns predicted salary.
- `POST /retrain` — accepts an uploaded CSV, re-fits the pipeline, swaps the live
  model (retraining criterion).
- **CORS:** explicit allowlist (not `*`): Flutter/dev origins + Render domain,
  methods restricted to `GET, POST, OPTIONS`, specific headers, `credentials`
  disabled. Reasoning documented in code + README.
- `requirements.txt` for Render; Swagger at `…/docs`.

## Task 3 — Flutter App (`summative/FlutterApp/`)

Single page: one text field per curated input feature, a **Predict** button, and
a result/error display area. Client-side range validation mirrors the API. Calls
`{RENDER_URL}/predict`. Clean, organized vertical layout.

## Task 4 — Video (user-produced)

7-minute demo: mobile app predicting, Swagger UI type/range tests, notebook
walkthrough, and the four rubric questions (loss level, hyperparameters, new-data
retraining, CORS basis).

## Tooling & Deployment

- `uv` for env/deps; Python pinned to 3.12 (3.14 lacks ML wheels).
- GitHub repo via `gh` (user commits/pushes — assistant never commits).
- Render Blueprint (`render.yaml`) for one-click deploy from the pushed repo.
