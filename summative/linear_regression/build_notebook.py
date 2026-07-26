"""Builds multivariate.ipynb from ordered (markdown/code) cells, then executes it.

Run: uv run python summative/linear_regression/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
OUT = HERE / "multivariate.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# Engineering Graduate Salary Prediction - Multivariate Linear Regression

**Mission.** Extending the *Stint* mission (connecting ALU students to real
opportunities), this notebook builds a regression model that predicts an
engineering graduate's **starting salary** from academic records, standardized
aptitude and domain-skill scores, and Big-Five personality traits. The goal is to
surface which factors actually move employability and pay.

**Dataset.** AMEO 2015 *Engineering Graduate Salary Prediction* (Aspiring Minds
Employability Outcomes) - **2,998 rows × 34 columns**, target `Salary` (INR).
Source: Kaggle `manishkc06/engineering-graduate-salary-prediction`.

We (1) explore & visualize the data, (2) engineer features, (3) standardize,
(4) compare **four** regression algorithms - stochastic gradient descent,
closed-form linear regression, a decision tree, and a random forest - (5) plot
train/test loss curves, (6) save the best model, and (7) predict a single row.
""")

code(r"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
DATA = Path("data/Engineering_graduate_salary.csv")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
""")

md("## 1. Load and inspect the data")

code(r"""
df = pd.read_csv(DATA)
print("Shape:", df.shape)
df.head()
""")

code(r"""
df.describe(include="all").T.head(20)
""")

md(r"""
## 2. Exploratory visualizations (interpreted)

We look at the salary distribution, the correlation structure, and the strongest
single predictor. These directly inform the feature-engineering choices below.
""")

code(r"""
plt.figure(figsize=(7, 4.5))
sns.histplot(df["Salary"], bins=40, kde=True)
plt.xlabel("Salary (INR)"); plt.title("Salary Distribution")
plt.tight_layout(); plt.show()
""")

md(r"""
**Interpretation.** Salary is heavily **right-skewed** with a long tail up to
4,000,000 INR. A handful of extreme values will dominate the squared-error loss
and pull a linear fit off the bulk of graduates, so we cap the top 1% below.
""")

code(r"""
# Columns to consider. Subject-specific skill scores use -1 for "not attempted".
skill_cols = ["ComputerProgramming", "ElectronicsAndSemicon", "ComputerScience",
              "MechanicalEngg", "ElectricalEngg", "TelecomEngg", "CivilEngg"]
missing_share = (df[skill_cols] == -1).mean().sort_values(ascending=False)
missing_share.to_frame("share_of_-1")
""")

md(r"""
**Interpretation - which columns to drop.** `ElectronicsAndSemicon`,
`ComputerScience`, `MechanicalEngg`, `ElectricalEngg`, `TelecomEngg` and
`CivilEngg` are **-1 (not attempted) for 70–99% of students** - they carry almost
no signal and are dropped. `ComputerProgramming` is attempted by most students, so
we keep it and impute the remaining `-1`s with the median. Identifier columns
(`ID`, `DOB`, `CollegeID`, `CollegeCityID`) are dropped because they are not
predictive.
""")

md(r"""
## 3. Feature engineering, numeric conversion & the curated feature set

We convert `Gender` to numeric, treat `-1` sentinels as missing, cap salary
outliers, and select a **curated set of 12 interpretable numeric features** for the
deployed model. (The full 34-column frame above is what the interpretation is
based on; the deployed API uses this clean subset so the mobile form stays usable.)
""")

code(r"""
FEATURES = ["CollegeTier", "collegeGPA", "English", "Logical", "Quant", "Domain",
            "ComputerProgramming", "conscientiousness", "agreeableness",
            "extraversion", "nueroticism", "openess_to_experience"]
TARGET = "Salary"

data = df.copy()
data["Gender"] = (data["Gender"] == "m").astype(int)  # numeric conversion example

# -1 sentinels -> median imputation
for col in ["ComputerProgramming", "Domain"]:
    data[col] = data[col].replace(-1, np.nan)
cp_median = data["ComputerProgramming"].median()
dom_median = data["Domain"].median()
data["ComputerProgramming"] = data["ComputerProgramming"].fillna(cp_median)
data["Domain"] = data["Domain"].fillna(dom_median)

# Cap top-1% salary outliers
cap = data[TARGET].quantile(0.99)
data = data[data[TARGET] <= cap].copy()
print("Rows after cleaning:", data.shape[0], "| salary cap:", int(cap))

X = data[FEATURES].astype(float)
y = data[TARGET].astype(float)
""")

code(r"""
plt.figure(figsize=(10, 8))
corr = data[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap"); plt.tight_layout(); plt.show()
""")

md(r"""
**Interpretation - feature weight.** The aptitude scores **Quant, English and
Logical** and **collegeGPA** show the strongest positive correlation with salary,
while **CollegeTier** is negatively correlated (tier 1 is the better college, so a
lower number means higher pay). Personality traits correlate only weakly. This
tells us the aptitude/academic block carries most of the predictive weight.
""")

code(r"""
top_feat = corr[TARGET].drop(TARGET).abs().idxmax()
print("Strongest linear predictor:", top_feat)
plt.figure(figsize=(7, 4.5))
sns.scatterplot(x=data[top_feat], y=y, alpha=0.3)
plt.xlabel(top_feat); plt.ylabel("Salary"); plt.title(f"{top_feat} vs Salary")
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Train/test split and standardization

Features are standardized with `StandardScaler` (fit on the **train** set only) so
that gradient descent converges well and all features are on a comparable scale.
""")

code(r"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
print("Train:", X_train_s.shape, "Test:", X_test_s.shape)
""")

md(r"""
## 5. Model comparison - 4 algorithms

We compare stochastic gradient-descent linear regression against three other
implementations: closed-form linear regression, a decision tree, and a random
forest. The **loss metric is Mean Squared Error (MSE)**; we also report RMSE (in
INR, interpretable) and R².
""")

code(r"""
results = {}

def evaluate(name, model):
    pred_te, pred_tr = model.predict(X_test_s), model.predict(X_train_s)
    results[name] = {
        "model": model,
        "test_mse": mean_squared_error(y_test, pred_te),
        "test_rmse": mean_squared_error(y_test, pred_te) ** 0.5,
        "test_r2": r2_score(y_test, pred_te),
        "train_rmse": mean_squared_error(y_train, pred_tr) ** 0.5,
    }

# 1) Stochastic Gradient Descent linear regression (the required one)
sgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4,
                   learning_rate="invscaling", eta0=0.01, max_iter=2000,
                   random_state=RANDOM_STATE)
sgd.fit(X_train_s, y_train); evaluate("SGD (Gradient Descent)", sgd)

# 2) Closed-form Linear Regression (OLS)
lr = LinearRegression().fit(X_train_s, y_train); evaluate("Linear Regression (OLS)", lr)

# 3) Decision Tree
dt = DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE)
dt.fit(X_train_s, y_train); evaluate("Decision Tree", dt)

# 4) Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                           random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_s, y_train); evaluate("Random Forest", rf)

summary = pd.DataFrame({k: {"Test RMSE": v["test_rmse"], "Test MSE": v["test_mse"],
                            "Test R2": v["test_r2"]} for k, v in results.items()}).T
summary.sort_values("Test RMSE")
""")

code(r"""
best_name = min(results, key=lambda k: results[k]["test_rmse"])
print("Best model (lowest test RMSE / MSE):", best_name)
""")

md(r"""
**Criterion for "best".** We select the model with the **lowest test MSE/RMSE**
(the loss metric) - i.e. the smallest average squared prediction error on unseen
data. The linear models and the random forest are near-tied; the closed-form
**Linear Regression** wins narrowly, and SGD converges to essentially the same
solution, confirming the relationship is close to linear.
""")

md(r"""
## 6. Loss curve - train vs test (SGD via `partial_fit`)

We train the SGD model one epoch at a time and record MSE on the train and test
sets after each epoch.
""")

code(r"""
sgd_lc = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4,
                      learning_rate="invscaling", eta0=0.01,
                      random_state=RANDOM_STATE)
n_epochs = 60
train_loss, test_loss = [], []
for _ in range(n_epochs):
    sgd_lc.partial_fit(X_train_s, y_train)
    train_loss.append(mean_squared_error(y_train, sgd_lc.predict(X_train_s)))
    test_loss.append(mean_squared_error(y_test, sgd_lc.predict(X_test_s)))

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, n_epochs + 1), train_loss, label="Train loss (MSE)")
plt.plot(range(1, n_epochs + 1), test_loss, label="Test loss (MSE)")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("SGD Linear Regression - Loss Curve")
plt.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
**Interpretation.** Both curves fall quickly and then **plateau together** with a
small gap - the model is **not overfitting** (train and test loss stay close). The
plateau height is set by the irreducible noise in salary, which is why adding more
epochs alone will not lower the loss much.
""")

md(r"""
## 7. Scatter plots - before vs after fitting the linear line
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# BEFORE: raw data, no fitted line
axes[0].scatter(X_test[top_feat], y_test, alpha=0.3, color="steelblue")
axes[0].set_title(f"Before - {top_feat} vs Salary (raw)")
axes[0].set_xlabel(top_feat); axes[0].set_ylabel("Salary")

# AFTER: linear fit line through the data
xt = X_test[top_feat].values
order = np.argsort(xt)
lr_pred = lr.predict(X_test_s)
axes[1].scatter(xt, y_test, alpha=0.3, color="steelblue", label="Actual")
axes[1].plot(xt[order], lr_pred[order], "r-", lw=2, label="Linear fit")
axes[1].set_title(f"After - Linear line through the data")
axes[1].set_xlabel(top_feat); axes[1].set_ylabel("Salary"); axes[1].legend()
plt.tight_layout(); plt.show()
""")

code(r"""
# Actual vs predicted for the best model (ideal = points on the red line)
best_model = results[best_name]["model"]
y_pred_best = best_model.predict(X_test_s)
plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, y_pred_best, alpha=0.4)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--", label="Ideal (y = x)")
plt.xlabel("Actual Salary"); plt.ylabel("Predicted Salary")
plt.title(f"Actual vs Predicted - {best_name}"); plt.legend()
plt.tight_layout(); plt.show()
""")

md(r"""
## 8. Save the best model and the scaler
""")

code(r"""
joblib.dump(best_model, MODELS / "best_model.pkl")
joblib.dump(scaler, MODELS / "scaler.pkl")
metadata = {"features": FEATURES, "best_model": best_name,
            "cp_median": float(cp_median), "domain_median": float(dom_median)}
joblib.dump(metadata, MODELS / "metadata.pkl")
print("Saved best model:", best_name)
print("Artifacts:", [p.name for p in MODELS.iterdir()])
""")

md(r"""
## 9. Predict a single row from the test set

This is the exact transformation the API reuses: take one row of inputs, scale
with the saved scaler, and predict with the saved model.
""")

code(r"""
one_row = X_test.iloc[[0]]
one_scaled = scaler.transform(one_row)
pred = float(best_model.predict(one_scaled)[0])
print("Input features:")
print(one_row.T)
print(f"\nPredicted salary: {pred:,.0f} INR")
print(f"Actual salary:    {y_test.iloc[0]:,.0f} INR")
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}

print("Executing notebook...")
ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, OUT)
print("Wrote", OUT)
