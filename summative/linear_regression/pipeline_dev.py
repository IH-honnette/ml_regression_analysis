"""Development script for the salary-prediction pipeline.

Run with: uv run python summative/linear_regression/pipeline_dev.py
Produces model artifacts and figures, and prints metrics used to author the
notebook. This mirrors the notebook cells exactly.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
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

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "Engineering_graduate_salary.csv"
MODELS = HERE / "models"
FIGS = HERE / "figures"
MODELS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Curated feature set used by the deployed model (clean, interpretable inputs).
FEATURES = [
    "CollegeTier", "collegeGPA", "English", "Logical", "Quant", "Domain",
    "ComputerProgramming", "conscientiousness", "agreeableness",
    "extraversion", "nueroticism", "openess_to_experience",
]
TARGET = "Salary"


def main() -> None:
    df = pd.read_csv(DATA)
    print("Raw shape:", df.shape)

    # --- Feature engineering / cleaning ---------------------------------
    # ComputerProgramming uses -1 for "section not attempted"; treat as missing
    # and impute with the median so it does not distort the scale.
    df["ComputerProgramming"] = df["ComputerProgramming"].replace(-1, np.nan)
    df["Domain"] = df["Domain"].replace(-1, np.nan)
    med_cp = df["ComputerProgramming"].median()
    med_dom = df["Domain"].median()
    df["ComputerProgramming"] = df["ComputerProgramming"].fillna(med_cp)
    df["Domain"] = df["Domain"].fillna(med_dom)

    # Cap extreme salary outliers (top 1%, up to 4M INR). These few rows are
    # not representative of a fresh graduate's pay and distort a linear fit.
    cap = df[TARGET].quantile(0.99)
    df = df[df[TARGET] <= cap].copy()
    print("Rows after 99th-pct salary cap:", df.shape[0])

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Models ----------------------------------------------------------
    results = {}

    def evaluate(name, model, Xtr, Xte):
        pred_te = model.predict(Xte)
        pred_tr = model.predict(Xtr)
        results[name] = {
            "model": model,
            "test_mse": mean_squared_error(y_test, pred_te),
            "test_rmse": mean_squared_error(y_test, pred_te) ** 0.5,
            "test_r2": r2_score(y_test, pred_te),
            "train_rmse": mean_squared_error(y_train, pred_tr) ** 0.5,
        }

    # 1) SGDRegressor — stochastic gradient descent linear regression.
    sgd = SGDRegressor(
        loss="squared_error", penalty="l2", alpha=1e-4,
        learning_rate="invscaling", eta0=0.01, max_iter=2000,
        random_state=RANDOM_STATE,
    )
    sgd.fit(X_train_s, y_train)
    evaluate("SGD (Gradient Descent)", sgd, X_train_s, X_test_s)

    # 2) LinearRegression — closed-form OLS.
    lr = LinearRegression().fit(X_train_s, y_train)
    evaluate("Linear Regression (OLS)", lr, X_train_s, X_test_s)

    # 3) Decision Tree.
    dt = DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE)
    dt.fit(X_train_s, y_train)
    evaluate("Decision Tree", dt, X_train_s, X_test_s)

    # 4) Random Forest.
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    evaluate("Random Forest", rf, X_train_s, X_test_s)

    print("\n=== Model comparison (test set) ===")
    print(f"{'Model':28s} {'RMSE':>12s} {'MSE':>16s} {'R2':>8s}")
    for name, r in results.items():
        print(f"{name:28s} {r['test_rmse']:12.0f} {r['test_mse']:16.0f} {r['test_r2']:8.3f}")

    best_name = min(results, key=lambda k: results[k]["test_rmse"])
    print(f"\nBest model (lowest test RMSE): {best_name}")

    # --- Loss curves (SGD via partial_fit) ------------------------------
    sgd_lc = SGDRegressor(
        loss="squared_error", penalty="l2", alpha=1e-4,
        learning_rate="invscaling", eta0=0.01, random_state=RANDOM_STATE,
    )
    n_epochs = 60
    train_loss, test_loss = [], []
    for _ in range(n_epochs):
        sgd_lc.partial_fit(X_train_s, y_train)
        train_loss.append(mean_squared_error(y_train, sgd_lc.predict(X_train_s)))
        test_loss.append(mean_squared_error(y_test, sgd_lc.predict(X_test_s)))

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, n_epochs + 1), train_loss, label="Train loss (MSE)")
    plt.plot(range(1, n_epochs + 1), test_loss, label="Test loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("SGD Linear Regression — Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "loss_curve.png", dpi=110)
    plt.close()

    # --- Correlation heatmap --------------------------------------------
    plt.figure(figsize=(10, 8))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGS / "heatmap.png", dpi=110)
    plt.close()

    # --- Salary distribution --------------------------------------------
    plt.figure(figsize=(7, 4.5))
    sns.histplot(y, bins=40, kde=True)
    plt.xlabel("Salary (INR)")
    plt.title("Salary Distribution")
    plt.tight_layout()
    plt.savefig(FIGS / "salary_dist.png", dpi=110)
    plt.close()

    # --- Scatter: strongest predictor vs salary -------------------------
    top_feat = corr[TARGET].drop(TARGET).abs().idxmax()
    print("Strongest linear predictor:", top_feat)

    # --- Scatter before/after: fitted line through data -----------------
    best_model = results[best_name]["model"]
    y_pred_best = best_model.predict(X_test_s)
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.4, label="Predictions")
    lims = [y_test.min(), y_test.max()]
    plt.plot(lims, lims, "r--", label="Ideal fit (y = x)")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "actual_vs_pred.png", dpi=110)
    plt.close()

    # Single-feature best-fit line (Linear model on top predictor).
    xt = X_test[top_feat].values
    order = np.argsort(xt)
    lr_pred_test = lr.predict(X_test_s)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(xt, y_test, alpha=0.3, label="Actual")
    plt.plot(xt[order], lr_pred_test[order], "r-", lw=2, label="Linear fit")
    plt.xlabel(top_feat)
    plt.ylabel("Salary")
    plt.title(f"Best-fit line: {top_feat} vs Salary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "bestfit_line.png", dpi=110)
    plt.close()

    # --- Save best model + scaler + metadata ----------------------------
    joblib.dump(best_model, MODELS / "best_model.pkl")
    joblib.dump(scaler, MODELS / "scaler.pkl")
    meta = {"features": FEATURES, "best_model": best_name,
            "cp_median": float(med_cp), "domain_median": float(med_dom)}
    joblib.dump(meta, MODELS / "metadata.pkl")
    print("\nSaved:", best_name, "->", MODELS / "best_model.pkl")

    # --- Single-row prediction demo -------------------------------------
    one = X_test.iloc[[0]]
    one_s = scaler.transform(one)
    print("\nSingle-row demo input:\n", one.to_dict("records")[0])
    print("Predicted salary:", float(best_model.predict(one_s)[0]))
    print("Actual salary:   ", float(y_test.iloc[0]))


if __name__ == "__main__":
    main()
