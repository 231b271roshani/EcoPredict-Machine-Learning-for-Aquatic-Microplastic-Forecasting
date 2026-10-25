import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ("Depthm", "Salnty", "O2ml_L")
TARGET_COLUMN = "T_degC"
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
MAX_TRAINING_ROWS = 60_000
Regressor = LinearRegression | Ridge | RandomForestRegressor | GradientBoostingRegressor


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing_columns)
        )

    cleaned = dataset.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.dropna()


def train_models(
    dataset: pd.DataFrame, sample_size: int = MAX_TRAINING_ROWS
) -> tuple[pd.DataFrame, StandardScaler, dict[str, Regressor]]:
    if sample_size < 1:
        raise ValueError("Sample size must be greater than zero.")

    cleaned = clean_dataset(dataset)
    if len(cleaned) < 5:
        raise ValueError("At least five complete numeric rows are required for training.")

    sample = cleaned.sample(
        n=min(sample_size, len(cleaned)), random_state=42
    )
    features = sample.loc[:, list(FEATURE_COLUMNS)]
    target = sample[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, max_depth=6, random_state=42
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        results.append(
            {
                "Model": name,
                "R2 Score": round(r2_score(y_test, predictions), 4),
                "RMSE": round(
                    np.sqrt(mean_squared_error(y_test, predictions)), 4
                ),
                "MAE": round(mean_absolute_error(y_test, predictions), 4),
            }
        )

    return pd.DataFrame(results), scaler, models


def main() -> None:
    print("[1/5] Loading CalCOFI dataset...")
    try:
        dataset = pd.read_csv("bottle.csv", usecols=list(REQUIRED_COLUMNS))
    except FileNotFoundError:
        print("Error: 'bottle.csv' not found. Ensure the dataset is in the ML directory.")
        return
    except ValueError as error:
        print(f"Error reading 'bottle.csv': {error}")
        return

    print(f"Original dataset shape: {dataset.shape}")
    try:
        cleaned = clean_dataset(dataset)
    except ValueError as error:
        print(f"Error: {error}")
        return

    print(f"Cleaned dataset shape: {cleaned.shape}")
    print("\n[2/5] Training models for comparative analysis...")
    try:
        results, scaler, models = train_models(cleaned)
    except ValueError as error:
        print(f"Error: {error}")
        return

    print("\n[3/5] Evaluation Complete. Summary Table:")
    print(results.to_string(index=False))

    print("\n[4/5] Generating publication charts...")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x="Model", y="R2 Score", data=results, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Algorithm Comparison: R² Score (Higher is Better)", fontsize=12)
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(x="Model", y="RMSE", data=results, ax=axes[1], palette="Reds_r")
    axes[1].set_title(
        "Algorithm Comparison: Root Mean Squared Error (Lower is Better)", fontsize=12
    )
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("model_comparison_metrics.png", dpi=300)
    print("Saved comparison chart as 'model_comparison_metrics.png'")

    print("\n[5/5] Exporting best model...")
    import joblib

    best_model_name = results.loc[results["R2 Score"].idxmax(), "Model"]
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(models[best_model_name], "best_ecopredict_model.pkl")
    print(f"Top Model: {best_model_name} (R²: {results['R2 Score'].max()})")
    print("Saved best model artifact as 'best_ecopredict_model.pkl'")


if __name__ == "__main__":
    main()
