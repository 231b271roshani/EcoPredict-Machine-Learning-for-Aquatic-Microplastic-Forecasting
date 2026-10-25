from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from ML.model import (
    FEATURE_COLUMNS,
    MAX_TRAINING_ROWS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    clean_dataset,
    train_models,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "ML" / "bottle.csv"
FEATURE_LABELS = {
    "Depthm": ("Depth", "m"),
    "Salnty": ("Salinity", "PSU"),
    "O2ml_L": ("Dissolved oxygen", "ml/L"),
}

st.set_page_config(
    page_title="EcoPredict | Ocean temperature",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False, max_entries=2)
def read_local_dataset(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS)


@st.cache_data(show_spinner=False, max_entries=2)
def read_uploaded_dataset(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(
        BytesIO(contents), usecols=lambda column: column in REQUIRED_COLUMNS
    )


def get_dataset() -> tuple[pd.DataFrame | None, str | None, str | None, str | None]:
    st.sidebar.title("EcoPredict")
    st.sidebar.caption("Environmental forecasting dashboard")
    uploaded_file = st.sidebar.file_uploader(
        "Load CalCOFI bottle data",
        type=["csv"],
        help="CSV must include the four model columns. Maximum file size: 300 MB.",
    )

    if uploaded_file is not None:
        contents = uploaded_file.getvalue()
        try:
            dataset = read_uploaded_dataset(contents)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
            return None, None, None, f"Could not read this CSV: {error}"
        dataset_id = sha256(contents).hexdigest()
        return dataset, f"{uploaded_file.name} · {len(dataset):,} rows", dataset_id, None

    if DEFAULT_DATASET.is_file():
        modified_ns = DEFAULT_DATASET.stat().st_mtime_ns
        try:
            dataset = read_local_dataset(str(DEFAULT_DATASET), modified_ns)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
            return None, None, None, f"Could not read ML/bottle.csv: {error}"
        return dataset, "ML/bottle.csv", f"{DEFAULT_DATASET}:{modified_ns}", None

    st.sidebar.info(
        "No dataset found yet. Upload `bottle.csv` here, or place it at "
        "`ML/bottle.csv`."
    )
    return None, None, None, None


dataset, dataset_label, dataset_id, load_error = get_dataset()

st.caption("CalCOFI · Regression workspace")
st.title("Ocean temperature forecasting")
st.info(
    f"This model predicts water temperature ({TARGET_COLUMN}) from depth, salinity, "
    "and dissolved oxygen. It does not currently predict microplastic contamination.",
    title="Current prediction target",
)
st.write("")

if dataset is None:
    if load_error:
        st.error(load_error)
    else:
        st.info(
            "Add the CalCOFI `bottle.csv` dataset to get started. The dashboard needs "
            "`Depthm`, `Salnty`, `O2ml_L`, and `T_degC` columns."
        )
        st.code(
            "pip install -r requirements.txt\nstreamlit run app.py",
            language="powershell",
        )
    st.stop()

try:
    cleaned_dataset = clean_dataset(dataset)
except ValueError as error:
    st.error(str(error))
    st.info("Expected columns: `Depthm`, `Salnty`, `O2ml_L`, and `T_degC`.")
    st.stop()

if st.session_state.get("dataset_id") != dataset_id:
    st.session_state["dataset_id"] = dataset_id
    st.session_state.pop("training", None)

header_left, header_right = st.columns([3, 1])
with header_left:
    st.subheader("Your dataset")
    st.caption(f"Source: {dataset_label}")
with header_right:
    st.metric(
        "Samples per model",
        f"{min(len(cleaned_dataset), MAX_TRAINING_ROWS):,}",
    )

metric_columns = st.columns(3)
metric_columns[0].metric("Rows loaded", f"{len(dataset):,}")
metric_columns[1].metric("Complete numeric rows", f"{len(cleaned_dataset):,}")
metric_columns[2].metric("Model inputs", "3 ocean measurements")

dashboard_view = st.segmented_control(
    "Dashboard view",
    ["Model comparison", "Temperature prediction"],
    default="Model comparison",
    label_visibility="collapsed",
)
if dashboard_view == "Model comparison":
    st.markdown("### Compare forecasting models")
    st.write(
        f"Train on up to {MAX_TRAINING_ROWS:,} complete rows, using a reproducible "
        "80/20 train/test split."
    )
    if st.button("Train and compare models", type="primary", width="stretch"):
        with st.spinner("Training four regressors and evaluating the holdout set..."):
            try:
                metrics, scaler, models = train_models(cleaned_dataset)
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state["training"] = {
                    "metrics": metrics,
                    "scaler": scaler,
                    "models": models,
                }

    st.dataframe(
        dataset.loc[:, list(REQUIRED_COLUMNS)].head(8),
        hide_index=True,
        alt="Preview of the first eight oceanographic observations.",
    )
    training = st.session_state.get("training")
    if training:
        results = training["metrics"]
        best_row = results.loc[results["R2 Score"].idxmax()]
        st.success(
            f"Best holdout R²: **{best_row['Model']}** "
            f"(R² {best_row['R2 Score']:.4f})",
            title="Top-performing model",
        )
        chart_data = results.set_index("Model")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("#### R² score · higher is better")
            st.bar_chart(
                chart_data,
                y="R2 Score",
                horizontal=True,
                alt="Holdout R-squared scores for the four regression models.",
            )
        with chart_right:
            st.markdown("#### RMSE · lower is better")
            st.bar_chart(
                chart_data,
                y="RMSE",
                horizontal=True,
                alt="Holdout root mean squared error for the four regression models.",
            )
        st.dataframe(
            results.rename(
                columns={"R2 Score": "R² score", "RMSE": "RMSE (°C)", "MAE": "MAE (°C)"}
            ),
            hide_index=True,
            alt="Holdout evaluation metrics for the four regression models.",
        )
    else:
        st.caption("Train the models to see holdout performance and enable predictions.")

else:
    st.markdown("### Estimate water temperature")
    st.write("Enter ocean measurements within the observed range of this dataset.")
    training = st.session_state.get("training")
    if not training:
        st.info("Train the models in **Model comparison** before making predictions.")
    else:
        observed = cleaned_dataset.loc[:, FEATURE_COLUMNS].agg(["min", "max", "mean"])
        best_model = training["metrics"].loc[
            training["metrics"]["R2 Score"].idxmax(), "Model"
        ]
        model_name = st.segmented_control(
            "Regression model",
            list(training["models"]),
            default=best_model,
        )
        with st.form("prediction_form"):
            input_columns = st.columns(len(FEATURE_COLUMNS))
            values = {}
            for column, container in zip(FEATURE_COLUMNS, input_columns):
                label, unit = FEATURE_LABELS[column]
                minimum = float(observed.loc["min", column])
                maximum = float(observed.loc["max", column])
                mean = float(observed.loc["mean", column])
                values[column] = container.number_input(
                    f"{label} ({unit})",
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(mean, minimum), maximum),
                    step=max((maximum - minimum) / 100, 0.01),
                    format="%.2f",
                )
            predict_clicked = st.form_submit_button(
                "Predict temperature", type="primary", width="stretch"
            )

        if predict_clicked:
            observation = pd.DataFrame([values], columns=FEATURE_COLUMNS)
            scaled_observation = training["scaler"].transform(observation)
            prediction = training["models"][model_name].predict(scaled_observation)[0]
            st.metric("Predicted water temperature", f"{prediction:.2f} °C")
            st.caption(f"Estimate from {model_name}; this is a model output, not a measurement.")
