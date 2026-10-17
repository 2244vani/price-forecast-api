
import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Load and prepare data
df = pd.read_csv("price_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

series = df.set_index("date")["avg_monthly_price"]
series = series.asfreq("MS")

if series.isna().any():
    raise ValueError("Missing monthly dates or prices in the dataset.")

if len(series) < 60:
    raise ValueError("At least 60 monthly observations are required.")


# 2. Calculate evaluation metrics
def calculate_metrics(actual, predicted):
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(
        mean_squared_error(actual_values, predicted_values)
    )

    if np.any(actual_values == 0):
        mape = np.nan
    else:
        mape = np.mean(
            np.abs(
                (actual_values - predicted_values) / actual_values
            )
        ) * 100

    return mae, rmse, mape


# 3. Rolling-origin evaluation settings
FORECAST_HORIZON = 12
STEP_SIZE = 3

# Use the latest 60 months as evaluation origins.
# Each origin predicts the following 12 months.
evaluation_start = max(48, len(series) - 60)

origins = list(
    range(evaluation_start, len(series) - FORECAST_HORIZON + 1,
          STEP_SIZE)
)

results = []

print("\nROLLING-ORIGIN MODEL EVALUATION")
print("=" * 65)
print("Total observations:", len(series))
print("Forecast horizon:", FORECAST_HORIZON, "months")
print("Step between origins:", STEP_SIZE, "months")
print("Number of forecast origins:", len(origins))


# 4. Test each model at every forecast origin
for run_number, train_end in enumerate(origins, start=1):

    train = series.iloc[:train_end]
    test = series.iloc[train_end:train_end + FORECAST_HORIZON]

    print(
        f"\nRun {run_number}/{len(origins)} | "
        f"Training through {train.index[-1].date()} | "
        f"Testing {test.index[0].date()} "
        f"to {test.index[-1].date()}"
    )

    predictions_by_model = {}

    # Model 1: Naive Forecasting
    predictions_by_model["Naive Forecasting"] = np.repeat(
        train.iloc[-1], FORECAST_HORIZON
    )

    # Model 2: Holt Additive
    holt_model = ExponentialSmoothing(
        train,
        trend="add",
        initialization_method="estimated"
    ).fit()

    predictions_by_model["Holt Additive"] = np.asarray(
        holt_model.forecast(FORECAST_HORIZON)
    )

    # Model 3: Holt Damped
    damped_model = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        initialization_method="estimated"
    ).fit()

    predictions_by_model["Holt Damped"] = np.asarray(
        damped_model.forecast(FORECAST_HORIZON)
    )

    # Model 4: Holt-Winters Seasonal
    if len(train) >= 36:
        seasonal_model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        ).fit()

        predictions_by_model["Holt-Winters Seasonal"] = np.asarray(
            seasonal_model.forecast(FORECAST_HORIZON)
        )

    # Calculate metrics for this origin and model
    for model_name, predictions in predictions_by_model.items():

        mae, rmse, mape = calculate_metrics(test, predictions)

        results.append({
            "Run": run_number,
            "Training End": train.index[-1].strftime("%Y-%m"),
            "Test Start": test.index[0].strftime("%Y-%m"),
            "Test End": test.index[-1].strftime("%Y-%m"),
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape
        })


# 5. Display results from every origin
results_df = pd.DataFrame(results)

print("\n\nRESULTS FOR EACH FORECAST ORIGIN")
print("=" * 110)
print(results_df.round(2).to_string(index=False))


# 6. Summarize average errors across origins
summary_df = (
    results_df
    .groupby("Model")[["MAE", "RMSE", "MAPE (%)"]]
    .mean()
    .sort_values("MAE")
)

print("\n\nAVERAGE ERRORS ACROSS ROLLING ORIGINS")
print("=" * 65)
print(summary_df.round(2).to_string())

print("\nBest model by average MAE:", summary_df.index[0])


# 7. Save results
results_df.to_csv(
    "rolling_origin_results.csv",
    index=False
)

summary_df.to_csv(
    "rolling_origin_summary.csv"
)

print("\nSaved files:")
print("- rolling_origin_results.csv")
print("- rolling_origin_summary.csv")

print("\nLower MAE, RMSE, and MAPE generally indicate better forecasts.")
print("The production model has not been changed.")
