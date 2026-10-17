
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the historical dataset
df = pd.read_csv("price_data.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# 2. Prepare the monthly price series
series = df.set_index("date")["avg_monthly_price"]
series = series.asfreq("MS")

if series.isna().any():
    raise ValueError(
        "Missing monthly dates or prices found. Check price_data.csv."
    )

# 3. Split the data chronologically
# Use the last 12 months as test data.
train = series.iloc[:-12]
test = series.iloc[-12:]

print("Training period:", train.index.min().date(), "to", train.index.max().date())
print("Testing period: ", test.index.min().date(), "to", test.index.max().date())

# 4. Train the model using only the training data
model = ExponentialSmoothing(
    train,
    trend="add",
    initialization_method="estimated"
).fit()

# 5. Forecast the test period
predictions = model.forecast(len(test))

# 6. Calculate evaluation metrics
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

# MAPE is meaningful when actual prices are non-zero.
if (test == 0).any():
    mape = np.nan
else:
    mape = np.mean(np.abs((test - predictions) / test)) * 100

# 7. Display actual prices and predicted prices
results = pd.DataFrame({
    "Date": test.index,
    "Actual Price": test.values,
    "Predicted Price": predictions.values
})

results["Absolute Error"] = abs(
    results["Actual Price"] - results["Predicted Price"]
)

print("\n--- Actual vs Predicted Prices ---")
print(results.round(2).to_string(index=False))

print("\n--- Model Evaluation Metrics ---")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

if np.isnan(mape):
    print("MAPE: Cannot calculate because an actual price is zero.")
else:
    print(f"MAPE: {mape:.2f}%")

# Save evaluation results for the Django dashboard
results.to_csv("evaluation_results.csv", index=False)

print("\nEvaluation results saved to evaluation_results.csv")
