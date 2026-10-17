
import pandas as pd
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read the original dataset
df = pd.read_csv("price_data.csv")

# Prepare the date column
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Create the monthly price series
series = df.set_index("date")["avg_monthly_price"]
series = series.asfreq("MS")

# Check for missing values
if series.isna().any():
    raise ValueError(
        "Missing monthly dates or prices found. "
        "Please check the dataset."
    )

# Train the forecasting model
model = ExponentialSmoothing(
    series,
    trend="add",
    initialization_method="estimated"
).fit()

# Save the model on this computer
joblib.dump(model, "price_forecast_model.joblib")

# Test that the saved model can be loaded
loaded_model = joblib.load("price_forecast_model.joblib")

print("Model trained and saved successfully!")
print("\nNext 3 monthly price predictions:")
print(loaded_model.forecast(3))
