from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI(title="Monthly Price Forecast API")


# Load the existing monthly price data
df = pd.read_csv("price_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


@app.get("/")
def home():
    return {
        "message": "Monthly Price Forecast API is running"
    }


@app.get("/forecast")
def forecast(months: int = 12):
    if months < 1 or months > 24:
        raise HTTPException(
            status_code=400,
            detail="Months must be between 1 and 24"
        )

    try:
        if df.empty:
            raise ValueError("Price data is empty")

        # Naive Forecasting: repeat the latest observed price
        latest_date = df["date"].iloc[-1]
        latest_price = float(df["avg_monthly_price"].iloc[-1])

        # Generate future monthly dates
        future_dates = pd.date_range(
            start=latest_date + pd.DateOffset(months=1),
            periods=months,
            freq="MS"
        )

        return {
            "latest_data_date": latest_date.strftime("%Y-%m-%d"),
            "forecast": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "predicted_price": round(latest_price, 2)
                }
                for date in future_dates
            ]
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate forecast"
        ) from exc