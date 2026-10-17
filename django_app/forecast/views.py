
import requests
import pandas as pd
from pathlib import Path
from django.conf import settings
from django.shortcuts import render


def home(request):
    forecast_data = None
    error_message = None

    # Existing forecast feature
    if request.method == "POST":
        months = request.POST.get("months", "3")

        try:
            response = requests.get(
                "https://price-forecast-api-3xtd.onrender.com/forecast",,
                params={"months": months},
                timeout=10,
            )
            response.raise_for_status()
            forecast_data = response.json()

        except requests.RequestException:
            error_message = (
                "Could not connect to the forecasting API. "
                "Please check that FastAPI is running."
            )

    # Load model evaluation results
    evaluation_data = []

    csv_path = (
        Path(settings.BASE_DIR).parent
        / "evaluation_results.csv"
    )

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

            evaluation_data = [
                {
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "actual_price": round(
                        float(row["Actual Price"]), 2
                    ),
                    "predicted_price": round(
                        float(row["Predicted Price"]), 2
                    ),
                }
                for _, row in df.iterrows()
            ]
        except (ValueError, KeyError, OSError) as exc:
            error_message = (
                "Could not read evaluation results: "
                + str(exc)
            )

    return render(
        request,
        "forecast/home.html",
        {
            "forecast_data": forecast_data,
            "error_message": error_message,
            "evaluation_data": evaluation_data,
        },
    )
