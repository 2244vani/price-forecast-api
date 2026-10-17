
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the monthly price data
df = pd.read_csv("price_data.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

df = df.set_index("date")
series = df["avg_monthly_price"].asfreq("MS")

if series.isna().any():
    raise ValueError(
        "Missing monthly dates or prices found. "
        "Please check price_data.csv."
    )

# 2. Calculate the average price for each calendar month
monthly_averages = series.groupby(
    series.index.month
).mean()

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

monthly_averages.index = [
    month_names[int(month) - 1]
    for month in monthly_averages.index
]

print("\nAVERAGE PRICE BY CALENDAR MONTH")
print("=" * 40)
print(monthly_averages.round(2).to_string())

# 3. Plot the average prices by month
plt.figure(figsize=(10, 5))

plt.bar(
    monthly_averages.index,
    monthly_averages.values
)

plt.title("Average Price by Calendar Month")
plt.xlabel("Month")
plt.ylabel("Average Monthly Price")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("seasonality_check.png")

print("\nGraph saved as seasonality_check.png")

plt.show()
