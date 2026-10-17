
import pandas as pd
import matplotlib.pyplot as plt

# Load the monthly price data
df = pd.read_csv("price_data.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Select the period with the larger forecasting errors
start_date = "2023-10-01"
end_date = "2024-09-01"

period = df[
    (df["date"] >= start_date)
    & (df["date"] <= end_date)
].copy()

print("\nPRICE DATA: OCTOBER 2023 TO SEPTEMBER 2024")
print("=" * 55)

if period.empty:
    print("No records found for the selected period.")
else:
    print(period.to_string(index=False))

    print("\nPRICE CHANGE ANALYSIS")
    print("=" * 55)

    first_price = period["avg_monthly_price"].iloc[0]
    last_price = period["avg_monthly_price"].iloc[-1]

    change = last_price - first_price

    if first_price != 0:
        change_percent = (change / first_price) * 100
    else:
        change_percent = float("nan")

    print(f"First month's price: {first_price:.2f}")
    print(f"Last month's price:  {last_price:.2f}")
    print(f"Total price change:  {change:.2f}")
    print(f"Percentage change:   {change_percent:.2f}%")

    print(
        "Largest monthly increase:",
        period["avg_monthly_price"].diff().max()
    )

    print(
        "Largest monthly decrease:",
        period["avg_monthly_price"].diff().min()
    )

    # Plot the selected monthly prices
    plt.figure(figsize=(12, 6))

    plt.plot(
        period["date"],
        period["avg_monthly_price"],
        marker="o",
        label="Monthly Price"
    )

    plt.title("Monthly Price Trend: Oct 2023 to Sep 2024")
    plt.xlabel("Date")
    plt.ylabel("Average Monthly Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("period2_price_trend.png")
    print("\nGraph saved as period2_price_trend.png")

    plt.show()
