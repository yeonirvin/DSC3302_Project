import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Modify file paths based on OS
file_path = "/Users/yeonirvin/Desktop/DSC3302 Project/Historical Data.xlsx"  # Adjust this path
output_path = "/Users/yeonirvin/Desktop/DSC3302 Project/ARIMA_Forecast_All_Products.xlsx"

# Load historical data
df = pd.read_excel(file_path, sheet_name="Historical Raw Data")

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Get a list of unique Product IDs
product_ids = df["Product ID"].unique()

# Forecasting settings
forecast_steps = 12  # Forecast for 12 months
all_forecasts = []  # Store forecasts for all products

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())  # Drop NaN before checking stationarity
    return result[1] < 0.05  # True if series is stationary

# Dictionary to store product forecasts
forecast_data = {}

# Loop through each product
for product_id in product_ids:
    print(f"\nProcessing Product ID: {product_id}")

    # Aggregate demand (Units Sold) by Date for the current Product ID
    df_product = df[df["Product ID"] == product_id].groupby("Date")["Units Sold"].sum().reset_index()

    # Ensure data is sorted
    df_product = df_product.sort_values(by="Date")

    # Skip if not enough data points
    if len(df_product) < 12:
        print(f"Skipping {product_id} - Not enough data points.")
        continue

    # Check stationarity
    is_stationary = check_stationarity(df_product["Units Sold"])

    # Apply differencing if needed
    if not is_stationary:
        df_product["Units Sold Diff"] = df_product["Units Sold"].diff()
        df_product.dropna(subset=["Units Sold Diff"], inplace=True)  # Ensure no NaN values remain
        is_stationary = check_stationarity(df_product["Units Sold Diff"])

    # Skip if still not enough data after differencing
    if len(df_product) < 12:
        print(f"Skipping {product_id} - Not enough data after differencing.")
        continue

    # Define ARIMA order (p, d, q) - Adjust based on ACF/PACF
    p, d, q = 1, 0, 1  

    # Fit ARIMA model
    print(f"Fitting ARIMA model for {product_id}...")
    model = ARIMA(df_product["Units Sold"], order=(p, d, q))
    model_fit = model.fit()

    # Forecast future demand
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create forecasted date range
    forecast_dates = pd.date_range(start=df_product["Date"].max(), periods=forecast_steps + 1, freq="ME")[1:]

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Product ID": product_id,
        "Forecasted Units Sold": forecast.values
    })

    # Store forecast for later use
    forecast_data[product_id] = {
        "historical": df_product,
        "forecast": forecast_df
    }

    # Append to list
    all_forecasts.append(forecast_df)

# Combine all forecasts
final_forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Save results to Excel
final_forecast_df.to_excel(output_path, index=False)
print(f"\nForecast for all products saved to '{output_path}'")

# Display forecast summary
print("\nForecast Summary:")
print(final_forecast_df.head())

# --- FILTERED PLOTTING BY PRODUCT ID ---
while True:
    selected_product_id = input("\nEnter a Product ID to visualize ACF, PACF, and Forecast (or type 'exit' to quit): ").strip()
    
    if selected_product_id.lower() == "exit":
        break

    if selected_product_id not in forecast_data:
        print("Invalid Product ID or no forecast data available. Please enter a valid one.")
        continue

    # Retrieve data for selected product
    df_product = forecast_data[selected_product_id]["historical"]
    product_forecast = forecast_data[selected_product_id]["forecast"]

    # Plot ACF and PACF together
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ACF Plot (for MA component q)
    plot_acf(df_product["Units Sold"], ax=axes[0], lags=10)
    axes[0].set_title(f"ACF Plot for {selected_product_id}")

    # PACF Plot (for AR component p)
    plot_pacf(df_product["Units Sold"], ax=axes[1], lags=10, method="ywm")
    axes[1].set_title(f"PACF Plot for {selected_product_id}")

    plt.tight_layout()
    plt.show()

    # Plot filtered forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df_product["Date"], df_product["Units Sold"], marker="o", linestyle="-", label="Historical Sales")
    plt.plot(product_forecast["Date"], product_forecast["Forecasted Units Sold"], marker="o", linestyle="--", color="red", label="Forecasted Sales")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.title(f"ARIMA Forecast for Product {selected_product_id}")
    plt.legend()
    plt.grid()
    plt.show()
