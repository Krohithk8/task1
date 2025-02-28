import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load the Dataset
# Load the unemployment dataset (replace 'file_path' with your actual file path)
data = pd.read_csv('unemployment analysis.csv')

# Step 2: Initial Data Exploration
print("Dataset Preview:")
print(data.head())  # Display the first 5 rows

# Check for missing values and basic information
print("\nData Information:")
print(data.info())

# Step 3: Data Cleaning (if necessary)
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values or drop rows with missing data if necessary
# In this case, we are dropping rows with missing values for simplicity
data = data.dropna()

# Step 4: Convert Date Column to DateTime Format (if necessary)
# Assuming your 'date' column is in the format 'YYYY-MM-DD'
data['date'] = pd.to_datetime(data['date'])

# Step 5: Set Date Column as the Index
data.set_index('date', inplace=True)

# Step 6: Plot Unemployment Rate Over Time
plt.figure(figsize=(10, 6))
plt.plot(data['unemployment_rate'], label='Unemployment Rate', color='blue')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.legend()
plt.show()

# Step 7: Analyze Unemployment Rate by Region (if available)
# If there is a 'region' column, you can analyze it like this:
if 'region' in data.columns:
    regional_data = data.groupby('region')['unemployment_rate'].mean()
    regional_data.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Unemployment Rate by Region')
    plt.xlabel('Region')
    plt.ylabel('Unemployment Rate (%)')
    plt.show()

# Step 8: Correlation with Other Economic Indicators (Optional)
# If there are other economic indicators like GDP or inflation rate
if 'gdp' in data.columns and 'inflation_rate' in data.columns:
    correlation_matrix = data[['unemployment_rate', 'gdp', 'inflation_rate']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Economic Indicators')
    plt.show()

# Step 9: Forecasting Unemployment Rate (Optional)
# Fit an ARIMA model (you might need to tune the parameters)
# For simplicity, let's assume an ARIMA(1,1,1) model
model = ARIMA(data['unemployment_rate'], order=(1,1,1))
model_fit = model.fit()

# Forecast the next 12 months
forecast = model_fit.forecast(steps=12)

# Plot the forecast alongside the observed data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['unemployment_rate'], label='Observed', color='blue')
plt.plot(pd.date_range(data.index[-1], periods=12, freq='M'), forecast, label='Forecast', linestyle='--', color='red')
plt.legend()
plt.title('Unemployment Rate Forecast')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.show()

# Step 10: Save the Forecasted Data (Optional)
# You can save the forecasted data to a CSV file for further analysis
forecast_df = pd.DataFrame({'forecast': forecast}, index=pd.date_range(data.index[-1], periods=12, freq='M'))
forecast_df.to_csv('unemployment_forecast.csv')
