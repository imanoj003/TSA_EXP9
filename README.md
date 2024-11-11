# Develepor name : Manoj M
# Reg no: 212221240027
# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Data Preparation: Load the data into a Pandas DataFrame. Encode any categorical features (like Gender and ParentalSupport) using label encoding, converting them into numerical values for compatibility with machine learning models.
3. Select FinalGrade as the target series to forecast. Split the series into training and testing sets, with 80% of the data for training and the remaining 20% for testing.
4. Use an ARIMA model to forecast FinalGrade. Specify the ARIMA parameters (p, d, q) (e.g., (1, 1, 1)), where p is the autoregressive term, d is the differencing term, and q is the moving average term. Train the ARIMA model on the training data.
5. Generate a forecast for the testing set length using the trained model. This step produces the predicted values for FinalGrade.
6.Plot both the actual FinalGrade values and the forecasted values to visualize the model's performance. 
7.Print the forecasted values to assess the model's accuracy against the actual test data.
### PROGRAM:
```# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


df = pd.DataFrame('/content/student_performance(1)(1).csv')

# Encode categorical features (like Gender and ParentalSupport)
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_parental_support = LabelEncoder()
df['ParentalSupport'] = le_parental_support.fit_transform(df['ParentalSupport'])

# Define the series we want to forecast (e.g., FinalGrade)
series = df['FinalGrade']

# Split data into training and test sets
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(1, 1, 1))  # order=(p,d,q) - adjust these parameters for tuning
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(series.index, series, label='Actual Final Grade')
plt.plot(test.index, forecast, color='red', label='Forecasted Final Grade')
plt.title('ARIMA Forecast for Final Grade')
plt.xlabel('Index')
plt.ylabel('Final Grade')
plt.legend()
plt.show()
# Print the forecasted values
print("Forecasted Final Grade values:", forecast)
```

### OUTPUT:

![Untitled](https://github.com/user-attachments/assets/724e7b49-379e-4e5e-8350-d6abf00a5853)


### RESULT:
thus the program run successfully based on the ARIMA model using python.
