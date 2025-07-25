import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv('dataset/homeprices-m.csv')

# Explore the dataset
print("First 5 rows:\n", data.head())
print("\nDataset statistics:\n", data.describe())
print("\nMissing values:\n", data.isnull().sum())

# handle missing value in bedrooms
median = data.bedrooms.median()
data.bedrooms = data.bedrooms.fillna(median)

X = data.drop('price',axis='columns')
y = data['price'] # Target

X_train, y_train = X,y

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)  # Slopes (b1, b2, ...)
print("Intercept:", model.intercept_)  # Bias (b0)


test_data = pd.read_csv('dataset/homeprices-test.csv')

# Make predictions
predictions = model.predict(test_data)

results = pd.DataFrame({
    'Area (sq ft)': test_data['area'],
    'Bedrooms': test_data['bedrooms'],
    'Age (years)': test_data['age'],
    'Predicted Price (Ksh)': predictions.round().astype(int)  # Remove decimals
})

# Save results
results.to_csv('dataset/price_predictions_m.csv', index=False)

