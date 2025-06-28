import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load the dataset
data = pd.read_csv('dataset/homeprices.csv')
 
# exploer the dataset
print(data.head())
print(data.describe())

# check for missing values
print("Missing values \n",data.isnull().sum())

# visulizing using scatter plot

# plt.scatter(data['area'], data['price']) # X and Y
# plt.xlabel('Feature (area)')
# plt.ylabel('Target (Price)')
# plt.title('Scatter Plot of Feature vs Target')
# plt.show()


# Defining Feature and Target for our model
X = data[['area']] # Feature (Independent variable)
y = data['price'] # Target (Dependent variable)

# use the whole of the dataset for training without splitting
X_train, y_train = X,y

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, 
#     test_size=0.3,       # 30% for testing
#     random_state=42      # Ensures reproducibility
# )

# Check shapes
print("Training set shape:", X_train.shape, y_train.shape)
# print("Test set shape:", X_test.shape, y_test.shape)

# create an instance of the LinearRegression class.
model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)  # Slopes (b1, b2, ...)
print("Intercept:", model.intercept_)  # Bias (b0)


# Generate predictions using the trained model on test features

# y_pred = model.predict(X_test)

# Calculate Mean Absolute Error (MAE) - average absolute difference between true and predicted values

# mae = mean_absolute_error(y_test, y_pred)

# Calculate Mean Squared Error (MSE) - average squared difference (penalizes large errors more)

# mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE) - square root of MSE (in original units)
# Uses backward-compatible approach for different sklearn versions

# rmse = (mean_squared_error(y_test, y_pred, squared=False) 
#         if hasattr(mean_squared_error, 'squared') 
#         else np.sqrt(mse))

# Calculate R² Score - proportion of variance explained (1 is perfect, 0 is baseline mean)
# Note: Will be NaN if test set has <2 samples or no variance

# r2 = r2_score(y_test, y_pred)

# Print all metrics with 2 decimal places for readability

# print(f"MAE: {mae:.2f}")   
# print(f"MSE: {mse:.2f}") 
# print(f"RMSE: {rmse:.2f}")  
# print(f"R² Score: {r2:.2f}")

# plt.figure(figsize=(10, 6))
# plt.scatter(X_test, y_test, color='orange', label='Actual price')
# plt.plot(X_test, y_pred, color='red', linewidth=1, label='Regression Line')
# plt.xlabel('Area (sq ft)')
# plt.ylabel('Price (Ksh)')
# plt.title('Linear Regression Model')
# plt.legend()
# plt.show()

areas = pd.read_csv('dataset/areas.csv')
prices = model.predict(areas)

results = pd.DataFrame({
    'Area (sq ft)': areas['area'],  # Use your actual column name
    'Predicted Price (Ksh)': prices.round().astype(int)
})

# Save to new CSV
results.to_csv('dataset/predictions.csv', index=False)


