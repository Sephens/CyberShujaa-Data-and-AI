

This project focuses on building a **Linear Regression model** using Python to predict an outcome based on a single feature. The goal is to:  
- Explore a real-world dataset  
- Preprocess and split data for training/testing  
- Train a Linear Regression model  
- Evaluate performance using key metrics (MAE, MSE, RMSE, R² Score)  
- Visualize the regression line  

The dataset used is **[Dataset Name]**, which contains **[brief description of dataset]**.  

---

## Table of Contents
- [House Price Prediction Project](#house-price-prediction-project)
  - [Project Overview](#project-overview)
  - [Code Explanation](#code-explanation)
    - [1. Import Libraries](#1-import-libraries)
    - [2. Data Loading \& Exploration](#2-data-loading--exploration)
    - [3. Data Preprocessing](#3-data-preprocessing)
    - [4. Feature-Target Separation](#4-feature-target-separation)
    - [5. Model Training](#5-model-training)
    - [6. Prediction Pipeline](#6-prediction-pipeline)
    - [7. Results Formatting \& Export](#7-results-formatting--export)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Prediction \& Results](#prediction--results)


---

### **2.1 Data Loading & Exploration**  
First, we load the dataset and explore its structure:  

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('dataset.csv')
print(data.head())
print(data.describe())
print(data.isnull().sum())  # Check for missing values
```

![load-explore-data](/AI/ML/screenshots/load-explore-data.png) 

### **2.2 Data Visualization**  
We visualize the relationship between the feature (X) and target (Y) using a scatter plot:  

```python
plt.scatter(data['X'], data['Y'])
plt.xlabel('Feature (X)')
plt.ylabel('Target (Y)')
plt.title('Scatter Plot of Feature vs Target')
plt.show()
```

![evaluation](/AI/ML/screenshots/visualize-data.png) 

---
### **2.3 Data Splitting**  

`train_test_split` function from scikit-learn is used to split a dataset into training and testing sets. Here's a breakdown:

**What it does**:
Splits the data: Takes your feature matrix X and target variable y, and divides them into two subsets:

- Training set (X_train, y_train): Used to train the machine learning model
- Testing set (X_test, y_test): Used to evaluate the model's performance

Parameters:

- test_size=0.2: 20% of the data will be allocated to the test set, and 80% to the training set
- random_state=42: Sets a random seed to ensure the split is reproducible (you'll get the same split every time you run the code)

We split the data into training (80%) and testing (20%) sets:  

```python
from sklearn.model_selection import train_test_split

X = data[['X']]
y = data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
![splitting](/AI/ML/screenshots/splitting.png)

---
### **2.4 Model Training**  
We train a Linear Regression model using `scikit-learn`: 

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

This code snippet trains a **linear regression model** using scikit-learn. Here's a detailed breakdown:

### **1. `model = LinearRegression()`**  
- Creates an instance of the `LinearRegression` class.  
- This initializes a **linear regression model** with default parameters.  
- At this point, the model is untrained (has not seen any data yet).  

### **2. `model.fit(X_train, y_train)`**  
- **Trains the model** on the provided training data:  
  - `X_train` → Features (input variables, e.g., square footage, age, etc.)  
  - `y_train` → Target variable (output to predict, e.g., house price)  
- The method **learns the coefficients** (weights) that best fit the data by minimizing the **sum of squared errors** (ordinary least squares method).  

### **What Happens Under the Hood?**  
1. The model computes the **best-fit line** (or hyperplane in higher dimensions) using:  
   \[
   y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n
   \]
   where:  
   - \(b_0\) = intercept (bias term)  
   - \(b_1, b_2, ..., b_n\) = coefficients for each feature  

2. After training, you can access:  
   - `model.coef_` → Learned coefficients (slopes for each feature).  
   - `model.intercept_` → Bias term (\(b_0\)).  

### **Example Workflow**  
```python
from sklearn.linear_model import LinearRegression

# 1. Create & train model
model = LinearRegression()
model.fit(X_train, y_train)

# 2. Check learned parameters
print("Coefficients:", model.coef_)  # Slopes (b1, b2, ...)
print("Intercept:", model.intercept_)  # Bias (b0)

# 3. Predict on new data
y_pred = model.predict(X_test)  # Predictions for test set
```

This output shows the learned parameters of your **linear regression model** after training. Let's break it down:



### **1. Coefficients (`model.coef_`)**
- **Value:** `[128.27102804]`  
- **Meaning:**  
  - This is the **slope** (\(b_1\)) of our regression line.  
  - Interpretation: *For every 1-unit increase in the feature (X), the predicted target (y) increases by ~128.27 units.*  
  - Example: If `X` represents **house size (sq. ft.)**, the model predicts that each additional square foot increases the price by $128.27.



### **2. Intercept (`model.intercept_`)**
- **Value:** `211542.05607476638`  
- **Meaning:**  
  - This is the **baseline prediction** (\(b_0\)) when all features are `0`.  
  - Interpretation: *If the feature (X) is 0, the predicted target (y) is ~211,542.06.*  
  - Example: If `X` is **house size**, this suggests a house with `0 sq. ft.` (theoretical) would "cost" $211,542.06 (likely capturing fixed costs like land value or model bias).  


### **Putting It All Together**
Your model’s equation is:  
\[
\hat{y} = 211542.06 + 128.27 \times X
\]  
- **Prediction Example:**  
  - For a house with `X = 1500 sq. ft.`:  
    \[
    \hat{y} = 211542.06 + (128.27 \times 1500) = 403,947.06
    \]  
  

![training](/AI/ML/screenshots/training.png)

---
### **2.5 Model Evaluation**  
This section evaluates the trained linear regression model using four key metrics to assess its performance on the test set (X_test, y_test).  

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
```

## **1. Making Predictions (`y_pred`)**
```python
y_pred = model.predict(X_test)
```
- **What it does**: Uses the trained model to predict target values (`y_pred`) for the test features (`X_test`).  
- **Why it matters**: These predictions are compared against the actual test values (`y_test`) to evaluate model accuracy.


## **2. Evaluation Metrics**
Four metrics are computed to assess model performance:

### **1. Mean Absolute Error (MAE)**
```python
mae = mean_absolute_error(y_test, y_pred)
```
- **Formula**:  
  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]
- **Interpretation**:  
  - Average **absolute difference** between predicted (`y_pred`) and actual (`y_test`) values.  
  - Lower MAE = better model.  
  - **Example**: `MAE = 5000` means predictions are, on average, off by **±5000 units** (e.g., dollars if predicting house prices).

### **2. Mean Squared Error (MSE)**
```python
mse = mean_squared_error(y_test, y_pred)
```
- **Formula**:  
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
- **Interpretation**:  
  - Average of **squared errors** (penalizes large errors more than MAE).  
  - **Lower MSE = better model**, but hard to interpret since units are squared.  
  - **Example**: `MSE = 40,000,000` (if `y` is in dollars, units are dollars²).

### **3. Root Mean Squared Error (RMSE)**
```python
rmse = mean_squared_error(y_test, y_pred, squared=False)
```
- **Formula**:  
  \[
  \text{RMSE} = \sqrt{\text{MSE}}
  \]
- **Interpretation**:  
  - **Square root of MSE** (brings units back to original scale).  
  - More interpretable than MSE.  
  - **Example**: `RMSE = 6,324` means predictions are off by **±6,324 units** on average.

### **4. R² Score (Coefficient of Determination)**
```python
r2 = r2_score(y_test, y_pred)
```
- **Formula**:  
  \[
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  \]
- **Interpretation**:  
  - Measures **how much variance** in `y_test` is explained by the model.  
  - **Range**: `0` (worst) to `1` (best).  
  - **Example**: `R² = 0.85` means **85% of variability** in `y_test` is explained by the model.


## **3. Printing the Results**
```python
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
```
- **Output Example**:
  ```
  MAE: 5000.00
  MSE: 40000000.00
  RMSE: 6324.56
  R² Score: 0.85
  ```
- **`:2f`** formats numbers to **2 decimal places** for readability.


| Metric | Interpretation | Preferred Value |
|--------|---------------|----------------|
| **MAE** | Average error magnitude | Closer to 0 |
| **MSE** | Squared errors (punishes outliers) | Closer to 0 |
| **RMSE** | Error in original units | Closer to 0 |
| **R²** | % of variance explained | Closer to 1 |

- **RMSE** is often the most useful for business decisions (same units as `y`).  
- **R²** tells how well the model fits compared to a simple mean prediction.  
- **MAE** is robust to outliers, while **MSE/RMSE** penalize large errors more.

![evaluation](/AI/ML/screenshots/evaluation.png)

---

### **2.6 Visualization of Results**  
We plot the regression line against the test data:  

```python
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Feature (X)')
plt.ylabel('Target (Y)')
plt.title('Linear Regression Model')
plt.legend()
plt.show()
```

![visualization](/AI/ML/screenshots/visualize-data2.png)

---

## **3. Conclusion**  
Through this project, I learned:  
- How to **preprocess** and **explore** a dataset for regression.  
- The importance of **train-test splitting** to avoid overfitting.  
- How to **train and evaluate** a Linear Regression model using key metrics.  
- The significance of **visualizing results** to interpret model performance.  

The model achieved an **R² score of [X]**, indicating **[good/moderate/poor] fit**. Future improvements could include feature engineering or trying different regression models.  

---
---

# House Price Prediction Project

This project implements a **Multiple Linear Regression** model to predict house prices based on features like area, bedrooms, and age. Below is a detailed explanation of the code and its components.



## Project Overview
This predictive model:
- Uses historical house price data
- Handles missing values intelligently
- Trains a linear regression model
- Generates price predictions for new properties
- Exports results in a clean CSV format

## Code Explanation

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```
- **Pandas/Numpy**: Data manipulation and numerical operations
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning components (models, metrics, utilities)

### 2. Data Loading & Exploration
```python
data = pd.read_csv('dataset/homeprices-m.csv')
print("First 5 rows:\n", data.head())
print("\nDataset statistics:\n", data.describe())
print("\nMissing values:\n", data.isnull().sum())
```
- Loads the training dataset from CSV
- Shows sample data, statistical summary, and missing value count
- Helps understand data distribution and quality

### 3. Data Preprocessing
```python
median = data.bedrooms.median()
data.bedrooms = data.bedrooms.fillna(median)
```
- Calculates median bedrooms value
- Fills missing bedroom values with the median (robust to outliers)
- Ensures complete data for model training

### 4. Feature-Target Separation
```python
X = data.drop('price', axis='columns')
y = data['price']
```
- **X**: Contains all features (area, bedrooms, age)
- **y**: Contains target variable (price)
- Prepares data for supervised learning

### 5. Model Training
```python
model = LinearRegression()
model.fit(X_train, y_train)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```
- Initializes Linear Regression model
- Trains model on all available data (no test split)
- Outputs learned coefficients and intercept
  - Coefficients show each feature's impact on price
  - Intercept represents base price when all features are zero

### 6. Prediction Pipeline
```python
test_data = pd.read_csv('dataset/homeprices-test.csv')
predictions = model.predict(test_data)
```
- Loads new property data for prediction
- Generates price estimates using trained model

### 7. Results Formatting & Export
```python
results = pd.DataFrame({
    'Area (sq ft)': test_data['area'],
    'Bedrooms': test_data['bedrooms'],
    'Age (years)': test_data['age'],
    'Predicted Price (Ksh)': predictions.round().astype(int)
})
results.to_csv('dataset/price_predictions_m.csv', index=False)
```
- Creates clean results dataframe
- Rounds prices to whole Kenyan Shillings
- Exports to CSV without row indices

## Data Preparation
The model expects input data with these columns:
1. `area`: Property size in square feet
2. `bedrooms`: Number of bedrooms
3. `age`: Property age in years

## Model Training
- Algorithm: Ordinary Least Squares (OLS) Regression
- Trained on all available data (no holdout set)
- Outputs linear equation:  
  `Price = (area_coef × area) + (bedrooms_coef × bedrooms) + (age_coef × age) + intercept`

## Prediction & Results
Example output CSV:

| Area (sq ft) | Bedrooms | Age (years) | Predicted Price (Ksh) |
|--------------|----------|-------------|-----------------------|
| 2500         | 3        | 10          | 4,231,456            |
| 3200         | 4        | 5           | 5,123,789            |


