import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data = pd.read_csv('dataset/Titanic-Dataset.csv')

# Explore the dataset
print("First 5 rows:\n", data.head())
print("\nDataset statistics:\n", data.describe())
print("\nMissing values:\n", data.isnull().sum())

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna('S', inplace=True)
data['Has_Cabin'] = data['Cabin'].notna().astype(int)
data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Convert categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'])

# Select features and target
# all features (input variables) by removing the 'Survived'
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42  # Added for reproducibility
)

print("\nTraining set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# After training the model (model.fit)
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# # Plot
# plt.figure(figsize=(10, 6))
# feature_importance.plot(kind='barh', color='skyblue')
# plt.title('Feature Importance for Survival Prediction')
# plt.xlabel('Importance Score')
# plt.ylabel('Features')
# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 1. Feature Importance
feature_importance.plot(kind='bar', ax=ax1, title='Feature Importance')

# 2. Survival Rate by Sex
data.groupby('Sex')['Survived'].mean().plot(
    kind='bar', color=['skyblue', 'pink'], ax=ax2,
    title='Survival Rate by Gender (0=Male, 1=Female)'
)
plt.show()