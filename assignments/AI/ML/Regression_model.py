import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% for testing
    random_state=42      # Ensures reproducibility
)

# Check shapes
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)


