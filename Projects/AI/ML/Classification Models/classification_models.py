# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler  # <-- ADDED FOR SCALING

# =============== Load the dataset and convert to DataFrame =======================
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

# Verify dataset shape and classes
print("=== Dataset Verification ===")
print(f"Features shape: {X.shape}")  
print(f"Target shape: {y.shape}")     
print(f"Class distribution:\n{y.value_counts()}")
print(f"Feature names:\n{wine.feature_names}")

# ============================ Exploratory Data Analysis (EDA) =====================
# Class distribution
sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()

# Statistical summary
print(X.describe())

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# ===================== Data Preparation =========================================
# Check missing values
print("Missing values:\n", X.isnull().sum())

# Scale the features (critical for models like Logistic Regression/SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # <-- DEFINE X_scaled HERE



# Train/Test split (now using X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Verify train-test split
print("\nTest set class counts:\n", y_test.value_counts())

# Verify split sizes
print("\n=== Train-Test Split Verification ===")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# DataFrame to store results
results = pd.DataFrame(columns=['Model', 'Accuracy'])

# Helper function for confusion matrix plotting
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()

# Train & evaluate models
# ================== Logistic Regression ==================

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Logistic Regression\n", classification_report(y_test, y_pred_lr))
plot_conf_matrix(y_test, y_pred_lr, "Logistic Regression")
results.loc[len(results)] = ['Logistic Regression', accuracy_score(y_test, y_pred_lr)]

# Add other models (Decision Tree, Random Forest, etc.) similarly...

# ================== Decision Tree ==================

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree \n", classification_report(y_test, y_pred_dt))
plot_conf_matrix(y_test, y_pred_dt, "Decision Tree")

# Visualize the tree (first 4 levels for clarity)
plt.figure(figsize=(12, 10))
plot_tree(dt, max_depth=4, feature_names=wine.feature_names, 
          class_names=wine.target_names, filled=True)
plt.title("Decision Tree (first 4 levels)")
plt.show()

results.loc[len(results)] = ['Decision Tree', accuracy_score(y_test, y_pred_dt)]

# ================== Random Forest ==================

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest \n", classification_report(y_test, y_pred_rf))
plot_conf_matrix(y_test, y_pred_rf, "Random Forest")
results.loc[len(results)] = ['Random Forest', accuracy_score(y_test, y_pred_rf)]

# ================== K-Nearest Neighbours ==================

kn = KNeighborsClassifier()
kn.fit(X_train, y_train)

y_pred_kn = kn.predict(X_test)

print("K-Nearest Neighbours \n", classification_report(y_test, y_pred_kn))
plot_conf_matrix(y_test, y_pred_kn, "K-Nearest Neighbours")
results.loc[len(results)] = ['K-Nearest Neighbours', accuracy_score(y_test, y_pred_kn)]

# ================== Naive Bayes ==================

gb = GaussianNB()
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

print("Gaussian Naive Bayess \n", classification_report(y_test, y_pred_gb))
plot_conf_matrix(y_test, y_pred_gb, "Gaussian Naive Bayes")
results.loc[len(results)] = ['Gaussian Naive Bayes', accuracy_score(y_test, y_pred_gb)]


# mb = MultinomialNB()
# mb.fit(X_train, y_train)

# y_pred_mb = mb.predict(X_test)

bb = BernoulliNB()
bb.fit(X_train, y_train)

y_pred_bb = bb.predict(X_test)

print("Bernoulli Naive Bayes \n", classification_report(y_test, y_pred_bb))
plot_conf_matrix(y_test, y_pred_bb, "Bernoulli Naive Bayes")
results.loc[len(results)] = ['Bernoulli Naive Bayes', accuracy_score(y_test, y_pred_bb)]

# ================== SVC ==================

# poly
# linear
# rbf
# sigmoid

# scale
# auto

sv_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
sv_rbf.fit(X_train, y_train)

y_pred_sv_rbf = sv_rbf.predict(X_test)

print("SVM with RBF Kernel \n", classification_report(y_test, y_pred_sv_rbf))
plot_conf_matrix(y_test, y_pred_sv_rbf, "SVM with RBF Kernel")
results.loc[len(results)] = ['SVM with RBF Kernel', accuracy_score(y_test, y_pred_sv_rbf)]


sv_linear = SVC(kernel="linear")
sv_linear.fit(X_train, y_train)

y_pred_sv_linear = sv_linear.predict(X_test)



print("SVM with Linear Kernell \n", classification_report(y_test, y_pred_sv_linear))
plot_conf_matrix(y_test, y_pred_sv_linear, "SVM with Linear Kernel")
results.loc[len(results)] = ['SVM with Linear Kernel', accuracy_score(y_test, y_pred_sv_linear)]


# ================== Results Summary ==================
print("\n\n" + "="*60)
print("Model Performance Summary")
print("="*60)
print(results.sort_values(by='Accuracy', ascending=False))

# Plot accuracy comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Accuracy', y='Model', data=results.sort_values('Accuracy'), palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1.1)
for index, value in enumerate(results.sort_values('Accuracy')['Accuracy']):
    plt.text(value, index, f'{value:.3f}')
plt.tight_layout()
plt.show()


