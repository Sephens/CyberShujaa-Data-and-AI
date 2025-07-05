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
from sklearn.preprocessing import StandardScaler

# Load the dataset and convert to DataFrame
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# DataFrame to store results
results = pd.DataFrame(columns=['Model', 'Accuracy'])

# Helper function for confusion matrix plotting
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()

# Helper function to display feature importance
def plot_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances: {title}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) > 1:  # for multi-class
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
            
        indices = np.argsort(coef)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Coefficients (absolute): {title}')
        plt.bar(range(len(coef)), coef[indices], align='center')
        plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

# ================== 1. Logistic Regression ==================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("="*60)
print("Logistic Regression")
print("="*60)
print(classification_report(y_test, y_pred_lr))
plot_conf_matrix(y_test, y_pred_lr, "Logistic Regression")
plot_feature_importance(lr, wine.feature_names, "Logistic Regression")
results.loc[len(results)] = ['Logistic Regression', accuracy_score(y_test, y_pred_lr)]

# ================== 2. Decision Tree ==================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("="*60)
print("Decision Tree")
print("="*60)
print(classification_report(y_test, y_pred_dt))
plot_conf_matrix(y_test, y_pred_dt, "Decision Tree")
plot_feature_importance(dt, wine.feature_names, "Decision Tree")

# Visualize the tree (first 2 levels for clarity)
plt.figure(figsize=(20, 10))
plot_tree(dt, max_depth=2, feature_names=wine.feature_names, 
          class_names=wine.target_names, filled=True)
plt.title("Decision Tree (first 2 levels)")
plt.show()

results.loc[len(results)] = ['Decision Tree', accuracy_score(y_test, y_pred_dt)]

# ================== 3. Random Forest ==================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("="*60)
print("Random Forest")
print("="*60)
print(classification_report(y_test, y_pred_rf))
plot_conf_matrix(y_test, y_pred_rf, "Random Forest")
plot_feature_importance(rf, wine.feature_names, "Random Forest")
results.loc[len(results)] = ['Random Forest', accuracy_score(y_test, y_pred_rf)]

# ================== 4. K-Nearest Neighbors ==================
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
y_pred_kn = kn.predict(X_test)

print("="*60)
print("K-Nearest Neighbors")
print("="*60)
print(classification_report(y_test, y_pred_kn))
plot_conf_matrix(y_test, y_pred_kn, "K-Nearest Neighbors")
results.loc[len(results)] = ['K-Nearest Neighbors', accuracy_score(y_test, y_pred_kn)]

# ================== 5. Naive Bayes ==================
# Gaussian NB
gb = GaussianNB()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("="*60)
print("Gaussian Naive Bayes")
print("="*60)
print(classification_report(y_test, y_pred_gb))
plot_conf_matrix(y_test, y_pred_gb, "Gaussian Naive Bayes")
results.loc[len(results)] = ['Gaussian Naive Bayes', accuracy_score(y_test, y_pred_gb)]

# Bernoulli NB
bb = BernoulliNB()
bb.fit(X_train, y_train)
y_pred_bb = bb.predict(X_test)

print("="*60)
print("Bernoulli Naive Bayes")
print("="*60)
print(classification_report(y_test, y_pred_bb))
plot_conf_matrix(y_test, y_pred_bb, "Bernoulli Naive Bayes")
results.loc[len(results)] = ['Bernoulli Naive Bayes', accuracy_score(y_test, y_pred_bb)]

# ================== 6. Support Vector Machines ==================
# RBF Kernel
sv_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
sv_rbf.fit(X_train, y_train)
y_pred_sv_rbf = sv_rbf.predict(X_test)

print("="*60)
print("SVM with RBF Kernel")
print("="*60)
print(classification_report(y_test, y_pred_sv_rbf))
plot_conf_matrix(y_test, y_pred_sv_rbf, "SVM (RBF Kernel)")
results.loc[len(results)] = ['SVM (RBF Kernel)', accuracy_score(y_test, y_pred_sv_rbf)]

# Linear Kernel
sv_linear = SVC(kernel='linear')
sv_linear.fit(X_train, y_train)
y_pred_sv_linear = sv_linear.predict(X_test)

print("="*60)
print("SVM with Linear Kernel")
print("="*60)
print(classification_report(y_test, y_pred_sv_linear))
plot_conf_matrix(y_test, y_pred_sv_linear, "SVM (Linear Kernel)")
plot_feature_importance(sv_linear, wine.feature_names, "SVM (Linear Kernel)")
results.loc[len(results)] = ['SVM (Linear Kernel)', accuracy_score(y_test, y_pred_sv_linear)]

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