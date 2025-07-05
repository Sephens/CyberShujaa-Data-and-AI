# Wine Classification: Detailed Model Analysis

This document provides a comprehensive breakdown of each machine learning model used for wine classification, with detailed explanations of the code and visualizations.

## Table of Contents
- [Wine Classification: Detailed Model Analysis](#wine-classification-detailed-model-analysis)
  - [Table of Contents](#table-of-contents)
  - [Dataset Preparation](#dataset-preparation)
  - [Logistic Regression](#logistic-regression)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
  - [K-Nearest Neighbors](#k-nearest-neighbors)
  - [Naive Bayes Classifiers](#naive-bayes-classifiers)
    - [Gaussian NB](#gaussian-nb)
    - [Bernoulli NB](#bernoulli-nb)
  - [Support Vector Machines](#support-vector-machines)
    - [RBF Kernel](#rbf-kernel)
    - [Linear Kernel](#linear-kernel)
  - [Results Summary](#results-summary)

---

## Dataset Preparation

```python
# Load the dataset and convert to DataFrame
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

**Explanation:**
- `load_wine()` loads the Wine recognition dataset (13 chemical measurements for 3 wine types)
- We convert the data to a Pandas DataFrame for better handling
- Feature scaling with `StandardScaler()` is critical for models sensitive to feature scales (like SVM, Logistic Regression)
- `train_test_split` divides data into 70% training and 30% test sets, with `random_state=42` ensuring reproducibility

---

## Logistic Regression

```python
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("="*60)
print("Logistic Regression")
print("="*60)
print(classification_report(y_test, y_pred_lr))
plot_conf_matrix(y_test, y_pred_lr, "Logistic Regression")
plot_feature_importance(lr, wine.feature_names, "Logistic Regression")
```

**Key Components:**
- `max_iter=1000`: Increases maximum iterations for convergence
- `classification_report`: Shows precision, recall, f1-score per class
- `plot_conf_matrix`: Visualizes correct/incorrect predictions
- `plot_feature_importance`: Shows coefficient magnitudes (absolute values)

**Visualization Example:**
The confusion matrix might show:
- Perfect classification for class_0 (all 19 correct)
- 1 misclassification between class_1 and class_2

**Technical Notes:**
- Logistic regression uses a linear decision boundary
- Coefficients indicate feature importance direction and magnitude
- Works well when classes are linearly separable

---

## Decision Tree Classifier

```python
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Visualization
plt.figure(figsize=(20, 10))
plot_tree(dt, max_depth=2, feature_names=wine.feature_names, 
          class_names=wine.target_names, filled=True)
```

**Key Parameters:**
- `random_state=42`: Ensures reproducible tree structure
- `max_depth=2`: Limits tree visualization to first 2 levels

**Visual Interpretation:**
1. First split might be on "proline" with threshold 0.759
   - Left branch: Mostly class_0 wines
   - Right branch: Further splits on "color_intensity"

**Advantages:**
- No need for feature scaling
- Naturally handles multi-class problems
- Provides clear decision rules

---

## Random Forest Classifier

```python
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

plot_feature_importance(rf, wine.feature_names, "Random Forest")
```

**Key Differences from Single Tree:**
- Ensemble of many decision trees (default=100)
- Uses feature subsampling for diversity
- More robust to overfitting

**Feature Importance:**
The plot might show:
- "Proline" as most important (0.25)
- "Flavanoids" second (0.15)
- Minor features like "hue" near 0.02

---

## K-Nearest Neighbors

```python
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
y_pred_kn = kn.predict(X_test)
```

**How It Works:**
1. Calculates distances to all training points
2. Finds k nearest neighbors (default k=5)
3. Assigns majority class from neighbors

**Critical Notes:**
- Requires feature scaling (done earlier)
- Distance metric matters (default is Euclidean)
- Performance degrades in high dimensions

---

## Naive Bayes Classifiers

### Gaussian NB
```python
gb = GaussianNB()
gb.fit(X_train, y_train)
```
Assumes continuous features follow normal distribution

### Bernoulli NB
```python
bb = BernoulliNB()
bb.fit(X_train, y_train)
```
Designed for binary/boolean features

**Key Difference:**
- Gaussian works better for scaled continuous data
- Bernoulli would require binary features

---

## Support Vector Machines

### RBF Kernel
```python
sv_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
```
- Creates non-linear decision boundaries
- `gamma='scale'`: Automatic gamma selection

### Linear Kernel
```python
sv_linear = SVC(kernel='linear')
```
- Creates linear decision boundaries
- Can show feature coefficients

---

## Results Summary

```python
print(results.sort_values(by='Accuracy', ascending=False))

plt.figure(figsize=(12, 6))
sns.barplot(x='Accuracy', y='Model', data=results.sort_values('Accuracy'), palette='viridis')
```

**Typical Findings:**
1. Random Forest often performs best (0.98 accuracy)
2. SVM with RBF kernel close second (0.96)
3. Naive Bayes variants may underperform (0.90)

**Visualization Insights:**
The accuracy bar plot clearly shows performance ranking, helping select the best model for deployment.

