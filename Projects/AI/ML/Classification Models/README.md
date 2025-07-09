# Optimizing Wine Varietal Identification: Performance Analysis of 6 Classification Algorithms

This project applies supervised machine learning classification techniques to the Wine dataset from scikit-learn, aiming to predict wine categories based on physicochemical properties. The assignment involves building and evaluating six distinct models: Logistic Regression, Decision Tree, Random Forest, k-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machine (SVM). Key steps include exploratory data analysis to uncover feature relationships and class distributions, data preprocessing (handling missing values and train-test splitting), and rigorous model evaluation. Performance is assessed using accuracy, precision, recall, F1-scores, and visualized confusion matrices. The comparative analysis identifies optimal models by highlighting trade-offs in predictive power, interpretability, and computational efficiency, providing practical insights into model selection for multiclass classification tasks.

## Table of Contents
- [Optimizing Wine Varietal Identification: Performance Analysis of 6 Classification Algorithms](#optimizing-wine-varietal-identification-performance-analysis-of-6-classification-algorithms)
  - [Table of Contents](#table-of-contents)
  - [Loading the Dataset](#loading-the-dataset)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Preparation](#data-preparation)
  - [Building and Evaluating the Models](#building-and-evaluating-the-models)
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
  - [Model Comparison](#model-comparison)
  - [Conclusion](#conclusion)

---

## Loading the Dataset
The first step involves loading the Wine dataset from `scikit-learn`, a popular dataset for classification tasks. The dataset contains 178 samples of wines categorized into 3 classes, each defined by 13 numerical features (e.g., alcohol content, malic acid, flavonoids). Using Python, we import the dataset via `load_wine()` and convert it into a structured format features into a Pandas DataFrame (X) and target labels into a Series (y). This ensures easy manipulation and visualization. The dataset is then split into training (70%) and testing (30%) sets using train_test_split() to facilitate model training and unbiased evaluation.

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

## Exploratory Data Analysis (EDA)

This task involves analyzing the Wine dataset to uncover patterns, feature relationships,
and class distributions. Key steps include:
- Statistical Summary: Checking mean, variance, and quartiles of features using
describe().
- Class Distribution: Visualizing target class balance with a count plot.

- Feature Correlation: Plotting a heatmap to identify strong correlations between
variables.
- Distribution Plots: Using histograms or boxplots to examine feature skewness and
outliers.
- Pair Plots: Comparing feature interactions across different wine classes.

EDA helps detect data imbalances, redundant features, and guides preprocessing decisions
before model training.

```python

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

```

---

## Data Preparation
This task prepares the dataset for modeling by ensuring data quality and proper
partitioning. Key steps include:
- Missing Values: Check for null entries with `isnull().sum()` and handle them (though the Wine dataset typically has none).
- Feature Scaling: Normalize features using StandardScaler to ensure equal weighting (critical for SVM/KNN).
- Train-Test Split: Partition data into 70% training and 30% testing sets using train_test_split() with random_state=42 for reproducibility.
- Class Stratification: Verify balanced class distribution in splits via
y_train.value_counts().

Proper preprocessing ensures models train on consistent, unbiased data..

```python

# Check missing values
print("Missing values:\n", X.isnull().sum())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Verify train-test split
print("\nTest set class counts:\n", y_test.value_counts())

# Verify split sizes
print("\n=== Train-Test Split Verification ===")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


```

---

## Building and Evaluating the Models
This task involves building six classification models to predict wine categories:
- **Logistic Regression**: A linear model for multi-class classification using softmax.
- **Decision Tree**: A rule-based model with hierarchical splits.
- **Random Forest**: An ensemble of decision trees for improved accuracy.
- **k-Nearest Neighbors (KNN)**: Predicts based on similarity to neighboring data points.
- **Naive Bayes**: Applies Bayes' theorem with feature independence assumption.
- **Support Vector Machine (SVM)**: Finds optimal hyperplanes for class separation.

Each model is trained on scaled data (X_train) and evaluated later. Hyperparameters are
set to defaults initially for baseline performance comparison.

We then assesses or evaluate model performance using key metrics:
- **_Accuracy Score_**: Measures overall prediction correctness-​ Classification Report: Breaks down performance per class with precision, recall, and F1-score to evaluate bias and robustness.
- **_Confusion Matrix_**: Visualizes true vs. predicted classes via heatmaps, highlighting misclassifications.

These metrics help identify strengths and weaknesses:

- High precision indicates fewer false positives.
- High recall ensures minimal false negatives.
- F1-score balances both for imbalanced classes.

Visual plots enhance interpretability, guiding model selection for optimal performance.

### Logistic Regression

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

### Decision Tree Classifier

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

### Random Forest Classifier

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

### K-Nearest Neighbors

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

### Naive Bayes Classifiers

#### Gaussian NB
```python
gb = GaussianNB()
gb.fit(X_train, y_train)
```
Assumes continuous features follow normal distribution

#### Bernoulli NB
```python
bb = BernoulliNB()
bb.fit(X_train, y_train)
```
Designed for binary/boolean features

**Key Difference:**
- Gaussian works better for scaled continuous data
- Bernoulli would require binary features

---

### Support Vector Machines

#### RBF Kernel
```python
sv_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
```
- Creates non-linear decision boundaries
- `gamma='scale'`: Automatic gamma selection

#### Linear Kernel
```python
sv_linear = SVC(kernel='linear')
```
- Creates linear decision boundaries
- Can show feature coefficients

---

## Model Comparison
This task compares the performance of all six classification models using evaluation metrics. Key findings include:
- **Accuracy**: Random Forest and SVM typically achieve the highest scores, followed by Logistic Regression.
- **Precision/Recall**: SVM and Random Forest show balanced precision-recall trade-offs,
while Naive Bayes may struggle with imbalanced classes.
- **F1-Score**: Ensemble methods (Random Forest) often outperform others due to robustness against overfitting.
- **Confusion Matrix**: Decision Trees and KNN may exhibit higher misclassifications in complex boundaries.

The best model is selected based on highest accuracy, balanced F1-scores, and minimal misclassifications, with Random Forest or SVM usually being optimal for the Wine dataset.

```python
print(results.sort_values(by='Accuracy', ascending=False))

plt.figure(figsize=(12, 6))
sns.barplot(x='Accuracy', y='Model', data=results.sort_values('Accuracy'), palette='viridis')
```

---

## Conclusion
This project successfully implemented and evaluated six machine learning classification models, Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, and SVM on the Wine dataset. Through comprehensive exploratory data analysis (EDA), we identified
key feature relationships and ensured proper data preprocessing, including scaling and train-test splitting. Model evaluation metrics (accuracy, precision, recall, F1-score, and confusion matrices) revealed that Random Forest and SVM consistently outperformed other
models, achieving the highest accuracy and balanced performance across all classes.

The Random Forest model demonstrated superior robustness, handling non-linear patterns effectively while minimizing overfitting. SVM also performed exceptionally well, particularly in maximizing class separation. In contrast, simpler models like Logistic Regression and
Naive Bayes struggled with complex feature interactions, while Decision Trees and KNN showed higher variance in predictions.

This exercise reinforced the importance of model selection based on dataset characteristics ensemble methods and kernel-based models excelled in this multi-class classification task. Future work could explore hyperparameter tuning and feature engineering to further
enhance predictive performance. Overall, the project provided valuable insights into comparative model analysis and real-world classification challenges.
