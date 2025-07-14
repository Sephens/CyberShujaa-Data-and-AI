# **Exploratory Data Analysis (EDA) of the Titanic Dataset**  
**Name:** Steven Odhiambo 
**Program:** CyberShuJaa Data and AI 
**Date:** 5-6-2025 

---

## **1. Introduction**  
The sinking of the Titanic in 1912 remains one of the most infamous maritime disasters in history. The dataset from Kaggle provides valuable insights into the passengers aboard the Titanic, including their survival status, demographics, and travel details.  

This report presents an **Exploratory Data Analysis (EDA)** of the Titanic dataset, covering:  
- Initial data exploration  
- Handling missing values and outliers  
- Univariate, bivariate, and multivariate analysis  
- Target variable (Survived) exploration  

The goal is to uncover patterns and relationships that may explain survival trends among passengers.  

---

## **2. Initial Data Exploration**  
The dataset contains **891 rows and 12 columns**, with the following features:  
- **PassengerId**: Unique identifier  
- **Survived**: Survival status (0 = No, 1 = Yes)  
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)  
- **Name**: Passenger name  
- **Sex**: Gender  
- **Age**: Age in years  
- **SibSp**: Number of siblings/spouses aboard  
- **Parch**: Number of parents/children aboard  
- **Ticket**: Ticket number  
- **Fare**: Passenger fare  
- **Cabin**: Cabin number  
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)  

### **2.1 Data Preview**  
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic.csv')
df.head()
```
**Output:**  
| PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
|------------|----------|--------|------|-----|-----|-------|-------|--------|------|-------|----------|
| 1 | 0 | 3 | Braund, Mr. Owen Harris | male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S |
| 2 | 1 | 1 | Cumings, Mrs. John Bradley | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |

### **2.2 Data Structure & Missing Values**  
```python
df.info()
```
**Output:**  
- **Missing Values:**  
  - **Age**: 177 missing  
  - **Cabin**: 687 missing  
  - **Embarked**: 2 missing  

### **2.3 Summary Statistics**  
```python
df.describe()
```
**Output:**  
| Feature | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|-------|------|-----|-----|-----|-----|-----|-----|
| Age | 714 | 29.70 | 14.53 | 0.42 | 20.12 | 28.00 | 38.00 | 80.00 |
| Fare | 891 | 32.20 | 49.69 | 0.00 | 7.91 | 14.45 | 31.00 | 512.33 |

---

## **3. Handling Missing Values & Outliers**  
### **3.1 Missing Value Treatment**  
- **Age**: Imputed with median (28.0)  
- **Embarked**: Filled with mode ('S')  
- **Cabin**: Dropped (too many missing values)  

```python
# Handle missing values safely
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Only drop Cabin if it exists
df = df.drop(columns=['Cabin'], errors='ignore')
```

### **3.2 Outlier Detection (Fare)**  
```python
sns.boxplot(x=df['Fare'])
plt.title('Fare Distribution with Outliers')
plt.show()
```
**Observation:**  
- Extreme outliers (Fare > 300) likely correspond to luxury-class passengers.  
- **Action:** Capped at the 99th percentile (Fare ≤ 300).  

```python
df['Fare'] = df['Fare'].clip(upper=300)
```

---

## **4. Univariate Analysis**  
### **4.1 Age Distribution**  
```python
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()
```
**Finding:**  
- Most passengers were between **20-40 years old**.  
- Few infants and elderly passengers.  

### **4.2 Embarked Location Counts**  
```python
sns.countplot(x='Embarked', data=df)
plt.title('Passengers by Embarkation Port')
plt.show()
```
**Finding:**  
- **Southampton (S)** had the most passengers.  

### **4.3 Fare Distribution (Skewed?)**  
```python
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title('Fare Distribution (Skewed Right)')
plt.show()
```
**Finding:**  
- **Right-skewed**: Most fares were below $50.  

---

## **5. Bivariate Analysis**  
### **5.1 Fare vs. Pclass**  
```python
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Passenger Class')
plt.show()
```
**Finding:**  
- **1st class** had the highest fares.  

### **5.2 Age vs. Survival**  
```python
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.show()
```
**Finding:**  
- **Children (<10) had higher survival rates.**  

### **5.3 Embarked vs. Survival**  
```python
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Embarkation Port')
plt.show()
```
**Finding:**  
- **Cherbourg (C)** had the highest survival rate.  

---

## **6. Multivariate Analysis**  
### **6.1 Pclass, Age, and Fare Impact on Survival**  
```python
sns.scatterplot(x='Age', y='Fare', hue='Survived', size='Pclass', data=df)
plt.title('Survival by Age, Fare, and Pclass')
plt.show()
```
**Finding:**  
- **1st-class passengers with higher fares had better survival odds.**  

### **6.2 Embarked & Pclass Interaction on Survival**  
```python
sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=df)
plt.title('Survival Rate by Embarked & Pclass')
plt.show()
```
**Finding:**  
- **1st-class passengers from Cherbourg (C) had the highest survival rate.**  

---

## **7. Target Variable (Survived) Analysis**  
### **7.1 Survival Distribution**  
```python
sns.countplot(x='Survived', data=df)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()
```
**Finding:**  
- **Imbalanced dataset**: Only **38% survived**.  

### **7.2 Key Survival Factors**  
- **Gender**: Females had a **74% survival rate** vs. males (19%).  
- **Pclass**: 1st-class passengers had **63% survival** vs. 3rd-class (24%).  

---

## **8. Conclusion**  
Key findings:  
✅ **Higher survival rates** for **women, children, and 1st-class passengers**.  
✅ **Fare and Pclass** strongly correlated with survival.  
✅ **Cherbourg (C) embarkation** had the highest survival rate.  

**Recommendation for predictive modeling:**  
- Include **Sex, Pclass, Age, and Fare** as key predictors.  
- Handle **class imbalance** (e.g., SMOTE or stratified sampling).  

🔗 **Kaggle Notebook Link:** [Insert Public Kaggle Notebook Link]  

---  
**End of Report**
