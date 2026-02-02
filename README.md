# AI and Data Science Workshop: Customer Shopping Behavior Analysis  
*End-to-End Workflow in Google Colab*

---

## **Workshop Overview**

### **Target Audience**
Business leaders, developers, and consultants seeking a reusable, step-by-step workflow for applying AI to tabular datasets.

### **Learning Objectives**
- Load and understand CSV data in Colab
- Perform Exploratory Data Analysis (EDA) to extract business insights
- Clean common data issues and document changes
- Build baseline ML models for regression and classification
- Interpret confusion matrices and improve classification performance
- Create customer segments using clustering
- Understand association rules and forecasting applicability
- Build MLP neural network baselines for tabular data
- Explore LSTM for forecasting (with required data considerations)

### **Key Outputs**
- `cleaned_shopping_data.csv`
- `encoded_shopping_data.csv`
- `clustered_customer_data.csv`
- Trained model files
- Visualization charts

---

## **Dataset Note**
*This dataset has been taken from https://www.kaggle.com/datasets/wardabilal/customer-shopping-behaviour-analysis it has no date column and most customers appear only once, meaning:*
- Real forecasting of future orders cannot be performed
- Association rules may not yield meaningful cross-sell insights without multi-item/multi-order customer data

---

## **Workshop Structure**
1. **Colab Setup & Data Upload** (09:00–09:25)
2. **EDA: Data Understanding & Quality Checks** (09:25–10:45)
3. **Data Cleaning & Saving** (10:45–11:15)
4. **Deeper EDA: Pivots, Correlations, Outliers** (11:15–12:15)
5. **Regression: Predict Purchase Amount** (13:00–14:10)
6. **Classification: Predict Category** (14:10–15:10)
7. **Segmentation: Clustering** (15:10–15:40)
8. **Association Rules & Forecasting Concepts** (15:40–16:40)
9. **Export Outputs & Next Steps** (16:40–17:00)

---

## **1. Colab Setup & Environment Checks**
*Ensuring reproducibility and avoiding environment/file issues.*

```python
import os, sys
print("Python version:", sys.version.split()[0])
print("Current folder:", os.getcwd())
print("Files in this folder:", os.listdir(".")[:20])
```

**Output:**
```
Python version: 3.12.12
Current folder: /content
Files in this folder: ['.config', 'shopping_behavior_updated (1) (1).csv', 'shopping_behavior_updated (1).csv', 'sample_data']
```

**Interpretation:**  
- Python 3.12.12 is running in the `/content` directory
- The dataset `shopping_behavior_updated (1).csv` is available
- Environment is ready for data loading and analysis

---

## **2. Upload Dataset**
*Options: direct upload or Google Drive mount.*

```python
from google.colab import files
uploaded = files.upload()
print("Uploaded:", list(uploaded.keys()))
```

**Output:**
```
Saving shopping_behavior_updated (1).csv to shopping_behavior_updated (1) (2).csv
Uploaded: ['shopping_behavior_updated (1) (2).csv']
```

---

## **3. Install Optional Libraries**
*Ensuring reproducibility across teams.*

```python
!pip -q install mlxtend joblib
print("Installed: mlxtend, joblib")
```

**Output:**
```
Installed: mlxtend, joblib
```

---

## **4. Import Key Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import joblib
```

---

## **5. Load CSV into DataFrame**

```python
import os
print("Files in /content:", os.listdir("/content")[:30])

file_path = "/content/shopping_behavior_updated (1).csv"
df = pd.read_csv(file_path)

print("Loaded dataset shape (rows, columns):", df.shape)
df.head()
```

**Output:**
```
Files in /content: ['.config', 'shopping_behavior_updated (1) (1).csv', 'shopping_behavior_updated (1).csv', 'shopping_behavior_updated (1) (2).csv', 'sample_data']
Loaded dataset shape (rows, columns): (3900, 16)
```

| Customer ID | Age | Gender | Item Purchased | Category | Purchase Amount (USD) | Location | Size | Color | Season | Review Rating | Subscription Status | Discount Applied | Previous Purchases | Payment Method | Frequency of Purchases |
|-------------|-----|--------|----------------|----------|----------------------|----------|------|-------|--------|---------------|---------------------|------------------|-------------------|----------------|------------------------|
| 1 | 55 | Male | Blouse | Clothing | 53 | Kentucky | L | Gray | Winter | 3.1 | Yes | Yes | 14 | Venmo | Fortnightly |
| 2 | 19 | Male | Sweater | Clothing | 64 | Maine | L | Maroon | Winter | 3.1 | Yes | Yes | 2 | Cash | Fortnightly |
| 3 | 50 | Male | Jeans | Clothing | 73 | Massachusetts | S | Maroon | Spring | 3.1 | Yes | Yes | 23 | Credit Card | Weekly |
| 4 | 21 | Male | Sandals | Footwear | 90 | Rhode Island | M | Maroon | Spring | 3.5 | Yes | Yes | 49 | PayPal | Weekly |
| 5 | 45 | Male | Blouse | Clothing | 49 | Oregon | M | Turquoise | Spring | 2.7 | Yes | Yes | 31 | PayPal | Annually |

**Interpretation:**  
- Dataset contains 3900 rows and 16 columns
- Each row represents a single purchase transaction
- Columns include customer demographics, product details, purchase metrics, and behavioral signals

---

## **6. Data Dictionary**

| Column | Description |
|--------|-------------|
| Customer ID | Unique identifier |
| Age, Gender, Location | Customer demographics |
| Item Purchased, Category, Size, Color, Season | Product attributes |
| Purchase Amount (USD) | Spend outcome (regression target) |
| Review Rating | Customer satisfaction |
| Subscription Status, Discount Applied | Commercial levers |
| Previous Purchases, Frequency of Purchases | Loyalty/behavior signals |
| Payment Method | Transaction channel |

---

## **7. Quick Structural Checks**

```python
print("Columns:")
for c in df.columns:
    print("- ", c)

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
display(df.isnull().sum())

print("\nDuplicate rows:", int(df.duplicated().sum()))

print("\nNumeric summary:")
display(df.describe())
```

**Output Summary:**
- **Columns:** 16 total (5 numeric, 11 categorical)
- **Missing values:** None (complete dataset)
- **Duplicates:** 0 (unique rows)
- **Numeric stats:**
  - Age: 18–70 years (mean 44.1)
  - Purchase Amount: $20–$100 (mean $59.76)
  - Review Rating: 2.5–5.0 (mean 3.75)
  - Previous Purchases: 1–50 (mean 25.35)

**Interpretation:**  
- No missing data or duplicates—clean starting point
- Purchase amounts show wide range ($20–$100)
- Ratings are generally positive (mean 3.75/5)
- Previous purchases indicate varied customer loyalty

---

## **8. EDA Part 1: Categorical Columns**

```python
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    print(f"\n--- {col} ---")
    print("Unique values:", df[col].nunique())
    display(df[col].value_counts().head(10))
```

**Key Findings:**

| Column | Unique Values | Top Values |
|--------|--------------|------------|
| Gender | 2 | Male (2652), Female (1248) |
| Item Purchased | 25 | Blouse (171), Pants (171), Jewelry (171) |
| Category | 4 | Clothing (1737), Accessories (1240), Footwear (599) |
| Location | 50 | Montana (96), California (95), Idaho (93) |
| Size | 4 | M (1755), L (1053), S (663) |
| Color | 25 | Olive (177), Yellow (174), Silver (173) |
| Season | 4 | Spring (999), Fall (975), Winter (971) |
| Subscription Status | 2 | No (2847), Yes (1053) |
| Discount Applied | 2 | No (2223), Yes (1677) |
| Payment Method | 6 | PayPal (677), Credit Card (671), Cash (670) |
| Frequency of Purchases | 7 | Every 3 Months (584), Annually (572), Quarterly (563) |

**Interpretation:**  
- Male customers dominate the dataset (68%)
- Clothing is the most popular category (44% of purchases)
- Medium size is most common (45%)
- Most customers are not subscribed (73%)
- Payment methods are evenly distributed
- Purchase frequencies show no strong seasonality

---

## **9. EDA Part 2: Numeric Distributions**

```python
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", num_cols)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    if col == 'Customer ID':  # Skip ID column
        continue
    df[col].hist(ax=axes[i], bins=30, edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    if col == 'Customer ID':
        continue
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()
```

**Interpretation of Distributions:**

### **Age Distribution:**
- Nearly uniform between 18–70
- Slight dip in middle ages (30–40)
- Good representation across age groups

### **Purchase Amount:**
- Bimodal distribution with peaks at $40–$50 and $80–$90
- Suggests different pricing tiers or product categories

### **Review Rating:**
- Left-skewed with concentration at 3.1–4.0
- Few ratings below 3.0
- Most customers are satisfied

### **Previous Purchases:**
- Uniform distribution from 1–50
- Indicates equal representation of new and loyal customers

### **Box Plot Insights:**
- No extreme outliers in any numeric column
- Purchase amounts show tight IQR ($39–$81)
- Ratings vary within reasonable range (2.5–5.0)

---

## **Next Steps in Workshop**
*The workshop continues with:*

### **Data Cleaning**
- Handling inconsistent values
- Encoding categorical variables
- Feature engineering

### **Predictive Modeling**
1. **Regression:** Predict purchase amount using demographics and product features
2. **Classification:** Predict product category based on customer attributes
3. **Clustering:** Segment customers for targeted marketing

### **Advanced Topics**
- Neural networks (MLP) for tabular data
- Association rule mining (if data permits)
- Time-series forecasting concepts with LSTM

### **Business Applications**
- Price optimization based on customer segments
- Personalized recommendations
- Inventory planning by category/season
- Customer retention strategies

---

## **Conclusion of EDA Phase**
The dataset shows:
- **High quality:** No missing values or duplicates
- **Good balance:** Across categories, though male-dominated
- **Clear patterns:** Purchase amounts cluster at price points, ratings are generally positive
- **Actionable insights:** Customer segments can be derived from demographics and purchase behavior

*Proceed to modeling with clean, well-understood data.*
