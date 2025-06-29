# ===========================================================
#  Exploratory Data Analysis on train.csv
# Author: Debargha Karmakar
# Internship Project @ Celebal
# ===========================================================

# -------------------------
#  Import the Libraries
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid")  


# -------------------------
#  Loading the Dataset
# -------------------------

df = pd.read_csv("train.csv")

# -------------------------
#  Basic Information
# -------------------------

print("\n Shape of the dataset:", df.shape)
print("\n First 5 rows:\n", df.head())
print("\n Data Types:\n", df.dtypes)
print("\n Summary Statistics:\n", df.describe(include='all'))
print("\n Null Values:\n", df.isnull().sum())

# -------------------------
#  Missing Value Analysis
# -------------------------

plt.figure(figsize=(10, 6))
msno.bar(df, fontsize=12)
plt.title("Missing Values Summary", fontsize=14)
plt.show()

plt.figure(figsize=(8, 5))
msno.heatmap(df)
plt.title("Missing Value Heatmap", fontsize=14)
plt.show()

missing_percent = df.isnull().mean() * 100
missing_df = pd.DataFrame({'Feature': df.columns, 'Missing %': missing_percent}).sort_values('Missing %', ascending=False)
print("\n Features with Missing Values:\n", missing_df[missing_df['Missing %'] > 0])

# -------------------------
#  Univariate Analysis
# -------------------------

# Numerical Features

num_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Categorical Features

cat_cols = df.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f"Count Plot for {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# -------------------------
#  Outlier Detection
# -------------------------

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.tight_layout()
    plt.show()

# Outlier Detection using IQR

def detect_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    return outliers.shape[0]

outlier_summary = pd.DataFrame({'Feature': num_cols,
                                'Outliers Count': [detect_outliers(col) for col in num_cols]})
print("\n Outlier Summary:\n", outlier_summary.sort_values("Outliers Count", ascending=False))

# -------------------------
#  Correlation & Multivariate Analysis
# -------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr().abs(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# Pairplot for top correlated features

corr_matrix = numeric_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
top_corr_features = [column for column in upper.columns if any(upper[column] > 0.75)]

if top_corr_features:
    sns.pairplot(numeric_df[top_corr_features].dropna())
    plt.suptitle("Pairplot of Highly Correlated Features", y=1.02)
    plt.show()
else:
    print("\n No features found with correlation > 0.75.")


# -------------------------
#  Key Insights
# -------------------------

print("\n KEY INSIGHTS :")
print("""
1. The dataset contains 891 entries with 12 variables.
2. Missing data exists in 'Age' (19.8%), 'Cabin' (77%), and 'Embarked' (0.2%). 'Cabin' has too many missing values for direct imputation.
3. The 'Fare' feature is highly right-skewed with several extreme values (outliers), suggesting log transformation might help.
4. Outliers are also detected in 'Age' and 'Fare' using IQR method.
5. Most passengers did not survive (around 62%), and survival rates appear to vary significantly across 'Sex' and 'Pclass'.
6. 'Sex', 'Pclass', and 'Fare' show moderate correlation with the target 'Survived' and are likely important predictors.
""")

# -------------------------
#  Recommendations
# -------------------------

print("\n RECOMMENDATIONS :")
print("""
1. Impute missing 'Age' values using median or model-based imputation. Drop or extract deck information from 'Cabin' instead of imputing.
2. Encode 'Sex' using label encoding and 'Embarked' using one-hot encoding for better machine learning compatibility.
3. Apply log transformation to the 'Fare' column to reduce skewness and the effect of outliers.
4. Create new features like:
   - 'FamilySize' = SibSp + Parch + 1
   - 'IsAlone' based on FamilySize
   - 'Title' extracted from 'Name'
5. Standardize continuous features (like 'Age' and 'Fare') using StandardScaler or MinMaxScaler.
6. Use Logistic Regression, Random Forest, or Gradient Boosting models with cross-validation for survival prediction.
""")
