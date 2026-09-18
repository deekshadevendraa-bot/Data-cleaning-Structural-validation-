# Data-cleaning-Structural-validation-
# ============================================
# DATA CLEANING & STRUCTURAL VALIDATION
# ============================================

import pandas as pd
import numpy as np

# --------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------

# Change this to your downloaded CSV file name
file_path = "sample_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Display first 5 rows
display(df.head())


# --------------------------------------------
# 2. INSPECT THE DATA
# --------------------------------------------

print("\n========== DATA INFORMATION ==========")
df.info()

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST 10 ROWS ==========")
display(df.head(10))

print("\n========== STATISTICAL SUMMARY ==========")
display(df.describe(include="all").T)


# --------------------------------------------
# 3. CHECK MISSING VALUES
# --------------------------------------------

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

display(missing_report)


# --------------------------------------------
# 4. CHECK DUPLICATE RECORDS
# --------------------------------------------

print("\n========== DUPLICATE RECORDS ==========")

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    print("\nDuplicate rows:")
    display(df[df.duplicated()])
else:
    print("No duplicate rows found.")


# --------------------------------------------
# 5. STANDARDIZE COLUMN HEADERS
# --------------------------------------------

print("\n========== STANDARDIZING COLUMN NAMES ==========")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("New column names:")
print(df.columns.tolist())


# --------------------------------------------
# 6. REMOVE DUPLICATES
# --------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\n========== DUPLICATE REMOVAL ==========")
print("Rows before:", before_duplicates)
print("Rows after:", after_duplicates)
print("Duplicates removed:", before_duplicates - after_duplicates)


# --------------------------------------------
# 7. CLEAN TEXT / CATEGORICAL COLUMNS
# --------------------------------------------

print("\n========== CLEANING TEXT COLUMNS ==========")

text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()

print("Text columns cleaned.")


# --------------------------------------------
# 8. HANDLE MISSING VALUES
# --------------------------------------------

print("\n========== HANDLING MISSING VALUES ==========")

# Separate numeric and categorical columns
numeric_columns = df.select_dtypes(include=np.number).columns
categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns

# Fill numeric missing values with median
for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)
        print(
            f"{column}: Missing