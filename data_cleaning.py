import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("iris.csv")

# Step 2: Inspect the data
print("First 5 rows of the dataset:")
print(df.head())

print("\nData Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Step 3: Handle missing values
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values (if any) with the mean of the column
df.fillna(df.mean(numeric_only=True), inplace=True)

# Step 4: Remove duplicates
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Step 5: Correct data types
# Example: Ensure 'species' is a categorical column
df['species'] = df['species'].astype('category')

# Step 6: Fix inconsistent data
# Example: Standardize species names
df['species'] = df['species'].str.lower()

# Step 7: Handle outliers
# Example: Remove outliers using the IQR method for 'sepal_length'
Q1 = df['sepal_length'].quantile(0.25)
Q3 = df['sepal_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['sepal_length'] >= lower_bound) & (df['sepal_length'] <= upper_bound)]

# Step 8: Save the cleaned dataset
df.to_csv("cleaned_iris.csv", index=False)

print("\nData Cleaning Completed. Cleaned data saved to 'cleaned_iris.csv'.")
