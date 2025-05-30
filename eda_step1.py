import pandas as pd

# Load the data
file_path = 'AES Metrics for AI v3.xlsx'
df = pd.read_excel(file_path)

# 1. Data Cleaning & Initial Overview
print('--- Data Types ---')
print(df.dtypes)

print('\n--- Missing Values (count) ---')
print(df.isnull().sum())

# Value counts for key categorical columns
def value_counts_with_nan(col):
    return df[col].value_counts(dropna=False)

categorical_cols = ['Gender', 'Primary Constituency', 'Additional Constituency ', 'City', 'State', 'Country', 'Class Year', 'Circle/Crescent House']

for col in categorical_cols:
    print(f'\n--- Value Counts: {col} ---')
    print(value_counts_with_nan(col).head(10))  # Show top 10 for brevity 