import pandas as pd

# Read the Excel file
file_path = 'AES Metrics for AI v3.xlsx'
df = pd.read_excel(file_path)

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst few rows of the data:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist()) 