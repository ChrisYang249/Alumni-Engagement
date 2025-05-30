import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'AES Metrics for AI v3.xlsx'
df = pd.read_excel(file_path)

# Set style
sns.set(style="whitegrid")

# Boxplot: Philanthropy by Class Year (show only most common years for clarity)
top_years = df['Class Year'].value_counts().index[:10]
plt.figure(figsize=(12, 6))
sns.boxplot(x='Class Year', y='Alumni Engagement Philanthropic FY25', data=df[df['Class Year'].isin(top_years)])
plt.title('Philanthropy by Top 10 Class Years')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('philanthropy_by_class_year.png')
plt.close()

# Boxplot: Philanthropy by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Alumni Engagement Philanthropic FY25', data=df)
plt.title('Philanthropy by Gender')
plt.tight_layout()
plt.savefig('philanthropy_by_gender.png')
plt.close()

# Boxplot: Philanthropy by House (show only most common houses for clarity)
top_houses = df['Circle/Crescent House'].value_counts().index[:10]
plt.figure(figsize=(12, 6))
sns.boxplot(x='Circle/Crescent House', y='Alumni Engagement Philanthropic FY25', data=df[df['Circle/Crescent House'].isin(top_houses)])
plt.title('Philanthropy by Top 10 Houses')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('philanthropy_by_house.png')
plt.close()

print("Saved boxplots: philanthropy_by_class_year.png, philanthropy_by_gender.png, philanthropy_by_house.png") 