import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'AES Metrics for AI v3.xlsx'
df = pd.read_excel(file_path)

# 2. Descriptive Statistics
print('--- Summary Statistics for Numeric Columns ---')
print(df.describe())

# 3. Correlation Matrix for Engagement Metrics and Philanthropy
engagement_cols = [
    'Alumni Engagement Score FY25',
    'Alumni Engagement Rating FY25',
    'Alumni Engagement Philanthropic FY25',
    'Alumni Engagement Volunteer FY25',
    'Alumni Engagement Experiential FY25',
    'Alumni Engagement Comms FY25'
]

corr = df[engagement_cols].corr()
print('\n--- Correlation Matrix (Engagement Metrics) ---')
print(corr)

# 4. Visualize Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix: Engagement Metrics & Philanthropy')
plt.tight_layout()
plt.savefig('engagement_correlation_heatmap.png')
plt.close()
print("\nHeatmap saved as 'engagement_correlation_heatmap.png'") 