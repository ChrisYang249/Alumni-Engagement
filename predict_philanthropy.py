import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'AES Metrics for AI v3.xlsx'
df = pd.read_excel(file_path)

# Binarize the target: 1 if Philanthropy > 0, else 0
philanthropy_col = 'Alumni Engagement Philanthropic FY25'
df['philanthropy_bin'] = (df[philanthropy_col].fillna(0) > 0).astype(int)

# Select features
features = [
    'Gender', 'Class Year', 'Circle/Crescent House',
    'Alumni Engagement Score FY25',
    'Alumni Engagement Rating FY25',
    'Alumni Engagement Volunteer FY25',
    'Alumni Engagement Experiential FY25',
    'Alumni Engagement Comms FY25'
]

# Fill missing values for engagement scores with 0
for col in features[3:]:
    df[col] = df[col].fillna(0)

# Fill missing categorical values with 'Unknown'
df['Gender'] = df['Gender'].fillna('Unknown')
df['Circle/Crescent House'] = df['Circle/Crescent House'].fillna('Unknown')

# Optionally group Class Year into bins (decades)
df['Class Year Group'] = (df['Class Year'] // 10) * 10
features[1] = 'Class Year Group'

# One-hot encode categorical variables
df_model = pd.get_dummies(df[features], drop_first=True)

# Prepare X and y
X = df_model
y = df['philanthropy_bin']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print('--- Classification Report (Test Set) ---')
print(classification_report(y_test, y_pred))
print('\n--- Confusion Matrix (Test Set) ---')
print(confusion_matrix(y_test, y_pred))

# Cross-validation to check for overfitting
print('\n--- Cross-Validation Results ---')
cv_scores_accuracy = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
cv_scores_f1 = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')

print(f'Cross-Validation Accuracy Scores: {cv_scores_accuracy}')
print(f'Cross-Validation Accuracy Mean: {cv_scores_accuracy.mean():.3f} (+/- {cv_scores_accuracy.std() * 2:.3f})')
print(f'Cross-Validation F1 Scores: {cv_scores_f1}')
print(f'Cross-Validation F1 Mean: {cv_scores_f1.mean():.3f} (+/- {cv_scores_f1.std() * 2:.3f})')

# Check for overfitting
print('\n--- Overfitting Analysis ---')
test_accuracy = accuracy_score(y_test, y_pred)
cv_accuracy_mean = cv_scores_accuracy.mean()
overfitting_diff = test_accuracy - cv_accuracy_mean

print(f'Test Set Accuracy: {test_accuracy:.3f}')
print(f'CV Mean Accuracy: {cv_accuracy_mean:.3f}')
print(f'Difference (Test - CV): {overfitting_diff:.3f}')

if overfitting_diff > 0.05:
    print('⚠️  Potential overfitting detected (difference > 0.05)')
elif overfitting_diff > 0.02:
    print('⚠️  Minor overfitting possible (difference > 0.02)')
else:
    print('✅ No significant overfitting detected')

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print('\n--- Top 10 Feature Importances ---')
print(importances.head(10))

# Visualize top 10 feature importances
plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='bar')
plt.title('Top 10 Feature Importances (Random Forest Classification)')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()
print("\nFeature importances plot saved as 'feature_importances.png'")

# Plot and save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Engaged', 'Engaged'], yticklabels=['Not Engaged', 'Engaged'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Philanthropy Engagement Prediction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix plot saved as 'confusion_matrix.png'")

