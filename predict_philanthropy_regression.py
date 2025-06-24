import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the data
file_path = 'AES Metrics for AI v3.xlsx'
df = pd.read_excel(file_path)

# Use the raw philanthropy values (1-10) as target
philanthropy_col = 'Alumni Engagement Philanthropic FY25'
# Fill missing philanthropy values with 0 for training
df[philanthropy_col] = df[philanthropy_col].fillna(0)

# Select features (same as classification model)
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

# Group Class Year into bins (decades)
df['Class Year Group'] = (df['Class Year'] // 10) * 10
features[1] = 'Class Year Group'
# One-hot encode categorical variables
df_model = pd.get_dummies(df[features], drop_first=True)

# Prepare X and y
X = df_model
y = df[philanthropy_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_regressor.predict(X_test)

# Calculate regression metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('--- Regression Metrics (Test Set) ---')
print(f'RMSE: {rmse:.3f}')
print(f'MAE: {mae:.3f}')
print(f'R² Score: {r2:.3f}')

# Cross-validation to check for overfitting
print('\n--- Cross-Validation Results ---')
cv_scores_r2 = cross_val_score(rf_regressor, X_train, y_train, cv=5, scoring='r2')
cv_scores_rmse = cross_val_score(rf_regressor, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_scores_mae = cross_val_score(rf_regressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

print(f'Cross-Validation R² Scores: {cv_scores_r2}')
print(f'Cross-Validation R² Mean: {cv_scores_r2.mean():.3f} (+/- {cv_scores_r2.std() * 2:.3f})')
print(f'Cross-Validation RMSE Mean: {-cv_scores_rmse.mean():.3f} (+/- {cv_scores_rmse.std() * 2:.3f})')
print(f'Cross-Validation MAE Mean: {-cv_scores_mae.mean():.3f} (+/- {cv_scores_mae.std() * 2:.3f})')

# Check for overfitting
print('\n--- Overfitting Analysis ---')
test_r2 = r2_score(y_test, y_pred)
cv_r2_mean = cv_scores_r2.mean()
overfitting_diff = test_r2 - cv_r2_mean

print(f'Test Set R²: {test_r2:.3f}')
print(f'CV Mean R²: {cv_r2_mean:.3f}')
print(f'Difference (Test - CV): {overfitting_diff:.3f}')

if overfitting_diff > 0.05:
    print('⚠️  Potential overfitting detected (difference > 0.05)')
elif overfitting_diff > 0.02:
    print('⚠️  Minor overfitting possible (difference > 0.02)')
else:
    print('✅ No significant overfitting detected')

# Feature importances
importances = pd.Series(rf_regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
print('\n--- Top 10 Feature Importances ---')
print(importances.head(10))

# Visualize top 10 feature importances
plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='bar')
plt.title('Top 10 Feature Importances (Random Forest Regression)')
plt.ylabel('Importance')
plt.tight_layout()
# plt.savefig('feature_importances_regression.png')
plt.close()
print("\nFeature importances plot saved as 'feature_importances_regression.png'")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Philanthropy Score')
plt.ylabel('Predicted Philanthropy Score')
plt.title('Actual vs Predicted Philanthropy Scores')
plt.tight_layout()
# plt.savefig('actual_vs_predicted_regression.png')
plt.close()
print("Actual vs predicted plot saved as 'actual_vs_predicted_regression.png'")

# Save the model and feature columns
model_bundle = {
    'model': rf_regressor,
    'feature_columns': X.columns.tolist(),
    'model_type': 'regression'
}

with open('philanthropy_model_regression.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)
print("Regression model saved as 'philanthropy_model_regression.pkl'")

# Add predictions to dataframe for analysis
df['Philanthropy_Prediction_Regression'] = rf_regressor.predict(X)
print("\nRegression predictions added to dataframe") 