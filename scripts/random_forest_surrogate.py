# random_forest_surrogate.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the training data
training_data = pd.read_csv('../data/real_forced_training_data.csv')

# Define input features and output variables
features = ['sst', 'day_of_year', 'latitude', 'longitude', 'N_0', 'time']
outputs = ['N', 'P', 'Z', 'D', 'chlorophyll']

# Split data into features (X) and outputs (y)
X = training_data[features]
y = training_data[outputs]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Calculate performance metrics for each output variable
metrics = {}
for i, output in enumerate(outputs):
    r2 = r2_score(y_test[output], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[output], y_pred[:, i]))
    metrics[output] = {'R²': r2, 'RMSE': rmse}
    print(f"{output} - R²: {r2:.3f}, RMSE: {rmse:.3f}")

# Plot feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=features)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title('Feature Importance in Random Forest Model')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# Save predictions for visualization
test_results = pd.DataFrame({
    'N_true': y_test['N'], 'N_pred': y_pred[:, 0],
    'P_true': y_test['P'], 'P_pred': y_pred[:, 1],
    'Z_true': y_test['Z'], 'Z_pred': y_pred[:, 2],
    'D_true': y_test['D'], 'D_pred': y_pred[:, 3],
    'chlorophyll_true': y_test['chlorophyll'], 'chlorophyll_pred': y_pred[:, 4]
})
test_results.to_csv('../data/rf_predictions.csv', index=False)
print("Predictions saved to '../data/rf_predictions.csv'")