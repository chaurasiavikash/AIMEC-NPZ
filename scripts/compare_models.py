import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load training data and predictions
data = pd.read_csv('../data/real_forced_training_data.csv')
rf_preds = pd.read_csv('../data/rf_predictions.csv')
nn_preds = pd.read_csv('../data/nn_predictions_pytorch.csv')

# Define features and targets
features = ['sst', 'day_of_year', 'latitude', 'longitude', 'N_0', 'time']
targets = ['N', 'P', 'Z', 'D', 'chlorophyll']

# Prepare original data and split (replicate the original split with random_state=42)
X = data[features].values
y = data[targets].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure nn_preds indices match the test set (assuming rows correspond to X_test order)
nn_test_data = pd.DataFrame(X_test, columns=features)
nn_full = pd.concat([nn_test_data, nn_preds], axis=1)

# Align RF predictions (assuming rf_preds has _true and _pred columns)
rf_full = rf_preds

# Calculate metrics for both models
rf_metrics = {}
nn_metrics = {}

for output in targets:
    # RF metrics
    rf_r2 = r2_score(rf_full[f'{output}_true'], rf_full[f'{output}_pred'])
    rf_rmse = np.sqrt(mean_squared_error(rf_full[f'{output}_true'], rf_full[f'{output}_pred']))
    rf_metrics[output] = {'R²': rf_r2, 'RMSE': rf_rmse}

    # NN metrics (use y_test as true values, aligned by index)
    nn_r2 = r2_score(y_test[:, targets.index(output)], nn_preds[f'{output}_pred'].values)
    nn_rmse = np.sqrt(mean_squared_error(y_test[:, targets.index(output)], nn_preds[f'{output}_pred'].values))
    nn_metrics[output] = {'R²': nn_r2, 'RMSE': nn_rmse}

# Print comparison
print("Model Performance Comparison:")
print("---------------------------")
for output in targets:
    print(f"{output}:")
    print(f"  Random Forest - R²: {rf_metrics[output]['R²']:.3f}, RMSE: {rf_metrics[output]['RMSE']:.3f}")
    print(f"  Neural Network - R²: {nn_metrics[output]['R²']:.3f}, RMSE: {nn_metrics[output]['RMSE']:.3f}")
    print()

# Visualize comparison with scatter plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot first 5 variables
for i, output in enumerate(targets):
    ax = axes[i]
    ax.scatter(rf_full[f'{output}_true'], rf_full[f'{output}_pred'], alpha=0.5, color='blue', label='RF')
    ax.scatter(y_test[:, targets.index(output)], nn_preds[f'{output}_pred'], alpha=0.5, color='orange', label='NN')
    min_val = min(rf_full[f'{output}_true'].min(), y_test[:, targets.index(output)].min())
    max_val = max(rf_full[f'{output}_true'].max(), y_test[:, targets.index(output)].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
    ax.set_xlabel(f'True {output}')
    ax.set_ylabel(f'Predicted {output}')
    ax.set_title(f'{output} (RF R²: {rf_metrics[output]["R²"]:.3f}, NN R²: {nn_metrics[output]["R²"]:.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Turn off the empty sixth subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('figures/model_comparison_scatter_plots.png', dpi=150)
plt.show()

print("Comparison plots saved to 'figures/model_comparison_scatter_plots.png'")