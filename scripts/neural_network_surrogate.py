import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error

# Define the neural network model
class NPZNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NPZNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# Load training data
data = pd.read_csv('../data/real_forced_training_data.csv')
features = ['sst', 'day_of_year', 'latitude', 'longitude', 'N_0', 'time']
targets = ['N', 'P', 'Z', 'D', 'chlorophyll']

# Prepare data
X = data[features].values
y = data[targets].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features and targets
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

# Initialize model, loss, and optimizer
input_size = len(features)
hidden_size = 64
output_size = len(targets)
model = NPZNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_X = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_test_orig = scaler_y.inverse_transform(y_test_tensor.numpy())

    # Calculate metrics
    metrics = {}
    for i, target in enumerate(targets):
        r2 = r2_score(y_test_orig[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_orig[:, i], y_pred[:, i]))
        metrics[target] = {'R²': r2, 'RMSE': rmse}
        print(f"{target} - R²: {r2:.3f}, RMSE: {rmse:.3f}")

    # Save predictions
    predictions = pd.DataFrame(y_pred, columns=[f'{t}_pred' for t in targets])
    test_data = pd.DataFrame(X_test, columns=features)
    results = pd.concat([test_data, predictions], axis=1)
    results.to_csv('../data/nn_predictions_pytorch.csv', index=False)

print("Predictions saved to '../data/nn_predictions_pytorch.csv'")