import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd


# Custom dataset
class UYPairsDataset(Dataset):
    def __init__(self, us, ys):
        self.us = us
        self.ys = ys

    def __len__(self):
        return len(self.us)

    def __getitem__(self, idx):
        u = self.us[idx]
        y = self.ys[idx]
        return torch.tensor(y, dtype=torch.float32), torch.tensor(u, dtype=torch.float32)


# Load data
us_df = pd.read_csv('../data/trajectories/steady_state/control_inputs_uniform.csv')
ys_df = pd.read_csv('../data/trajectories/steady_state/observations_steady_state_src_demo_17oct24.csv')
rest_positions = np.array([0.1005, -0.10698, 0.10445, -0.10302, -0.20407, 0.10933, 0.10581, -0.32308, 0.10566])
ys_df = ys_df - rest_positions
N = len(ys_df)
if len(us_df) != N:
    us_df = us_df[:N]

# Convert to numpy arrays
us = us_df.to_numpy()[:, 1:]
ys = ys_df.to_numpy()

# Randomly split dataset into train and test
dataset = UYPairsDataset(us, ys)
train_size = int(0.85 * N)
test_size = N - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Neural Network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, y):
        x = torch.relu(self.fc1(y))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model, loss function, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0025, weight_decay=1e-6)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for ys_batch, us_batch in train_loader:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(ys_batch)
        loss = criterion(outputs, us_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], RMSE: {np.sqrt(running_loss/len(train_loader)):.4f}')

# Evaluation on test dataset
model.eval()
test_loss = 0.0
with torch.no_grad():
    for ys_batch, us_batch in test_loader:
        outputs = model(ys_batch)
        loss = criterion(outputs, us_batch)
        test_loss += loss.item()

print(f'Test RMSE: {np.sqrt(test_loss)/len(test_loader):.4f}')

# Evaluate random model performance
u_min, u_max = -0.35, 0.35
test_loss_random = 0.0
mse_criterion = nn.MSELoss()
model.eval()
with torch.no_grad():
    for ys_batch, us_batch in test_loader:
        random_preds = torch.FloatTensor(us_batch.size()).uniform_(u_min, u_max)
        mse_loss = mse_criterion(random_preds, us_batch)
        test_loss_random += mse_loss.item()

print(f'Random Model Test RMSE: {np.sqrt(test_loss_random/len(test_loader)):.4f}')

# Simple linear IK model
us_train = us[:train_size]
us_test = us[train_size:]
ys_train = ys[:train_size]
ys_test = ys[train_size:]
G = np.linalg.lstsq(ys_train, us_train, rcond=None)[0].T
us_pred = ys_test @ G.T
rmse_lsq = np.sqrt(np.mean(np.square(us_test - us_pred)))
print(f"Test RMSE of least-squares model: {rmse_lsq:.4f}")
