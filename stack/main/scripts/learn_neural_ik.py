import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
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
num_seeds = 7

us_df = pd.read_csv('../data/trajectories/steady_state/control_inputs_beta_seed0.csv')
for seed in range(1, num_seeds + 1):
    us_df = pd.concat([us_df, pd.read_csv(f'../data/trajectories/steady_state/control_inputs_beta_seed{seed}.csv')])
us_df = us_df.drop(columns=['ID'])

# Observations
ys_df = pd.read_csv('../data/trajectories/steady_state/observations_steady_state_beta_seed0.csv')
for seed in range(1, num_seeds + 1):
    ys_df = pd.concat([ys_df, pd.read_csv(f'../data/trajectories/steady_state/observations_steady_state_beta_seed{seed}.csv')])
ys_df = ys_df.drop(columns=['ID'])

rest_positions = np.array([0.1005, -0.10698, 0.10445, -0.10302, -0.20407, 0.10933, 0.10581, -0.32308, 0.10566])
ys_df = ys_df - rest_positions
N = len(ys_df)
if len(us_df) != N:
    us_df = us_df[:N]

# Convert to numpy arrays
us = us_df.to_numpy()
ys = ys_df.to_numpy()

# Randomly split dataset into train and test
dataset = UYPairsDataset(us, ys)
train_size = int(0.8 * N)
val_size = int(0.1 * N)
test_size = N - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
# Spectrally-normalized Relu MLP
class MLP(nn.Module):
    def __init__(self,
                 num_inputs: int = 9,
                 num_outputs: int = 6,
                 num_neurons: list = [32, 32],
                 act: nn.Module = nn.ReLU(),
                 spectral_normalize: bool = False):
        super(MLP, self).__init__()

        layers = []
        
        # Input layer
        if spectral_normalize:
            input_layer = spectral_norm(nn.Linear(num_inputs, num_neurons[0]))
        else:
            input_layer = nn.Linear(num_inputs, num_neurons[0])
        layers.append(input_layer)
        layers.append(act)
        
        # Hidden layers
        for i in range(len(num_neurons) - 1):
            if spectral_normalize:
                hidden_layer = spectral_norm(nn.Linear(num_neurons[i], num_neurons[i + 1]))
            else:
                hidden_layer = nn.Linear(num_neurons[i], num_neurons[i + 1])
            layers.append(hidden_layer)
            layers.append(act)
        
        # Output layer
        if spectral_normalize:
            output_layer = spectral_norm(nn.Linear(num_neurons[-1], num_outputs))
        else:
            output_layer = nn.Linear(num_neurons[-1], num_outputs)
        
        layers.append(output_layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, y):
        x = y
        for layer in self.layers:
            x = layer(x)
        u = x
        return u



# Instantiate the model, loss function, and optimizer
inverse_kinematic_model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(inverse_kinematic_model.parameters(), lr=0.0025, weight_decay=1e-6)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    inverse_kinematic_model.train()
    running_loss = 0.0
    for ys_batch, us_batch in train_loader:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = inverse_kinematic_model(ys_batch)
        loss = criterion(outputs, us_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], RMSE: {np.sqrt(running_loss/len(train_loader)):.4f}')

# Evaluation on test dataset
inverse_kinematic_model.eval()
test_loss = 0.0
with torch.no_grad():
    for ys_batch, us_batch in test_loader:
        outputs = inverse_kinematic_model(ys_batch)
        loss = criterion(outputs, us_batch)
        test_loss += loss.item()

print(f'Test RMSE: {np.sqrt(test_loss)/len(test_loader):.4f}')

# Evaluate random model performance
u_min, u_max = -0.35, 0.35
test_loss_random = 0.0
mse_criterion = nn.MSELoss()
inverse_kinematic_model.eval()
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
