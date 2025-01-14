import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
from typing import Sequence
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass


@dataclass(frozen=True)
class NeuralNetworkConfig:
    """
    Configuration for standard MLP neural network.
    """
    input_size: int
    hidden_sizes: tuple
    output_shape: tuple
    activation: callable
    learning_rate: float
    momentum: float


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron model.
    """
    hidden_sizes: Sequence[int]
    output_shape: tuple
    activation: callable = nn.relu
    training: bool = True

    def setup(self):
        """
        Initialize by obtaining the output size and defining layers.
        """
        self.output_size = 1
        for dim in self.output_shape:
            self.output_size *= dim

        # Create hidden layers
        self.hidden_layers = [nn.Dense(num_neurons) for num_neurons in self.hidden_sizes]

        # Create output layer
        self.output_layer = nn.Dense(self.output_size)

    def __call__(self, input):
        """
        Forward pass through the MLP.
        """
        output = input
        for hidden_layer in self.hidden_layers:
            output = hidden_layer(output)
            output = self.activation(output)
        output = self.output_layer(output)
        return output.reshape(-1, *self.output_shape).squeeze()


class DynamicsDataset(Dataset):
    """
    PyTorch dataset for dynamics data.
    """
    def __init__(self, inputs, outputs):
        """
        Initialize the dataset, inputs are states and controls concatenated.
        """
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        """
        Return the number of data points.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Return a single data point.
        """
        return self.inputs[idx], self.outputs[idx]


def custom_collate_fn(batch, device):
    """
    Converts batches of PyTorch tensors to JAX numpy arrays.
    """
    transposed_data = list(zip(*batch))
    inputs_np = jnp.array(transposed_data[0])
    outputs_np = jnp.array(transposed_data[1])

    # If a device is specified, transfer the data to that device
    if device is not None:
        inputs_np = jax.device_put(inputs_np, device)
        outputs_np = jax.device_put(outputs_np, device)

    return inputs_np, outputs_np


def create_data_loaders(xs_flat, us_flat, delta_x_dots_flat, batch_size, data_split, device=None):
    """
    Create PyTorch data loaders for training, validation, and testing.
    """
    if not isinstance(xs_flat, np.ndarray):
        xs_flat = np.array(xs_flat)
        us_flat = np.array(us_flat)
        delta_x_dots_flat = np.array(delta_x_dots_flat)
    nn_inputs = np.concatenate([xs_flat, us_flat], axis=1)
    nn_outputs = delta_x_dots_flat
    dataset = DynamicsDataset(nn_inputs, nn_outputs)
    N_data = len(dataset)
    N_train = int(N_data * data_split[0])
    N_val = int(N_data * data_split[1])
    N_test = N_data - N_train - N_val
    device_collate_fn = partial(custom_collate_fn, device=device)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [N_train, N_val, N_test])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=device_collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=device_collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=device_collate_fn, shuffle=False)
    return train_loader, val_loader, test_loader


def calculate_loss(params, inputs, gt_outputs, config):
    """
    Calculate the loss for a single batch.
    """
    model = MLP(config.hidden_sizes, config.output_shape, config.activation)
    if len(config.output_shape) == 1:
        predictions = model.apply({'params': params}, inputs)
    else:
        x, u = jnp.split(inputs, [config.output_shape[0]], axis=1)
        output = model.apply({'params': params}, x)
        predictions = jax.vmap(jnp.matmul, in_axes=(0, 0))(output, u)
    loss = jnp.mean(optax.l2_loss(predictions, gt_outputs))
    return loss


def train_step(state, inputs, gt_outputs, config):
    """Train for a single batch."""
    loss, grads = jax.value_and_grad(calculate_loss)(state.params, inputs, gt_outputs, config)
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval_step(state, inputs, gt_outputs, config):
    """Calculate the loss for a single batch."""
    loss = calculate_loss(state.params, inputs, gt_outputs, config)
    return loss


def train_one_epoch(train_step_device, state, data_loader, config, device):
    """Train for 1 epoch on the training set."""
    batch_losses = []
    for inputs, gt_outputs in data_loader:
        state, loss = train_step_device(state, inputs, gt_outputs, config)
        batch_losses.append(loss)
    epoch_loss = np.mean(batch_losses)
    return state, epoch_loss


def evaluate_model(eval_step_device, state, data_loader, config):
    """Evaluate on a held out set."""
    batch_losses = []
    for inputs, gt_outputs in data_loader:
        loss = eval_step_device(state, inputs, gt_outputs, config)
        batch_losses.append(loss)
    return np.mean(batch_losses)


def create_train_state(key, config, device=None):
    """Create the initial TrainState on specified device."""
    model = MLP(config.hidden_sizes, config.output_shape, config.activation)
    params = model.init(key, jnp.ones([1, config.input_size]))['params']
    if device:
        params = jax.tree_map(lambda x: jax.device_put(x, device), params)
    adam_opt = optax.adam(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=adam_opt)


def train_model(state, config, train_loader, val_loader, num_epochs, device=None):
    """Train model on training data and log performance on validation data."""
    train_step_jit = jax.jit(train_step, device=device, static_argnums=-1)
    eval_step_jit = jax.jit(eval_step, device=device, static_argnums=-1)

    train_losses, val_losses = [], []
    init_train_loss = evaluate_model(eval_step_jit, state, train_loader, config)
    train_losses.append(init_train_loss)
    init_val_loss = evaluate_model(eval_step_jit, state, val_loader, config)
    val_losses.append(init_val_loss)
    print(f'Initial train loss: {init_train_loss:.6f}, Initial val loss: {init_val_loss:.6f}')
    
    for epoch in range(1, num_epochs+1):
        state, train_loss = train_one_epoch(train_step_jit, state, train_loader, config, device)
        train_losses.append(train_loss)
        val_loss = evaluate_model(eval_step_jit, state, val_loader, config)
        val_losses.append(val_loss)
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}')
    print(f'Final train loss: {train_loss:.6f}, Final val loss: {val_loss:.6f}')
    return state, train_losses, val_losses
