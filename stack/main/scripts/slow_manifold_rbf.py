import torch
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
import pandas as pd
import numpy as np
# ============================================================================
# PART 1: FIT WITH PYTORCH (Memory-efficient)
# ============================================================================

SCALING = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)

def fit_linear_rbf_pytorch(U_data, Y_data, scaling, epsilon=1.0, device='cuda', 
                           chunk_size=1000, reg=1e-8):
    """
    Fit a linear RBF using PyTorch with chunked computation for memory efficiency.
    
    Args:
        U_data: (N, 6) - numpy or torch tensor of input data
        Y_data: (N, 6) - numpy or torch tensor of output data
        scaling: (6,) - scaling vector to apply to each input dimension
        epsilon: scalar - shape parameter (not used for linear RBF)
        device: 'cuda' or 'cpu'
        chunk_size: process this many rows at a time for distance matrix
        reg: regularization term for numerical stability
    
    Returns:
        U_centers: (6, N) - UNSCALED centers as numpy array (ready for JAX)
        W: (N, 6) - weights as numpy array (ready for JAX)
        epsilon: scalar
        scaling: (6,) - scaling vector (for use in evaluation)
    """
    # Convert to torch if needed
    if isinstance(U_data, np.ndarray):
        U_data = torch.from_numpy(U_data).float()
        Y_data = torch.from_numpy(Y_data).float()
    
    if isinstance(scaling, np.ndarray):
        scaling_torch = torch.from_numpy(scaling).float()
    else:
        scaling_torch = scaling.float()
    
    U_data = U_data.to(device)
    Y_data = Y_data.to(device)
    scaling_torch = scaling_torch.to(device)
    
    N = U_data.shape[0]
    print(f"Fitting RBF with {N} centers using PyTorch...")
    print(f"Device: {device}")
    print(f"Scaling vector: {scaling}")
    
    # Apply scaling: U_scaled = U / scaling (only for distance computation)
    U_scaled = U_data / scaling_torch[None, :]  # (N, 6) / (1, 6)
    
    # Build distance matrix in chunks to save memory
    print("Computing distance matrix (chunked) on SCALED inputs...")
    
    # Preallocate distance matrix (still large, but we build it chunk by chunk)
    A = torch.zeros((N, N), device=device, dtype=torch.float32)
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        chunk_i = U_scaled[i:end_i]  # (chunk_size, 6)
        
        for j in range(0, N, chunk_size):
            end_j = min(j + chunk_size, N)
            chunk_j = U_scaled[j:end_j]  # (chunk_size, 6)
            
            # Compute pairwise distances for this block
            # (chunk_size_i, 6) vs (chunk_size_j, 6)
            diff = chunk_i[:, None, :] - chunk_j[None, :, :]  # (ci, cj, 6)
            dist = torch.sqrt(torch.sum(diff**2, dim=2))  # (ci, cj)
            
            A[i:end_i, j:end_j] = dist
        
        if (i // chunk_size) % 10 == 0:
            print(f"  Progress: {i}/{N} rows computed")
    
    print("Distance matrix complete!")
    
    # Add regularization for numerical stability
    A = A + reg * torch.eye(N, device=device)
    
    # Solve linear system: A @ W = Y
    print("Solving linear system...")
    W = torch.linalg.solve(A, Y_data)  # (N, 6)
    
    print("Converting to numpy for JAX...")
    # Convert to numpy for JAX - store ORIGINAL UNSCALED centers
    U_centers_np = U_data.T.cpu().numpy()  # (6, N) - UNSCALED
    W_np = W.cpu().numpy()  # (N, 6)
    scaling_np = scaling_torch.cpu().numpy()  # (6,)
    
    print("RBF fitting complete!")
    
    # Clear GPU memory
    del A, U_data, Y_data, W
    torch.cuda.empty_cache()
    
    return U_centers_np, W_np, epsilon


def fit_linear_rbf_pytorch_fast(U_data, Y_data, scaling, epsilon=1.0, device='cuda', reg=1e-8):
    """
    Fast version - computes entire distance matrix at once.
    Only use if you have enough GPU memory (~4GB for 100k points in float32).
    
    Args:
        U_data: (N, 6) - input data
        Y_data: (N, 6) - output data
        scaling: (6,) - scaling vector to apply to each input dimension
        device: 'cuda' or 'cpu'
        reg: regularization term
    
    Returns:
        U_centers, W, epsilon, scaling (all as numpy arrays ready for JAX)
    """
    if isinstance(U_data, np.ndarray):
        U_data = torch.from_numpy(U_data).float()
        Y_data = torch.from_numpy(Y_data).float()
    
    if isinstance(scaling, np.ndarray):
        scaling_torch = torch.from_numpy(scaling).float()
    else:
        scaling_torch = scaling.float()
    
    U_data = U_data.to(device)
    Y_data = Y_data.to(device)
    scaling_torch = scaling_torch.to(device)
    
    N = U_data.shape[0]
    print(f"Fitting RBF with {N} centers (fast mode)...")
    print(f"Scaling vector: {scaling}")
    
    # Apply scaling: U_scaled = U / scaling (only for distance computation)
    U_scaled = U_data / scaling_torch[None, :]  # (N, 6) / (1, 6)
    
    # Compute entire distance matrix at once on SCALED inputs
    print("Computing distance matrix on SCALED inputs...")
    diff = U_scaled[:, None, :] - U_scaled[None, :, :]  # (N, N, 6)
    A = torch.sqrt(torch.sum(diff**2, dim=2))  # (N, N)
    
    # Add regularization
    A = A + reg * torch.eye(N, device=device)
    
    # Solve
    print("Solving linear system...")
    W = torch.linalg.solve(A, Y_data)
    
    # Convert to numpy - store ORIGINAL UNSCALED centers
    U_centers_np = U_data.T.cpu().numpy()  # UNSCALED
    W_np = W.cpu().numpy()
    scaling_np = scaling_torch.cpu().numpy()
    
    # Cleanup
    del A, U_data, U_scaled, Y_data, W
    torch.cuda.empty_cache()
    
    return U_centers_np, W_np, epsilon, scaling_np


# ============================================================================
# PART 2: EVALUATE WITH JAX (Fast inference)
# ============================================================================

@jax.jit
def rbf_eval_batch_jax(u_batch, U_centers, W, epsilon, scaling):
    """
    Batched JAX-compatible linear RBF evaluation with input scaling.
    
    Args:
        u_batch: (N, 6) - batch of evaluation points (UNSCALED)
        U_centers: (6, M) - each column is a center (UNSCALED) (M = 100,000)
        W: (M, 6) - weights
        epsilon: scalar - shape parameter (not used for linear RBF)
        scaling: (6,) - scaling vector
    
    Returns:
        result: (N, 6) - batch of RBF evaluations
    """
    if u_batch.ndim == 1:
        u_batch = u_batch[None, :]
    
    # Scale the input and centers internally: scaled = unscaled / scaling
    # u_batch: (N, 6), scaling: (6,) -> broadcast to (N, 6) / (1, 6)
    u_scaled = u_batch / scaling[None, :]  # (N, 6)
    
    # Scale the centers: U_centers: (6, M), scaling: (6,) -> (6, M) / (6, 1)
    centers_scaled = U_centers / scaling[:, None]  # (6, M)
    
    # Compute distances using SCALED inputs
    u_expanded = u_scaled[:, :, None]  # (N, 6, 1)
    centers_expanded = centers_scaled[None, :, :]  # (1, 6, M)
    
    diff = u_expanded - centers_expanded  # (N, 6, M)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=1))  # (N, M)
    
    # Linear RBF evaluation
    result = dist @ W  # (N, M) @ (M, 6) = (N, 6)
    
    return result


@partial(jax.jit, static_argnums=(4,))
def rbf_eval_chunked_jax(u_batch, U_centers, W, epsilon, scaling, max_centers_per_chunk=10000):
    """
    Chunked evaluation for memory efficiency during inference with input scaling.
    Splits the centers into chunks to reduce memory usage.
    
    Args:
        u_batch: (N, 6) - evaluation points (UNSCALED)
        U_centers: (6, M) - UNSCALED centers
        W: (M, 6) - weights
        epsilon: scalar
        scaling: (6,) - scaling vector
        max_centers_per_chunk: process this many centers at a time
    
    Returns:
        result: (N, 6)
    """
    M = U_centers.shape[1]
    n_chunks = (M + max_centers_per_chunk - 1) // max_centers_per_chunk
    
    if u_batch.ndim == 1:
        u_batch = u_batch[None, :]
    
    # Scale the input
    u_scaled = u_batch / scaling[None, :]  # (N, 6)
    
    # Scale the centers
    centers_scaled = U_centers / scaling[:, None]  # (6, M)
    
    N = u_scaled.shape[0]
    result = jnp.zeros((N, 6))
    
    for i in range(n_chunks):
        start = i * max_centers_per_chunk
        end = min(start + max_centers_per_chunk, M)
        
        # Chunk of scaled centers and weights
        U_chunk = centers_scaled[:, start:end]  # (6, chunk_size)
        W_chunk = W[start:end, :]  # (chunk_size, 6)
        
        # Compute distances for this chunk using SCALED inputs
        u_expanded = u_scaled[:, :, None]  # (N, 6, 1)
        centers_expanded = U_chunk[None, :, :]  # (1, 6, chunk_size)
        
        diff = u_expanded - centers_expanded  # (N, 6, chunk_size)
        dist = jnp.sqrt(jnp.sum(diff**2, axis=1))  # (N, chunk_size)
        
        # Accumulate result
        result += dist @ W_chunk  # (N, 6)
    
    return result


# SCALING = np.array([30, 90, 50, 90, 50, 30], dtype=np.float32)

# def fit_linear_rbf_pytorch(U_data, Y_data, epsilon=1.0, device='cuda', 
#                            chunk_size=1000, reg=1e-8):
#     """
#     Fit a linear RBF using PyTorch with chunked computation for memory efficiency.
    
#     Args:
#         U_data: (N, 6) - numpy or torch tensor of input data
#         Y_data: (N, 6) - numpy or torch tensor of output data
#         epsilon: scalar - shape parameter (not used for linear RBF)
#         device: 'cuda' or 'cpu'
#         chunk_size: process this many rows at a time for distance matrix
#         reg: regularization term for numerical stability
    
#     Returns:
#         U_centers: (6, N) - centers as numpy array (ready for JAX)
#         W: (N, 6) - weights as numpy array (ready for JAX)
#         epsilon: scalar
#     """
#     # Convert to torch if needed
#     if isinstance(U_data, np.ndarray):
#         U_data = torch.from_numpy(U_data).float()
#         Y_data = torch.from_numpy(Y_data).float()
    
#     U_data = U_data.to(device)
#     Y_data = Y_data.to(device)
    
#     N = U_data.shape[0]
#     print(f"Fitting RBF with {N} centers using PyTorch...")
#     print(f"Device: {device}")
    
#     # Build distance matrix in chunks to save memory
#     print("Computing distance matrix (chunked)...")
    
#     # Preallocate distance matrix (still large, but we build it chunk by chunk)
#     A = torch.zeros((N, N), device=device, dtype=torch.float32)
    
#     for i in range(0, N, chunk_size):
#         end_i = min(i + chunk_size, N)
#         chunk_i = U_data[i:end_i]  # (chunk_size, 6)
        
#         for j in range(0, N, chunk_size):
#             end_j = min(j + chunk_size, N)
#             chunk_j = U_data[j:end_j]  # (chunk_size, 6)
            
#             # Compute pairwise distances for this block
#             # (chunk_size_i, 6) vs (chunk_size_j, 6)
#             diff = chunk_i[:, None, :] - chunk_j[None, :, :]  # (ci, cj, 6)
#             dist = torch.sqrt(torch.sum(diff**2, dim=2))  # (ci, cj)
            
#             A[i:end_i, j:end_j] = dist
        
#         if (i // chunk_size) % 10 == 0:
#             print(f"  Progress: {i}/{N} rows computed")
    
#     print("Distance matrix complete!")
    
#     # Add regularization for numerical stability
#     A = A + reg * torch.eye(N, device=device)
    
#     # Solve linear system: A @ W = Y
#     print("Solving linear system...")
#     W = torch.linalg.solve(A, Y_data)  # (N, 6)
    
#     print("Converting to numpy for JAX...")
#     # Convert to numpy for JAX
#     U_centers_np = U_data.T.cpu().numpy()  # (6, N)
#     W_np = W.cpu().numpy()  # (N, 6)
    
#     print("RBF fitting complete!")
    
#     # Clear GPU memory
#     del A, U_data, Y_data, W
#     torch.cuda.empty_cache()
    
#     return U_centers_np, W_np, epsilon


# def fit_linear_rbf_pytorch_fast(U_data, Y_data, epsilon=1.0, device='cuda', reg=1e-8):
#     """
#     Fast version - computes entire distance matrix at once.
#     Only use if you have enough GPU memory (~4GB for 100k points in float32).
    
#     Args:
#         U_data: (N, 6) - input data
#         Y_data: (N, 6) - output data
#         device: 'cuda' or 'cpu'
#         reg: regularization term
    
#     Returns:
#         U_centers, W, epsilon (all as numpy arrays ready for JAX)
#     """
#     if isinstance(U_data, np.ndarray):
#         U_data = torch.from_numpy(U_data).float()
#         Y_data = torch.from_numpy(Y_data).float()
    
#     U_data = U_data.to(device)
#     Y_data = Y_data.to(device)
    
#     N = U_data.shape[0]
#     print(f"Fitting RBF with {N} centers (fast mode)...")
    
#     # Compute entire distance matrix at once
#     print("Computing distance matrix...")
#     diff = U_data[:, None, :] - U_data[None, :, :]  # (N, N, 6)
#     A = torch.sqrt(torch.sum(diff**2, dim=2))  # (N, N)
    
#     # Add regularization
#     A = A + reg * torch.eye(N, device=device)
    
#     # Solve
#     print("Solving linear system...")
#     W = torch.linalg.solve(A, Y_data)
    
#     # Convert to numpy
#     U_centers_np = U_data.T.cpu().numpy()
#     W_np = W.cpu().numpy()
    
#     # Cleanup
#     del A, U_data, Y_data, W
#     torch.cuda.empty_cache()
    
#     return U_centers_np, W_np, epsilon


# # ============================================================================
# # PART 2: EVALUATE WITH JAX (Fast inference)
# # ============================================================================

# @jax.jit
# def rbf_eval_batch_jax(u_batch, U_centers, W, epsilon):
#     """
#     Batched JAX-compatible linear RBF evaluation.
    
#     Args:
#         u_batch: (N, 6) - batch of evaluation points
#         U_centers: (6, M) - each column is a center (M = 100,000)
#         W: (M, 6) - weights
#         epsilon: scalar - shape parameter (not used for linear RBF)
    
#     Returns:
#         result: (N, 6) - batch of RBF evaluations
#     """
#     if u_batch.ndim == 1:
#         u_batch = u_batch[None, :]
    
#     # Compute distances
#     u_expanded = u_batch[:, :, None]  # (N, 6, 1)
#     centers_expanded = U_centers[None, :, :]  # (1, 6, M)
    
#     diff = u_expanded - centers_expanded  # (N, 6, M)
#     dist = jnp.sqrt(jnp.sum(diff**2, axis=1))  # (N, M)
    
#     # Linear RBF evaluation
#     result = dist @ W  # (N, M) @ (M, 6) = (N, 6)
    
#     return result


# @partial(jax.jit, static_argnums=(3,))
# def rbf_eval_chunked_jax(u_batch, U_centers, W, epsilon, max_centers_per_chunk=10000):
#     """
#     Chunked evaluation for memory efficiency during inference.
#     Splits the centers into chunks to reduce memory usage.
    
#     Args:
#         u_batch: (N, 6) - evaluation points
#         U_centers: (6, M) - centers
#         W: (M, 6) - weights
#         epsilon: scalar
#         max_centers_per_chunk: process this many centers at a time
    
#     Returns:
#         result: (N, 6)
#     """
#     M = U_centers.shape[1]
#     n_chunks = (M + max_centers_per_chunk - 1) // max_centers_per_chunk
    
#     if u_batch.ndim == 1:
#         u_batch = u_batch[None, :]
    
#     N = u_batch.shape[0]
#     result = jnp.zeros((N, 6))
    
#     for i in range(n_chunks):
#         start = i * max_centers_per_chunk
#         end = min(start + max_centers_per_chunk, M)
        
#         # Chunk of centers and weights
#         U_chunk = U_centers[:, start:end]  # (6, chunk_size)
#         W_chunk = W[start:end, :]  # (chunk_size, 6)
        
#         # Compute distances for this chunk
#         u_expanded = u_batch[:, :, None]  # (N, 6, 1)
#         centers_expanded = U_chunk[None, :, :]  # (1, 6, chunk_size)
        
#         diff = u_expanded - centers_expanded  # (N, 6, chunk_size)
#         dist = jnp.sqrt(jnp.sum(diff**2, axis=1))  # (N, chunk_size)
        
#         # Accumulate result
#         result += dist @ W_chunk  # (N, 6)
    
#     return result


# ============================================================================
# PART 3: USAGE EXAMPLE
# ============================================================================


# ============================================================================
# Load Robot Data from CSV files
# ============================================================================

def load_robot_data(control_path, observation_path):
    """
    Load robot control inputs and observations from CSV files.
    
    Args:
        control_path: path to control inputs CSV
        observation_path: path to observations CSV
    
    Returns:
        U_data: (N, 6) numpy array of control inputs [u1, u2, u3, u4, u5, u6]
        Y_data: (N, 6) numpy array of positions [x3, y3, z3, x2, y2, z2]
    """
    print("Loading control inputs...")
    control_df = pd.read_csv(control_path)
    
    print("Loading observations...")
    obs_df = pd.read_csv(observation_path)
    
    # Extract control inputs (u1-u6)
    U_data = control_df[['u1', 'u2', 'u3', 'u4', 'u5', 'u6']].values
    
    # Extract positions in the order: x3, y3, z3, x2, y2, z2
    Y_data = obs_df[['x3', 'y3', 'z3', 'x2', 'y2', 'z2']].values
    
    # Verify shapes match
    # assert U_data.shape[0] == Y_data.shape[0], \
    #     f"Mismatch: {U_data.shape[0]} control points vs {Y_data.shape[0]} observation points"
    
    N = U_data.shape[0]
    print(f"Loaded {N} data points")
    print(f"Control inputs shape: {U_data.shape}")
    print(f"Observations shape: {Y_data.shape}")
    
    # Convert to float32 for memory efficiency
    U_data = U_data.astype(np.float32)
    Y_data = Y_data.astype(np.float32)
    
    U_data = U_data[300:100000:5,:]
    Y_data = Y_data[300:100000:5,:]
    # Print some statistics
    print("\nControl inputs (U) statistics:")
    print(f"  Min: {U_data.min(axis=0)}")
    print(f"  Max: {U_data.max(axis=0)}")
    print(f"  Mean: {U_data.mean(axis=0)}")
    
    print("\nObservations (Y) statistics:")
    print(f"  Min: {Y_data.min(axis=0)}")
    print(f"  Max: {Y_data.max(axis=0)}")
    print(f"  Mean: {Y_data.mean(axis=0)}")
    
    return U_data, Y_data


def load_robot_data_test(control_path, observation_path):
    """
    Load robot control inputs and observations from CSV files.
    
    Args:
        control_path: path to control inputs CSV
        observation_path: path to observations CSV
    
    Returns:
        U_data: (N, 6) numpy array of control inputs [u1, u2, u3, u4, u5, u6]
        Y_data: (N, 6) numpy array of positions [x3, y3, z3, x2, y2, z2]
    """
    print("Loading control inputs...")
    control_df = pd.read_csv(control_path)
    
    print("Loading observations...")
    obs_df = pd.read_csv(observation_path)
    
    # Extract control inputs (u1-u6)
    U_data = control_df[['u1', 'u2', 'u3', 'u4', 'u5', 'u6']].values
    
    # Extract positions in the order: x3, y3, z3, x2, y2, z2
    Y_data = obs_df[['x3', 'y3', 'z3', 'x2', 'y2', 'z2']].values
    
    # Verify shapes match
    # assert U_data.shape[0] == Y_data.shape[0], \
    #     f"Mismatch: {U_data.shape[0]} control points vs {Y_data.shape[0]} observation points"
    
    N = U_data.shape[0]
    print(f"Loaded {N} data points")
    print(f"Control inputs shape: {U_data.shape}")
    print(f"Observations shape: {Y_data.shape}")
    
    # Convert to float32 for memory efficiency
    U_data = U_data.astype(np.float32)
    Y_data = Y_data.astype(np.float32)
    
    U_data = U_data
    Y_data = Y_data[:-1,:]
    # Print some statistics
    print("\nControl inputs (U) statistics:")
    print(f"  Min: {U_data.min(axis=0)}")
    print(f"  Max: {U_data.max(axis=0)}")
    print(f"  Mean: {U_data.mean(axis=0)}")
    
    print("\nObservations (Y) statistics:")
    print(f"  Min: {Y_data.min(axis=0)}")
    print(f"  Max: {Y_data.max(axis=0)}")
    print(f"  Mean: {Y_data.mean(axis=0)}")
    
    return U_data, Y_data

import matplotlib.pyplot as plt

def plot_predictions_vs_true(Y_true, Y_pred, column_idx=1, column_name='y3'):
    """
    Plot predicted vs true values for a specific column.
    
    Args:
        Y_true: (N, 6) true values
        Y_pred: (N, 6) predicted values
        column_idx: which column to plot (1 for y3)
        column_name: name for the plot title
    """
    # Extract the column
    y_true = Y_true[:, column_idx]
    y_pred = np.array(Y_pred)[:, column_idx]
    
    # Calculate metrics
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(np.abs(errors))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Scatter plot - Predicted vs True
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax1.set_xlabel(f'True {column_name}', fontsize=12)
    ax1.set_ylabel(f'Predicted {column_name}', fontsize=12)
    ax1.set_title(f'Predicted vs True: {column_name}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Error distribution
    ax2 = axes[1]
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax2.set_xlabel(f'Prediction Error ({column_name})', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Error Distribution: {column_name}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error vs Index (to see if errors are systematic)
    ax3 = axes[2]
    ax3.scatter(range(len(errors)), errors, alpha=0.5, s=10)
    ax3.axhline(0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel(f'Prediction Error ({column_name})', fontsize=12)
    ax3.set_title(f'Error vs Sample: {column_name}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add text box with metrics
    textstr = f'MAE: {mae:.6f}\nRMSE: {rmse:.6f}\nMax Error: {max_error:.6f}\nN samples: {len(y_true)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'rbf_predictions_{column_name}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: rbf_predictions_{column_name}.png")
    plt.show()
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"Statistics for {column_name}:")
    print(f"{'='*60}")
    print(f"Number of samples: {len(y_true)}")
    print(f"True {column_name} range: [{y_true.min():.6f}, {y_true.max():.6f}]")
    print(f"Predicted {column_name} range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print(f"\nError metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.6f}")
    print(f"  Max Absolute Error: {max_error:.6f}")
    print(f"  Mean Error (bias): {np.mean(errors):.6f}")
    print(f"  Std of Error: {np.std(errors):.6f}")
    print(f"\nPercentiles of absolute error:")
    print(f"  50th (median): {np.percentile(np.abs(errors), 50):.6f}")
    print(f"  90th: {np.percentile(np.abs(errors), 90):.6f}")
    print(f"  95th: {np.percentile(np.abs(errors), 95):.6f}")
    print(f"  99th: {np.percentile(np.abs(errors), 99):.6f}")


if __name__ == "__main__":
    import os
    
    # Define paths
    control_path = 'stack/main/data/trajectories/dynamic/control_inputs_controlled_410.csv'
    observation_path = 'stack/main/data/trajectories/dynamic/observations_controlled_410.csv'
    
    # Check if files exist
    if not os.path.exists(control_path):
        print(f"Error: Control file not found at {control_path}")
        exit(1)
    if not os.path.exists(observation_path):
        print(f"Error: Observation file not found at {observation_path}")
        exit(1)
    
    # Load the robot data
    print("="*60)
    print("Loading Robot Data")
    print("="*60)
    U_data_np, Y_data_np = load_robot_data(control_path, observation_path)
    
    N = U_data_np.shape[0]
    print(f"\nTotal data points: {N}")
    
    # Define scaling vector
    scaling = np.array([1,1, 1,1,1, 1], dtype=np.float32)
    print(f"Using scaling vector: {scaling}")

    # Option 1: Chunked fitting (more memory-efficient)
    print("\n" + "="*60)
    print("METHOD 1: Chunked PyTorch fitting (memory-efficient)")
    print("="*60)
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    U_centers, W, epsilon = fit_linear_rbf_pytorch(
        U_data_np, Y_data_np, scaling=scaling,
        device=device,
        chunk_size=1000,
        reg=1e-8
    )
    
    # Convert to JAX arrays
    print("\nConverting to JAX arrays...")
    import jax.numpy as jnp
    U_centers_jax = jnp.array(U_centers)
    W_jax = jnp.array(W)
    
    # Save the fitted model
    print("\nSaving model...")
    np.save('stack/main/data/models/ssm/rbf_centers.npy', U_centers)
    np.save('stack/main/data/models/ssm/rbf_weights.npy', W)
    print("Saved rbf_centers.npy and rbf_weights.npy")
    
    # Test evaluation on some actual data points
    print("\n" + "="*60)
    print("Testing JAX evaluation on real data")
    print("="*60)
    
    # Use first 10 data points as test
    u_test = jnp.array(U_data_np[:10])
    y_true = Y_data_np[:10]
    
    # Evaluate
    print("Running inference...")
    predictions = rbf_eval_batch_jax(u_test, U_centers_jax, W_jax, epsilon, scaling)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"\nSample comparison (first point):")
    print(f"  True:      {y_true[0]}")
    print(f"  Predicted: {predictions[0]}")
    print(f"  Error:     {np.abs(y_true[0] - np.array(predictions[0]))}")
    
    # Calculate overall error on test points
    errors = np.abs(y_true - np.array(predictions))
    print(f"\nTest set errors:")
    print(f"  Mean absolute error: {errors.mean(axis=0)}")
    print(f"  Max absolute error: {errors.max(axis=0)}")
    
    # Benchmark
    print("\n" + "="*60)
    print("Benchmarking")
    print("="*60)
    
    import time
    
    # Warm-up
    _ = rbf_eval_batch_jax(u_test, U_centers_jax, W_jax, epsilon, scaling).block_until_ready()
    
    # Time it
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        _ = rbf_eval_batch_jax(u_test, U_centers_jax, W_jax, epsilon, scaling).block_until_ready()
    elapsed = time.time() - start
    
    print(f"Average evaluation time for {len(u_test)} points: {elapsed/n_iters*1000:.2f} ms")
    print(f"Throughput: {len(u_test)*n_iters/elapsed:.0f} evaluations/sec")
    
    # Test on a larger batch
    print("\nTesting larger batch...")
    u_test_large = jnp.array(U_data_np[:1000])
    
    start = time.time()
    predictions_large = rbf_eval_batch_jax(u_test_large, U_centers_jax, W_jax, epsilon, scaling)
    predictions_large.block_until_ready()
    elapsed = time.time() - start
    
    print(f"Time for 1000 points: {elapsed*1000:.2f} ms")
    print(f"Time per point: {elapsed/1000*1000:.3f} ms")

    control_path_test = 'stack/main/data/trajectories/dynamic/control_inputs_controlled_310.csv'
    observation_path_test = 'stack/main/data/trajectories/dynamic/observations_controlled_310.csv'

    print("="*60)
    print("Loading Robot Test Data")
    print("="*60)
    U_data_test, Y_data_test = load_robot_data_test(control_path_test, observation_path_test)
    U_data_test_jax = jnp.array(U_data_test)
    predictions = rbf_eval_batch_jax(U_data_test_jax, U_centers_jax, W_jax, epsilon, scaling)
    print(f"Predictions shape: {predictions.shape}")
    print("\n" + "="*60)
    print("Plotting y3 trajectory")
    print("="*60)

    # Extract y3 (column index 1)
    y3_true = Y_data_test[:, 1]
    y3_pred = np.array(predictions)[:, 1]

    # Create time vector (assuming sequential samples)
    time = np.arange(len(y3_true))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time, y3_true, 'b-', linewidth=2, label='True y3', alpha=0.7)
    plt.plot(time, y3_pred, 'r--', linewidth=2, label='Predicted y3', alpha=0.7)
    plt.xlabel('Time (sample index)', fontsize=12)
    plt.ylabel('y3', fontsize=12)
    plt.title('y3 Trajectory: True vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('y3_trajectory.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: y3_trajectory.png")
    plt.show()

    # Print simple error metrics
    mae = np.mean(np.abs(y3_pred - y3_true))
    rmse = np.sqrt(np.mean((y3_pred - y3_true)**2))
    print(f"\ny3 Error Metrics:")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    # # Plot for y3 (column index 1)
    # print("\n" + "="*60)
    # print("Plotting Results for y3")
    # print("="*60)
    # plot_predictions_vs_true(Y_data_test, predictions, column_idx=1, column_name='y3')
    