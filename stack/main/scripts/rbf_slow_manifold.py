import numpy as np

def rbf_eval(u, U_select, W, epsilon, case_rbf):
    """
    Radial Basis Function (RBF) evaluation.
    
    Parameters:
    u (ndarray): d x N matrix, where each column is a d-dimensional evaluation point
    U_select (ndarray): d x M matrix, where each column is a d-dimensional center
    W (ndarray): M x d' matrix, where each row is the weight vector for a center
    epsilon (float): shape parameter for the Gaussian RBF
    case_rbf (bool): flag to select Gaussian RBF (True) or distance matrix (False)
    
    Returns:
    final_value (ndarray): d' x N matrix, where each column is the RBF interpolation
    """
    # Get dimensions
    d, N = u.shape
    _, M = U_select.shape
    _, d_prime = W.shape
    
    # Validate dimensions
    if U_select.shape[0] != d:
        raise ValueError('Dimension of U_select must match dimension of u')
    if W.shape[0] != M:
        raise ValueError('Number of rows in W must match number of centers in U_select')
    
    # Compute pairwise Euclidean distances between u and U_select
    # dist: M x N matrix, where dist[i,j] is the distance between U_select[:,i] and u[:,j]
    dist = np.linalg.norm(U_select[:, :, np.newaxis] - u[:, np.newaxis, :], axis=0)
    
    # Compute Gaussian RBF: phi(r) = exp(-(epsilon * r)^2)
    # A: M x N matrix, where A[i,j] = phi(||U_select[:,i] - u[:,j]||)
    if case_rbf:
        A = np.exp(-(epsilon * dist)**2)
    else:
        A = dist
    
    # Compute RBF interpolation: final_value = W' * A
    # A: M x N, W: M x d' -> final_value: d' x N
    final_value = W.T @ A
    
    return final_value