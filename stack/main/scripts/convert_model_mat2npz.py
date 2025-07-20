import os
import scipy.io
import numpy as np


def mat_to_npz_slow(mat_filepath, npz_filepath):
    """
    Loads a.mat file and saves its contents as a.npz file.

    Args:
        mat_filepath (str): The path to the.mat file.
        npz_filepath (str): The path to save the.npz file. 

    Returns:
        str: The path to the saved.npz file.
        None: If there is an error during the process.
    """
    mat_data = scipy.io.loadmat(mat_filepath)
    decoder_coeff, decoder_exp = mat_data['Slow_manifold_coeff'], mat_data['exps_sm']
    Const_coeff = mat_data['Const_coeff']
    np.savez(
        npz_filepath,
        decoder_coeff=decoder_coeff,
        decoder_exp=decoder_exp,
        Const_coeff=Const_coeff,
    )
    print(f"Successfully saved as.npz at {npz_filepath}")
    return npz_filepath


def mat_to_npz(mat_filepath, npz_filepath, perf_var="tip_3D"):
    """
    Loads a.mat file and saves its contents as a.npz file.

    Args:
        mat_filepath (str): The path to the.mat file.
        npz_filepath (str): The path to save the.npz file. 

    Returns:
        str: The path to the saved.npz file.
        None: If there is an error during the process.
    """
    mat_data = scipy.io.loadmat(mat_filepath)

    dynamics_coeff, dynamics_exp = mat_data['R'], mat_data['exps_r']  # (n_x, ...) & (..., n_x)
    decoder_coeff, decoder_exp = mat_data['M'], mat_data['exps']  # (n_y, ...) & (..., n_x)
    encoder_coeff, encoder_exp = mat_data['Vfinal'], mat_data['exps_V']  # (n_x, ...) & (..., n_y)

    # Print all shape information (check manually if they're correct)
    print(f"dynamics_coeff shape: {dynamics_coeff.shape}, dynamics_exp shape: {dynamics_exp.shape}")
    print(f"decoder_coeff shape: {decoder_coeff.shape}, decoder_exp shape: {decoder_exp.shape}")
    print(f"encoder_coeff shape: {encoder_coeff.shape}, encoder_exp shape: {encoder_exp.shape}")

    n_y = decoder_coeff.shape[0]
    if perf_var == "tip_2D":
        n_z = 2
    elif perf_var == "tip_3D":
        # Nominally we care about the 3D tip position
        n_z = 3
    else:
        raise ValueError(f"Unknown performance variable: {perf_var}")
    
    obs_perf_matrix = np.zeros((n_z, n_y))
    obs_perf_matrix[:, :n_z] = np.eye(n_z)

    if 'B_red' in mat_data:
        B_r_coeff = mat_data['B_red']
        np.savez(
            npz_filepath,
            encoder_coeff=encoder_coeff,
            encoder_exp=encoder_exp,
            decoder_coeff=decoder_coeff,
            decoder_exp=decoder_exp,
            dynamics_coeff=dynamics_coeff,
            dynamics_exp=dynamics_exp,
            B_r_coeff=B_r_coeff,
            obs_perf_matrix=obs_perf_matrix,
        )
    else:
        np.savez(
            npz_filepath,
            encoder_coeff=encoder_coeff,
            encoder_exp=encoder_exp,
            decoder_coeff=decoder_coeff,
            decoder_exp=decoder_exp,
            dynamics_coeff=dynamics_coeff,
            dynamics_exp=dynamics_exp,
            obs_perf_matrix=obs_perf_matrix,
        )
    print(f"Successfully saved as.npz at {npz_filepath}")
    return npz_filepath


def main():
    model_name = 'adiabatic/first_slow_aSSM'
    data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
    mat_filepath = os.path.join(data_dir, f'models/ssm/{model_name}.mat')
    npz_filepath = os.path.join(data_dir, f'models/ssm/{model_name}.npz')
    saved_path = mat_to_npz(mat_filepath, npz_filepath)
    data = np.load(saved_path)
    print('Keys: ', list(data.keys()))


if __name__ == '__main__':
    main()
