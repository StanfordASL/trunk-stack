import os
import scipy.io
import numpy as np


def mat_to_npz(mat_filepath, npz_filepath):
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
    encoder_coeff, encoder_exp = mat_data['Vfinal'], mat_data['exps_V']
    decoder_coeff, decoder_exp = mat_data['M'], mat_data['exps']
    dynamics_coeff, dynamics_exp = mat_data['R'], mat_data['exps_r']
    B_r_coeff = mat_data['B_red']
    obs_perf_matrix = np.zeros((3, 12))  # TODO: generalize this
    obs_perf_matrix[:, :3] = np.eye(3)
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
    print(f"Successfully saved as.npz at {npz_filepath}")
    return npz_filepath


def main():
    model_name = 'ssm_origin_300g_fast'
    data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
    mat_filepath = os.path.join(data_dir, f'models/ssm/{model_name}.mat')
    npz_filepath = os.path.join(data_dir, f'models/ssm/{model_name}.npz')
    saved_path = mat_to_npz(mat_filepath, npz_filepath)
    data = np.load(saved_path)
    print('Keys: ', list(data.keys()))
    print('B_r matrix: ', data['B_r_coeff'])

if __name__ == '__main__':
    main()
