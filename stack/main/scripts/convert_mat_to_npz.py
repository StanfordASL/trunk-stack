import os
import scipy.io
import numpy as np


def mat_to_npz(mat_filepath, npz_filepath=None):
    """
    Loads a.mat file and saves its contents as a.npz file.

    Args:
        mat_filepath (str): The path to the.mat file.
        npz_filepath (str, optional): The path to save the.npz file. 
                                      If None, the npz file will be saved in the same 
                                      directory as the mat file with the same name 
                                      but the.npz extension. Defaults to None.

    Returns:
        str: The path to the saved.npz file.
        None: If there is an error during the process.
    """

    try:
        mat_data = scipy.io.loadmat(mat_filepath)  # Load the.mat file
    except FileNotFoundError:
        print(f"Error:.mat file not found at {mat_filepath}")
        return None
    except Exception as e:  # Catch other potential loading errors
        print(f"Error loading.mat file: {e}")
        return None

    if npz_filepath is None:
        base_name = os.path.splitext(os.path.basename(mat_filepath))
        npz_filepath = os.path.join(os.path.dirname(mat_filepath), base_name + ".npz")

    try:
        np.savez(npz_filepath, **mat_data)  # Save as.npz, unpacking the dictionary
        print(f"Successfully saved as.npz at {npz_filepath}")
        return npz_filepath
    except Exception as e:
        print(f"Error saving.npz file: {e}")
        return None


def main(args=None):
    model_name = "origin_ssmr_200g"
    data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
    mat_filepath = os.path.join(data_dir, f'models/ssmr/{model_name}.mat')
    saved_path = mat_to_npz(mat_filepath)

if __name__ == '__main__':
    main()
