import os
import scipy.io
import numpy as np

def mat_to_npz_rbf(mat_filepath, npz_filepath):
    """
    Loads a .mat file containing U_select, W, epsilon, and case_rbf, and saves them as a .npz file.

    Args:
        mat_filepath (str): The path to the .mat file.
        npz_filepath (str): The path to save the .npz file.

    Returns:
        str: The path to the saved .npz file if successful.
        None: If an error occurs during the process.
    """
    try:
        # Check if the .mat file exists
        if not os.path.exists(mat_filepath):
            print(f"Error: The file {mat_filepath} does not exist.")
            return None

        # Load the .mat file
        mat_data = scipy.io.loadmat(mat_filepath)

        # Verify that required variables exist in the .mat file
        required_vars = ['U_select', 'W', 'epsilon', 'case_rbf']
        missing_vars = [var for var in required_vars if var not in mat_data]
        if missing_vars:
            print(f"Error: The following required variables are missing in the .mat file: {missing_vars}")
            return None

        # Extract the required variables
        U_select = mat_data['U_select']
        W = mat_data['W']
        epsilon = mat_data['epsilon']
        case_rbf = mat_data['case_rbf']

        # Save variables to .npz file
        np.savez(
            npz_filepath,
            U_select=U_select,
            W=W,
            epsilon=epsilon,
            case_rbf=case_rbf
        )

        print(f"Successfully saved as .npz at {npz_filepath}")
        return npz_filepath

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Example file paths (replace with your actual file paths)
    mat_file = "path/to/your/input.mat"  # Update this path
    npz_file = "path/to/your/output.npz"  # Update this path

    # Convert the .mat file to .npz
    result = mat_to_npz_rbf(mat_file, npz_file)
    if result:
        print(f"Conversion completed: {result}")
    else:
        print("Conversion failed.")