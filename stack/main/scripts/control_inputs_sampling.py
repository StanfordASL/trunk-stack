import os
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from itertools import product


def save_to_csv(df, control_inputs_file):
    df['ID'] = df['ID'].astype(int)
    df.to_csv(control_inputs_file, index=False)
    print(f'Control inputs have been saved to {control_inputs_file}')


def sinusoidal_sampling(control_variables):
    num_controls = len(control_variables)
    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    points_per_period = 15
    amplitude = 0.30
    
    for i in range(len(control_variables)):
        for j in range(points_per_period):
            values = np.zeros(num_controls)
            t = np.pi/8 * np.arange(1, points_per_period + 1)
            values[i] = amplitude * np.sin(t[j])
            control_inputs_df = control_inputs_df._append(dict(zip(['ID'] + control_variables, [j + i * points_per_period] + list(values))), ignore_index=True)
    
    return control_inputs_df


def uniform_sampling(control_variables):
    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    #TODO: this code currently doesn't correctly assign segments to actuators
    #reasonable for SRC demo
    tip_range = 0.15
    mid_range = 0.25
    base_range = 0.35
    ranges = [np.linspace(-val, val, 3) for val in [tip_range, mid_range, base_range]*2]
    combinations = list(product(*ranges))

    for i, combo in enumerate(combinations):
        control_inputs_df = control_inputs_df._append(dict(zip(['ID'] + control_variables, [i] + list(combo))), ignore_index=True)

    return control_inputs_df
    

def beta_sampling(control_variables, seed, sample_size=100):
    np.random.seed(seed)
    tip_range, mid_range, base_range = 0.4, 0.35, 0.3

    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    
    # Beta parameters
    a, b = 0.5, 0.5

    # Initialize an empty list to collect valid samples
    valid_samples = []
    num_valid_samples = 0

    rejection_count = 0
    while num_valid_samples < sample_size:
        # Sample from Beta distribution, then shift and scale to match desired ranges
        u1 = (np.random.beta(a, b) - 0.5) * 2 * tip_range
        u6 = (np.random.beta(a, b) - 0.5) * 2 * tip_range
        u2 = (np.random.beta(a, b) - 0.5) * 2 * mid_range
        u5 = (np.random.beta(a, b) - 0.5) * 2 * mid_range
        u3 = (np.random.beta(a, b) - 0.5) * 2 * base_range
        u4 = (np.random.beta(a, b) - 0.5) * 2 * base_range

        # Compute control input vectors
        u1_vec = u1 * np.array([-np.cos(15 * np.pi/180), np.sin(15 * np.pi/180)])
        u2_vec = u2 * np.array([np.cos(45 * np.pi/180), np.sin(45 * np.pi/180)])
        u3_vec = u3 * np.array([-np.cos(15 * np.pi/180), -np.sin(15 * np.pi/180)])
        u4_vec = u4 * np.array([-np.cos(75 * np.pi/180), np.sin(75 * np.pi/180)])
        u5_vec = u5 * np.array([np.cos(45 * np.pi/180), -np.sin(45 * np.pi/180)])
        u6_vec = u6 * np.array([-np.cos(75 * np.pi/180), -np.sin(75 * np.pi/180)])

        # Calculate the norm based on the constraint
        vector_sum = (
            0.75 * (u3_vec + u4_vec) +
            1.0 * (u2_vec + u5_vec) +
            1.4 * (u1_vec + u6_vec)
        )
        norm_value = np.linalg.norm(vector_sum)

        # Check the constraint: if the sample is valid, keep it
        if norm_value <= 0.7:
            valid_samples.append([u1, u2, u3, u4, u5, u6])
            num_valid_samples += 1
        else:   
            rejection_count += 1
            print(f'Rejected {u1, u2, u3, u4, u5, u6} count: {rejection_count}')

    # Convert valid samples to a DataFrame
    ids = np.arange(0, sample_size)
    valid_samples = np.array(valid_samples)
    control_inputs_df = pd.DataFrame(valid_samples, columns=control_variables)
    control_inputs_df.insert(0, 'ID', ids)

    return control_inputs_df


def visualize_samples(control_inputs_df):
    control_variables = control_inputs_df.columns[1:]
    num_vars = len(control_variables)
    _, axes = plt.subplots(num_vars, 1, figsize=(8, 8))
    
    for i, var in enumerate(control_variables):
        axes[i].hist(control_inputs_df[var], bins=30, alpha=0.7, color='blue')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def main(data_type='dynamic', sampling_type='uniform', seed=None):
    control_variables = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
    if seed is not None:
        control_inputs_file = os.path.join(data_dir, f'trajectories/{data_type}/control_inputs_{sampling_type}_seed{seed}.csv')
    else:
        control_inputs_file = os.path.join(data_dir, f'trajectories/{data_type}/control_inputs_{sampling_type}.csv')
    
    if sampling_type=='sinusoidal':
        control_inputs_df = sinusoidal_sampling(control_variables)
    elif sampling_type=='uniform':
        control_inputs_df = uniform_sampling(control_variables)
    elif sampling_type=='beta':
        control_inputs_df = beta_sampling(control_variables, seed)
    else:
        raise ValueError(f"Invalid sampling_type: {sampling_type}")

    save_to_csv(control_inputs_df, control_inputs_file)
    visualize_samples(control_inputs_df)


if __name__ == '__main__':
    data_type = 'steady_state'       # 'steady_state' or 'dynamic'
    sampling_type = 'beta'           # 'beta', 'uniform' or 'sinusoidal'
    seed = 8                         # choose integer seed number
    main(data_type, sampling_type, seed)
