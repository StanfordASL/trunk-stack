import os
import numpy as np
import pandas as pd  # type: ignore


def random_trajectory():
    # Parameters
    n_u = 6
    sample_interval = 1   # [s]
    total_duration = 150  # [s]
    sampling_rate = 100   # [Hz]
    u_min, u_max = -0.4, 0.4
    num_trajectories = 10  # Number of sinusoidal trajectories

    # Number of samples
    num_samples = total_duration // sample_interval
    total_samples = total_duration * sampling_rate

    # Generate sinusoidal trajectories
    us_sin = np.zeros((num_trajectories, total_samples, n_u))
    for i in range(num_trajectories):
        for j in range(n_u):
            amplitude = np.random.uniform(u_min, u_max)  # Random amplitude
            frequency = np.random.uniform(0.01, 0.1)  # Random frequency
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase
            us_sin[i, :, j] = amplitude * np.sin(2 * np.pi * frequency * ts + phase)

    # Time vector for interpolation
    ts = np.arange(0, total_duration, 1/sampling_rate)

    # Interpolated sinusoidal trajectories
    us_interp = np.zeros((num_trajectories, total_samples, n_u))
    for i in range(num_trajectories):
        for j in range(n_u):
            us_interp[i, :, j] = np.interp(ts, np.arange(0, total_duration, sample_interval), us_sin[i, :, j])

    # Create DataFrames for each trajectory
    dfs = []
    for i in range(num_trajectories):
        IDs = np.arange(total_samples)
        df = pd.DataFrame(us_interp[i, :, :], columns=[f'u{i+1}' for i in range(n_u)])
        df['ID'] = IDs
        df = df[['ID'] + [f'u{i+1}' for i in range(n_u)]]
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined


def sine_trajectory():
    # Parameters
    n_u = 6
    num_cycles = 4  # number of cycles per control
    sampling_rate = 100  # [Hz]
    period = 5  # [s], period of the sine wave
    max_amp = 0.55  # maximum amplitude of the sine wave
    gamma = 0.9  # decrease amplitude with this factor

    ts = np.arange(0, period, 1/sampling_rate)

    u_order = [0, 5, 1, 4, 2, 3]
    us_sine = np.zeros((len(ts)*num_cycles*n_u, n_u))
    for u_i in u_order:
        amplitude = max_amp
        for cycle_i in range(num_cycles):
            start_index = u_i * len(ts) * num_cycles + cycle_i * len(ts)
            end_index = start_index + len(ts)
            
            us_sine[start_index:end_index, u_i] = amplitude * np.sin(2 * np.pi * ts / period)
            amplitude *= gamma
    
    IDs = np.arange(len(ts)*num_cycles*n_u)
    df = pd.DataFrame(us_sine, columns=[f'u{i+1}' for i in range(n_u)])

    df['ID'] = IDs
    df = df[['ID'] + [f'u{i+1}' for i in range(n_u)]]
    return df


def interpolated_beta(seed=0):
    # Parameters
    sample_size = 45
    sec_per_sample = 2
    sampling_rate = 100  # [Hz]
    np.random.seed(seed)
    tip_range, mid_range, base_range = 0.45, 0.35, 0.3

    N = sample_size * sec_per_sample * sampling_rate
    control_variables = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
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
        if norm_value <= 0.75:
            valid_samples.append([u1, u2, u3, u4, u5, u6])
            num_valid_samples += 1
        else:   
            rejection_count += 1
            print(f'Rejected {u1, u2, u3, u4, u5, u6} count: {rejection_count}')
    
    valid_samples = np.array(valid_samples)

    # Interpolate these sampled control inputs to get smoother signal at high sampling rate
    interp_points = np.linspace(0, 1, sec_per_sample * sampling_rate)
    interpolated_data = []
    for i in range(sample_size - 1):
        start = valid_samples[i]
        end = valid_samples[i + 1]
        interpolated = np.outer(1 - interp_points, start) + np.outer(interp_points, end)
        interpolated_data.append(interpolated)

    interpolated_data = np.array(interpolated_data).reshape(-1, 6)

    IDs = np.arange(len(interpolated_data))
    control_inputs_df = pd.DataFrame(interpolated_data, columns=control_variables)
    control_inputs_df.insert(0, 'ID', IDs)
    return control_inputs_df


def main(control_inputs_file, control_type):
    if control_type == 'random':
        df = random_trajectory()
    elif control_type == 'interp_beta':
        df = interpolated_beta()
    elif control_type == 'sinusoidal':
        df = sine_trajectory()
    df.to_csv(control_inputs_file, index=False)


if __name__ == '__main__':
    control_type = 'interp_beta'  # 'random', 'interp_beta' or 'sinusoidal'
    data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
    # control_inputs_file = os.path.join(data_dir, f'trajectories/dynamic/control_inputs_controlled_{control_type}.csv')
    control_inputs_file = os.path.join(data_dir, f'trajectories/dynamic/control_inputs_controlled_1.csv')
    main(control_inputs_file, control_type=control_type)
