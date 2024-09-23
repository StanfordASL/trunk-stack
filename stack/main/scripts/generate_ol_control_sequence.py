import os
import numpy as np
import pandas as pd  # type: ignore


def random_trajectory():
    # Parameters
    n_u = 6
    sample_interval = 1   # [s]
    total_duration = 150  # [s]
    sampling_rate = 100   # [Hz]
    u_min, u_max = -0.35, 0.35
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
    max_amp = 0.35  # maximum amplitude of the sine wave
    gamma = 0.8  # decrease amplitude with this factor

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


def main(control_inputs_file, control_type):
    if control_type == 'random':
        df = random_trajectory()
    elif control_type == 'sinusoidal':
        df = sine_trajectory()
    df.to_csv(control_inputs_file, index=False)


if __name__ == '__main__':
    control_type = 'sinusoidal'
    data_dir = os.getenv('TRUNK_DATA', '/home/asl/Documents/asl_trunk_ws/data')
    control_inputs_file = os.path.join(data_dir, f'trajectories/dynamic/control_inputs_controlled_{control_type}.csv')
    main(control_inputs_file, control_type=control_type)
