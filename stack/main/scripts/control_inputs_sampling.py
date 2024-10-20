import os
import numpy as np
import pandas as pd  # type: ignore
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

def random_sampling(control_variables):
    sample_size = 250
    np.random.seed(9)
    tip_range = 0.15
    mid_range = 0.25
    base_range = 0.35 

    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    
    u1 = np.random.uniform(low=-tip_range, high=tip_range, size=(sample_size,))
    u6 = np.random.uniform(low=-tip_range, high=tip_range, size=(sample_size,))
    u2 = np.random.uniform(low=-mid_range, high=mid_range, size=(sample_size,))
    u5 = np.random.uniform(low=-mid_range, high=mid_range, size=(sample_size,))
    u3 = np.random.uniform(low=-base_range, high=base_range, size=(sample_size,))
    u4 = np.random.uniform(low=-base_range, high=base_range, size=(sample_size,))

    data = np.column_stack([u1, u2, u3, u4, u5, u6])
    ids = np.arange(0, sample_size)
    control_inputs_df = pd.DataFrame(data, columns=control_variables)
    control_inputs_df.insert(0, 'ID', ids)

    return control_inputs_df
    




def main(data_type='dynamic', sampling_type='uniform'):
    control_variables = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
    control_inputs_file = os.path.join(data_dir, f'trajectories/{data_type}/control_inputs_{sampling_type}.csv')
    
    if sampling_type=='sinusoidal':
        control_inputs_df = sinusoidal_sampling(control_variables)
    elif sampling_type=='uniform':
        control_inputs_df = uniform_sampling(control_variables)
    elif sampling_type=='random':
        control_inputs_df = random_sampling(control_variables)
    else:
        raise ValueError(f"Invalid sampling_type: {sampling_type}")

    save_to_csv(control_inputs_df, control_inputs_file)


if __name__ == '__main__':
    data_type = 'steady_state'       # 'steady_state' or 'dynamic'
    sampling_type = 'random'   # 'uniform' or 'sinusoidal' or 'random'
    main(data_type, sampling_type)
