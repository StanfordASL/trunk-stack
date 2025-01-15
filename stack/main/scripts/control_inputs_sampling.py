import os
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from itertools import product


def save_to_csv(df, control_inputs_file):
    df['ID'] = df['ID'].astype(int)
    df.to_csv(control_inputs_file, index=False)
    print(f'Control inputs have been saved to {control_inputs_file}')

# sets the DC offset for an equilibrium of adiabatic data collection
def set_adiabatic_control_offset(n_samples):
    # tip
    u1_offset = 0.4 # for now just in one control input
    u6_offset = 0 
    u1 = np.full(n_samples, u1_offset)
    u6 = np.full(n_samples, u6_offset)

    # mid
    u2_offset = 0
    u5_offset = 0
    u2 = np.full(n_samples, u2_offset)
    u5 = np.full(n_samples, u5_offset)

    # base
    u3_offset = 0
    u4_offset = 0
    u3 = np.full(n_samples, u3_offset)
    u4 = np.full(n_samples, u4_offset)

    const_input = np.array([u1, u2, u3, u4, u5, u6])

    return const_input

# TODO: write this function
def adiabatic_jolt_sampling(control_variables, seed):
    pass

# TODO: maybe use check_settled instead of this simplification where I just wait a certain number of seconds
# for sampling n_perturbations perturbations about a single constant equilibrium point for automatic adiabatic data collection
# jolt will just be the same except it has a shorter t_step
def adiabatic_step_sampling(control_variables, seed):
    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    np.random.seed(seed)
    n_samples = 1 
    const_input = set_adiabatic_control_offset(n_samples) 

    # add perturbations
    n_perturbations = 20    # number of perturbations per data collection round
    perturb_min = - 0.1
    perturb_max = 0.1
    t_settle = 1 * 300  # number of timesteps to allow for settling [seconds * 100Hz]
    t_step = 1 * 100 # number of timesteps to allow for step input [seconds * 100Hz]

    for i in range(n_perturbations):
        # sample random (small) perturbation in each control input
        perturbation = np.random.uniform(perturb_min, perturb_max, size=6)

        # set perturbation and hold for t_step seconds
        perturbed_control_input = const_input + perturbation
        step_inputs = np.tile(perturbed_control_input, (t_step, 1))

        # release to equilibrium and hold at equilibrium for t_settle seconds
        const_inputs = np.tile(const_inputs, (t_settle, 1))
        inputs = np.vstack((step_inputs, const_inputs)) # concatenate the step inputs and the return to constant

        control_inputs_df = control_inputs_df._append(dict(zip(['ID'] + control_variables, inputs)))

    return control_inputs_df



# for creating control inputs for a single constant equilibrium point for manual adiabatic data collection
def adiabatic_manual_sampling(control_variables):
    # set constant offset
    n_samples = 12000 # 120s * 100Hz = 12000 (2 min)
    const_input = set_adiabatic_control_offset(n_samples) 

    ids = np.arange(0, n_samples)
    control_inputs_df = pd.DataFrame(const_input.T, columns=control_variables)
    control_inputs_df.insert(0, 'ID', ids)

    return control_inputs_df

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

def circle_sampling(control_variables, random_seed):
    np.random.seed(random_seed)
    tip_radius, mid_radius, base_radius = 0.4, 0.35, 0.3 # always fits within check control inputs with 0.45, 0.35, 0.30 - change these for bigger/smaller circles
    noise_amplitude = 0.05

    num_samples_on_circle = 40
    sampled_angles = np.linspace(0, 2*np.pi, num_samples_on_circle)  # no flipping for now
    # sampled_angles_fwd = np.linspace(0, 2*np.pi, num_samples_on_circle)
    # sampled_angles_bkwd = sampled_angles_fwd[::-1] # flip it
    # sampled_angles = np.concatenate((sampled_angles_fwd, sampled_angles_bkwd))

    angle_offset = (1/6)*np.pi # 30 deg
    
    # set control inputs based on geometry of cable arrangement
    u1s = tip_radius * np.cos(sampled_angles)
    u6s = - tip_radius * np.sin(sampled_angles)
    u5s = - mid_radius * np.cos(sampled_angles + angle_offset) 
    u2s = mid_radius * np.sin(sampled_angles + angle_offset)
    u4s = base_radius * np.cos(sampled_angles + 2 * angle_offset) 
    u3s = - base_radius * np.sin(sampled_angles + 2 * angle_offset)

    circle_samples = np.column_stack((u1s, u2s, u3s, u4s, u5s, u6s))
    circle_samples += np.random.uniform(-noise_amplitude, noise_amplitude, circle_samples.shape)
    # we are not checking the circle values with check_control_inputs

    control_inputs_df = pd.DataFrame(circle_samples, columns=control_variables)
    control_inputs_df.insert(0, 'ID', np.arange(0, len(circle_samples)))

    return control_inputs_df

def targeted_sampling(control_variables, random_seed):
    # Load data
    num_seeds = 8
    data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')

    us_df = pd.read_csv(os.path.join(data_dir, 'trajectories/steady_state/control_inputs_beta_seed0.csv'))
    for seed in range(1, num_seeds + 1):
        us_df = pd.concat([us_df, pd.read_csv(os.path.join(data_dir, f'trajectories/steady_state/control_inputs_beta_seed{seed}.csv'))])
    us_df = pd.concat([us_df, pd.read_csv(data_dir + '/trajectories/steady_state/control_inputs_targeted_seed0.csv')])
    us_df = us_df.drop(columns=['ID'])

    # Observations
    ys_df = pd.read_csv(os.path.join(data_dir, 'trajectories/steady_state/observations_steady_state_beta_seed0.csv'))
    for seed in range(1, num_seeds + 1):
        ys_df = pd.concat([ys_df, pd.read_csv(os.path.join(data_dir, f'trajectories/steady_state/observations_steady_state_beta_seed{seed}.csv'))])
    ys_df = pd.concat([ys_df, pd.read_csv(data_dir + '/trajectories/steady_state/observations_steady_state_targeted_seed0.csv')])
    ys_df = ys_df.drop(columns=['ID'])

    rest_positions = np.array([0.1005, -0.10698, 0.10445, -0.10302, -0.20407, 0.10933, 0.10581, -0.32308, 0.10566])
    ys_df = ys_df - rest_positions

    # Convert to numpy arrays
    us = us_df.to_numpy()
    ys = ys_df.to_numpy()

    above_bins_idx = np.where(np.abs(ys[:, -3]) > 0.12)
    control_inputs_above_bins = us[above_bins_idx]
    print(f'Number of data points above bins: {len(control_inputs_above_bins)}')

    num_samples_per_point = 0 # was 10 - i dont think the above sampling is very effective, better to just do more circle sampling
    noise_range = 0.10 # was 0.05

    # Generate random samples around the previously found control inputs
    np.random.seed(random_seed)
    repeated_inputs = np.repeat(control_inputs_above_bins, num_samples_per_point, axis=0)
    bin_samples = repeated_inputs + np.random.uniform(-noise_range, noise_range, (len(control_inputs_above_bins) * num_samples_per_point, us.shape[1]))

    for i, bin_sample in enumerate(bin_samples):
        bin_samples[i] = check_control_inputs(bin_sample, repeated_inputs[i])

    # Sort bin samples
    sorted_bin_samples = bin_samples[bin_samples[:, 0].argsort()]

    # Collect samples on a circle of radius 0.5 (i.e. 50%) for u1, u6 and near zero for u2-u5
    num_samples_on_circle = 60
    sampled_angles = np.linspace(0, 2*np.pi, num_samples_on_circle)
    radius = 0.6 #probably should not increase this (was 0.5)
    u1s = radius * np.cos(sampled_angles)
    u6s = radius * np.sin(sampled_angles)
    circle_samples = np.zeros((num_samples_on_circle, 6))
    circle_samples[:, 0] = u1s
    circle_samples[:, -1] = u6s
    circle_samples += np.random.uniform(-noise_range/2, noise_range/2, (num_samples_on_circle, us.shape[1]))
    # we are not checking the circle values with check_control_inputs

    targeted_samples = np.vstack([sorted_bin_samples, circle_samples])

    control_inputs_df = pd.DataFrame(targeted_samples, columns=control_variables)
    control_inputs_df.insert(0, 'ID', np.arange(0, len(targeted_samples)))

    return control_inputs_df


def check_control_inputs(u_opt, u_opt_previous):
    # reject vector norms of u that are too large
    tip_range, mid_range, base_range = 0.45, 0.35, 0.3 #changed tip range to 0.55 to account for circle samples

    u1, u2, u3, u4, u5, u6 = u_opt[0], u_opt[1], u_opt[2], u_opt[3], u_opt[4], u_opt[5]

    # First we clip to max and min values
    u1 = np.clip(u1, -tip_range, tip_range)
    u6 = np.clip(u6, -tip_range, tip_range)
    u2 = np.clip(u2, -mid_range, mid_range)
    u5 = np.clip(u5, -mid_range, mid_range)
    u3 = np.clip(u3, -base_range, base_range)
    u4 = np.clip(u4, -base_range, base_range)

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

    # Check the constraint: if the constraint is met, then keep previous control command
    if norm_value > 0.8:
        print(f'Sample {u_opt} got rejected')
        u_opt = u_opt_previous
    else:
        # Else the clipped command is published
        u_opt = np.array([u1, u2, u3, u4, u5, u6])

    return u_opt


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


def main(data_type, sampling_type, seed=None):
    control_variables = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    # data_dir for mark's mac starts with '/Users/asltrunk/trunk-stack/stack/main/data'
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
    elif sampling_type=='targeted':
        control_inputs_df = targeted_sampling(control_variables, seed)
    elif sampling_type =='circle':
        control_inputs_df = circle_sampling(control_variables, seed)
    elif sampling_type == 'adiabatic_manual':
        control_inputs_df = adiabatic_manual_sampling(control_variables)
    elif sampling_type == 'adiabatic_step':
        control_inputs_df = adiabatic_step_sampling(control_variables, seed)
    elif sampling_type == 'adiabatic_jolt':
        control_inputs_df = adiabatic_jolt_sampling(control_variables, seed)
    else:
        raise ValueError(f"Invalid sampling_type: {sampling_type}")

    save_to_csv(control_inputs_df, control_inputs_file)
    visualize_samples(control_inputs_df)


if __name__ == '__main__':
    data_type = 'dynamic'                   # 'steady_state' or 'dynamic'
    sampling_type = 'adiabatic_manual'      # 'circle', 'beta', 'targeted', 'uniform', 'sinusoidal', 'adiabatic_manual', 'adiabatic_step', or 'adiabatic_jolt'
    seed = None                             # choose integer seed number
    main(data_type, sampling_type, seed)
