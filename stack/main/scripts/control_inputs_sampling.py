import os
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from itertools import product
from perlin_noise import PerlinNoise
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from collections import Counter


def save_to_csv(df, control_inputs_file):
    df['ID'] = df['ID'].astype(int)
    df.to_csv(control_inputs_file, index=False)
    print(f'Control inputs have been saved to {control_inputs_file}')

# generate open loop test trajectories with high actuation magnitude points.
def OL_test_high_z_sampling(control_variables, random_seed, N=10):
    # Define bounds
    tip_radius = 80
    mid_radius = 50
    base_radius = 30
    maxs = np.array([mid_radius, tip_radius, base_radius, tip_radius, base_radius, mid_radius])
    mins = -maxs

    # Latin Hypercube Sampling
    np.random.seed(random_seed)
    sampler = qmc.LatinHypercube(d=6, seed=random_seed)
    sample = sampler.random(n=1000)
    scaled_sample = qmc.scale(sample, mins, maxs)

    # Create distance matrix and find point pairs
    remaining_indices = list(range(1000))
    pairs = []

    for _ in range(N): # N is number of trajectories (pairs of points)
        subset = scaled_sample[remaining_indices]
        magnitudes = np.linalg.norm(subset, axis=1)
        idx1_local = np.argmax(magnitudes)
        point1_idx = remaining_indices[idx1_local]
        point1 = scaled_sample[point1_idx]

        distances = cdist([point1], subset)[0]
        idx2_local = np.argmax(distances)
        point2_idx = remaining_indices[idx2_local]
        point2 = scaled_sample[point2_idx]

        pairs.append((point1, point2))
        remaining_indices.remove(point1_idx)
        remaining_indices.remove(point2_idx)

    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    ID_counter = 0

    for point1, point2 in pairs:
        control_inputs_df, ID_counter = add_trajectory(control_inputs_df, ID_counter, point1, point2)

    control_inputs = control_inputs_df[control_variables].values

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # 2 rows, 3 columns

    for i, ax in enumerate(axes.flat):
        ax.plot(control_inputs[:, i])
        ax.set_title(f"U_{i+1}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Actuation")

    plt.tight_layout()
    plt.show()

    return control_inputs_df

# helper function for OL_test_high_z_sampling. Makes a smooth trajectory between two points
def add_trajectory(df, ID_counter, point1, point2):
        zero = np.zeros(6)

        def append_rows(pts):
            nonlocal df, ID_counter
            for pt in pts:
                df.loc[len(df)] = [ID_counter] + pt.tolist()
                ID_counter += 1

        append_rows([zero] * 200)  # 2s of zero input to start each traj
        append_rows(smooth_cosine_interp(zero, point1, 100)) # Smooth up to point1
        append_rows(smooth_cosine_interp(point1, zero, 100)) # Smooth down to zero
        append_rows(smooth_cosine_interp(zero, point2, 100)) # Smooth up to point2
        append_rows(smooth_cosine_interp(point2, zero, 100)) # Smooth back down to zero

        return df, ID_counter

# helper function for add_trajectory - smoothly interpolates between two points with cosine ramp
def smooth_cosine_interp(start, end, num_steps):
    return [
        start + 0.5 * (1 - np.cos(np.pi * t)) * (end - start)
        for t in np.linspace(0, 1, num_steps)
    ]


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

def hypercube_controlled_sampling(control_variables, random_seed, num_points=10, visits_per_point=6, excluded_neighbors=2):
    points_df = latin_hypercube_adiabatic_sampling(control_variables, random_seed, num_points=num_points, visits_per_point=visits_per_point, excluded_neighbors=excluded_neighbors)
    len_traj = 200  # 100 = 1s

    # repeat each row len_traj times
    control_inputs_df = points_df.loc[points_df.index.repeat(len_traj)].reset_index(drop=True)

    # 2 sec of zero control inputs
    num_zeros = 200
    zero_data = pd.DataFrame(np.zeros((num_zeros, len(control_variables))), columns=control_variables)
    zero_data.insert(0, 'ID', np.arange(num_zeros)) 

    # shift ids of the remaining trajectories and concatenate
    control_inputs_df['ID'] = np.arange(num_zeros, num_zeros + len(control_inputs_df))
    control_inputs_df = pd.concat([zero_data, control_inputs_df], ignore_index=True)

    return control_inputs_df



def latin_hypercube_adiabatic_sampling(control_variables, random_seed, num_points=100, visits_per_point=20, excluded_neighbors=20):
    np.random.seed(random_seed)

    # Define bounds
    tip_radius = 80
    mid_radius = 50
    base_radius = 30
    maxs = np.array([mid_radius, tip_radius, base_radius, tip_radius, base_radius, mid_radius])
    mins = -maxs

    # Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=6, seed=random_seed)
    sample = sampler.random(n=num_points)
    scaled_sample = qmc.scale(sample, mins, maxs)
    control_inputs = pd.DataFrame(scaled_sample, columns=control_variables)

    # Precompute neighbor sets and distance matrix
    distances = cdist(scaled_sample, scaled_sample)
    np.fill_diagonal(distances, np.inf)
    neighbor_indices = np.argsort(distances, axis=1)
    near_neighbors = [set(neighbor_indices[i, :excluded_neighbors]) for i in range(num_points)]

    # Build visit pool
    visit_pool = list(np.tile(np.arange(num_points), visits_per_point))
    visit_counts = Counter(visit_pool)

    # Start sequence
    current = np.random.choice(visit_pool)
    sequence = [current]
    visit_counts[current] -= 1
    if visit_counts[current] == 0:
        visit_pool.remove(current)

    for _ in range(num_points * visits_per_point - 1):
        # Valid candidates are not in the near-neighbor list and have remaining visits
        candidates = [p for p in set(visit_pool) if p not in near_neighbors[current]]

        if not candidates:
            print(f"Relaxation fallback triggered at step {_ + 1}, current node index = {current}")
            remaining = list(set(visit_pool))
            distances_to_current = distances[current, remaining]
            next_point = remaining[np.argmax(distances_to_current)]
        else:
            next_point = np.random.choice(candidates)

        sequence.append(next_point)
        visit_counts[next_point] -= 1
        if visit_counts[next_point] == 0:
            visit_pool = [p for p in visit_pool if p != next_point]
        current = next_point

    # Build control_inputs_df
    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    for i, idx in enumerate(sequence):
        row = control_inputs.iloc[idx].copy()
        row["ID"] = i
        control_inputs_df = pd.concat([control_inputs_df, pd.DataFrame([row])], ignore_index=True)

    # plot trajectory graphs
    plot_trajectory_graphs(control_inputs_df, control_variables, focus_node_idx=0)

    return control_inputs_df

def adiabatic_global_sampling(control_variables, random_seed):
    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    np.random.seed(random_seed)

    phase_shift_large = (np.pi/4)*np.pi 
    phase_shift_small = 0.0 # small circle rotated by 45 deg from large circle
    control_inputs_large_circle = circle_sampling(control_variables, random_seed,  tip_radius = 60, mid_radius = 35, base_radius = 15, phase_shift=phase_shift_large, noise_amplitude=0.0, num_samples_on_circle=4, num_circles=1)
    control_inputs_small_circle = circle_sampling(control_variables, random_seed, tip_radius = 20, mid_radius = 15, base_radius = 5, phase_shift=phase_shift_small, noise_amplitude=0.0, num_samples_on_circle=4, num_circles=1)
    
    # form the df of control input options
    origin_df = pd.DataFrame([[0]*7], columns=control_inputs_df.columns) # add in origin as one set of inputs
    control_inputs_equil_points = pd.concat([control_inputs_large_circle, control_inputs_small_circle, origin_df], ignore_index=True)
    n_equil_points = len(control_inputs_equil_points)

    # #uncomment this block and comment out for loop to check what control inputs look like
    # control_inputs_df = control_inputs_equil_points
    # control_inputs_df["ID"] = np.arange(n_equil_points)

    run_time = 10 * 60 * 100 # total run time (10 mins * 60s * 100Hz)
    t_settle = 3 * 100  # number of timesteps to allow for settling [seconds * 100Hz]
    num_traj = int(run_time/t_settle) # total number of trajectories
    print(f'Total run timesteps: {run_time}, n trajectories: {num_traj}')

    last_rand = None
    for i in range(num_traj):
        rand = np.random.randint(0, n_equil_points)

        if last_rand is not None:
            while True:
                if last_rand == rand:
                    rand = np.random.randint(0, n_equil_points)
                else:
                    break
                
        row = control_inputs_equil_points.iloc[rand]
        row["ID"] = i
        df = pd.DataFrame([row]*1, columns=control_inputs_df.columns) # change to [row]*t_settle if you want to manually do control inputs
        control_inputs_df = pd.concat([control_inputs_df, df], ignore_index=True)
        last_rand = rand

    return control_inputs_df

# TODO: maybe use check_settled instead of this simplification where I just wait a certain number of seconds
# for sampling n_perturbations perturbations about a single constant equilibrium point for automatic adiabatic data collection
# jolt will just be the same except it has a shorter t_step
def adiabatic_step_sampling(control_variables, seed):
    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    np.random.seed(seed)
    n_samples = 1 
    const_input = set_adiabatic_control_offset(n_samples).flatten()
    print(const_input)
    print(const_input.shape)

    # add perturbations, positions[3]
    n_perturbations = 20    # number of perturbations per data collection round
    perturb_min = - 0.1
    perturb_max = 0.1
    t_initial = 3 * 100 # number of timesteps at initial point [seconds * 100Hz]
    t_settle = 3 * 100  # number of timesteps to allow for settling [seconds * 100Hz]
    t_step = 1 * 100 # number of timesteps to allow for step input [seconds * 100Hz]

    # initial time at equilibrium point
    initial_inputs = np.tile(const_input, (t_initial, 1))
    ids = np.zeros(t_initial)
    ids = ids[:, np.newaxis]
    initial_inputs = np.hstack((ids, initial_inputs))
    initial_inputs_df = pd.DataFrame(initial_inputs, columns = control_inputs_df.columns)
    control_inputs_df = pd.concat([control_inputs_df, initial_inputs_df], ignore_index=True)

    # create perturbation inputs
    for i in range(n_perturbations):
        # sample random (small) perturbation in each control input
        perturbation = np.random.uniform(perturb_min, perturb_max, size=6)

        # set perturbation and hold for t_step seconds
        perturbed_control_input = const_input + perturbation
        step_inputs = np.tile(perturbed_control_input, (t_step, 1))

        # release to equilibrium and hold at equilibrium for t_settle seconds
        const_inputs = np.tile(const_input, (t_settle, 1))
        inputs = np.vstack((step_inputs, const_inputs)) # concatenate the step inputs and the return to constant

        ids = np.full(t_step + t_settle, i + 1)
        ids = ids[:, np.newaxis]
        inputs = np.hstack((ids, inputs))
        inputs_df = pd.DataFrame(inputs, columns = control_inputs_df.columns)
        control_inputs_df = pd.concat([control_inputs_df, inputs_df], ignore_index=True)


    return control_inputs_df


# for creating smooth random control trajectories
def perlin_noise_sampling(control_variables, seed, tip_radius = 80, mid_radius = 50, base_radius = 30, n_samples=15000):
    control_inputs_df = pd.DataFrame(columns=['ID'] + control_variables)
    
    n_octaves = 120 # more octaves = more peaks in the graph (less smooth)
    seeds = seed * np.arange(1,7) # one for each control input

    # max displacement from rest pos in degrees
    maxs = [mid_radius, tip_radius, base_radius, tip_radius, base_radius, mid_radius] # tip: 2, 4; mid: 1, 6; base: 3, 5
    mins = [-x for x in maxs]

    control_inputs = np.zeros((n_samples, 6))
    for i in range(len(seeds)):
        noise = PerlinNoise(octaves=n_octaves, seed=int(seeds[i])) # noise for each control input
        ctrl = np.array([noise(x / n_samples) for x in range(n_samples)])

        # normalize perlin noise to safe range
        ctrl = (ctrl - ctrl.min()) / (ctrl.max() - ctrl.min()) # normalize from 0 to 1
        ctrl = ctrl * (maxs[i] - mins[i]) + mins[i] # scale from min to max
        control_inputs[:,i] = ctrl

    single_motor = True # true if you want all other motor inputs to be zero, default false
    active_motor = 6
    if single_motor:
        mask = np.zeros(6)
        mask[active_motor - 1] = 1
        control_inputs = control_inputs * mask # keep only column of active motor, rest set to 0

    # convert to df
    ids = np.arange(n_samples)
    ids = ids[:, np.newaxis]
    inputs = np.hstack((ids, control_inputs))
    inputs_df = pd.DataFrame(inputs, columns = control_inputs_df.columns)
    control_inputs_df = pd.concat([control_inputs_df, inputs_df], ignore_index=True)

    # plot the control inputs
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # 2 rows, 3 columns

    for i, ax in enumerate(axes.flat):
        ax.plot(control_inputs[:,i]) 
        ax.set_title(f"U_{i+1}") 

    plt.tight_layout()
    plt.show()

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


def circle_sampling(control_variables, random_seed, tip_radius = 50, mid_radius = 30, base_radius = 10, phase_shift=0.0, noise_amplitude=0.00, num_samples_on_circle=2*500, num_circles = 2):
    np.random.seed(random_seed)

    sampled_angles = np.linspace(0, num_circles*2*np.pi, num_samples_on_circle*num_circles + 1) + phase_shift  # CHange back to 2*np.pi to get only one circle no flipping for now
    sampled_angles = sampled_angles[:-1] # cut off the last value (repeated since 0 deg = 360 deg)
    print(sampled_angles * 180/np.pi)
    # sampled_angles_fwd = np.linspace(0, 2*np.pi, num_samples_on_circle)
    # sampled_angles_bkwd = sampled_angles_fwd[::-1] # flip it
    # sampled_angles = np.concatenate((sampled_angles_fwd, sampled_angles_bkwd))

    offset = (1/6)*np.pi # 30 degree angle offset between cables
    # set control inputs based on geometry of cable arrangement - circle should start in direction of +u4
    u4s = tip_radius * np.cos(sampled_angles)
    u2s = - tip_radius * np.sin(sampled_angles)
    u1s = - mid_radius * np.cos(sampled_angles + offset)
    u6s = mid_radius * np.sin(sampled_angles + offset)
    u5s = base_radius * np.cos(sampled_angles + 2*offset)
    u3s = - base_radius * np.sin(sampled_angles + 2*offset)

    circle_samples = np.column_stack((u1s, u2s, u3s, u4s, u5s, u6s))
    circle_samples += np.random.uniform(-noise_amplitude, noise_amplitude, circle_samples.shape) # add noise if noise_amplitude != 0

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
    tip_range, mid_range, base_range = 0.45, 0.35, 0.3

    u1, u2, u3, u4, u5, u6 = u_opt[0], u_opt[1], u_opt[2], u_opt[3], u_opt[4], u_opt[5]

    # First we clip to max and min values
    u1 = np.clip(u1, -tip_range, tip_range)
    u6 = np.clip(u6, -tip_range, tip_range)
    u2 = np.clip(u2, -mid_range, mid_range)
    u4 = np.clip(u5, -mid_range, mid_range)
    u3 = np.clip(u3, -base_range, base_range)
    u5 = np.clip(u4, -base_range, base_range)

    # Compute control input vectors
    u1_vec = u1 * np.array([-np.cos(15 * np.pi/180), np.sin(15 * np.pi/180)])
    u2_vec = u2 * np.array([np.cos(45 * np.pi/180), np.sin(45 * np.pi/180)])
    u3_vec = u3 * np.array([-np.cos(15 * np.pi/180), -np.sin(15 * np.pi/180)])
    u4_vec = u4 * np.array([-np.cos(45 * np.pi/180), np.sin(45 * np.pi/180)])
    u5_vec = u5 * np.array([np.cos(75 * np.pi/180), -np.sin(75 * np.pi/180)])
    u6_vec = u6 * np.array([-np.cos(75 * np.pi/180), -np.sin(75 * np.pi/180)])

    # Calculate the norm based on the constraint
    vector_sum = (
        0.75 * (u3_vec + u5_vec) +
        1.0 * (u2_vec + u4_vec) +
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

def plot_trajectory_graphs(control_inputs_df, control_variables, focus_node_idx=None):
    controls = control_inputs_df[control_variables].values
    n_steps = controls.shape[0]

    # Color map setup
    cmap = plt.get_cmap('plasma')
    c_norm = mcolors.Normalize(vmin=0, vmax=n_steps)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cmap)
    colors = [scalar_map.to_rgba(i) for i in range(n_steps)]

    # PCA
    pca = PCA(n_components=2)
    controls_pca = pca.fit_transform(controls)

    # Extract key dimensions
    idx_u1 = control_variables.index('u1')
    idx_u2 = control_variables.index('u2')
    idx_u3 = control_variables.index('u3')
    idx_u4 = control_variables.index('u4')
    idx_u5 = control_variables.index('u5')
    idx_u6 = control_variables.index('u6')

    u1, u2, u3, u4, u5, u6 = controls[:, idx_u1], controls[:, idx_u2], controls[:, idx_u3], controls[:, idx_u4], controls[:, idx_u5], controls[:, idx_u6]

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    def plot_path(ax, x, y, title, xlabel, ylabel):
        ax.scatter(x, y, s=10, color='gray', alpha=0.6)
        for i in range(n_steps - 1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis('equal')

    plot_path(axs[0, 0], controls_pca[:, 0], controls_pca[:, 1], "Trajectory in PCA Space", "PC1", "PC2")
    plot_path(axs[0, 1], u2, u4, "Trajectory in U2, U4 (Tip) Space", "U2", "U4")
    plot_path(axs[1, 0], u1, u6, "Trajectory in U1, U6 (Mid) Space", "U1", "U6")
    plot_path(axs[1, 1], u3, u5, "Trajectory in U3, U5 (Base) Space", "U3", "U5")
    plt.tight_layout()
    plt.show()

    if focus_node_idx is not None:
        # Find the 20 transitions that end at the first appearance of that node
        target_node = control_inputs_df.loc[focus_node_idx, control_variables].values
        matches = (controls == target_node).all(axis=1)
        target_indices = np.where(matches)[0][:20]

        fig2, axs2 = plt.subplots(2, 2, figsize=(16, 12))

        def plot_focus(ax, x, y, title, xlabel, ylabel):
            ax.scatter(x, y, s=10, color='gray', alpha=0.3)
            for idx in target_indices:
                if idx == 0:
                    continue  # skip if no prior step
                ax.plot([x[idx - 1], x[idx]], [y[idx - 1], y[idx]], color='red', linewidth=2)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axis('equal')

        plot_focus(axs2[0, 0], controls_pca[:, 0], controls_pca[:, 1], "Focus Transitions in PCA Space", "PC1", "PC2")
        plot_focus(axs2[0, 1], u2, u4, "Focus Transitions in U2, U4 (Tip) Space", "U2", "U4")
        plot_focus(axs2[1, 0], u1, u6, "Focus Transitions in U1, U6 (Mid) Space", "U1", "U6")
        plot_focus(axs2[1, 1], u3, u5, "Focus Transitions in U3, U5 (Base) Space", "U3", "U5")

        plt.tight_layout()
        plt.show()


def main(data_type, sampling_type, seed=None):
    control_variables = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    # data_dir for mark's mac starts with '/Users/markleone/Documents/Stanford/ASL/trunk-stack/stack/main/data'
    # data_dir for workstation is '/home/trunk/Documents/trunk-stack/stack/main/data'
    data_dir = os.getenv('TRUNK_DATA', '/home/trunk/Documents/trunk-stack/stack/main/data')
    if seed is not None:
        control_inputs_file = os.path.join(data_dir, f'trajectories/{data_type}/control_inputs_{sampling_type}_{seed}.csv')
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
    elif sampling_type == 'adiabatic_global':
        control_inputs_df = adiabatic_global_sampling(control_variables, seed)
    elif sampling_type == 'random_smooth':
        control_inputs_df = perlin_noise_sampling(control_variables, seed)
    elif sampling_type == 'latin_hypercube':
        control_inputs_df = latin_hypercube_adiabatic_sampling(control_variables, seed)
    elif sampling_type == 'latin_hypercube_controlled':
        control_inputs_df = hypercube_controlled_sampling(control_variables, seed)
    elif sampling_type == 'ol_test_high_z':
        control_inputs_df = OL_test_high_z_sampling(control_variables, seed)
    else:
        raise ValueError(f"Invalid sampling_type: {sampling_type}")

    save_to_csv(control_inputs_df, control_inputs_file)
    visualize_samples(control_inputs_df)


if __name__ == '__main__':
    data_type = 'dynamic'                   # 'steady_state' or 'dynamic'
    sampling_type = 'latin_hypercube_controlled'      # 'circle', 'beta', 'targeted', 'uniform', 'sinusoidal', 'adiabatic_manual', 'adiabatic_step', 'adiabatic_global', 'random_smooth', 'latin_hypercube', or 'ol_test_high_z'
    seed = 10                            # choose integer seed number
    main(data_type, sampling_type, seed)
