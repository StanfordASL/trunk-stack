import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

base_path = "/home/trunk/Documents/trunk-stack/stack/main/data/trajectories/dynamic/"
base_name = "trialv2-201_zeropad_not_mod_"
n_trials = 5

xs = []
ys = []
zs = []
times = []

# load all trials
for i in range(1, n_trials + 1):
    filename = f"{base_path}{base_name}{i}.csv"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} not found")
    df = pd.read_csv(filename)

    # same column indexing you had: columns 1,2,3 are x,y,z
    x = df.iloc[:, 1].to_numpy()
    y = df.iloc[:, 2].to_numpy()
    z = df.iloc[:, 3].to_numpy()
    xs.append(x)
    ys.append(y)
    zs.append(z)

    # try to capture time column if present in col 0
    times.append(df.iloc[:, 0].to_numpy())

# align lengths by trimming to minimum length across all trials
min_len = min(map(len, xs + ys + zs))
xs = np.array([x[:min_len] for x in xs])  # shape: (n_trials, T)
ys = np.array([y[:min_len] for y in ys])
zs = np.array([z[:min_len] for z in zs])

# decide time axis: use first trial's time column if all time arrays are "close", else use sample index
time_axis = None
if all(len(t) >= min_len for t in times):
    # check if time columns are nearly equal across trials (within tolerance)
    all_time_trimmed = np.array([t[:min_len] for t in times])
    if np.allclose(all_time_trimmed, all_time_trimmed[0], rtol=1e-6, atol=1e-8):
        time_axis = all_time_trimmed[0]
if time_axis is None:
    time_axis = np.arange(min_len)

# compute statistics across trials (axis=0 is the time axis after stacking trials into rows)
# xs shape: (n_trials, T) -> np.std(xs, axis=0) gives length-T vector: std at each time point
std_x = np.std(xs, axis=0)    # default ddof=0; use ddof=1 for sample std if you prefer
std_y = np.std(ys, axis=0)
std_z = np.std(zs, axis=0)

mean_x = np.mean(xs, axis=0)
mean_y = np.mean(ys, axis=0)
mean_z = np.mean(zs, axis=0)

# put into a DataFrame
results_df = pd.DataFrame({
    "time": time_axis,
    "std_x": std_x,
    "std_y": std_y,
    "std_z": std_z,
    "mean_x": mean_x,
    "mean_y": mean_y,
    "mean_z": mean_z
})

print("Per-time-point std (first 10 rows):")
print(results_df.head(10))

# save if you want
results_df.to_csv(os.path.join(base_path, "per_time_std_xyz.csv"), index=False)

# Plot std over time
plt.figure(figsize=(12,6))
plt.plot(time_axis, std_x, marker='.', label='std_x')
plt.plot(time_axis, std_y, marker='.', label='std_y')
plt.plot(time_axis, std_z, marker='.', label='std_z')
plt.xlabel("time (or sample index)")
plt.ylabel("Std across trials")
plt.title("Standard Deviation at each timepoint (across trials)")
plt.legend()
plt.grid(True)
plt.show()

# Plot mean +/- std x
plt.figure(figsize=(12,6))
plt.plot(time_axis, mean_x, label='mean_x')
plt.fill_between(time_axis, mean_x - std_x, mean_x + std_x, alpha=0.25, label='mean ± std')
plt.xlabel("time (or sample index)")
plt.ylabel("X (mean ± std across trials)")
plt.title("Mean and variability of X across trials")
plt.legend()
plt.grid(True)
plt.show()

# Y axis
plt.figure(figsize=(12,6))
plt.plot(time_axis, mean_y, label='mean_y', color='g')
plt.fill_between(time_axis, mean_y - std_y, mean_y + std_y, color='g', alpha=0.25, label='mean ± std')
plt.xlabel("time (or sample index)")
plt.ylabel("Y (mean ± std across trials)")
plt.title("Mean and variability of Y across trials")
plt.legend()
plt.grid(True)
plt.show()

# Z axis
plt.figure(figsize=(12,6))
plt.plot(time_axis, mean_z, label='mean_z', color='b')
plt.fill_between(time_axis, mean_z - std_z, mean_z + std_z, color='b', alpha=0.25, label='mean ± std')
plt.xlabel("time (or sample index)")
plt.ylabel("Z (mean ± std across trials)")
plt.title("Mean and variability of Z across trials")
plt.legend()
plt.grid(True)
plt.show()

# Mean ± std shading for all axes
plt.figure(figsize=(12,6))
plt.plot(time_axis, mean_x, label='mean_x', color='r')
plt.fill_between(time_axis, mean_x - std_x, mean_x + std_x, color='r', alpha=0.25)

plt.plot(time_axis, mean_y, label='mean_y', color='g')
plt.fill_between(time_axis, mean_y - std_y, mean_y + std_y, color='g', alpha=0.25)

plt.plot(time_axis, mean_z, label='mean_z', color='b')
plt.fill_between(time_axis, mean_z - std_z, mean_z + std_z, color='b', alpha=0.25)

plt.xlabel("time (or sample index)")
plt.ylabel("Mean ± Std across trials")
plt.title("Mean and Variability of X, Y, Z across Trials")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Choose a colormap for 10 trials
colors = cm.tab10(np.linspace(0, 1, n_trials))

plt.figure(figsize=(12,6))

# Plot mean signal as dark, thick line
plt.plot(time_axis, mean_x, color='k', linewidth=2.5, label='mean_x')

# Overlay all 10 trials as points
for i in range(n_trials):
    plt.scatter(time_axis, xs[i], color=colors[i], s=15, label=f'Trial {i+1}')

plt.xlabel("time (or sample index)")
plt.ylabel("X signal")
plt.title("Mean X signal with all 10 trial points")
plt.legend(ncol=2, fontsize=9)
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))

# Plot all 10 trials as transparent red lines
for i in range(n_trials):
    plt.plot(time_axis, xs[i], color='red', alpha=0.5)

# Plot mean signal as solid, opaque red line on top
plt.plot(time_axis, mean_x, color='red', linewidth=2.5, label='Mean X')

plt.xlabel("time (or sample index)")
plt.ylabel("X signal")
plt.title("Mean X signal with all 10 trials overlay")
plt.legend()
plt.grid(True)
plt.show()


