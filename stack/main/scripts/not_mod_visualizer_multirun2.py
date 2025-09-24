import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_path = "/home/trunk/Documents/trunk-stack/stack/main/data/trajectories/dynamic/"
base_name = "trialv2-201_zeropad_not_mod_"

# === Figure 1: time plots ===
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# === Figure 2: 3D plot ===
fig3d = plt.figure(figsize=(8, 6))
ax3d = fig3d.add_subplot(111, projection="3d")

# Store all trials for mean/std computation
all_x, all_y, all_z = [], [], []
all_3d = []

for i in range(1, 6):  # loop over 5 trials
    filename = f"{base_path}{base_name}{i}.csv"
    df = pd.read_csv(filename)

    t = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    z = df.iloc[:, 3].values

    all_x.append(x)
    all_y.append(y)
    all_z.append(z)
    all_3d.append((x, y, z))

# Convert to numpy arrays for easy computation
all_x = np.array(all_x)
all_y = np.array(all_y)
all_z = np.array(all_z)

# Function to plot time series with mean ± std
def plot_with_mean_std(ax, t, data, color, label):
    # Plot all trials transparent
    for trial in data:
        ax.plot(t, trial, color=color, alpha=0.2)
    # Compute mean and std
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Plot mean
    ax.plot(t, mean, color=color, label=f"{label} Mean", alpha=1.0)
    # Optional: fill between std
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.3)

# Plot X, Y, Z
plot_with_mean_std(axs[0], t, all_x, 'red', 'X')
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("X Position")
axs[0].set_title("X Position over Time")
axs[0].legend()

plot_with_mean_std(axs[1], t, all_y, 'green', 'Y')
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Y Position")
axs[1].set_title("Y Position over Time")
axs[1].legend()

plot_with_mean_std(axs[2], t, all_z, 'blue', 'Z')
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Z Position")
axs[2].set_title("Z Position over Time")
axs[2].legend()

# 3D plot: all trajectories transparent, mean trajectory opaque purple
for x, y, z in all_3d:
    ax3d.plot(x, y, z, color='purple', alpha=0.2)

mean_x = np.mean(all_x, axis=0)
mean_y = np.mean(all_y, axis=0)
mean_z = np.mean(all_z, axis=0)
ax3d.plot(mean_x, mean_y, mean_z, color='purple', alpha=1.0, label="Mean Trajectory")

ax3d.set_xlabel("X Position")
ax3d.set_ylabel("Y Position")
ax3d.set_zlabel("Z Position")
ax3d.set_title("3D Trajectory Visualization")
ax3d.legend()

plt.tight_layout()
plt.show()
