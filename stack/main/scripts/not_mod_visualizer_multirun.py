import pandas as pd
import matplotlib.pyplot as plt

base_path = "/home/trunk/Documents/trunk-stack/stack/main/data/trajectories/dynamic/"
base_name = "trialv2-201_zeropad_not_mod_"

# === Figure 1: time plots ===
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# === Figure 2: 3D plot ===
fig3d = plt.figure(figsize=(8, 6))
ax3d = fig3d.add_subplot(111, projection="3d")

for i in range(1, 11):  # loop over 10 trials
    filename = f"{base_path}{base_name}{i}.csv"
    df = pd.read_csv(filename)

    t = df.iloc[:, 0]
    x = df.iloc[:, 1]
    y = df.iloc[:, 2]
    z = df.iloc[:, 3]

    # Time plots
    axs[0].plot(t, x, label=f"Trial {i}")
    axs[1].plot(t, y, label=f"Trial {i}")
    axs[2].plot(t, z, label=f"Trial {i}")

    # 3D trajectory
    ax3d.plot(x, y, z, label=f"Trial {i}")

# Format time plots
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("X Position")
axs[0].set_title("X Position over Time")
axs[0].legend()

axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Y Position")
axs[1].set_title("Y Position over Time")
axs[1].legend()

axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Z Position")
axs[2].set_title("Z Position over Time")
axs[2].legend()

# Format 3D plot
ax3d.set_xlabel("X Position")
ax3d.set_ylabel("Y Position")
ax3d.set_zlabel("Z Position")
ax3d.set_title("3D Trajectory Visualization")
ax3d.legend()

plt.tight_layout()
plt.show()
