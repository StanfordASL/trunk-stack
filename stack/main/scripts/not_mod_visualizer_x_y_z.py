import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/trunk/Documents/trunk-stack/stack/main/data/trajectories/dynamic/trialv2-201_zeropad_not_mod_1.csv')

t = df.iloc[:, 0]
x = df.iloc[:, 1]
y = df.iloc[:, 2]
z = df.iloc[:, 3]

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(t, x, label='X Position', color='r')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('X Position')
axs[0].set_title('X Position over Time')
axs[0].legend()

axs[1].plot(t, y, label='Y Position', color='g')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Y Position')
axs[1].set_title('Y Position over Time')
axs[1].legend()

axs[2].plot(t, z, label='Z Position', color='b')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Z Position')
axs[2].set_title('Z Position over Time')
axs[2].legend()

x = df.iloc[:, 1]
y = df.iloc[:, 2]
z = df.iloc[:, 3]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='3D Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectory Visualization')
ax.legend()

plt.tight_layout()
plt.show()