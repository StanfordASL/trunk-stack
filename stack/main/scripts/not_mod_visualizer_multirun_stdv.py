import pandas as pd
import numpy as np

base_path = "/home/trunk/Documents/trunk-stack/stack/main/data/trajectories/dynamic/"
base_name = "trialv2-201_zeropad_not_mod_"

results = []

for i in range(1, 11):
    filename = f"{base_path}{base_name}{i}.csv"
    df = pd.read_csv(filename)

    x = df.iloc[:, 1].to_numpy()
    y = df.iloc[:, 2].to_numpy()
    z = df.iloc[:, 3].to_numpy()

    std_x = np.std(x)
    std_y = np.std(y)
    std_z = np.std(z)

    results.append([i, std_x, std_y, std_z])

# Convert to a nice table
import pandas as pd
results_df = pd.DataFrame(results, columns=["Trial", "Std_X", "Std_Y", "Std_Z"])
print(results_df)
import matplotlib.pyplot as plt
from tabulate import tabulate

print(tabulate(results_df, headers="keys", tablefmt="pretty", floatfmt=".5f"))

# Plot bar chart
fig, ax = plt.subplots(figsize=(10,6))

width = 0.25
trials = results_df["Trial"]

ax.bar(trials - 0.25, results_df["Std_X"], width, label="Std X")
ax.bar(trials,         results_df["Std_Y"], width, label="Std Y")
ax.bar(trials + 0.25, results_df["Std_Z"], width, label="Std Z")

ax.set_xlabel("Trial")
ax.set_ylabel("Standard Deviation")
ax.set_title("Per-Trial Standard Deviations of X, Y, Z")
ax.legend()
plt.show()
plt.figure(figsize=(10,6))
plt.plot(trials, results_df["Std_X"], marker="o", label="Std X")
plt.plot(trials, results_df["Std_Y"], marker="o", label="Std Y")
plt.plot(trials, results_df["Std_Z"], marker="o", label="Std Z")

plt.xlabel("Trial")
plt.ylabel("Standard Deviation")
plt.title("Standard Deviation per Trial")
plt.legend()
plt.grid(True)
plt.show()

