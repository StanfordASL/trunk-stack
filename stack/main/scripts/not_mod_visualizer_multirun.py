import argparse, glob, os
import pandas as pd
import matplotlib.pyplot as plt

def pick_cols(df, time_col = None, xyz_cols = None):
    if time_col and time_col in df.columns:
        t = df[time_col]
    else:
        t = df.iloc[:, 0]
    
    if xyz_cols and all(c in df.columns for c in xyz_cols):
        x, y, z = (df[xyz_cols[0]], df[xyz_cols[1]], df[xyz_cols[2]])
    else:
        x, y, z = df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3]
    return t, x, y, z

def main():
    p = argparse.ArgumentParser(description="Plot x,y,z vs time from multiple CSVx")
    p.add_argument("paths", nargs="+", help="CSV paths or globs (e.g., data/*.csv)")
    p.add_argument("--time-col", help="Name of time column (default:first column)")
    p.add_argument("--xyz-cols", nargs=3, metavar=("X","Y","Z"),
                   help="Names of x/y/z columns (default: columns 2-4)")
    args = p.parse_args()

    files = []
    for pat in args.paths:
        matched = sorted(glob.glob(pat))
        if matched:
            files.extend(matched)
        elif os.path.isfile(pat):
            files.append(pat)
    if not files:
        raise SystemExit("No CSV files found for given paths/globs.")
    
    fig, axs = plt.subplots(3, 1, figsize=(11,8), sharex=True)
    xlab = "time"

    for f in files:
        df = pd.read_csv(f)
        t, x, y, z = pick_cols(df, args.time_col, args.xyz_cols)

        s = t.argsort(kind = "mergesort")
        t, x, y, z = t.iloc[s], x.iloc[s], y.iloc[s], z.iloc[s]

        label = os.path.basename(f)
        axs[0].plot(t, x, label=label)
        axs[1].plot(t, y, label=label)
        axs[2].plot(t, z, label=label)
    
    axs[0].set_ylabel(args.xyz_cols[0] if args.xyz_cols else "x")
    axs[0].set_title("x vs time (multiple runs)")
    axs[1].set_ylabel(args.xyz_cols[1] if args.xyz_cols else "y")
    axs[1].set_title("y vs time (multiple runs)")
    axs[2].set_ylabel(args.xyz_cols[2] if args.xyz_cols else "z")
    axs[2].set_title("z vs time (multiple runs)")
    axs[2].set_xlabel(xlab)

    for ax in axs:
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "main":
    main()
