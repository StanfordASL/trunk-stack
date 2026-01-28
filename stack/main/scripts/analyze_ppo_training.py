#!/usr/bin/env python3
"""
Analyze PPO training logs produced by TrainRLExecutor.

Reads:
  - metrics.csv (required)
  - rollouts/*.csv (optional; only used for extra sanity checks)

Plots:
  - ISE over episodes
  - Episode return over episodes
  - Rolling averages

Usage:
  python analyze_ppo_training.py \
    --ppo_root /home/trunk/Documents/trunk-stack/stack/main/data/ppo \
    --window 20 \
    --save /tmp/ppo_training.png
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(metrics_path: str) -> pd.DataFrame:
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    df = pd.read_csv(metrics_path)

    # Your file includes both episode rows and "update rows" with episode_idx_in_rollout = -1
    # Keep only actual episodes
    df_ep = df[df["episode_idx_in_rollout"] >= 1].copy()

    # Sort robustly
    sort_cols = ["rollout_idx", "episode_idx_in_rollout", "timestamp_unix"]
    sort_cols = [c for c in sort_cols if c in df_ep.columns]
    df_ep = df_ep.sort_values(sort_cols).reset_index(drop=True)

    # Add a global episode counter
    df_ep["episode_global"] = np.arange(1, len(df_ep) + 1)

    return df_ep


def rolling_mean(x: pd.Series, w: int) -> pd.Series:
    if w <= 1:
        return x
    return x.rolling(window=w, min_periods=max(1, w // 4)).mean()


def add_rollout_boundaries(ax, df_ep: pd.DataFrame):
    if "rollout_idx" not in df_ep.columns or len(df_ep) == 0:
        return
    # Vertical lines at rollout changes
    changes = df_ep["rollout_idx"].ne(df_ep["rollout_idx"].shift(1))
    idxs = df_ep.loc[changes, "episode_global"].tolist()
    for e in idxs[1:]:
        ax.axvline(e, linewidth=1, alpha=0.25)


def summarize(df_ep: pd.DataFrame):
    # Basic stats
    out = {}
    if len(df_ep) == 0:
        return out

    out["n_episodes"] = int(len(df_ep))
    out["n_rollouts"] = int(df_ep["rollout_idx"].nunique()) if "rollout_idx" in df_ep.columns else None

    if "episode_ise" in df_ep.columns:
        out["ise_min"] = float(df_ep["episode_ise"].min())
        out["ise_median"] = float(df_ep["episode_ise"].median())
        out["ise_last"] = float(df_ep["episode_ise"].iloc[-1])

        best_row = df_ep.loc[df_ep["episode_ise"].idxmin()]
        out["best_ise_episode_global"] = int(best_row["episode_global"])
        out["best_ise_rollout"] = int(best_row["rollout_idx"]) if "rollout_idx" in best_row else None
        out["best_ise_episode_in_rollout"] = int(best_row["episode_idx_in_rollout"]) if "episode_idx_in_rollout" in best_row else None

    if "episode_return" in df_ep.columns:
        out["return_max"] = float(df_ep["episode_return"].max())
        out["return_median"] = float(df_ep["episode_return"].median())
        out["return_last"] = float(df_ep["episode_return"].iloc[-1])

        best_row = df_ep.loc[df_ep["episode_return"].idxmax()]
        out["best_return_episode_global"] = int(best_row["episode_global"])
        out["best_return_rollout"] = int(best_row["rollout_idx"]) if "rollout_idx" in best_row else None
        out["best_return_episode_in_rollout"] = int(best_row["episode_idx_in_rollout"]) if "episode_idx_in_rollout" in best_row else None

    return out


def plot_training(df_ep: pd.DataFrame, window: int, save_path: str = None, show: bool = True):
    if len(df_ep) == 0:
        raise RuntimeError("No episode rows found in metrics.csv (episode_idx_in_rollout >= 1).")

    # Make plots
    fig1 = plt.figure()
    ax1 = plt.gca()
    ax1.plot(df_ep["episode_global"], df_ep["episode_ise"], label="ISE (episode)")
    ax1.plot(df_ep["episode_global"], rolling_mean(df_ep["episode_ise"], window), label=f"ISE rolling mean (w={window})")
    ax1.set_xlabel("Episode (global)")
    ax1.set_ylabel("ISE (∫ ||p - p*||^2 dt)")
    ax1.set_title("Tracking error over training")
    ax1.grid(True, alpha=0.3)
    add_rollout_boundaries(ax1, df_ep)
    ax1.legend()

    fig2 = plt.figure()
    ax2 = plt.gca()
    ax2.plot(df_ep["episode_global"], df_ep["episode_return"], label="Episode return")
    ax2.plot(df_ep["episode_global"], rolling_mean(df_ep["episode_return"], window), label=f"Return rolling mean (w={window})")
    ax2.set_xlabel("Episode (global)")
    ax2.set_ylabel("Episode return (sum of rewards)")
    ax2.set_title("Reward over training")
    ax2.grid(True, alpha=0.3)
    add_rollout_boundaries(ax2, df_ep)
    ax2.legend()

    if save_path:
        base, ext = os.path.splitext(save_path)
        if ext.lower() not in [".png", ".jpg", ".jpeg", ".pdf", ".svg"]:
            ext = ".png"
        fig1.savefig(base + "_ise" + ext, dpi=200, bbox_inches="tight")
        fig2.savefig(base + "_return" + ext, dpi=200, bbox_inches="tight")
        print(f"Saved: {base + '_ise' + ext}")
        print(f"Saved: {base + '_return' + ext}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo_root", type=str, required=True,
                    help="Path to .../data/ppo (the folder that contains metrics/ and rollouts/).")
    ap.add_argument("--window", type=int, default=20, help="Rolling window for smoothing.")
    ap.add_argument("--save", type=str, default=None,
                    help="If set, saves plots. Two files will be written: *_ise and *_return.")
    ap.add_argument("--no_show", action="store_true", help="Do not open interactive plot window.")
    args = ap.parse_args()

    metrics_path = os.path.join(args.ppo_root, "metrics", "metrics.csv")
    df_ep = load_metrics(metrics_path)

    print("\n=== PPO Training Summary ===")
    s = summarize(df_ep)
    for k, v in s.items():
        print(f"{k}: {v}")

    # Quick “is it improving?” heuristic: compare early vs late rolling means
    if "episode_ise" in df_ep.columns and len(df_ep) >= max(10, args.window * 2):
        early = df_ep["episode_ise"].iloc[:args.window].mean()
        late = df_ep["episode_ise"].iloc[-args.window:].mean()
        print(f"\nISE early(mean first {args.window} eps): {early:.6g}")
        print(f"ISE late (mean last  {args.window} eps): {late:.6g}")
        print(f"ISE improvement factor (early/late): {early/late:.3f}" if late > 0 else "ISE late is 0?")

    if "episode_return" in df_ep.columns and len(df_ep) >= max(10, args.window * 2):
        early = df_ep["episode_return"].iloc[:args.window].mean()
        late = df_ep["episode_return"].iloc[-args.window:].mean()
        print(f"\nReturn early(mean first {args.window} eps): {early:.6g}")
        print(f"Return late (mean last  {args.window} eps): {late:.6g}")

    plot_training(df_ep, window=args.window, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
