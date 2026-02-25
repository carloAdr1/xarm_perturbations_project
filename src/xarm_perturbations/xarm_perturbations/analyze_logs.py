#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rmse(a):
    return math.sqrt(np.mean(np.square(a)))


def load_csv(path):
    df = pd.read_csv(path)
    # numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def resample_to_timebase(t_ref, t_src, x_src):
    """
    Linear interpolation of x_src(t_src) evaluated at t_ref.
    """
    # Ensure increasing time
    order = np.argsort(t_src)
    t_src = np.array(t_src)[order]
    x_src = np.array(x_src)[order]
    # Clamp t_ref to src bounds
    t_ref = np.clip(t_ref, t_src[0], t_src[-1])
    return np.interp(t_ref, t_src, x_src)


def analyze_one(csv_path, out_dir):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    df = load_csv(csv_path)

    # Use actual timebase
    t = df["t_actual"].to_numpy()

    # Actual
    xa = df["x_act"].to_numpy()
    ya = df["y_act"].to_numpy()
    za = df["z_act"].to_numpy()

    # Desired sampled at its own timestamps, interpolate to actual timebase
    td = df["t_des"].to_numpy()
    xd = resample_to_timebase(t, td, df["x_des"].to_numpy())
    yd = resample_to_timebase(t, td, df["y_des"].to_numpy())
    zd = resample_to_timebase(t, td, df["z_des"].to_numpy())

    # Commanded velocity magnitude (interpolate cmd to actual timebase)
    tc = df["t_cmd"].to_numpy()
    vxc = resample_to_timebase(t, tc, df["vx_cmd"].to_numpy())
    vyc = resample_to_timebase(t, tc, df["vy_cmd"].to_numpy())
    vzc = resample_to_timebase(t, tc, df["vz_cmd"].to_numpy())
    vmag = np.sqrt(vxc**2 + vyc**2 + vzc**2)

    # Errors
    ex = xd - xa
    ey = yd - ya
    ez = zd - za
    en = np.sqrt(ex**2 + ey**2 + ez**2)

    metrics = {
        "rmse_x": rmse(ex),
        "rmse_y": rmse(ey),
        "rmse_z": rmse(ez),
        # total RMSE as norm RMSE (common and meaningful)
        "rmse_total": rmse(en),
        # max absolute position error as max norm
        "max_abs_error": float(np.max(en)),
        # optional: per-axis maxima
        "max_abs_x": float(np.max(np.abs(ex))),
        "max_abs_y": float(np.max(np.abs(ey))),
        "max_abs_z": float(np.max(np.abs(ez))),
    }

    os.makedirs(out_dir, exist_ok=True)

    # Plot desired vs actual (one plot per axis)
    for axis, a_des, a_act in [("x", xd, xa), ("y", yd, ya), ("z", zd, za)]:
        plt.figure()
        plt.plot(t - t[0], a_des, label=f"{axis}_desired")
        plt.plot(t - t[0], a_act, label=f"{axis}_actual")
        plt.xlabel("time (s)")
        plt.ylabel(f"{axis} (m)")
        plt.title(f"{name}: desired vs actual ({axis})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_pos_{axis}.png"), dpi=160)
        plt.close()

    # Plot error over time (norm + axes)
    plt.figure()
    plt.plot(t - t[0], ex, label="ex")
    plt.plot(t - t[0], ey, label="ey")
    plt.plot(t - t[0], ez, label="ez")
    plt.plot(t - t[0], en, label="||e||")
    plt.xlabel("time (s)")
    plt.ylabel("error (m)")
    plt.title(f"{name}: error over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_error.png"), dpi=160)
    plt.close()

    # Plot commanded velocity magnitude
    plt.figure()
    plt.plot(t - t[0], vmag)
    plt.xlabel("time (s)")
    plt.ylabel("|v_cmd| (m/s)")
    plt.title(f"{name}: commanded velocity magnitude")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_vmag.png"), dpi=160)
    plt.close()

    return metrics


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=os.path.expanduser("~/xarm_logs"))
    ap.add_argument("--out_dir", default=os.path.expanduser("~/xarm_logs/plots"))
    args = ap.parse_args()

    experiments = ["baseline", "sine", "gaussian"]
    results = {}

    for exp in experiments:
        csv_path = os.path.join(args.log_dir, f"{exp}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing: {csv_path}")
            continue
        metrics = analyze_one(csv_path, args.out_dir)
        results[exp] = metrics

    # Print comparison table
    if results:
        print("\n=== Metrics Comparison ===")
        header = ["exp", "rmse_x", "rmse_y", "rmse_z", "rmse_total", "max_abs_error"]
        print(",".join(header))
        for exp, m in results.items():
            print(",".join([
                exp,
                f"{m['rmse_x']:.6f}",
                f"{m['rmse_y']:.6f}",
                f"{m['rmse_z']:.6f}",
                f"{m['rmse_total']:.6f}",
                f"{m['max_abs_error']:.6f}",
            ]))
        print(f"\nPlots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
