#!/usr/bin/env python3
"""
Fit a 2D exponential model to K.csv data on a pressure–power grid.
Model: I = A * p^alpha * P^beta * exp(-B p - C P)
This is an alternative to the rational model; it is separable.
Usage: python fit_2d_grid_exponential.py /path/to/data_root
"""

import os
import sys
import csv
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Mapping of column names (as they appear in K.csv) to wavelength (nm)
# ----------------------------------------------------------------------
COLUMN_INFO = {
    "r549: O_1s -> O_1d":   557.7,
    "r550: O_5s -> O":      135.6,
    "r551: O_5s -> O":      135.9,
    "r552: O_3s -> O":      130.2,
    "r553: O_3s -> O":      130.5,
    "r554: O_3s -> O":      130.6,
    "r555: O_5p -> O_5s":   777.2,
    "r556: O_5p -> O_5s":   777.4,
    "r557: O_5p -> O_5s":   777.5,
    "r558: O_3p -> O_3s":   844.6,
    "r559: O_3p -> O_3s":   844.6,
    "r560: O_3p -> O_3s":   844.7,
}

WAVELENGTH_THRESHOLD = 200.0   # <200 nm = VUV

# ----------------------------------------------------------------------
# 2D exponential model (separable)
# ----------------------------------------------------------------------
def exponential_2d(P_grid, A, alpha, beta, B, C):
    """
    I = A * p^alpha * P^beta * exp(-B * p - C * P)
    P_grid: tuple (p, P) where p, P are arrays.
    """
    p, P = P_grid
    return A * (p ** alpha) * (P ** beta) * np.exp(-B * p - C * P)


def fit_line_exponential(power_grid, pressure_grid, intensity_grid):
    """Fit the exponential model to grid data. Returns popt, R^2."""
    P_flat = power_grid.flatten()
    p_flat = pressure_grid.flatten()
    I_flat = intensity_grid.flatten()

    mask = np.isfinite(I_flat)
    P_flat = P_flat[mask]
    p_flat = p_flat[mask]
    I_flat = I_flat[mask]

    if len(I_flat) < 5:   # we have 5 parameters
        return None, None

    # Normalize data
    I_scale = np.median(np.abs(I_flat)) if np.median(np.abs(I_flat)) > 0 else 1.0
    I_norm = I_flat / I_scale

    # Several initial guesses
    guess_list = [
        [1.0, 1.0, 1.0, 0.001, 0.001],
        [1.0, 0.5, 0.5, 0.0, 0.0],
        [1.0, 2.0, 2.0, 0.01, 0.01],
    ]

    best_popt = None
    best_r2 = -np.inf

    for p0 in guess_list:
        try:
            popt, _ = curve_fit(
                exponential_2d,
                (p_flat, P_flat),
                I_norm,
                p0=p0,
                maxfev=20000
            )
            I_pred_norm = exponential_2d((p_flat, P_flat), *popt)
            ss_res = np.sum((I_norm - I_pred_norm) ** 2)
            ss_tot = np.sum((I_norm - np.mean(I_norm)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 > best_r2:
                best_r2 = r2
                # de‑normalize A
                best_popt = list(popt)
                best_popt[0] = popt[0] * I_scale
        except Exception:
            continue

    if best_popt is not None and best_r2 > 0.5:
        return best_popt, best_r2
    else:
        return None, None


# ----------------------------------------------------------------------
# CSV saving helper
# ----------------------------------------------------------------------
def save_fitting_results(fitted_params, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Emission line", "A", "alpha", "beta", "B", "C", "R^2"])
        for col_name, (popt, r2) in fitted_params.items():
            A, alpha, beta, B, C = popt
            writer.writerow([col_name, A, alpha, beta, B, C, r2])
    print(f"Fitting results saved to {output_csv}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python fit_2d_grid_exponential.py /path/to/data_root")
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    if not root_dir.is_dir():
        print(f"Error: {root_dir} is not a directory.")
        sys.exit(1)

    # Discover folders
    power_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    power_levels = []
    pressure_levels_set = set()
    dir_map = {}

    for pwr_dir in power_dirs:
        pwr_name = pwr_dir.name.lower().rstrip('w')
        try:
            power = float(pwr_name)
        except ValueError:
            print(f"Skipping directory {pwr_dir.name}")
            continue
        power_levels.append(power)

        for pres_dir in sorted(pwr_dir.iterdir()):
            if not pres_dir.is_dir():
                continue
            parts = pres_dir.name.split('_')
            if len(parts) < 2:
                continue
            try:
                pressure = float(parts[1])
            except ValueError:
                continue
            pressure_levels_set.add(pressure)
            k_file = pres_dir / "K.csv"
            if k_file.is_file():
                dir_map[(power, pressure)] = k_file

    if not dir_map:
        print("No K.csv files found.")
        sys.exit(1)

    pressure_levels = sorted(pressure_levels_set)
    power_levels = sorted(power_levels)
    print(f"Found {len(power_levels)} power levels: {power_levels}")
    print(f"Found {len(pressure_levels)} pressure levels: {pressure_levels}")

    P_grid, p_grid = np.meshgrid(power_levels, pressure_levels)
    all_columns = list(COLUMN_INFO.keys())
    intensity_data = {col: np.full_like(P_grid, np.nan) for col in all_columns}

    # Fill grids
    for (power, pressure), filepath in dir_map.items():
        i_power = power_levels.index(power)
        i_press = pressure_levels.index(pressure)

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers is None:
                continue
            col_indices = {}
            for idx, h in enumerate(headers):
                h_clean = h.strip()
                if h_clean in all_columns:
                    col_indices[h_clean] = idx
            last_row = None
            for row in reader:
                if len(row) > 0 and any(cell.strip() for cell in row):
                    last_row = row
            if last_row is None:
                continue
            for col_name, idx in col_indices.items():
                try:
                    val = float(last_row[idx].strip())
                except (ValueError, IndexError):
                    val = np.nan
                intensity_data[col_name][i_press, i_power] = val

    # Output directory
    out_dir = root_dir / "fits_and_plots_exponential"
    out_dir.mkdir(exist_ok=True)

    # Fit each line
    fitted_params = {}
    for col_name in all_columns:
        print(f"\nFitting {col_name} (exponential model) ...")
        I_grid = intensity_data[col_name]
        popt, r2 = fit_line_exponential(P_grid, p_grid, I_grid)
        if popt is not None:
            fitted_params[col_name] = (popt, r2)
            A, alpha, beta, B, C = popt
            print(f"  R² = {r2:.5f}")
            print(f"  A={A:.4e}, α={alpha:.4f}, β={beta:.4f}, B={B:.4e}, C={C:.4e}")
        else:
            print("  Fit failed, skipping.")

    # Save CSV
    if fitted_params:
        output_csv = out_dir / "fitting_exponential.csv"
        save_fitting_results(fitted_params, output_csv)

    # Plot individual lines
    if fitted_params:
        p_fine = np.linspace(min(pressure_levels), max(pressure_levels), 100)
        P_fine = np.linspace(min(power_levels), max(power_levels), 100)
        Pp_fine, pp_fine = np.meshgrid(P_fine, p_fine)

        for col_name, (popt, r2) in fitted_params.items():
            I_surface = exponential_2d((pp_fine, Pp_fine), *popt)
            plt.figure(figsize=(8, 6))
            plt.contourf(P_fine, p_fine, I_surface, levels=20, cmap='viridis')
            plt.colorbar(label='Intensity (arb. units)')
            plt.xlabel('Power (W)')
            plt.ylabel('Pressure (mTorr)')
            plt.title(f'{col_name} (exponential model)\nR² = {r2:.4f}')
            # Overlay grid points
            for i, p in enumerate(pressure_levels):
                for j, P in enumerate(power_levels):
                    val = intensity_data[col_name][i, j]
                    if np.isfinite(val):
                        plt.plot(P, p, 'ko', markersize=3)
            safe_name = col_name.replace(':', '_').replace(' ', '_').replace('->', 'to')
            plt.tight_layout()
            plt.savefig(out_dir / f"{safe_name}.png", dpi=150)
            plt.close()
        print("Individual line plots saved.")

        # Aggregate VUV / non‑VUV / total
        vuv_cols = [col for col, wl in COLUMN_INFO.items() if wl < WAVELENGTH_THRESHOLD]
        nonvuv_cols = [col for col, wl in COLUMN_INFO.items() if wl >= WAVELENGTH_THRESHOLD]

        vuv_surface = np.zeros_like(pp_fine)
        nonvuv_surface = np.zeros_like(pp_fine)
        for col_name, (popt, _) in fitted_params.items():
            I_line = exponential_2d((pp_fine, Pp_fine), *popt)
            if col_name in vuv_cols:
                vuv_surface += I_line
            else:
                nonvuv_surface += I_line
        total_surface = vuv_surface + nonvuv_surface

        for label, data, fname in [
            ("VUV emission (< 200 nm)", vuv_surface, "VUV_total_exponential.png"),
            ("Non‑VUV emission (> 200 nm)", nonvuv_surface, "nonVUV_total_exponential.png"),
            ("Total emission (all lines)", total_surface, "Total_emission_exponential.png")
        ]:
            plt.figure(figsize=(8, 6))
            plt.contourf(P_fine, p_fine, data, levels=20, cmap='plasma')
            plt.colorbar(label='Intensity (arb. units)')
            plt.xlabel('Power (W)')
            plt.ylabel('Pressure (mTorr)')
            plt.title(label)
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=150)
            plt.close()
        print("Aggregate plots saved.")

    print("\nAll done. Results are in:", out_dir)


if __name__ == "__main__":
    main()