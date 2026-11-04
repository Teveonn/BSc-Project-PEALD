#!/usr/bin/env python3
"""
Fit a 2D rational model to K.csv data on a pressure–power grid.
Usage: python fit_2d_grid.py /path/to/data_root
The data root must contain subdirectories for each power (e.g., 200w, 500w, ...).
Inside each power directory there are subdirectories for each pressure,
named like "200_5" (power_pressure). Each of those contains a "K.csv" file
with the emission columns as described.
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
# The column names are expected to be of the form: r549: O_1s -> O_1d
# (no '+' and no wavelength in parentheses).
# ----------------------------------------------------------------------
COLUMN_INFO = {
    "r549: O_1s -> O_1d":   557.7,
    "r550: O_5s -> O":      135.6,
    "r551: O_5s -> O":      135.9,   # distinguish by r-number
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
# 2D rational model
# ----------------------------------------------------------------------
def rational_2d(P_grid, A, alpha, beta, B, C, D, E):
    """
    I = A * p^alpha * P^beta / (1 + B*p + C*P + D*p^2 + E*P^2)
    P_grid: tuple (p, P) where p, P are arrays (flattened).
    """
    p, P = P_grid
    numerator = A * (p ** alpha) * (P ** beta)
    denominator = 1.0 + B * p + C * P + D * p**2 + E * P**2
    return numerator / denominator

def save_fitting_results(fitted_params, output_csv):
    """
    Save fitted parameters and R² to a CSV file.
    fitted_params: dict mapping column_name -> (popt, r2)
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Emission line", "A", "alpha", "beta", "B", "C", "D", "E", "R^2"])
        for col_name, (popt, r2) in fitted_params.items():
            A, alpha, beta, B, C, D, E = popt
            writer.writerow([col_name, A, alpha, beta, B, C, D, E, r2])
    print(f"Fitting results saved to {output_csv}")

def fit_line(power_grid, pressure_grid, intensity_grid):
    from scipy.optimize import curve_fit
    import numpy as np

    P_flat = power_grid.flatten()
    p_flat = pressure_grid.flatten()
    I_flat = intensity_grid.flatten()

    # Remove non‑finite values
    mask = np.isfinite(I_flat)
    P_flat = P_flat[mask]
    p_flat = p_flat[mask]
    I_flat = I_flat[mask]

    if len(I_flat) < 7:
        return None, None

    # Normalize the data to avoid huge magnitude differences
    I_scale = np.median(np.abs(I_flat)) if np.median(np.abs(I_flat)) > 0 else 1.0
    I_norm = I_flat / I_scale

    # Try several initial guesses
    guess_list = [
        [1.0, 1.0, 1.0, 0.01, 0.001, 0.0, 0.0],
        [1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 2.0, 0.1, 0.1, 0.0, 0.0],
    ]

    best_popt = None
    best_r2 = -np.inf
    success = False

    for p0 in guess_list:
        try:
            popt, _ = curve_fit(
                rational_2d,
                (p_flat, P_flat),
                I_norm,
                p0=p0,
                maxfev=20000
            )
            I_pred_norm = rational_2d((p_flat, P_flat), *popt)
            ss_res = np.sum((I_norm - I_pred_norm) ** 2)
            ss_tot = np.sum((I_norm - np.mean(I_norm)) ** 2)
            r2 = 1 - ss_res / ss_tot
            if r2 > best_r2:
                best_r2 = r2
                # De‑normalize the amplitude A
                best_popt = list(popt)
                best_popt[0] = popt[0] * I_scale   # A = A_norm * I_scale
        except Exception:
            continue

    if best_popt is not None and best_r2 > 0.5:   # accept if R² is reasonable
        return best_popt, best_r2
    else:
        return None, None


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python fit_2d_grid.py /path/to/data_root")
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    if not root_dir.is_dir():
        print(f"Error: {root_dir} is not a directory.")
        sys.exit(1)

    # Discover power values and pressure values
    power_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    if not power_dirs:
        print("No power subdirectories found.")
        sys.exit(1)

    power_levels = []   # list of float (W)
    pressure_levels_set = set()
    dir_map = {}        # (power, pressure) -> path to K.csv

    for pwr_dir in power_dirs:
        # Extract power from directory name (e.g., "200w" => 200)
        pwr_name = pwr_dir.name.lower().rstrip('w')
        try:
            power = float(pwr_name)
        except ValueError:
            print(f"Skipping directory {pwr_dir.name} (cannot parse power)")
            continue
        power_levels.append(power)

        # Inside, find pressure subdirectories
        for pres_dir in sorted(pwr_dir.iterdir()):
            if not pres_dir.is_dir():
                continue
            # Expect name like "200_5" -> power_pressure
            parts = pres_dir.name.split('_')
            if len(parts) < 2:
                continue
            try:
                pressure = float(parts[1])
            except ValueError:
                continue
            pressure_levels_set.add(pressure)
            # Check for K.csv
            k_file = pres_dir / "K.csv"
            if k_file.is_file():
                dir_map[(power, pressure)] = k_file
            else:
                print(f"Warning: K.csv not found in {pres_dir}")

    if not dir_map:
        print("No K.csv files found.")
        sys.exit(1)

    # Sort pressure and power arrays
    pressure_levels = sorted(pressure_levels_set)
    power_levels = sorted(power_levels)

    print(f"Found {len(power_levels)} power levels: {power_levels}")
    print(f"Found {len(pressure_levels)} pressure levels: {pressure_levels}")
    print(f"Total grid points: {len(dir_map)}")

    # Create meshgrid arrays
    P_grid, p_grid = np.meshgrid(power_levels, pressure_levels)

    # We'll store intensity grids for each column
    all_columns = list(COLUMN_INFO.keys())
    intensity_data = {col: np.full_like(P_grid, np.nan) for col in all_columns}

    # Fill grids
    for (power, pressure), filepath in dir_map.items():
        i_power = power_levels.index(power)
        i_press = pressure_levels.index(pressure)

        # Read K.csv
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers is None:
                continue
            # Find column indices
            col_indices = {}
            for idx, h in enumerate(headers):
                h_clean = h.strip()
                if h_clean in all_columns:
                    col_indices[h_clean] = idx
            # Read all rows, get last valid row
            last_row = None
            for row in reader:
                if len(row) > 0 and any(cell.strip() for cell in row):
                    last_row = row
            if last_row is None:
                continue

            # Extract values
            for col_name, idx in col_indices.items():
                try:
                    val = float(last_row[idx].strip())
                except (ValueError, IndexError):
                    val = np.nan
                intensity_data[col_name][i_press, i_power] = val
    

    # Diagnostic: show r549 grid
    col_check = "r549: O_1s -> O_1d"
    if col_check in intensity_data:
        print(f"\nDiagnostic for {col_check}:")
        print("Pressure levels (rows):", pressure_levels)
        print("Power levels (cols):", power_levels)
        print("Data grid:")
        print(intensity_data[col_check])
        print("Finite count:", np.sum(np.isfinite(intensity_data[col_check])))
    else:
        print(f"{col_check} not found in any K.csv.")


    # Create output directory for plots
    out_dir = root_dir / "fits_and_plots"
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Fit each emission line
    # ------------------------------------------------------------------
    fitted_params = {}
    for col_name in all_columns:
        print(f"\nFitting {col_name} ...")
        I_grid = intensity_data[col_name]
        popt, r2 = fit_line(P_grid, p_grid, I_grid)
        if popt is not None:
            fitted_params[col_name] = (popt, r2)
            A, alpha, beta, B, C, D, E = popt
            print(f"  R² = {r2:.5f}")
            print(f"  A={A:.4e}, α={alpha:.4f}, β={beta:.4f}, B={B:.4e}, C={C:.4e}, D={D:.4e}, E={E:.4e}")
        else:
            print("  Fit failed, skipping plots for this line.")

    # Save fitting parameters to CSV
    output_csv = out_dir / "fitting_results.csv"
    save_fitting_results(fitted_params, output_csv)

    # ------------------------------------------------------------------
    # Plot individual lines
    # ------------------------------------------------------------------
    n_plots = len(fitted_params)
    if n_plots > 0:
        # Create a finer mesh for smooth plots
        p_fine = np.linspace(min(pressure_levels), max(pressure_levels), 100)
        P_fine = np.linspace(min(power_levels), max(power_levels), 100)
        Pp_fine, pp_fine = np.meshgrid(P_fine, p_fine)

        for col_name, (popt, r2) in fitted_params.items():
            # Compute surface
            I_surface = rational_2d((pp_fine, Pp_fine), *popt)

            plt.figure(figsize=(8, 6))
            plt.contourf(P_fine, p_fine, I_surface, levels=20, cmap='viridis')
            plt.colorbar(label='Intensity (arb. units)')
            plt.xlabel('Power (W)')
            plt.ylabel('Pressure (mTorr)')
            plt.title(f'{col_name}\nR² = {r2:.4f}')
            # Overlay grid points
            for i, p in enumerate(pressure_levels):
                for j, P in enumerate(power_levels):
                    val = intensity_data[col_name][i, j]
                    if np.isfinite(val):
                        plt.plot(P, p, 'ko', markersize=3)
            # Save
            safe_name = col_name.replace(':', '_').replace(' ', '_').replace('->', 'to')
            plot_file = out_dir / f"{safe_name}.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"Saved {plot_file}")

    # ------------------------------------------------------------------
    # Aggregate VUV, non‑VUV, total
    # ------------------------------------------------------------------
    vuv_cols = [col for col, wl in COLUMN_INFO.items() if wl < WAVELENGTH_THRESHOLD]
    nonvuv_cols = [col for col, wl in COLUMN_INFO.items() if wl >= WAVELENGTH_THRESHOLD]

    # For each grid point, we can compute summed intensities using the fitted models
    # Only for grid points where all fits succeeded and original data existed
    # We'll use the measured data directly (if available) for the aggregate plots,
    # but we can also use the fitted surfaces. To stay consistent with the program,
    # we'll compute the aggregated surface from the sum of individual fitted models.
    p_fine = np.linspace(min(pressure_levels), max(pressure_levels), 100)
    P_fine = np.linspace(min(power_levels), max(power_levels), 100)
    Pp_fine, pp_fine = np.meshgrid(P_fine, p_fine)

    vuv_surface = np.zeros_like(pp_fine)
    nonvuv_surface = np.zeros_like(pp_fine)

    for col_name, (popt, _) in fitted_params.items():
        I_line = rational_2d((pp_fine, Pp_fine), *popt)
        if col_name in vuv_cols:
            vuv_surface += I_line
        else:
            nonvuv_surface += I_line

    total_surface = vuv_surface + nonvuv_surface

    # Plot VUV
    plt.figure(figsize=(8, 6))
    plt.contourf(P_fine, p_fine, vuv_surface, levels=20, cmap='plasma')
    plt.colorbar(label='Total VUV intensity (arb. units)')
    plt.xlabel('Power (W)')
    plt.ylabel('Pressure (mTorr)')
    plt.title('VUV emission (< 200 nm)')
    plt.tight_layout()
    plt.savefig(out_dir / "VUV_total.png", dpi=150)
    plt.close()

    # Plot non‑VUV
    plt.figure(figsize=(8, 6))
    plt.contourf(P_fine, p_fine, nonvuv_surface, levels=20, cmap='plasma')
    plt.colorbar(label='Total non‑VUV intensity (arb. units)')
    plt.xlabel('Power (W)')
    plt.ylabel('Pressure (mTorr)')
    plt.title('Non‑VUV emission (> 200 nm)')
    plt.tight_layout()
    plt.savefig(out_dir / "nonVUV_total.png", dpi=150)
    plt.close()

    # Total
    plt.figure(figsize=(8, 6))
    plt.contourf(P_fine, p_fine, total_surface, levels=20, cmap='plasma')
    plt.colorbar(label='Total intensity (arb. units)')
    plt.xlabel('Power (W)')
    plt.ylabel('Pressure (mTorr)')
    plt.title('Total emission (all lines)')
    plt.tight_layout()
    plt.savefig(out_dir / "Total_emission.png", dpi=150)
    plt.close()

    print("\nAggregate plots saved.")
    print("All done. Results in:", out_dir)


if __name__ == "__main__":
    main()