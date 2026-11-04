#!/usr/bin/env python3
"""
Find peaks in the 2D parameter space from the interaction‑model fit
and save separate contour plots for each quantity.

Usage: python find_peaks_2d.py <fitting_interaction_log.csv>
"""

import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Wavelength mapping – must match the CSV "Emission line" column exactly
# ----------------------------------------------------------------------
COLUMN_WAVELENGTH = {
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

VUV_THRESHOLD = 200.0   # nm

# ----------------------------------------------------------------------
# Range for peak search (default)
# ----------------------------------------------------------------------
P_MIN, P_MAX = 2.0, 200.0           # mTorr
POWER_MIN, POWER_MAX = 100.0, 2000.0  # W
N_GRID = 300   # resolution of search grid

# ----------------------------------------------------------------------
def load_csv_params(filepath):
    """Read fitting CSV and return list of dicts with parameters and wl."""
    lines = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            lines.append(row)
    return lines

def parse_lines(rows):
    """Extract valid lines with parameters and wavelengths."""
    emission_data = []
    for row in rows:
        name = row.get("Emission line", "")
        A = safe_float(row.get("A"))
        alpha = safe_float(row.get("alpha"))
        beta  = safe_float(row.get("beta"))
        B = safe_float(row.get("B"))
        C = safe_float(row.get("C"))
        D = safe_float(row.get("D"))
        E = safe_float(row.get("E"))
        F = safe_float(row.get("F"))
        if None in (A, alpha, beta, B, C, D, E, F):
            continue
        wl = COLUMN_WAVELENGTH.get(name)
        if wl is None:
            continue
        emission_data.append({
            "name": name,
            "wl": wl,
            "params": (A, alpha, beta, B, C, D, E, F)
        })
    return emission_data

def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def intensity(p, P, params):
    """Interaction model: I = A p^α P^β / (1 + B p + C P + D p^2 + E P^2 + F p P)."""
    A, alpha, beta, B, C, D, E, F = params
    denom = 1.0 + B * p + C * P + D * p**2 + E * P**2 + F * p * P
    # Replace non‑physical points with zero
    return np.where(denom > 0, A * (p ** alpha) * (P ** beta) / denom, 0.0)

def find_peak(p_arr, P_arr, values):
    """Return (p, P, value) of maximum in a 2D array."""
    idx = np.unravel_index(np.argmax(values), values.shape)
    return p_arr[idx[0]], P_arr[idx[1]], values[idx]

def plot_and_save(p_vals, P_vals, data, peak, title, filename):
    """Create and save a single contour plot with the peak marked and labelled."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(P_vals, p_vals, data, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='Intensity (arb. units)')
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Pressure (mTorr)")
    ax.set_title(title)
    
    # Mark the peak with a red dot
    p_peak, P_peak, val = peak
    ax.plot(P_peak, p_peak, 'ro', markersize=8, markeredgecolor='white')
    
    # Place an annotation with an arrow that stays clear of the colour bar
    # Move the text 10% of the axis range to the left and up from the peak
    dp = 0.1 * (P_vals[-1] - P_vals[0])   # offset in power direction
    dP = 0.1 * (p_vals[-1] - p_vals[0])   # offset in pressure direction
    
    ax.annotate(
        f'{p_peak:.1f} mTorr, {P_peak:.0f} W',
        xy=(P_peak, p_peak),
        xytext=(P_peak - dp, p_peak + dP),   # offset left and up
        arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
        color='white',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
        ha='right'  # right‑align the text so it sits neatly to the left of the arrow tip
    )
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved plot: {filename}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python find_peaks_2d.py <fitting_interaction_log.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.")
        sys.exit(1)

    rows = load_csv_params(csv_path)
    lines = parse_lines(rows)

    if not lines:
        print("No valid emission lines found. Exiting.")
        sys.exit(1)

    # Separate VUV / non‑VUV
    vuv_lines = [l for l in lines if l["wl"] < VUV_THRESHOLD]
    nonvuv_lines = [l for l in lines if l["wl"] >= VUV_THRESHOLD]

    print(f"Loaded {len(vuv_lines)} VUV lines and {len(nonvuv_lines)} non‑VUV lines.")

    # Create search grid
    p_vals = np.linspace(P_MIN, P_MAX, N_GRID)
    P_vals = np.linspace(POWER_MIN, POWER_MAX, N_GRID)
    Pp, pp = np.meshgrid(P_vals, p_vals)   # Pp: (n_press, n_power)

    # Compute total VUV & non‑VUV
    vuv_total = np.zeros_like(pp)
    for line in vuv_lines:
        vuv_total += intensity(pp, Pp, line["params"])

    nonvuv_total = np.zeros_like(pp)
    for line in nonvuv_lines:
        nonvuv_total += intensity(pp, Pp, line["params"])

    total = vuv_total + nonvuv_total

    # Ratios (protect against division by zero)
    ratio_vuv_over_nonvuv = np.where(nonvuv_total > 0, vuv_total / nonvuv_total, 0.0)
    ratio_nonvuv_over_vuv = np.where(vuv_total > 0, nonvuv_total / vuv_total, 0.0)

    # Find peaks
    peak_vuv = find_peak(p_vals, P_vals, vuv_total)
    peak_nonvuv = find_peak(p_vals, P_vals, nonvuv_total)
    peak_total = find_peak(p_vals, P_vals, total)
    peak_ratio_vuv = find_peak(p_vals, P_vals, ratio_vuv_over_nonvuv)
    peak_ratio_nonvuv = find_peak(p_vals, P_vals, ratio_nonvuv_over_vuv)

    # Print summary to console
    def print_peak(label, p, P, value, extra=""):
        print(f"{label:30s}  p = {p:6.2f} mTorr  P = {P:7.1f} W  value = {value:.4e} {extra}")

    print("\n" + "="*80)
    print("PEAK LOCATIONS IN THE 2D PARAMETER SPACE")
    print("="*80)
    print_peak("Peak VUV flux", *peak_vuv)
    print_peak("Peak non‑VUV flux", *peak_nonvuv)
    print_peak("Peak total flux", *peak_total)
    print_peak("Max VUV / non‑VUV ratio", *peak_ratio_vuv, extra="(ratio)")
    print_peak("Max non‑VUV / VUV ratio", *peak_ratio_nonvuv, extra="(ratio)")

    # Save individual plots
    base_name = csv_path.stem  # "fitting_interaction_log"
    out_dir = csv_path.parent

    plot_and_save(p_vals, P_vals, vuv_total, peak_vuv,
                  "VUV total flux", str(out_dir / f"{base_name}_vuv_total.png"))
    plot_and_save(p_vals, P_vals, nonvuv_total, peak_nonvuv,
                  "non‑VUV total flux", str(out_dir / f"{base_name}_nonvuv_total.png"))
    plot_and_save(p_vals, P_vals, total, peak_total,
                  "Total flux", str(out_dir / f"{base_name}_total_flux.png"))
    plot_and_save(p_vals, P_vals, ratio_vuv_over_nonvuv, peak_ratio_vuv,
                  "VUV / non‑VUV ratio", str(out_dir / f"{base_name}_vuv_ratio.png"))
    plot_and_save(p_vals, P_vals, ratio_nonvuv_over_vuv, peak_ratio_nonvuv,
                  "non‑VUV / VUV ratio", str(out_dir / f"{base_name}_nonvuv_ratio.png"))

    print("\nAll plots saved. Done.")

if __name__ == "__main__":
    main()