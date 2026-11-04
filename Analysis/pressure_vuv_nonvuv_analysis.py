#!/usr/bin/env python3
"""
VUV and non‑VUV photon flux analysis using the Extended model.
Now also finds and displays the peak of total flux (VUV + non‑VUV).

Usage: python vuv_nonvuv_analysis.py <csv_file>
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Column names (adjust if your CSV uses different capitalisation/spelling)
# ----------------------------------------------------------------------
COL_EMISSION = "Emission"
COL_A = "Extended A"
COL_ALPHA = "Extended alpha"
COL_B = "Extended B"
COL_C = "Extended C"

# Pressure range for analysis (mTorr) – inclusive
PRESSURE_MIN = 2.0
PRESSURE_MAX = 200.0
N_POINTS = 2000   # resolution for finding peaks

WAVELENGTH_THRESHOLD = 200.0   # below this is VUV


def safe_float(value):
    """Convert string to float, return None on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def extract_wavelength(emission_str):
    """
    Extract the wavelength in nm from a string like:
    'O_5s -> O (135.6_nm)' or 'O_3p -> O_3s (844.6_nm).1'
    Returns None if not found.
    """
    match = re.search(r'\((\d+\.?\d*)_nm\)', emission_str)
    if match:
        return float(match.group(1))
    return None


def load_csv(filepath):
    """Read CSV and return list of dicts with stripped keys."""
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def intensity(p, A, alpha, B, C):
    """
    Extended model: I(P) = A * P^alpha / (1 + B*P + C*P^2).
    Handles both scalar and array pressure inputs.
    """
    p = np.asarray(p, dtype=float)
    denom = 1.0 + B * p + C * p * p
    result = np.where(denom <= 0, 0.0, A * (p ** alpha) / denom)
    return result


def find_peak_in_range(p_vals, y_vals, mode='max'):
    """
    Return the pressure and y value where y is maximal (mode='max') or minimal (mode='min').
    """
    if mode == 'max':
        idx = np.argmax(y_vals)
    else:
        idx = np.argmin(y_vals)
    return p_vals[idx], y_vals[idx]


def main():
    if len(sys.argv) != 2:
        print("Usage: python vuv_nonvuv_analysis.py <csv_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not Path(filepath).is_file():
        print(f"Error: file '{filepath}' not found.")
        sys.exit(1)

    rows = load_csv(filepath)

    vuv_lines = []
    nonvuv_lines = []

    for row in rows:
        emission = row.get(COL_EMISSION, "")
        wl = extract_wavelength(emission)
        if wl is None:
            print(f"Warning: could not parse wavelength from '{emission}', skipping.")
            continue

        A = safe_float(row.get(COL_A))
        alpha = safe_float(row.get(COL_ALPHA))
        B = safe_float(row.get(COL_B))
        C = safe_float(row.get(COL_C))

        if None in (A, alpha, B, C):
            print(f"Warning: missing parameters for '{emission}', skipping.")
            continue

        line_data = {"emission": emission, "wl": wl, "A": A, "alpha": alpha, "B": B, "C": C}
        if wl < WAVELENGTH_THRESHOLD:
            vuv_lines.append(line_data)
        else:
            nonvuv_lines.append(line_data)

    print(f"Found {len(vuv_lines)} VUV lines and {len(nonvuv_lines)} non‑VUV lines.")

    if not vuv_lines or not nonvuv_lines:
        print("Need at least one VUV and one non‑VUV line to proceed.")
        sys.exit(1)

    # Pressure array (mTorr)
    p = np.linspace(PRESSURE_MIN, PRESSURE_MAX, N_POINTS)

    # Compute total intensities
    vuv_total = np.zeros_like(p)
    nonvuv_total = np.zeros_like(p)

    for line in vuv_lines:
        vuv_total += intensity(p, line["A"], line["alpha"], line["B"], line["C"])

    for line in nonvuv_lines:
        nonvuv_total += intensity(p, line["A"], line["alpha"], line["B"], line["C"])

    total_flux = vuv_total + nonvuv_total
    ratio = np.divide(vuv_total, nonvuv_total,
                      out=np.zeros_like(vuv_total), where=nonvuv_total != 0)

    # ---------- Find peaks ----------
    p_vuv_max, max_vuv = find_peak_in_range(p, vuv_total, 'max')
    p_nonvuv_max, max_nonvuv = find_peak_in_range(p, nonvuv_total, 'max')
    p_total_max, max_total = find_peak_in_range(p, total_flux, 'max')
    p_ratio_max, max_ratio = find_peak_in_range(p, ratio, 'max')
    p_ratio_min, min_ratio = find_peak_in_range(p, ratio, 'min')

    # Helper to get values at a given pressure
    def get_values_at(p_target):
        idx = np.argmin(np.abs(p - p_target))
        return vuv_total[idx], nonvuv_total[idx], ratio[idx]

    vuv_at_vuv, nonvuv_at_vuv, _ = get_values_at(p_vuv_max)
    vuv_at_nonvuv, nonvuv_at_nonvuv, _ = get_values_at(p_nonvuv_max)
    vuv_at_total, nonvuv_at_total, ratio_total = get_values_at(p_total_max)
    vuv_at_ratmax, nonvuv_at_ratmax, _ = get_values_at(p_ratio_max)
    vuv_at_ratmin, nonvuv_at_ratmin, _ = get_values_at(p_ratio_min)

    # Print results
    print("\n" + "="*75)
    print("PRESSURE ANALYSIS (2 – 200 mTorr)")
    print("="*75)

    def print_point(label, p_val, vuv, nonvuv, ratio_val):
        print(f"\n{label}")
        print(f"  Pressure          : {p_val:.3f} mTorr")
        print(f"  VUV flux          : {vuv:.4e}")
        print(f"  Non‑VUV flux      : {nonvuv:.4e}")
        print(f"  VUV / non‑VUV ratio: {ratio_val:.4f}")

    print_point("A) Peak VUV flux", p_vuv_max, vuv_at_vuv, nonvuv_at_vuv,
                vuv_at_vuv / nonvuv_at_vuv if nonvuv_at_vuv != 0 else float('inf'))
    print_point("B) Peak non‑VUV flux", p_nonvuv_max, vuv_at_nonvuv, nonvuv_at_nonvuv,
                vuv_at_nonvuv / nonvuv_at_nonvuv if nonvuv_at_nonvuv != 0 else float('inf'))
    print_point("C) Peak total flux (VUV + non‑VUV)", p_total_max, vuv_at_total, nonvuv_at_total, ratio_total)
    print_point("D) Maximum VUV / non‑VUV ratio", p_ratio_max, vuv_at_ratmax, nonvuv_at_ratmax, max_ratio)
    print_point("E) Minimum VUV / non‑VUV ratio", p_ratio_min, vuv_at_ratmin, nonvuv_at_ratmin, min_ratio)

    # ---------- Plot ----------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(p, vuv_total, 'b-', label='Total VUV flux (<200 nm)')
    ax1.plot(p, nonvuv_total, 'r-', label='Total non‑VUV flux (>200 nm)')
    ax1.plot(p, total_flux, 'g-', label='Total flux (VUV + non‑VUV)')
    ax1.set_xlabel('Pressure (mTorr)')
    ax1.set_ylabel('Predicted flux (m^-3 s^-1)')
    ax1.set_title('VUV and non‑VUV photon fluxes vs. pressure (Extended model)')

    # Mark the peaks
    ax1.axvline(p_vuv_max, color='blue', linestyle='--', alpha=0.7)
    ax1.axvline(p_nonvuv_max, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(p_total_max, color='green', linestyle='--', alpha=0.8,
                label=f'Peak total at {p_total_max:.1f} mTorr')
    ax1.plot(p_vuv_max, vuv_at_vuv, 'bo', markersize=8)
    ax1.plot(p_nonvuv_max, nonvuv_at_nonvuv, 'ro', markersize=8)
    ax1.plot(p_total_max, max_total, 'go', markersize=8)

    # Secondary y-axis for ratio
    ax2 = ax1.twinx()
    ax2.plot(p, ratio, 'k:', label='VUV / non‑VUV ratio', alpha=0.6)
    ax2.set_ylabel('VUV / non‑VUV ratio')
    # ax2.axvline(p_ratio_max, color='gray', linestyle=':', alpha=0.8)
    # ax2.axvline(p_ratio_min, color='gray', linestyle=':', alpha=0.8)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plot_file = "pressure_vuv_nonvuv_intensity.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\nPlot saved as '{plot_file}'.")
    plt.show()


if __name__ == "__main__":
    main()