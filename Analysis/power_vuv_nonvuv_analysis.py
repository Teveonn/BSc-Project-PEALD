#!/usr/bin/env python3
"""
VUV / non‑VUV ratio and total flux peak analysis from POWER regression (Extended model).
Usage: python power_vuv_nonvuv_analysis.py <csv_file>
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CSV column names (adjust if needed)
# ----------------------------------------------------------------------
COL_EMISSION = "Emission"
COL_A = "Extended A"
COL_ALPHA = "Extended alpha"
COL_B = "Extended B"
COL_C = "Extended C"

# Power range for analysis (Watts) – inclusive
POWER_MIN = 100.0
POWER_MAX = 2000.0
N_POINTS = 2000                # resolution for scanning the range

WAVELENGTH_THRESHOLD = 200.0   # nm; below this is VUV


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
    Works with scalar or array power values.
    """
    p = np.asarray(p, dtype=float)
    denom = 1.0 + B * p + C * p * p
    # Denominator ≤ 0 → physically invalid, set intensity = 0
    return np.where(denom <= 0, 0.0, A * (p ** alpha) / denom)


def find_ratio_extrema(p_vals, ratio_vals):
    """Return (power_max, max_ratio) and (power_min, min_ratio)."""
    idx_max = np.argmax(ratio_vals)
    idx_min = np.argmin(ratio_vals)
    return (p_vals[idx_max], ratio_vals[idx_max]), (p_vals[idx_min], ratio_vals[idx_min])


def main():
    if len(sys.argv) != 2:
        print("Usage: python power_vuv_nonvuv_analysis.py <csv_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not Path(filepath).is_file():
        print(f"Error: file '{filepath}' not found.")
        sys.exit(1)

    rows = load_csv(filepath)

    # Separate lines by wavelength
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

        line_data = {"emission": emission, "wl": wl,
                     "A": A, "alpha": alpha, "B": B, "C": C}
        if wl < WAVELENGTH_THRESHOLD:
            vuv_lines.append(line_data)
        else:
            nonvuv_lines.append(line_data)

    print(f"Found {len(vuv_lines)} VUV lines and {len(nonvuv_lines)} non‑VUV lines.")

    if not vuv_lines or not nonvuv_lines:
        print("Need at least one VUV and one non‑VUV line to proceed.")
        sys.exit(1)

    # Power array (W)
    power = np.linspace(POWER_MIN, POWER_MAX, N_POINTS)

    # Compute total intensities for each group over the power range
    vuv_total = np.zeros_like(power)
    nonvuv_total = np.zeros_like(power)

    for line in vuv_lines:
        vuv_total += intensity(power, line["A"], line["alpha"],
                               line["B"], line["C"])

    for line in nonvuv_lines:
        nonvuv_total += intensity(power, line["A"], line["alpha"],
                                  line["B"], line["C"])

    # Total flux (VUV + non‑VUV)
    total_flux = vuv_total + nonvuv_total

    # Ratio VUV / non‑VUV (set to 0 where non‑VUV is 0)
    ratio = np.divide(vuv_total, nonvuv_total,
                      out=np.zeros_like(vuv_total), where=nonvuv_total != 0)

    # Find maximum and minimum ratio in [100, 2000] W
    (p_ratio_max, max_ratio), (p_ratio_min, min_ratio) = find_ratio_extrema(power, ratio)

    # Find peak of total flux
    idx_total_max = np.argmax(total_flux)
    p_total_max = power[idx_total_max]
    max_total_flux = total_flux[idx_total_max]

    # Helper to fetch all values at a given power (nearest grid point)
    def get_values_at(p_target):
        idx = np.argmin(np.abs(power - p_target))
        return vuv_total[idx], nonvuv_total[idx], ratio[idx]

    vuv_ratmax, nonvuv_ratmax, _ = get_values_at(p_ratio_max)
    vuv_ratmin, nonvuv_ratmin, _ = get_values_at(p_ratio_min)
    vuv_totalpeak, nonvuv_totalpeak, ratio_totalpeak = get_values_at(p_total_max)

    # ---- Print results ----
    print("\n" + "="*70)
    print("POWER ANALYSIS RESULTS (100 – 2000 W)")
    print("="*70)

    def print_point(label, p_val, vuv, nonvuv, ratio_val, extra=""):
        print(f"\n{label}")
        print(f"  Power             : {p_val:.2f} W")
        print(f"  VUV flux          : {vuv:.4e}")
        print(f"  Non‑VUV flux      : {nonvuv:.4e}")
        print(f"  VUV / non‑VUV ratio: {ratio_val:.6f}" + extra)

    print_point("PEAK TOTAL FLUX (VUV + non‑VUV)",
                p_total_max, vuv_totalpeak, nonvuv_totalpeak, ratio_totalpeak)

    print_point("MAXIMUM VUV / non‑VUV ratio",
                p_ratio_max, vuv_ratmax, nonvuv_ratmax, max_ratio)

    print_point("MINIMUM VUV / non‑VUV ratio",
                p_ratio_min, vuv_ratmin, nonvuv_ratmin, min_ratio)

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Intensities on primary y-axis
    ax1.plot(power, vuv_total, 'b-', label='Total VUV flux (<200 nm)')
    ax1.plot(power, nonvuv_total, 'r-', label='Total non‑VUV flux (>200 nm)')
    ax1.plot(power, total_flux, 'g-', label='Total flux (VUV + non‑VUV)')
    ax1.set_xlabel('Power (W)')
    ax1.set_ylabel('Predicted flux (m^-3 s^-1)')
    ax1.set_title('VUV and non‑VUV photon fluxes vs. power (Extended model)')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Mark the total flux peak
    ax1.axvline(p_total_max, color='green', linestyle='--', alpha=0.8,
                label=f'Peak total flux at {p_total_max:.0f} W')
    ax1.plot(p_total_max, max_total_flux, 'go', markersize=8)

    # Secondary y-axis for ratio
    ax2 = ax1.twinx()
    ax2.plot(power, ratio, 'k:', label='VUV / non‑VUV ratio', alpha=0.7)
    ax2.set_ylabel('VUV / non‑VUV ratio')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plot_file = "power_vuv_nonvuv_ratio.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\nPlot saved as '{plot_file}'.")
    plt.show()


if __name__ == "__main__":
    main()