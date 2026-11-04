#!/usr/bin/env python3
"""
Predict emission line fluxes from the 2D exponential model parameters
(saved in fitting_exponential.csv). User supplies (pressure, power).

Usage: python predict_from_fit_exponential.py fitting_exponential.csv
"""

import csv
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Wavelength mapping (same as before)
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


def load_fit_csv(filepath):
    """Read the fitting CSV, return list of dicts."""
    lines = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            lines.append(row)
    return lines


def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def predict_intensity(p, P, A, alpha, beta, B, C):
    """Exponential model: I = A * p^alpha * P^beta * exp(-B p - C P)."""
    return A * (p ** alpha) * (P ** beta) * np.exp(-B * p - C * P)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_from_fit_exponential.py <fitting_exponential.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.")
        sys.exit(1)

    rows = load_fit_csv(csv_path)
    print(f"Read {len(rows)} rows from CSV.")

    # Show all line names found
    print("Emission lines found:")
    for i, row in enumerate(rows):
        print(f"  Row {i+1}: '{row.get('Emission line', '??')}'")

    # Check for r549
    r549_present = any("r549" in row.get("Emission line", "") for row in rows)
    if not r549_present:
        print("\n*** WARNING: 'r549: O_1s -> O_1d' is not in the CSV. ***")
        print("    The exponential fit may have failed for that line.\n")

    # Parse parameters
    emission_data = []
    missing_wl = []
    skipped_incomplete = []
    for row in rows:
        name = row.get("Emission line", "")
        if not name:
            continue
        A = safe_float(row.get("A"))
        alpha = safe_float(row.get("alpha"))
        beta  = safe_float(row.get("beta"))
        B = safe_float(row.get("B"))
        C = safe_float(row.get("C"))
        if None in (A, alpha, beta, B, C):
            print(f"  Skipping '{name}' – incomplete parameters.")
            skipped_incomplete.append(name)
            continue
        wl = COLUMN_WAVELENGTH.get(name)
        if wl is None:
            missing_wl.append(name)
            continue
        emission_data.append({
            "name": name,
            "wl": wl,
            "A": A, "alpha": alpha, "beta": beta,
            "B": B, "C": C
        })

    if missing_wl:
        print("\nWarning: no wavelength mapping for:")
        for ln in missing_wl:
            print(f"  - {ln}")
    if skipped_incomplete:
        print("\nSkipped because of missing parameters:")
        for ln in skipped_incomplete:
            print(f"  - {ln}")

    if not emission_data:
        print("No valid emission lines to predict.")
        sys.exit(1)

    # User inputs
    print("\nEnter target conditions:")
    try:
        p_target = float(input("  Pressure (mTorr) : "))
        P_target = float(input("  Power (W)        : "))
    except ValueError:
        print("Invalid number.")
        sys.exit(1)

    # Compute intensities
    print(f"\n{'='*90}")
    print(f"PREDICTED FLUXES (EXPONENTIAL MODEL) AT p={p_target:.1f} mTorr, P={P_target:.1f} W")
    print(f"{'='*90}")
    header = f"{'Emission line':<30} {'Wavelength':>10} {'VUV/Non':>8} {'Intensity':>14}"
    print(header)
    print("-"*90)

    vuv_total = 0.0
    nonvuv_total = 0.0

    for line in emission_data:
        name = line["name"]
        wl = line["wl"]
        I = predict_intensity(p_target, P_target,
                              line["A"], line["alpha"], line["beta"],
                              line["B"], line["C"])
        is_vuv = "VUV" if wl < VUV_THRESHOLD else "nonVUV"
        print(f"{name:<30} {wl:>10.1f} {is_vuv:>8} {I:>14.4e}")
        if is_vuv == "VUV":
            vuv_total += I
        else:
            nonvuv_total += I

    total = vuv_total + nonvuv_total
    print("-"*90)
    print(f"{'VUV total':<30} {'':>10} {'':>8} {vuv_total:>14.4e}")
    print(f"{'Non‑VUV total':<30} {'':>10} {'':>8} {nonvuv_total:>14.4e}")
    print(f"{'GRAND TOTAL':<30} {'':>10} {'':>8} {total:>14.4e}")
    print("\nDone.")


if __name__ == "__main__":
    main()