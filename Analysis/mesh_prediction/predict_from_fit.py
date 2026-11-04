#!/usr/bin/env python3
"""
Predict emission line fluxes from the 2D rational model parameters 
(saved in a CSV file). The user specifies a (pressure, power) point.

Usage: python predict_from_fit.py fitting_results.csv
"""

import csv
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Mapping of emission column names to wavelength (nm) for VUV/non‑VUV
# Must match exactly the names in the first column of the CSV.
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
    """Read the fitting results CSV, return list of dicts."""
    lines = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values
            row = {k.strip(): v.strip() for k, v in row.items()}
            lines.append(row)
    return lines


def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def predict_intensity(p, P, A, alpha, beta, B, C, D, E):
    """2D rational model: I = A * p^alpha * P^beta / (1 + B p + C P + D p^2 + E P^2)."""
    denom = 1.0 + B * p + C * P + D * p * p + E * P * P
    if denom <= 0:
        return 0.0
    return A * (p ** alpha) * (P ** beta) / denom


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_from_fit.py <fitting_results.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file '{csv_path}' not found.")
        sys.exit(1)

    # Load parameters
    rows = load_fit_csv(csv_path)
    if not rows:
        print("Error: CSV is empty.")
        sys.exit(1)

    # Parse parameters
    emission_data = []
    missing_wl = []
    for row in rows:
        name = row.get("Emission line", "")
        if not name:
            continue
        A = safe_float(row.get("A"))
        alpha = safe_float(row.get("alpha"))
        beta  = safe_float(row.get("beta"))
        B = safe_float(row.get("B"))
        C = safe_float(row.get("C"))
        D = safe_float(row.get("D"))
        E = safe_float(row.get("E"))
        if None in (A, alpha, beta, B, C, D, E):
            print(f"Warning: incomplete parameters for '{name}', skipping.")
            continue
        # Get wavelength
        wl = COLUMN_WAVELENGTH.get(name)
        if wl is None:
            missing_wl.append(name)
            continue
        emission_data.append({
            "name": name,
            "wl": wl,
            "A": A, "alpha": alpha, "beta": beta,
            "B": B, "C": C, "D": D, "E": E
        })

    if missing_wl:
        print("Warning: no wavelength mapping for lines:", missing_wl)
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
    print(f"PREDICTED FLUXES AT p = {p_target:.1f} mTorr, P = {P_target:.1f} W")
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
                              line["B"], line["C"], line["D"], line["E"])
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
    print("Done.")


if __name__ == "__main__":
    main()