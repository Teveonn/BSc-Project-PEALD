#!/usr/bin/env python3
"""
Predict line fluxes at a chosen (pressure, power) using the Extended model.
Combines pressure and power dependencies via a separable model:
    I(p,P) = I_press(p) * I_pow(P) / I_cross
where I_cross = sqrt( I_press(p0) * I_pow(P0) ),
with p0 = reference pressure of the power scan,
     P0 = reference power of the pressure scan.

Usage: python predict_flux_v2.py pressure_regression.csv power_regression.csv
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Column names (same as previous scripts)
# ----------------------------------------------------------------------
COL_EMISSION = "Emission"
COL_A = "Extended A"
COL_ALPHA = "Extended alpha"
COL_B = "Extended B"
COL_C = "Extended C"

WAVELENGTH_THRESHOLD = 200.0   # nm


def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def extract_wavelength(emission_str):
    match = re.search(r'\((\d+\.?\d*)_nm\)', emission_str)
    return float(match.group(1)) if match else None


def load_csv(filepath):
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def intensity_model(p, A, alpha, B, C):
    """Extended model: I = A * p^alpha / (1 + B*p + C*p^2)."""
    denom = 1.0 + B * p + C * p * p
    if denom <= 0:
        return 0.0
    return A * (p ** alpha) / denom


def parse_lines(rows, model_name):
    """Extract parameters from rows; return list of dicts."""
    lines = []
    for row in rows:
        emission = row.get(COL_EMISSION, "")
        wl = extract_wavelength(emission)
        if wl is None:
            print(f"Warning ({model_name}): cannot parse wavelength from '{emission}', skipping.")
            continue
        A = safe_float(row.get(COL_A))
        alpha = safe_float(row.get(COL_ALPHA))
        B = safe_float(row.get(COL_B))
        C = safe_float(row.get(COL_C))
        if None in (A, alpha, B, C):
            print(f"Warning ({model_name}): missing parameters for '{emission}', skipping.")
            continue
        lines.append({
            "emission": emission,
            "wl": wl,
            "A": A, "alpha": alpha, "B": B, "C": C
        })
    return lines


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_flux_v2.py pressure_csv power_csv")
        sys.exit(1)

    press_file = sys.argv[1]
    pow_file = sys.argv[2]

    if not Path(press_file).is_file():
        print(f"Error: pressure file '{press_file}' not found.")
        sys.exit(1)
    if not Path(pow_file).is_file():
        print(f"Error: power file '{pow_file}' not found.")
        sys.exit(1)

    press_rows = load_csv(press_file)
    pow_rows = load_csv(pow_file)

    press_lines = parse_lines(press_rows, "pressure")
    pow_lines = parse_lines(pow_rows, "power")

    # Build lookup for power lines by emission string
    pow_dict = {line["emission"]: line for line in pow_lines}

    # Combine matching lines
    combined = []
    skipped = 0
    for pline in press_lines:
        emission = pline["emission"]
        if emission in pow_dict:
            combined.append({
                "emission": emission,
                "wl": pline["wl"],
                "press_params": pline,
                "pow_params": pow_dict[emission]
            })
        else:
            print(f"Warning: emission '{emission}' present in pressure file but not in power file, skipping.")
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} lines due to missing power counterpart.")
    if not combined:
        print("Error: no matching emission lines found.")
        sys.exit(1)

    # ================= User inputs =================
    print("\nEnter the target operating conditions:")
    try:
        p_target = float(input("  Target Pressure (mTorr) : "))
        P_target = float(input("  Target Power (W)        : "))
    except ValueError:
        print("Invalid number.")
        sys.exit(1)

    print("\nEnter the reference conditions of the scans:")
    print("  (Pressure scan was done at a constant power; power scan at a constant pressure.)")
    try:
        P0 = float(input("  Power during pressure scan (W) : "))
        p0 = float(input("  Pressure during power scan (mTorr) : "))
    except ValueError:
        print("Invalid number.")
        sys.exit(1)

    # ================= Flux calculations =================
    print(f"\n{'='*100}")
    print(f"PREDICTED FLUXES AT {p_target:.1f} mTorr, {P_target:.1f} W")
    print(f"Reference conditions: p0 = {p0:.1f} mTorr, P0 = {P0:.1f} W")
    print(f"{'='*100}")

    header = (f"{'Emission':<30} {'Wavelength':>10} {'VUV/Non':>8} "
              f"{'I_press':>14} {'I_pow':>14} {'I_comb':>14}")
    print(header)
    print("-"*100)

    vuv_press_total = 0.0
    nonvuv_press_total = 0.0
    vuv_pow_total = 0.0
    nonvuv_pow_total = 0.0
    vuv_comb_total = 0.0
    nonvuv_comb_total = 0.0

    # For cross‑check display
    print("\nCross‑check at (p0, P0) – should ideally agree:")
    print(f"{'Emission':<30} {'I_press(p0)':>14} {'I_pow(P0)':>14} {'Geom. mean':>14}")
    print("-"*75)

    for entry in combined:
        emission = entry["emission"]
        wl = entry["wl"]
        if wl < WAVELENGTH_THRESHOLD:
            is_vuv = "VUV"
        else:
            is_vuv = "nonVUV"

        # Parameters
        pp = entry["press_params"]
        powp = entry["pow_params"]

        # Intensity from pressure model at target p (still at P0)
        I_press = intensity_model(p_target, pp["A"], pp["alpha"], pp["B"], pp["C"])
        # Intensity from power model at target P (still at p0)
        I_pow = intensity_model(P_target, powp["A"], powp["alpha"], powp["B"], powp["C"])

        # Cross intensities at reference point
        I_press_ref = intensity_model(p0, pp["A"], pp["alpha"], pp["B"], pp["C"])
        I_pow_ref = intensity_model(P0, powp["A"], powp["alpha"], powp["B"], powp["C"])

        # Geometric mean as best estimate of I(p0, P0)
        if I_press_ref > 0 and I_pow_ref > 0:
            I_cross = np.sqrt(I_press_ref * I_pow_ref)
        else:
            I_cross = 0.0   # fallback, but shouldn't happen for valid models

        # Combined prediction
        if I_cross > 0:
            I_comb = (I_press * I_pow) / I_cross
        else:
            I_comb = 0.0

        # Print cross‑check for this line
        print(f"{emission:<30} {I_press_ref:>14.4e} {I_pow_ref:>14.4e} {I_cross:>14.4e}")

        # Print flux row later
        # (We store values and print after cross-check block)
        # Temporary: accumulate totals
        if is_vuv == "VUV":
            vuv_press_total += I_press
            vuv_pow_total += I_pow
            vuv_comb_total += I_comb
        else:
            nonvuv_press_total += I_press
            nonvuv_pow_total += I_pow
            nonvuv_comb_total += I_comb

    # Now print main flux table
    print("\n" + "="*100)
    print("FLUX TABLE")
    print("="*100)
    print(header)
    print("-"*100)

    for entry in combined:
        emission = entry["emission"]
        wl = entry["wl"]
        is_vuv = "VUV" if wl < WAVELENGTH_THRESHOLD else "nonVUV"

        pp = entry["press_params"]
        powp = entry["pow_params"]
        I_press = intensity_model(p_target, pp["A"], pp["alpha"], pp["B"], pp["C"])
        I_pow = intensity_model(P_target, powp["A"], powp["alpha"], powp["B"], powp["C"])
        I_press_ref = intensity_model(p0, pp["A"], pp["alpha"], pp["B"], pp["C"])
        I_pow_ref = intensity_model(P0, powp["A"], powp["alpha"], powp["B"], powp["C"])
        I_cross = np.sqrt(I_press_ref * I_pow_ref) if (I_press_ref>0 and I_pow_ref>0) else 0.0
        I_comb = (I_press * I_pow) / I_cross if I_cross > 0 else 0.0

        print(f"{emission:<30} {wl:>10.1f} {is_vuv:>8} {I_press:>14.4e} {I_pow:>14.4e} {I_comb:>14.4e}")

    # Totals
    print("-"*100)
    total_press = vuv_press_total + nonvuv_press_total
    total_pow = vuv_pow_total + nonvuv_pow_total
    total_comb = vuv_comb_total + nonvuv_comb_total

    print(f"{'VUV total':<30} {'':>10} {'':>8} {vuv_press_total:>14.4e} {vuv_pow_total:>14.4e} {vuv_comb_total:>14.4e}")
    print(f"{'non-VUV total':<30} {'':>10} {'':>8} {nonvuv_press_total:>14.4e} {nonvuv_pow_total:>14.4e} {nonvuv_comb_total:>14.4e}")
    print(f"{'GRAND TOTAL':<30} {'':>10} {'':>8} {total_press:>14.4e} {total_pow:>14.4e} {total_comb:>14.4e}")

    print(f"\nCombined model: I_comb = I_press(p) * I_pow(P) / sqrt( I_press(p0) * I_pow(P0) )")
    print("This assumes separability: I(p,P) = f(p) * g(P).")
    print("Done.")


if __name__ == "__main__":
    main()