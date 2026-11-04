#!/usr/bin/env python3
"""
2‑D flux prediction for oxygen emission lines.
Uses a single non‑separable rational model:
    I(p,P) = A * p^α * P^β / (1 + B p + C P + D p^2 + E P^2)

Parameters are derived from the two 1‑D Extended models (pressure scan + power scan)
so that the slices at P0 and p0 exactly match the 1‑D fits.

Usage: python predict_flux_2d.py pressure_regression.csv power_regression.csv
"""

import csv
import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# CSV column names (adjust if your files use different headers)
# ----------------------------------------------------------------------
COL_EMISSION = "Emission"
COL_A = "Extended A"
COL_ALPHA = "Extended alpha"
COL_B = "Extended B"
COL_C = "Extended C"          # coefficient of p² in pressure model

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


def intensity_press(p, A1, a1, B1, C1):
    """1‑D pressure model: I = A1 p^a1 / (1 + B1 p + C1 p^2)."""
    denom = 1.0 + B1 * p + C1 * p * p
    if denom <= 0:
        return 0.0
    return A1 * (p ** a1) / denom


def intensity_pow(P, A2, a2, B2, C2):
    """1‑D power model: I = A2 P^a2 / (1 + B2 P + C2 P^2)."""
    denom = 1.0 + B2 * P + C2 * P * P
    if denom <= 0:
        return 0.0
    return A2 * (P ** a2) / denom


def build_2d_model(p0, P0, A1, a1, B1, C1, A2, a2, B2, C2):
    """
    Compute the parameters of the 2‑D rational function that exactly
    matches the two 1‑D Extended models at the reference slices.
    Returns (A, α, β, B, C_den, D, E) where C_den is the coefficient of P.
    """
    # Exponents directly come from the 1-D fits
    alpha = a1
    beta = a2

    # Auxiliary constants
    K1 = B2 * P0 + C2 * P0 * P0   # C2 is coeff of P^2 in power model
    K2 = B1 * p0 + C1 * p0 * p0   # C1 is coeff of p^2 in pressure model

    # Solve for S1 = 1 + C*P0 + E*P0^2, S2 = 1 + B*p0 + D*p0^2
    denom_factor = 1.0 - K1 * K2
    if abs(denom_factor) < 1e-15:
        raise ValueError("Singular system: cannot build 2D model for this line.")
    S1 = (1.0 + K1) / denom_factor
    S2 = 1.0 + S1 * K2

    # Denominator coefficients
    B = B1 * S1
    D = C1 * S1          # note: C1 from pressure model becomes D (p² coeff)
    C_den = B2 * S2
    E = C2 * S2

    # Amplitude A – computed from both slices (they should agree)
    A_from_press = A1 * S1 / (P0 ** beta)
    A_from_pow = A2 * S2 / (p0 ** alpha)
    # Use average for robustness against tiny fitting inconsistencies
    A = (A_from_press + A_from_pow) / 2.0

    return A, alpha, beta, B, C_den, D, E


def intensity_2d(p, P, A, alpha, beta, B, C_den, D, E):
    """
    Evaluate 2‑D rational function:
    I = A * p^α * P^β / (1 + B p + C_den P + D p^2 + E P^2)
    """
    denom = 1.0 + B * p + C_den * P + D * p * p + E * P * P
    if denom <= 0:
        return 0.0
    return A * (p ** alpha) * (P ** beta) / denom


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_flux_2d.py pressure_regression.csv power_regression.csv")
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

    # Parse lines from pressure file
    press_dict = {}
    for row in press_rows:
        emission = row.get(COL_EMISSION, "")
        wl = extract_wavelength(emission)
        if wl is None:
            continue
        A1 = safe_float(row.get(COL_A))
        a1 = safe_float(row.get(COL_ALPHA))
        B1 = safe_float(row.get(COL_B))
        C1 = safe_float(row.get(COL_C))
        if None in (A1, a1, B1, C1):
            continue
        press_dict[emission] = {
            "emission": emission,
            "wl": wl,
            "A1": A1, "a1": a1, "B1": B1, "C1": C1
        }

    # Parse lines from power file
    pow_dict = {}
    for row in pow_rows:
        emission = row.get(COL_EMISSION, "")
        wl = extract_wavelength(emission)
        if wl is None:
            continue
        A2 = safe_float(row.get(COL_A))
        a2 = safe_float(row.get(COL_ALPHA))
        B2 = safe_float(row.get(COL_B))
        C2 = safe_float(row.get(COL_C))
        if None in (A2, a2, B2, C2):
            continue
        pow_dict[emission] = {
            "emission": emission,
            "wl": wl,
            "A2": A2, "a2": a2, "B2": B2, "C2": C2
        }

    # Matched lines
    common_emissions = set(press_dict.keys()).intersection(pow_dict.keys())
    if not common_emissions:
        print("Error: no matching emission lines between the two files.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # User inputs
    # ------------------------------------------------------------------
    print("\nEnter the target operating conditions:")
    try:
        p_target = float(input("  Target Pressure (mTorr) : "))
        P_target = float(input("  Target Power (W)        : "))
    except ValueError:
        print("Invalid input.")
        sys.exit(1)

    print("\nEnter the reference conditions at which the two scans were performed:")
    try:
        P0 = float(input("  Power during pressure scan (W) : "))
        p0 = float(input("  Pressure during power scan (mTorr) : "))
    except ValueError:
        print("Invalid input.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build 2‑D models and compute fluxes
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print(f"2‑D FLUX PREDICTION (p = {p_target:.1f} mTorr, P = {P_target:.1f} W)")
    print(f"Reference slices at P0 = {P0:.1f} W, p0 = {p0:.1f} mTorr")
    print(f"{'='*110}")

    header = (f"{'Emission':<30} {'Wavelength':>10} {'VUV/Non':>8} "
              f"{'I_press':>14} {'I_pow':>14} {'I_2D':>14}")
    print(header)
    print("-"*110)

    vuv_press = 0.0; nonvuv_press = 0.0
    vuv_pow   = 0.0; nonvuv_pow   = 0.0
    vuv_2d    = 0.0; nonvuv_2d    = 0.0

    for em in sorted(common_emissions, key=lambda e: extract_wavelength(e)):
        pdata = press_dict[em]
        podata = pow_dict[em]
        wl = pdata["wl"]
        is_vuv = "VUV" if wl < WAVELENGTH_THRESHOLD else "nonVUV"

        # 1‑D model predictions at the target point
        I_press = intensity_press(p_target, pdata["A1"], pdata["a1"],
                                  pdata["B1"], pdata["C1"])
        I_pow   = intensity_pow(P_target, podata["A2"], podata["a2"],
                                podata["B2"], podata["C2"])

        # Build 2‑D model for this line
        try:
            A, alpha, beta, B, C_den, D, E = build_2d_model(
                p0, P0,
                pdata["A1"], pdata["a1"], pdata["B1"], pdata["C1"],
                podata["A2"], podata["a2"], podata["B2"], podata["C2"]
            )
            I_2d = intensity_2d(p_target, P_target, A, alpha, beta, B, C_den, D, E)
        except Exception as e:
            print(f"Error building 2D model for {em}: {e}")
            I_2d = 0.0

        # Print row
        print(f"{em:<30} {wl:>10.1f} {is_vuv:>8} {I_press:>14.4e} {I_pow:>14.4e} {I_2d:>14.4e}")

        # Accumulate totals
        if is_vuv == "VUV":
            vuv_press += I_press
            vuv_pow   += I_pow
            vuv_2d    += I_2d
        else:
            nonvuv_press += I_press
            nonvuv_pow   += I_pow
            nonvuv_2d    += I_2d

    # Totals
    total_press = vuv_press + nonvuv_press
    total_pow   = vuv_pow + nonvuv_pow
    total_2d    = vuv_2d + nonvuv_2d

    print("-"*110)
    print(f"{'VUV total':<30} {'':>10} {'':>8} {vuv_press:>14.4e} {vuv_pow:>14.4e} {vuv_2d:>14.4e}")
    print(f"{'non-VUV total':<30} {'':>10} {'':>8} {nonvuv_press:>14.4e} {nonvuv_pow:>14.4e} {nonvuv_2d:>14.4e}")
    print(f"{'GRAND TOTAL':<30} {'':>10} {'':>8} {total_press:>14.4e} {total_pow:>14.4e} {total_2d:>14.4e}")

    print(f"\nThe 2‑D model is: I(p,P) = A p^α P^β / (1 + B p + C P + D p^2 + E P^2)")
    print("Derived from slice constraints; non‑separable because of additive denominator.")
    print("Done.")


if __name__ == "__main__":
    main()