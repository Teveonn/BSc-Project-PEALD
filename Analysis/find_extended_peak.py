#!/usr/bin/env python3
"""
Compute the peak pressure and peak intensity for the Extended model:
    I(P) = A * P^alpha / (1 + B * P + C * P^2)

Reads parameters from a CSV file like the one provided.
"""

import csv
import math
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Column names (adjust to match your CSV exactly if needed)
# ----------------------------------------------------------------------
COL_EMISSION = "Emission"
COL_A = "Extended A"
COL_ALPHA = "Extended alpha"
COL_B = "Extended B"
COL_C = "Extended C"
COL_R2 = "Extended R^2"        # optional, for reference


def safe_float(value):
    """Convert string to float, returning None if invalid/missing."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_csv(filepath):
    """Read CSV and return a list of dicts with stripped headers."""
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def find_extended_peak(A, alpha, B, C):
    """
    Find the positive pressure P that maximises I(P) = A * P^alpha / (1 + B*P + C*P^2).

    Returns:
        (peak_pressure, peak_intensity)  or  (None, None) if no valid peak.
    """
    if A is None or alpha is None or B is None or C is None:
        return None, None

    # Coefficients of the quadratic: a2*P^2 + a1*P + a0 = 0
    a2 = C * (alpha - 2)
    a1 = B * (alpha - 1)
    a0 = alpha

    candidates = []

    # Case 1: quadratic (alpha != 2)
    if alpha != 2:
        # a2 == 0 would imply C == 0 (since alpha ≠ 2) -> degenerates to linear
        if a2 == 0:
            if a1 != 0:
                p_cand = -a0 / a1
                if p_cand > 0:
                    candidates.append(p_cand)
        else:
            discriminant = a1 * a1 - 4 * a2 * a0
            if discriminant < 0:
                pass  # no real root
            elif discriminant == 0:
                p_cand = -a1 / (2 * a2)
                if p_cand > 0:
                    candidates.append(p_cand)
            else:
                sqrt_disc = math.sqrt(discriminant)
                p1 = (-a1 + sqrt_disc) / (2 * a2)
                p2 = (-a1 - sqrt_disc) / (2 * a2)
                for p in (p1, p2):
                    if p > 0:
                        candidates.append(p)
    # Case 2: alpha = 2 -> equation becomes B*P + 2 = 0
    else:
        # a2 = 0, a1 = B, a0 = 2
        if B != 0:
            p_cand = -2 / B
            if p_cand > 0:
                candidates.append(p_cand)
        # else no solution

    if not candidates:
        return None, None

    # Evaluate I(P) for all positive candidates, pick the one with highest value
    best_p = None
    best_val = -float("inf")
    for p in candidates:
        # Safeguard against very small/zero p (though our formula should be fine)
        if p <= 0:
            continue
        try:
            denom = 1 + B * p + C * p * p
            if denom <= 0:
                continue   # model undefined (should not happen for a maximum)
            val = A * (p ** alpha) / denom
            if val > best_val:
                best_val = val
                best_p = p
        except (OverflowError, ValueError):
            continue

    if best_p is None:
        return None, None
    return best_p, best_val


def main():
    if len(sys.argv) != 2:
        print("Usage: python find_extended_peak.py <csv_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not Path(filepath).is_file():
        print(f"Error: file '{filepath}' not found.")
        sys.exit(1)

    rows = load_csv(filepath)

    print(f"\n{'='*80}")
    print(f"EXTENDED MODEL PEAK ANALYSIS")
    print(f"I(P) = A * P^α / (1 + B*P + C*P²)")
    print(f"{'='*80}")

    header = f"{'Emission':<35} {'A':>12} {'α':>8} {'B':>10} {'C':>12} {'Peak P':>12} {'Peak I':>14} {'R²':>8}"
    print(header)
    print("-"*80)

    for row in rows:
        emission = row.get(COL_EMISSION, "Unknown")
        A = safe_float(row.get(COL_A))
        alpha = safe_float(row.get(COL_ALPHA))
        B = safe_float(row.get(COL_B))
        C = safe_float(row.get(COL_C))
        r2 = safe_float(row.get(COL_R2))   # optional, may be None

        # Find peak
        peak_p, peak_i = find_extended_peak(A, alpha, B, C)

        # Format numbers for display
        a_str = f"{A:.4e}" if A is not None else "N/A"
        b_str = f"{B:.4e}" if B is not None else "N/A"
        c_str = f"{C:.4e}" if C is not None else "N/A"
        alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"

        if peak_p is None:
            peak_p_str = "No peak"
            peak_i_str = "N/A"
        else:
            peak_p_str = f"{peak_p:.4e}"
            peak_i_str = f"{peak_i:.4e}"

        print(f"{emission:<35} {a_str:>12} {alpha_str:>8} {b_str:>10} {c_str:>12} {peak_p_str:>12} {peak_i_str:>14} {r2_str:>8}")

    print("="*80)


if __name__ == "__main__":
    main()