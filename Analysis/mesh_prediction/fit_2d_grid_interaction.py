#!/usr/bin/env python3
"""
Fit the interaction model in log‑space:
    log(I) = log(A) + α*log(p) + β*log(P) - log(1 + B p + C P + D p^2 + E P^2 + F p P)

This minimises *relative* errors, fixing the 10‑25% discrepancies.
Usage: python fit_2d_grid_interaction_log.py /path/to/data_root
"""

import os, sys, csv
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ----- column info -----
COLUMN_INFO = {
    "r549: O_1s -> O_1d": 557.7, "r550: O_5s -> O": 135.6, "r551: O_5s -> O": 135.9,
    "r552: O_3s -> O": 130.2, "r553: O_3s -> O": 130.5, "r554: O_3s -> O": 130.6,
    "r555: O_5p -> O_5s": 777.2, "r556: O_5p -> O_5s": 777.4, "r557: O_5p -> O_5s": 777.5,
    "r558: O_3p -> O_3s": 844.6, "r559: O_3p -> O_3s": 844.6, "r560: O_3p -> O_3s": 844.7,
}
VUV_THRESHOLD = 200.0

# ----- model in log space (the function returns log(I)) -----
def log_interaction_model(P_grid, logA, alpha, beta, B, C, D, E, F):
    p, P = P_grid
    denom = 1.0 + B * p + C * P + D * p**2 + E * P**2 + F * p * P
    # log of intensity
    return logA + alpha * np.log(p) + beta * np.log(P) - np.log(denom)

# ----- fitting routine -----
def fit_line_log(p_grid, P_grid, I_grid):
    """Fit in log space; returns popt (A, alpha, beta, B, C, D, E, F) and R2 on log scale."""
    p_flat = p_grid.flatten()
    P_flat = P_grid.flatten()
    I_flat = I_grid.flatten()
    mask = (I_flat > 0) & np.isfinite(I_flat)
    p_flat, P_flat, I_flat = p_flat[mask], P_flat[mask], I_flat[mask]
    if len(I_flat) < 8:
        return None, None, None

    logI = np.log(I_flat)
    # initial guesses for logA, alpha, beta, B, C, D, E, F
    p0 = [np.log(np.median(I_flat)), 1.0, 1.0, 0.01, 0.001, 0.0, 0.0, 0.0]

    try:
        popt_log, _ = curve_fit(log_interaction_model, (p_flat, P_flat), logI,
                                 p0=p0, maxfev=20000)
        # Compute R2 in log space
        logI_pred = log_interaction_model((p_flat, P_flat), *popt_log)
        ss_res = np.sum((logI - logI_pred)**2)
        ss_tot = np.sum((logI - np.mean(logI))**2)
        r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Convert parameters back to original A
        A = np.exp(popt_log[0])
        full_popt = [A] + list(popt_log[1:])
        return full_popt, r2_log, logI_pred  # logI_pred for diagnostics
    except Exception as e:
        print(f"    Log‑fit failed: {e}")
        return None, None, None

# ----- save CSV -----
def save_csv(fitted, outpath):
    with open(outpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Emission line", "A", "alpha", "beta", "B", "C", "D", "E", "F", "R^2_log"])
        for name, (popt, r2, _) in fitted.items():
            w.writerow([name] + popt + [r2])
    print(f"Saved parameters to {outpath}")

# ----- main -----
def main():
    if len(sys.argv) != 2:
        print("Usage: python fit_2d_grid_interaction_log.py /path/to/data_root")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print("Error: not a directory")
        sys.exit(1)

    # Discover grid (same as before)
    power_levels = sorted(set(
        float(d.name.lower().rstrip('w'))
        for d in root.iterdir() if d.is_dir() and d.name.lower().rstrip('w').replace('.','',1).isdigit()
    ))
    pressure_levels_set = set()
    dir_map = {}
    for pwr_dir in root.iterdir():
        if not pwr_dir.is_dir(): continue
        try: power = float(pwr_dir.name.lower().rstrip('w'))
        except: continue
        for pres_dir in pwr_dir.iterdir():
            if not pres_dir.is_dir(): continue
            parts = pres_dir.name.split('_')
            if len(parts) < 2: continue
            try: pressure = float(parts[1])
            except: continue
            pressure_levels_set.add(pressure)
            k_file = pres_dir / "K.csv"
            if k_file.is_file(): dir_map[(power, pressure)] = k_file

    pressure_levels = sorted(pressure_levels_set)
    power_levels = sorted(power_levels)
    print(f"Found {len(power_levels)} powers, {len(pressure_levels)} pressures")

    P_grid, p_grid = np.meshgrid(power_levels, pressure_levels)
    all_cols = list(COLUMN_INFO.keys())
    intensity_data = {c: np.full_like(P_grid, np.nan) for c in all_cols}

    for (power, pressure), fpath in dir_map.items():
        i_pwr = power_levels.index(power)
        i_prs = pressure_levels.index(pressure)
        with open(fpath) as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers: continue
            col_idx = {}
            for ic, h in enumerate(headers):
                h = h.strip()
                if h in all_cols: col_idx[h] = ic
            last_row = None
            for row in reader:
                if any(cell.strip() for cell in row): last_row = row
            if last_row is None: continue
            for cname, ic in col_idx.items():
                try: val = float(last_row[ic].strip())
                except: val = np.nan
                intensity_data[cname][i_prs, i_pwr] = val

    # Output folder
    out_dir = root / "fits_log"
    out_dir.mkdir(exist_ok=True)

    fitted_params = {}
    for col in all_cols:
        print(f"\nFitting {col} (log‑space) ...")
        popt, r2_log, log_pred = fit_line_log(p_grid, P_grid, intensity_data[col])
        if popt is not None:
            fitted_params[col] = (popt, r2_log, log_pred)
            A, alpha, beta, B, C, D, E, F = popt
            print(f"  R²_log = {r2_log:.6f}")
            print(f"  A={A:.4e}, α={alpha:.4f}, β={beta:.4f}")
            print(f"  B={B:.4e}, C={C:.4e}, D={D:.4e}, E={E:.4e}, F={F:.4e}")

            # Show relative errors at grid points
            I_obs = intensity_data[col].flatten()
            mask = np.isfinite(I_obs) & (I_obs > 0)
            I_obs = I_obs[mask]
            I_pred = np.exp(log_pred)  # log_pred already for valid mask
            rel_err = np.abs(I_pred - I_obs) / I_obs * 100.0
            print(f"  Max relative error: {np.max(rel_err):.2f}%, Mean: {np.mean(rel_err):.2f}%")
        else:
            print("  Fit failed.")

    if fitted_params:
        # Save CSV
        save_csv(fitted_params, out_dir / "fitting_interaction_log.csv")

        # Plot individual contours
        p_fine = np.linspace(min(pressure_levels), max(pressure_levels), 100)
        P_fine = np.linspace(min(power_levels), max(power_levels), 100)
        Pp_f, pp_f = np.meshgrid(P_fine, p_fine)
        for col, (popt, _, _) in fitted_params.items():
            A, alpha, beta, B, C, D, E, F = popt
            # Compute intensity directly on fine grid
            denom = 1.0 + B*pp_f + C*Pp_f + D*pp_f**2 + E*Pp_f**2 + F*pp_f*Pp_f
            I_surf = A * (pp_f**alpha) * (Pp_f**beta) / denom
            plt.figure(figsize=(8,6))
            plt.contourf(P_fine, p_fine, I_surf, levels=20, cmap='viridis')
            plt.colorbar(label='Intensity (arb. units)')
            plt.xlabel('Power (W)'); plt.ylabel('Pressure (mTorr)')
            _, r2_for_title, _ = fitted_params[col]
            plt.title(f'{col} (log‑fit)\nR²_log = {r2_for_title:.4f}')
            for i, p in enumerate(pressure_levels):
                for j, P in enumerate(power_levels):
                    if np.isfinite(intensity_data[col][i,j]):
                        plt.plot(P, p, 'ko', markersize=3)
            safe = col.replace(':','_').replace(' ','_').replace('->','to')
            plt.tight_layout(); plt.savefig(out_dir / f"{safe}.png", dpi=150); plt.close()

        # Aggregates...
        vuv_cols = [c for c,wl in COLUMN_INFO.items() if wl<VUV_THRESHOLD]
        nonvuv_cols = [c for c,wl in COLUMN_INFO.items() if wl>=VUV_THRESHOLD]
        vuv_surf = np.zeros_like(pp_f)
        nonvuv_surf = np.zeros_like(pp_f)
        for col, (popt,_,_) in fitted_params.items():
            A,alpha,beta,B,C,D,E,F = popt
            denom = 1.0 + B*pp_f + C*Pp_f + D*pp_f**2 + E*Pp_f**2 + F*pp_f*Pp_f
            I_line = A*(pp_f**alpha)*(Pp_f**beta)/denom
            if col in vuv_cols: vuv_surf += I_line
            else: nonvuv_surf += I_line
        total_surf = vuv_surf + nonvuv_surf
        for lab,data,fname in [("VUV",vuv_surf,"VUV_log.png"),
                               ("nonVUV",nonvuv_surf,"nonVUV_log.png"),
                               ("Total",total_surf,"Total_log.png")]:
            plt.figure(figsize=(8,6))
            plt.contourf(P_fine,p_fine,data,20,cmap='plasma')
            plt.colorbar(); plt.xlabel('Power'); plt.ylabel('Pressure')
            plt.title(lab); plt.tight_layout()
            plt.savefig(out_dir/fname,dpi=150); plt.close()
        print("Plots saved.")

    print("\nDone. Results in", out_dir)

if __name__ == "__main__":
    main()