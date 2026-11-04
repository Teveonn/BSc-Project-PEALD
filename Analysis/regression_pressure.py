import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.optimize import curve_fit

# what regressions to perform
linear = True 
power_law = True
saturation = True
extended = True
exponential = True
generalised = True



# creating variables incase regressions aren't ran
column = 0
r2_linear = 0
alpha = 0
r2_power = 0
A_sat = 0
alpha_sat = 0
B_sat = 0
r2_sat = 0
A_ext = 0
alpha_ext = 0
B_ext = 0
C_ext = 0
r2_ext = 0

# -------------------
# Helper: safe filename
# -------------------
def safe_filename(name):
    name = name.replace("->", "to")
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    return name.replace(" ", "_")

# -------------------
# Saturation model
# -------------------
def saturation_model(x, A, alpha, B):
    return A * x**alpha / (1 + B * x)

# -------------------
# Extended model
# -------------------
def extended_model(x, A, alpha, B, C):
    return A * x**alpha / (1 + B*x + C*x**2)

# -------------------
# Exponential model
# -------------------
def exp_model(x, A, alpha, gamma):
    return A * x**alpha * np.exp(-gamma * x)

# generalised model
def generalized_model(x, A, alpha, B):
    return A * x**alpha * np.exp(-B*x)

# -------------------
# Setup
# -------------------
os.makedirs("pressure_regression_plots", exist_ok=True)

df = pd.read_csv("pressure_analysed_data_2_400.csv")
x = df["Pressure (mTorr)"].values

results = []

# -------------------
# Main loop
# -------------------
for column in df.columns:
    if column == "Pressure (mTorr)":
        continue

    y = df[column].values



    # -------------------
    # Linear fit
    # -------------------
    m, c = np.polyfit(x, y, 1)
    y_linear = m * x + c

    ss_tot = np.sum((y - np.mean(y))**2)
    r2_linear = 1 - np.sum((y - y_linear)**2) / ss_tot


    # -------------------
    # Power-law fit
    # -------------------
    mask = (x > 0) & (y > 0)
    alpha, logA = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    A_power = np.exp(logA)
    y_power = A_power * x**alpha

    r2_power = 1 - np.sum((y - y_power)**2) / ss_tot


    # -------------------
    # Saturation fit
    # -------------------
    try:
        popt, _ = curve_fit(
            saturation_model,
            x,
            y,
            p0=[A_power, alpha, 0.001],  # good initial guess
            maxfev=10000
        )

        A_sat, alpha_sat, B_sat = popt
        y_sat = saturation_model(x, *popt)

        r2_sat = 1 - np.sum((y - y_sat)**2) / ss_tot

    except RuntimeError:
        A_sat, alpha_sat, B_sat, r2_sat = np.nan, np.nan, np.nan, np.nan
        y_sat = None


# -------------------
# Extended model fit (SCALED)
# -------------------
    try:
        # --- Scale data ---
        x_scale = np.max(x)
        y_scale = np.max(y)

        if x_scale == 0 or y_scale == 0:
            raise ValueError("Scaling failed: zero max value")

        x_scaled = x / x_scale
        y_scaled = y / y_scale

        # --- Fit using scaled data ---
        popt_scaled, _ = curve_fit(
            extended_model,
            x_scaled,
            y_scaled,
            p0=[1.0, 1.0, 0.1, 0.01],   # stable, scale-independent guesses
            maxfev=200000
        )

        # --- Extract scaled parameters ---
        A_s, alpha_ext, B_s, C_s = popt_scaled

        # --- Convert parameters back to original scale ---
        A_ext = A_s * y_scale / (x_scale ** alpha_ext)
        B_ext = B_s / x_scale
        C_ext = C_s / (x_scale ** 2)

        # --- Generate fit in original units ---
        y_ext = extended_model(x, A_ext, alpha_ext, B_ext, C_ext)

        # --- Compute R² ---
        r2_ext = 1 - np.sum((y - y_ext)**2) / ss_tot

    except Exception as e:
        print(f"Extended fit failed for {column}: {e}")
        A_ext, alpha_ext, B_ext, C_ext, r2_ext = np.nan, np.nan, np.nan, np.nan, np.nan
        y_ext = None


    # -------------------
    # Exponential model fit
    # -------------------
    try:
        popt_exp, _ = curve_fit(
            exp_model,
            x,
            y,
            p0=[A_power, alpha, 1e-4],  # initial guess for gamma is important
            maxfev=20000
        )

        A_exp, alpha_exp, gamma_exp = popt_exp
        y_exp = exp_model(x, *popt_exp)

        r2_exp = 1 - np.sum((y - y_exp)**2) / ss_tot
    
    except RuntimeError:
        A_exp, alpha_exp, gamma_exp, r2_exp = np.nan, np.nan, np.nan, np.nan
        y_exp = None


   # -------------------
    # Generalised (peak) model fit – robust version
    # -------------------
    try:
        # 1. Scale y to avoid gigantic A values
        y_max = np.max(y)
        if y_max == 0:
            raise ValueError("All y values are zero")
        y_scaled = y / y_max

        # 2. Intelligent initial guess for B from data peak location
        idx_peak = np.argmax(y)
        x_peak_guess = x[idx_peak]
        alpha_guess = alpha if not np.isnan(alpha) else 1.0
        B0 = alpha_guess / x_peak_guess if x_peak_guess > 0 else 0.05
        B0 = max(B0, 1e-6)   # ensure positive

        # Scale A initial guess accordingly
        A_power_scaled = A_power / y_max
        p0 = [A_power_scaled, alpha_guess, B0]

        # 3. Fit using Levenberg‑Marquardt (no bounds, more stable)
        popt_scaled, pcov = curve_fit(
            generalized_model,   # original function works because y is scaled
            x, y_scaled,
            p0=p0,
            method='lm',         # <<< key change
            maxfev=20000
        )

        A_scaled, alpha_gen, B_gen = popt_scaled
        A_gen = A_scaled * y_max
        y_gen = generalized_model(x, A_gen, alpha_gen, B_gen)

        # 4. Compute R²
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_gen = 1 - np.sum((y - y_gen)**2) / ss_tot

    except Exception as e:
        # Detailed error message helps diagnose remaining issues
        print(f"Generalised fit failed for {column}: {e}")
        print(f"  Data: x_peak={x[np.argmax(y)]:.1f}, y_max={y_max:.2e}")
        print(f"  p0 = {p0}")
        A_gen = alpha_gen = B_gen = r2_gen = np.nan
        y_gen = None

    # -------------------
    # Plot
    # -------------------
    plt.figure()

    plt.scatter(x, y, label="Data")

    if linear:
        plt.plot(x, y_linear, label=f"Linear (R²={r2_linear:.3f})")

    if power_law:
        plt.plot(x, y_power, linestyle="--",
                label=f"Power-law (α={alpha:.2f}, R²={r2_power:.3f})")

    if saturation:
        plt.plot(x, y_sat, linestyle=":",
                 label=f"Saturation (α={alpha_sat:.2f}, R²={r2_sat:.3f})")
        
    if extended:
        plt.plot(x, y_ext, linestyle="-.",
                 label=f"Extended (α={alpha_ext:.2f}, R²={r2_ext:.3f})")
        
    if exponential:
        plt.plot(x, y_exp, linestyle="--",
             label=f"Exp (α={alpha_exp:.2f}, R²={r2_exp:.3f})")
        
    if generalised and y_gen is not None:
        plt.plot(x, y_gen, linestyle="--",
                label=f"Gen (α={alpha_gen:.2f}, R²={r2_gen:.3f})")

    plt.xlabel("Pressure (mTorr)")
    plt.ylabel("Flux")
    plt.title(f"{column} flux vs Pressure")
    plt.legend()

    filename = safe_filename(column)
    plt.savefig(f"pressure_regression_plots/{filename}regression.png", bbox_inches="tight", dpi=400)
    plt.close()

    # -------------------
    # Store results
    # -------------------
    results.append({
        "Emission": column,
        "Linear R^2": r2_linear,
        "Power-law alpha": alpha,
        "Power-law R^2": r2_power,
        "Saturation A": A_sat,
        "Saturation alpha": alpha_sat,
        "Saturation B": B_sat,
        "Saturation R^2": r2_sat,
        "Extended A": A_ext,
        "Extended alpha": alpha_ext,
        "Extended B": B_ext,
        "Extended C": C_ext,
        "Extended R^2": r2_ext,
        "Exp A": A_exp,
        "Exp alpha": alpha_exp,
        "Exp gamma": gamma_exp,
        "Exp R^2": r2_exp,
        "Gen A": A_gen,
        "Gen alpha": alpha_gen,
        "Gen B": B_gen,
        "Gen R^2": r2_gen
        })

# -------------------
# Save results
# -------------------
results_df = pd.DataFrame(results)
results_df.to_csv("pressure_regression_results.csv", index=False)

