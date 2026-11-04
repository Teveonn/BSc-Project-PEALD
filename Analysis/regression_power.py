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
generalised = False



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
def generalized_model(x, A, alpha, B, n):
    return A * x**alpha / (1 + B*x)**n

# -------------------
# Setup
# -------------------
os.makedirs("power_regression_plots", exist_ok=True)

df = pd.read_csv("power_analysed_data_100_1000.csv")
x = df["Power (W)"].values

results = []

# -------------------
# Main loop
# -------------------
for column in df.columns:
    if column == "Power (W)":
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
    # Extended model fit
    # -------------------
    try:
        popt_ext, _ = curve_fit(
            extended_model,
            x,
            y,
            p0=[A_power, alpha, 0.001, 1e-6],  # initial guess is important!
            maxfev=20000
        )

        A_ext, alpha_ext, B_ext, C_ext = popt_ext
        y_ext = extended_model(x, *popt_ext)

        r2_ext = 1 - np.sum((y - y_ext)**2) / ss_tot

    except RuntimeError:
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
    # Generalised model fit
    # -------------------
    try:
        popt_gen, _ = curve_fit(
            generalized_model,
            x,
            y,
            p0=[A_power, alpha, 0.001, 1.5],
            bounds=(
                [0, 0, 0, 0],        # lower bounds
                [np.inf, 5, 1, 5]    # upper bounds
            ),
            maxfev=20000
)

        A_gen, alpha_gen, B_gen, n_gen = popt_gen
        y_gen = generalized_model(x, *popt_gen)

        r2_gen = 1 - np.sum((y - y_gen)**2) / ss_tot

    except RuntimeError:
        A_gen, alpha_gen, B_gen, n_gen, r2_gen = np.nan, np.nan, np.nan, np.nan, np.nan
        y_gen = None
    except ValueError:
        print(f"Infeasible initial guess for {column}")
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
        
    if generalised:
        plt.plot(x, y_gen, linestyle="--",
                label=f"Gen (n={n_gen:.2f}, R²={r2_gen:.3f})")

    plt.xlabel("Power (W)")
    plt.ylabel("Flux")
    plt.title(f"{column} flux vs Power")
    plt.legend()

    filename = safe_filename(column)
    plt.savefig(f"power_regression_plots/{filename}regression.png", bbox_inches="tight", dpi=400)
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
        "Gen n": n_gen,
        "Gen R^2": r2_gen
    })

# -------------------
# Save results
# -------------------
results_df = pd.DataFrame(results)
results_df.to_csv("power_regression_results.csv", index=False)

