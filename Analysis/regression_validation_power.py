import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


saturation = True
extended = True
exponential = True
generalised = False

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

# -------------------
# Generalised model
# -------------------
def generalized_model(x, A, alpha, B, n):
    return A * x**alpha / (1 + B*x)**n


# -------------------
# Setup
# -------------------
os.makedirs("power_validation_plots", exist_ok=True)

# Load new dataset
df_new = pd.read_csv("power_analysed_data_100_2000.csv")
x_new = df_new["Power (W)"].values

# Load regression parameters
df_reg = pd.read_csv("power_regression_results.csv")

results = []

# -------------------
# Loop through emissions
# -------------------
for _, row in df_reg.iterrows():
    column = row["Emission"]

    if column not in df_new.columns:
        print(f"Skipping {column} (not in new dataset)")
        continue

    y_new = df_new[column].values

    # Get fitted parameters
    A = row["Saturation A"]
    alpha = row["Saturation alpha"]
    B = row["Saturation B"]

    # Extended model parameters
    A_ext = row.get("Extended A", np.nan)
    alpha_ext = row.get("Extended alpha", np.nan)
    B_ext = row.get("Extended B", np.nan)
    C_ext = row.get("Extended C", np.nan)

    # Exponential model parameters
    A_exp = row.get("Exp A", np.nan)
    alpha_exp = row.get("Exp alpha", np.nan)
    gamma_exp = row.get("Exp gamma", np.nan)

    # Generalised model parameters
    A_gen = row.get("Gen A", np.nan)
    alpha_gen = row.get("Gen alpha", np.nan)
    B_gen = row.get("Gen B", np.nan)
    n_gen = row.get("Gen n", np.nan)


    # Skip if fit failed previously
    if np.isnan(A):
        continue

    # -------------------
    # Predict using OLD model
    # -------------------
    y_pred = saturation_model(x_new, A, alpha, B)


    ss_res = np.sum((y_new - y_pred)**2)
    ss_tot = np.sum((y_new - np.mean(y_new))**2)

    # -------------------
    # Extended model prediction
    # -------------------
    if not np.isnan(A_ext):
        y_pred_ext = extended_model(x_new, A_ext, alpha_ext, B_ext, C_ext)

        ss_res_ext = np.sum((y_new - y_pred_ext)**2)
        r2_ext = 1 - (ss_res_ext / ss_tot)
    else:
        y_pred_ext = None
        r2_ext = np.nan

    # -------------------
    # Exponential model prediction
    # -------------------
    if not np.isnan(A_exp):
        y_pred_exp = exp_model(x_new, A_exp, alpha_exp, gamma_exp)

        ss_res_exp = np.sum((y_new - y_pred_exp)**2)
        r2_exp = 1 - (ss_res_exp / ss_tot)
    else:
        y_pred_exp = None
        r2_exp = np.nan

    # -------------------
    # Generalised model prediction
    # -------------------
    if not np.isnan(A_gen):
        y_pred_gen = generalized_model(x_new, A_gen, alpha_gen, B_gen, n_gen)

        ss_res_gen = np.sum((y_new - y_pred_gen)**2)
        r2_gen = 1 - (ss_res_gen / ss_tot)
    else:
        y_pred_gen = None
        r2_gen = np.nan

    # -------------------
    # Compute NEW R²
    # -------------------

    r2_new = 1 - (ss_res / ss_tot)

    # -------------------
    # Plot comparison
    # -------------------
    plt.figure()

    # New data
    plt.scatter(x_new, y_new, label="New data")

    if saturation:
        plt.plot(x_new, y_pred, linestyle="--",
                label=f"Saturation (R²={r2_new:.3f})")

    # Extended model
    if extended:
        plt.plot(x_new, y_pred_ext, linestyle="-.",
                 label=f"Extended (R²={r2_ext:.3f})")
        
    if exponential:
        plt.plot(x_new, y_pred_exp, linestyle=":",
             label=f"Exp (R²={r2_exp:.3f})")
        
    if generalised:
        plt.plot(x_new, y_pred_gen, linestyle="-.",
                label=f"Gen (n={n_gen:.2f}, R²={r2_gen:.3f})")

    plt.xlabel("Power (W)")
    plt.ylabel(column)
    plt.title(f"Validation: {column}")
    plt.legend()

    filename = safe_filename(column)
    plt.savefig(f"power_validation_plots/{filename}.png", bbox_inches="tight", dpi=400)
    plt.close()

    # -------------------
    # Store results
    # -------------------
    results.append({
        "Emission": column,
        "Saturation Validation R^2": r2_new,
        "Extended Validation R^2": r2_ext,
        "Exponential Validation R^2": r2_exp,
        "Generalised Validation R^2": r2_gen,
        "Original Saturation R^2": row["Saturation R^2"],
        "Original Extended R^2": row.get("Extended R^2", np.nan),
        "Original Exponential R^2": row.get("Exp R^2", np.nan),
        "Original Generalised R^2": row.get("Gen R^2", np.nan)
    })
# Save validation results
# -------------------
results_df = pd.DataFrame(results)
results_df.to_csv("power_validation_results.csv", index=False)

print("Validation complete! Check validation_plots/ and validation_results.csv")