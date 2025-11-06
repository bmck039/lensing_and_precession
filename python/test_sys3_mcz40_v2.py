#!/usr/bin/env python3
"""
Test script to reproduce sys3, mcz=40 contour using PrecessingV2 (odeint)
and compare behavior against the default Precessing implementation.

- Single-threaded grid to avoid Pool deadlocks
- Optional reduced gamma grid for speed (gamma_points=21)
- Prints SHGO vs. grid minima (both evaluated with PrecessingV2)
"""
import sys
sys.path.insert(0, "./lensing_and_precession/")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import shgo

from modules.default_params_ver2 import *
from modules.Classes_ver2 import PrecessingV2
from helper_classes import *
from helper_functions import evaluate_mismatch, evaluate_mismatch_2D, set_to_location_class

# System 3 (Taman random) parameters
sky_location = {
    "theta_S": np.pi / 4,
    "phi_S": 0,
    "theta_J": 8 * np.pi / 9,
    "phi_J": np.pi / 4,
}

# Set parameters
lens_params, NP_params, RP_params = set_to_location_class(
    sky_location, lens_params_1, NP_params_1, RP_params_1
)

# System 3 with mcz=40, td=0.03, I=0.5 (from reference PDF)
mcz = 40
I = 0.5
td = 0.03

lens_params["mcz"] = mcz * solar_mass
RP_params["mcz"] = mcz * solar_mass
lens_params["I"] = I
lens_params["td"] = td

# Grid parameters matching reference (0-4 omega, 0-15 theta)
min_omega, max_omega = 0, 4
min_theta, max_theta = 0, 15
o_res, t_res = 41, 151  # Match Tien's resolution

print(f"[V2] Generating {o_res}x{t_res} grid for sys3, mcz={mcz}, td={td}, I={I}")
print(f"Sky location: theta_S={sky_location['theta_S']:.3f}, phi_S={sky_location['phi_S']:.3f}")
print(f"              theta_J={sky_location['theta_J']:.3f}, phi_J={sky_location['phi_J']:.3f}\n")

# Test a known-good point
v2_params = RP_params.copy()
v2_params["omega_tilde"] = 1.5
v2_params["theta_tilde"] = 7.0
v2_eps = evaluate_mismatch(v2_params, lens_params, prec_Class=PrecessingV2)
print(f"[V2] Test point (1.5, 7.0): epsilon = {v2_eps}")

# Test another point often near ridges
v2_params2 = RP_params.copy()
v2_params2["omega_tilde"] = 2.5
v2_params2["theta_tilde"] = 11.0
v2_eps2 = evaluate_mismatch(v2_params2, lens_params, prec_Class=PrecessingV2)
print(f"[V2] Test point (2.5, 11.0): epsilon = {v2_eps2}")

# Generate mismatch grid with PrecessingV2
dict_v2 = evaluate_mismatch_2D(
    RP_params,
    lens_params,
    ("omega_tilde", "theta_tilde"),
    ((min_omega, max_omega), (min_theta, max_theta)),
    (o_res, t_res),
    multithread=False,  # force sequential to rule out Pool-related deadlocks
    pbar=True,
    gamma_points=21,
    prec_Class=PrecessingV2,
)

omega_list = dict_v2["omega_tilde"]
theta_list = dict_v2["theta_tilde"]
eps_data = np.transpose(dict_v2["results"])  # shape: (len(theta), len(omega))

omega_data, theta_data = np.meshgrid(omega_list, theta_list)

# Find minimum in grid
grid_min_idx = np.unravel_index(np.argmin(eps_data), eps_data.shape)
grid_min_eps = eps_data[grid_min_idx]
grid_min_omega = omega_data[grid_min_idx]
grid_min_theta = theta_data[grid_min_idx]

print(f"\n[V2] Grid minimum:")
print(f"  epsilon = {grid_min_eps:.6f}")
print(f"  omega   = {grid_min_omega:.6f}")
print(f"  theta   = {grid_min_theta:.6f}")

# Optional: run SHGO using PrecessingV2 to compare with grid min
def _v2_objective(x):
    om, th = float(x[0]), float(x[1])
    p = RP_params.copy()
    p["omega_tilde"], p["theta_tilde"] = om, th
    return evaluate_mismatch(p, lens_params, prec_Class=PrecessingV2)

print("\n[V2] Running SHGO (reduced iterations for speed)...")
res_v2 = shgo(_v2_objective, ((min_omega, max_omega), (min_theta, max_theta)), iters=2, options={"ftol": 1e-4})
if not res_v2.success:
    print("[V2] Warning: SHGO did not converge successfully:", res_v2.message)

print(f"[V2] SHGO minimum:")
print(f"  epsilon = {res_v2.fun:.6f}")
print(f"  omega   = {res_v2.x[0]:.6f}")
print(f"  theta   = {res_v2.x[1]:.6f}")

# Data quality
num_nan = np.sum(~np.isfinite(eps_data))
print(f"\n[V2] Data quality:")
print(f"  NaN/Inf count: {num_nan}/{eps_data.size}")
print(f"  Min epsilon: {np.nanmin(eps_data):.6f}")
print(f"  Max epsilon: {np.nanmax(eps_data):.6f}")

# Plot
plt.figure(figsize=(8, 6))
contourPlot = plt.contourf(omega_data, theta_data, eps_data, levels=100, cmap=cm.jet)
plt.colorbar(contourPlot) #.set_label(label=r"$\\epsilon(\\tilde{h}_{\\rm L}, \\tilde{h}_{\\rm P})$")
plt.plot(res_v2.x[0], res_v2.x[1], 'r.', markersize=10, label='SHGO minimum (V2)')
plt.plot(grid_min_omega, grid_min_theta, 'g.', markersize=10, label='Grid minimum (V2)')
plt.xlabel(r"omega tilde")
plt.ylabel(r"theta tilde")
plt.title(f"sys3 (V2), mcz={mcz}, td={td}, I={I}")
plt.legend()
plt.tight_layout()

output_file = f"./Figures/test_sys3_mcz{mcz}_td{td}_I{I}_v2.png"
plt.savefig(output_file, dpi=150)
print(f"\n[V2] Plot saved to: {output_file}")
