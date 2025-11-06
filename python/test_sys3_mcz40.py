#!/usr/bin/env python3
"""
Test script to reproduce sys3, mcz=40 contour and compare with reference.
This will help diagnose where mismatches are coming from.
"""
import sys
sys.path.insert(0, "./lensing_and_precession/")

from modules.default_params_ver2 import *
from helper_classes import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

print(f"Generating {o_res}x{t_res} grid for sys3, mcz={mcz}, td={td}, I={I}")
print(f"Sky location: theta_S={sky_location['theta_S']:.3f}, phi_S={sky_location['phi_S']:.3f}")
print(f"              theta_J={sky_location['theta_J']:.3f}, phi_J={sky_location['phi_J']:.3f}")
print()

# Test a known-good point
test_params = RP_params.copy()
test_params["omega_tilde"] = 1.5
test_params["theta_tilde"] = 7.0
test_eps = evaluate_mismatch(test_params, lens_params)
print(f"Test point (1.5, 7.0): epsilon = {test_eps}")

# Test a spike location
test_params2 = RP_params.copy()
test_params2["omega_tilde"] = 2.5  
test_params2["theta_tilde"] = 11.0
test_eps2 = evaluate_mismatch(test_params2, lens_params)
print(f"Test point (2.5, 11.0): epsilon = {test_eps2}")

# Generate mismatch grid
mismatch_dict = evaluate_mismatch_2D(
    RP_params, 
    lens_params, 
    ("omega_tilde", "theta_tilde"), 
    ((min_omega, max_omega), (min_theta, max_theta)), 
    (o_res, t_res),
    multithread=False,  # force sequential to rule out Pool-related deadlocks
    pbar=True,
    gamma_points=21     # reduce per-cell gamma grid for speed during sweeps
)

omega_list = mismatch_dict["omega_tilde"]
theta_list = mismatch_dict["theta_tilde"]
eps_data = np.transpose(mismatch_dict["results"])

omega_data, theta_data = np.meshgrid(omega_list, theta_list)

# Find global minimum
eps, omega_best, theta_best = find_optimal_RP_mismatch(
    lens_params, RP_params, 
    omega_bounds=(min_omega, max_omega), 
    theta_bounds=(min_theta, max_theta)
)

print(f"\nGlobal minimum from SHGO:")
print(f"  epsilon = {eps:.6f}")
print(f"  omega   = {omega_best:.6f}")
print(f"  theta   = {theta_best:.6f}")

# Find minimum in grid
grid_min_idx = np.unravel_index(np.argmin(eps_data), eps_data.shape)
grid_min_eps = eps_data[grid_min_idx]
grid_min_omega = omega_data[grid_min_idx]
grid_min_theta = theta_data[grid_min_idx]

print(f"\nGrid minimum:")
print(f"  epsilon = {grid_min_eps:.6f}")
print(f"  omega   = {grid_min_omega:.6f}")
print(f"  theta   = {grid_min_theta:.6f}")

# Check for NaN/Inf
num_nan = np.sum(~np.isfinite(eps_data))
print(f"\nData quality:")
print(f"  NaN/Inf count: {num_nan}/{eps_data.size}")
print(f"  Min epsilon: {np.nanmin(eps_data):.6f}")
print(f"  Max epsilon: {np.nanmax(eps_data):.6f}")

# Plot
plt.figure(figsize=(8, 6))
contourPlot = plt.contourf(omega_data, theta_data, eps_data, levels=100, cmap=cm.jet)
plt.colorbar(contourPlot).set_label(label=r"$\epsilon(\tilde{h}_{\rm L}, \tilde{h}_{\rm P})$")
plt.plot(omega_best, theta_best, 'r.', markersize=10, label='SHGO minimum')
plt.plot(grid_min_omega, grid_min_theta, 'g.', markersize=10, label='Grid minimum')
plt.xlabel(r"$\tilde{\Omega}$")
plt.ylabel(r"$\tilde{\theta}$")
plt.title(f"sys3, mcz={mcz}, td={td}, I={I}")
plt.legend()
plt.tight_layout()

output_file = f"./Figures/test_sys3_mcz{mcz}_td{td}_I{I}.png"
plt.savefig(output_file, dpi=150)
print(f"\nPlot saved to: {output_file}")
