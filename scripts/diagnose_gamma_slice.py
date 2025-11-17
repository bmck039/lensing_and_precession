#!/usr/bin/env python3
"""Diagnose mismatch vs gamma_P for selected (omega_tilde, theta_tilde) points.
Helps identify whether discontinuities come from gamma optimization.
Run:
  python ./python/diagnose_gamma_slice.py
Optionally set GW_INTERP_NAN=0 for raw phase diagnostics.
"""
import numpy as np
import matplotlib.pyplot as plt
from lensing_and_precession.modules.default_params_ver2 import *
from lensing_and_precession.modules.functions_ver2 import mismatch
from scripts.helper_functions import set_to_location_class
from lensing_and_precession.modules.Classes_ver2 import Precessing

# Sky/location baseline
sky_location = {
    "theta_S": np.pi / 4,
    "phi_S": 0,
    "theta_J": 8 * np.pi / 9,
    "phi_J": np.pi / 4,
}

lens_params, NP_params, RP_params = set_to_location_class(
    sky_location, lens_params_1, NP_params_1, RP_params_1
)

# System 3 specifics
lens_params["mcz"] = 40 * solar_mass
RP_params["mcz"] = 40 * solar_mass
lens_params["I"] = 0.5
lens_params["td"] = 0.03

# Points to sample (omega, theta)
points = [
    (1.5, 7.0),
    (2.5, 11.0),
    (3.3, 7.6),  # grid minimum location
    (1.43, 14.76),  # SHGO minimum location
]

num_gamma = 181  # 2 degree resolution ~ smooth sampling
gamma_arr = np.linspace(0, 2*np.pi, num_gamma, endpoint=False)

plt.figure(figsize=(10, 6))

for (om, th) in points:
    mismatches = []
    for g in gamma_arr:
        t_params = RP_params.copy()
        t_params["omega_tilde"] = om
        t_params["theta_tilde"] = th
        t_params["gamma_P"] = g
        res = mismatch(t_params, lens_params, prec_Class=Precessing)
        mismatches.append(res["mismatch"])
    mismatches = np.asarray(mismatches)
    plt.plot(gamma_arr, mismatches, label=f"(Ω={om}, θ={th})")

plt.xlabel(r"$gamma_P$")
plt.ylabel("mismatch")
plt.title("Mismatch vs $gamma_P$ for selected points (sys3, mcz=40, td=0.03, I=0.5)")
plt.legend()
plt.tight_layout()
outfile = "./Figures/gamma_slice_diagnostic.png"
plt.savefig(outfile, dpi=150)
print("Diagnostic plot saved:", outfile)
