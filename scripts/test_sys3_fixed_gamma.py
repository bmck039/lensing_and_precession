#!/usr/bin/env python3
"""Compute sys3 contour with fixed gamma_P to isolate source of ribbed artifacts.
Runs multiple contours for a small set of fixed gamma values and saves plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from lensing_and_precession.modules.default_params_ver2 import *
from scripts.helper_functions import evaluate_mismatch_2D, set_to_location_class


def main():
    sky_location = {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": 8 * np.pi / 9,
        "phi_J": np.pi / 4,
    }

    lens_params, NP_params, RP_params = set_to_location_class(
        sky_location, lens_params_1, NP_params_1, RP_params_1
    )

    # System 3 with mcz=40, td=0.03, I=0.5
    mcz = 40
    I = 0.5
    td = 0.03
    lens_params["mcz"] = mcz * solar_mass
    RP_params["mcz"] = mcz * solar_mass
    lens_params["I"] = I
    lens_params["td"] = td

    # Grid parameters
    min_omega, max_omega = 0, 4
    min_theta, max_theta = 0, 15
    o_res, t_res = 41, 151

    # Choose a few fixed gamma values to test
    # fixed_gammas = [0.0, np.pi/2, np.pi, 3*np.pi/2]
    fixed_gammas = [0.0]

    for g in fixed_gammas:
        rp = RP_params.copy()
        rp["gamma_P"] = g

        print(f"Computing fixed-gamma contour: gamma_P={g:.3f}")
        result = evaluate_mismatch_2D(
            rp,
            lens_params,
            ("omega_tilde", "theta_tilde"),
            ((min_omega, max_omega), (min_theta, max_theta)),
            (o_res, t_res),
            multithread=False,
            pbar=True,
            optimize_gamma=False,
        )

        omega_list = result["omega_tilde"]
        theta_list = result["theta_tilde"]
        eps_data = np.transpose(result["results"])  # (len(theta), len(omega))
        omega_data, theta_data = np.meshgrid(omega_list, theta_list)

        grid_min_idx = np.unravel_index(np.argmin(eps_data), eps_data.shape)
        grid_min_eps = eps_data[grid_min_idx]
        grid_min_omega = omega_data[grid_min_idx]
        grid_min_theta = theta_data[grid_min_idx]

        plt.figure(figsize=(8, 6))
        contourPlot = plt.contourf(omega_data, theta_data, eps_data, levels=100, cmap=cm.jet)
        plt.colorbar(contourPlot).set_label(label=r"$epsilon(tilde{h}_{rm L}, tilde{h}_{rm P})$")
        plt.plot(grid_min_omega, grid_min_theta, 'g.', markersize=10, label='Grid minimum')
        plt.xlabel(r"$tilde{Omega}$")
        plt.ylabel(r"$tilde{theta}$")
        plt.title(f"sys3 fixed gamma, mcz={mcz}, td={td}, I={I}, gamma={g:.3f}")
        plt.legend()
        plt.tight_layout()

        outfile = f"./Figures/test_sys3_fixed_gamma_{g:.3f}.png"
        plt.savefig(outfile, dpi=150)
        print("Saved:", outfile)


if __name__ == "__main__":
    main()
