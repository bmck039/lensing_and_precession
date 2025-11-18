from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import glob

from lensing_and_precession.modules.Classes_ver2 import *
from lensing_and_precession.modules.functions_ver2 import *
from lensing_and_precession.modules.functions_Precessing import *

from scripts.helper_classes import *
from scripts.helper_functions import *

def main():
    parser = argparse.ArgumentParser(
        description="Test convergence of delta phi for different cutoff angles."
    )
    parser.add_argument('-I', type=float, default=0.6, help='Inclination angle I (default: 0.6)')
    parser.add_argument('-td', type=float, default=0.022, help='Time delay td (default: 0.022)')
    parser.add_argument('-m', type=float, default=20, help='Chirp mass mcz (default: 20)')
    args = parser.parse_args()

    I = args.I
    td = args.td
    mass = args.m
    sky_location = {
        "theta_S": np.pi / 3,
        "phi_S": np.pi / 4,
        "theta_J": np.pi / 6,
        "phi_J": np.pi / 3
    }

    lens_params, NP_params, RP_params = set_to_location_class(
        sky_location, lens_params_1, NP_params_1, RP_params_1
    )

# td = 0.022
# m = 20 * solar_mass

# lens_params["I"] = 0.6
# lens_params["td"] = td
# lens_params["mcz"] = m
# RP_params["mcz"] = m

# # RP_params["omega_tilde"] = 3.8
# # RP_params["theta_tilde"] = 8
# RP_params["omega_tilde"] = 4
# RP_params["theta_tilde"] = 9.5
# # RP_params["gamma_P"] = optimize_mismatch_gammaP(RP_params, lens_params)["ep_min_gammaP"]

# updated_params = find_optimized_coalescence_params(
#         RP_params,
#         lens_params,
#     )
# lens_params = updated_params["updated_s_params"]
# RP_params = updated_params["updated_t_params"]
# epsilon = updated_params["updated_mismatch_results"]["mismatch"]
# idx = updated_params["updated_mismatch_results"]["index"]
# phi = updated_params["updated_mismatch_results"]["phi"]

    # print(RP_params["gamma_P"])

    I = 0.6
    td = 0.022
    mass = 20

    sky_location = {
        "theta_S": np.pi / 3,
        "phi_S": np.pi / 4,
        "theta_J": np.pi / 6,
        "phi_J": np.pi / 3
    }
    lens_params, NP_params, RP_params = set_to_location_class(
            sky_location, lens_params_1, NP_params_1, RP_params_1
        )
    lens_params["I"] = I
    lens_params["td"] = td
    lens_params["mcz"] = mass * solar_mass
    RP_params["mcz"] = mass * solar_mass
    RP_params["theta_tilde"] = 7.7
    RP_params["omega_tilde"] = 3.1
    RP_params['gamma_P'] = 5.529203070318037

    prec = Precessing(RP_params)
    # print(mismatch(lens_params, RP_params))
    # e, o, t = find_optimal_RP_mismatch(lens_params, RP_params, td, m)
    # print("Found optimal at omega: {o}, theta: {t}, epsilon: {e}".format(o=o, t=t, e=e))
    plt.figure()

    # f_list = np.linspace(20, prec.f_cut(), 1000)
    f_list = np.arange(32, 40, 0.01)
    # cutoff_list = np.flip(np.linspace(0.001, 0.0001, 5))
    start = 1.0
    n = 5
    cutoff_list = [start]
    for i in range(n):
        cutoff_list.append(cutoff_list[-1] / 2)
    for c in cutoff_list:
        prec.LdotN_threshold = c
        # y_list = prec.phase_delta_phi(f_list)
        y_list = np.unwrap(np.angle(prec.strain(f_list)))
        plt.plot(f_list, y_list, label="cutoff angle: {c}".format(c=c))

    plt.legend(loc="upper right")
    plt.xlabel(r"f $[Hz]$")
    plt.ylabel(r"$\delta \Phi$")
    plt.savefig("test.png")

if __name__ == "__main__":
    main()