from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pickle
import argparse
from os.path import exists
from os import makedirs

import time

from lensing_and_precession.modules.default_params_ver2 import *

from scripts.helper_classes import *
from scripts.helper_functions import *

def create_contour_plot(data, m, td):

    lens_params["mcz"] = m * solar_mass
    RP_params["mcz"] = m * solar_mass


    eps, omega_best, theta_best = find_optimal_RP_mismatch(lens_params, RP_params, omega_bounds=(min_omega, max_omega), theta_bounds=(min_theta, max_theta))

    y_opt, x_opt = theta_best, omega_best

    plt.figure()

    contourPlot = plt.contourf(data["omega_matrix"], data["theta_matrix"], data["epsilon_matrix"], levels=100, cmap=cm.jet)

    plt.colorbar(contourPlot).set_label(
            label=r"$\epsilon(\~h_{\rm L}, \~h_{\rm P})$",
        )

    plt.plot(x_opt, y_opt, 'r.')

    plt.xlabel(r"$\~\Omega$")
    plt.ylabel(r"$\~\theta$")

    folder = "./Figures/minimization_contours/mcz{mcz}_td{td}_I{I}/".format(mcz=mass, td=td, I=I)
    makedirs(folder, exist_ok=True)

    print("saving graph")

    plt.savefig(folder + "mcz{mcz}_td{td}_I{I}_omega{omin}-{omax}_theta{tmin}-{tmax}_{xres}x{yres}_{loc}.png".format(mcz=mass, td=td, I=I, omin=min_omega, omax=max_omega, tmin=min_theta, tmax=max_theta, xres=o_res, yres=t_res, loc=str(sky_location)))

def create_fft_plot(data, o_spacing, t_spacing):
    fft_vals = np.fft.rfft2(data["epsilon_matrix"])

    fft_vals = np.fft.fftshift(fft_vals) # moves 0Hz to center of array

    N_o = len(fft_vals)
    N_t = len(fft_vals[0])

    o_freq = np.fft.fftfreq(N_o, o_spacing)
    t_freq = np.fft.fftfreq(N_t, t_spacing)

    o_freq_matrix, t_freq_matrix = np.meshgrid(o_freq, t_freq)

    fft_vals = np.abs(fft_vals)

    fft_vals = fft_vals.T

    plt.figure()

    contourPlot = plt.contourf(o_freq_matrix, t_freq_matrix, fft_vals, levels=100, cmap=cm.jet)

    plt.colorbar(contourPlot)

    plt.xlabel(r"Frequency in $\~\Omega$")
    plt.ylabel(r"Frequency in $\~\theta$")

    folder = "./Figures/minimization_contours/mcz{mcz}_td{td}_I{I}/".format(mcz=mass, td=td, I=I)
    makedirs(folder, exist_ok=True)

    plt.savefig(folder + "fourier_mcz{mcz}_td{td}_I{I}_omega{omin}-{omax}_theta{tmin}-{tmax}_{xres}x{yres}.png".format(mcz=mass, td=td, I=I, omin=min_omega, omax=max_omega, tmin=min_theta, tmax=max_theta, xres=o_res, yres=t_res))



# Using template bank created with contours_ver3.py
def load_Tien_data(mass):
    files = {
        "20": "sys3_indiv_contour_mcz20_2024-07-14_06-02-21.pkl",
        "30": "sys3_indiv_contour_mcz30_2024-07-25_04-13-21.pkl",
        "40": "sys3_indiv_contour_mcz40_2024-07-14_06-20-57.pkl"
    }

    with open("./lensing_and_precession/data/" + files[str(mass)], "rb") as f:
        data = pickle.load(f)
    return data

def main():
    sky_locations = {
        # "20": {
        # "theta_S": np.pi / 4,
        # "phi_S": 0,
        # "theta_J": np.pi / 2,
        # "phi_J": np.pi / 2
        # },
        "20": {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": 8*np.pi / 9,
        "phi_J": np.pi / 4
        },
        "30": {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": 8*np.pi / 9,
        "phi_J": np.pi / 4
        },
        "40": {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": 8*np.pi / 9,
        "phi_J": np.pi / 4
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action='store_true')
    parser.add_argument('-time', action='store_true')
    parser.add_argument('-I', default=0.6, type=float)
    parser.add_argument('-mcz', default=30, type=float)
    parser.add_argument('-td', default=0.03, type=float)

    parser.add_argument('-t-S', default='pi/3', type=str)
    parser.add_argument('-p-S', default='pi/4', type=str)
    parser.add_argument('-t-J', default='pi/6', type=str)
    parser.add_argument('-p-J', default='pi/3', type=str)


    parser.add_argument('-o-bounds', type=float, nargs=2, default=[0, 5])
    parser.add_argument('-t-bounds', type=float, nargs=2, default=[0, 15])
    parser.add_argument('-resolution', type=int, nargs=2, default=[51, 151])


    args = parser.parse_args()
    mass = float(args.mcz)
    I = float(args.I)
    td = float(args.td)
    min_omega, max_omega = args.o_bounds
    min_theta, max_theta = args.t_bounds
    o_res, t_res = args.resolution

    sky_location = parse_sky_location(args.t_S, args.p_S, args.t_J, args.p_J)

    # test_minimization.py -n -mcz 20 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 11 41

    # test_minimization.py -n -mcz 20 -td 0.02 -resolution 101 101
    # test_minimization.py -n -mcz 20 -td 0.02 -resolution 101 101 -p-S pi/2
    # test_minimization.py -n -mcz 20 -td 0.02 -resolution 101 101 -p-S pi/2 -p-J pi

    # test_minimization.py -n -I 0.6 -mcz 20 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101

    # test_minimization.py -n -I 0.4 -mcz 20 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101


    # test_minimization.py -n -I 0.6 -mcz 40 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101

    # test_minimization.py -n -I 0.6 -mcz 20 -td 0.011 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101
    # test_minimization.py -n -I 0.6 -mcz 20 -td 0.044 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101

    # test_minimization.py -n -I 0.3 -mcz 20 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101
    # test_minimization.py -n -I 0.1 -mcz 20 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101
    # test_minimization.py -n -I 0 -mcz 20 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101

    # test_minimization.py -n -I 0.6 -mcz 25 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101
    # test_minimization.py -n -I 0.6 -mcz 30 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101
    # test_minimization.py -n -I 0.6 -mcz 35 -td 0.022 -o-bounds 3 4 -t-bounds 6 10 -resolution 101 101



    # test_minimization.py -n -mcz 20 -td 0.0622


    if args.time:
        lens_params, NP_params, RP_params = set_to_location_class(
                sky_location, lens_params_1, NP_params_1, RP_params_1
            )
        lens_params["I"] = I
        lens_params["td"] = td
        lens_params["mcz"] = mass * solar_mass
        RP_params["mcz"] = mass * solar_mass

        theta_arr = []
        omega_arr = []
        eps_arr = []
        start = time.time()
        eps, theta_best, omega_best = find_optimal_RP_mismatch(lens_params, RP_params, omega_bounds=(min_omega, max_omega), theta_bounds=(min_theta, max_theta))
        end = time.time()
        print((start - end) * 1000, "ms")


    elif args.new:
        lens_params, NP_params, RP_params = set_to_location_class(
                sky_location, lens_params_1, NP_params_1, RP_params_1
            )
        
        lens_params["I"] = I
        lens_params["td"] = td
        lens_params["mcz"] = mass * solar_mass
        RP_params["mcz"] = mass * solar_mass

        mismatch_dict = evaluate_mismatch_2D(RP_params, lens_params, ("omega_tilde", "theta_tilde"), ((min_omega, max_omega), (min_theta, max_theta)), (o_res, t_res))
        omega_list = mismatch_dict["omega_tilde"]
        theta_list = mismatch_dict["theta_tilde"]
        eps_data = np.transpose(mismatch_dict["results"])

        omega_data, theta_data = np.meshgrid(omega_list, theta_list)
        
        data = {}
        data["omega_matrix"] = omega_data
        data["theta_matrix"] = theta_data
        data["epsilon_matrix"] = eps_data
        data["s_params"] = lens_params
        data["t_params"] = RP_params

        folder = "./output/"

        filename = folder + "mcz{mcz}_td{td}_I{I}_omega{omin}-{omax}_theta{tmin}-{tmax}_{xres}x{yres}_{loc}.pkl".format(mcz=mass, td=td, I=I, omin=min_omega, omax=max_omega, tmin=min_theta, tmax=max_theta, xres=o_res, yres=t_res, loc=str(sky_location))

        with open(filename, "wb") as file:
            pickle.dump(data, file)

        create_contour_plot(data, mass, td)
        # o_spacing = data["omega_matrix"][0, 1] - data["omega_matrix"][0, 0]
        # t_spacing = data["theta_matrix"][1, 0] - data["theta_matrix"][0, 0]
        # print(o_spacing, t_spacing)
        # create_fft_plot(data, o_spacing, t_spacing)

    else:
        folder = "./output/"
        filename = folder + "mcz{mcz}_td{td}_I{I}_omega{omin}-{omax}_theta{tmin}-{tmax}_{xres}x{yres}_{loc}.pkl".format(mcz=mass, td=td, I=I, omin=min_omega, omax=max_omega, tmin=min_theta, tmax=max_theta, xres=o_res, yres=t_res, loc=str(sky_location))

        if(exists(filename)):

            with open(filename, "rb") as file:
                data = pickle.load(file)
            lens_params = data["s_params"]
            RP_params = data["t_params"]
            print("creating contour")
            create_contour_plot(data, mass, td)

            # o_spacing = data["omega_matrix"][0, 1] - data["omega_matrix"][0, 0]
            # t_spacing = data["theta_matrix"][1, 0] - data["theta_matrix"][0, 0]
            # print(o_spacing, t_spacing)
            # create_fft_plot(data, o_spacing, t_spacing)
        else:
            for mass in [20, 30, 40]:
                data = load_Tien_data(mass)
                sky_location = sky_locations[str(mass)]

                lens_params, NP_params, RP_params = set_to_location_class(
                    sky_location, lens_params_1, NP_params_1, RP_params_1
                )

                lens_params["I"] = 0.5
                td = 0.03

                create_contour_plot(data, mass, td)

if __name__ == "__main__":
    main()