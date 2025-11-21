import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import fsolve
from tqdm import tqdm

from modules.Classes_ver2 import *
from modules.functions_ver2 import *
from modules.functions_Precessing import *
from scripts.helper_classes import *
from scripts.helper_functions import *

# Heavy project imports are deferred into main() so `-h/--help` works even if
# optional modules have issues or are slow to import.


# from hanging_threads import start_monitoring
# start_monitoring(seconds_frozen=10, test_interval=100)

def create_contour(filename, plt, zlabel=r"$\min_{\~\theta, \~\Omega, \gamma_P} \epsilon(\~h_{\rm L}, \~h_{\rm RP})$"):
    theta_best = None
    omega_best = None
    with open(filename, 'rb') as file:
        if("NP" not in filename):
            td_list, mcz_list, mismatch_list, theta_best, omega_best = pickle.load(file)
            
        else:
            td_list, mcz_list, mismatch_list = pickle.load(file)

    mismatch_list = np.transpose(mismatch_list)
    plt.figure()

    contourPlot = plt.contourf(td_list, mcz_list, mismatch_list, levels=100, cmap=cm.jet)

    plt.colorbar(contourPlot).set_label(
            label=zlabel,
        )

    td_bounds = Bounds(np.min(td_list), np.max(td_list))

    plt = plot_boundary_curve(plt, td_bounds)
    plt = plot_max_and_min(plt, td_bounds)

    plt.title(r"Effects of $\Delta t_d$ and $\mathcal{M}_{\text{s}}$ $[M_{\odot}]$ on Waveform Mismatch")
    plt.xlabel(r"$\Delta t_d$")
    plt.ylabel(r"$\mathcal{M}_{\text{s}}$ $[M_{\odot}]$")

    pltName = "./Figures/" + filename[8:-3] + "png"
    plt.savefig(pltName)

    if(theta_best is not None):
        plt.figure()
        theta_best = np.transpose(theta_best)
        contourPlot = plt.contourf(td_list, mcz_list, theta_best, levels=100, cmap=cm.jet)
        
        plt.colorbar(contourPlot).set_label(
                label=r"$\~\theta_{best}$",
            )
        
        plt = plot_boundary_curve(plt, td_bounds)

        plt.title(r"Effects of $\Delta t_d$ and $\mathcal{M}_{\text{s}}$ $[M_{\odot}]$ on $\~\theta_{best}$")
        plt.xlabel(r"$\Delta t_d$")
        plt.ylabel(r"$\mathcal{M}_{\text{s}}$ $[M_{\odot}]$")

        pltName = "./Figures/" + "theta_best_" + filename[9:-3] + "png"
        plt.savefig(pltName)


    if(omega_best is not None):
        plt.figure()
        omega_best = np.transpose(omega_best)
        contourPlot = plt.contourf(td_list, mcz_list, omega_best, levels=100, cmap=cm.jet)
        
        plt.colorbar(contourPlot).set_label(
                label=r"$\~\Omega_{best}$",
            )
        plt = plot_boundary_curve(plt, td_bounds)
        
        plt.title(r"Effects of $\Delta t_d$ and $\mathcal{M}_{\text{s}}$ $[M_{\odot}]$ on $\~\Omega_{best}$")
        plt.xlabel(r"$\Delta t_d$")
        plt.ylabel(r"$\mathcal{M}_{\text{s}}$ $[M_{\odot}]$")

        pltName = "./Figures/" + "omega_best_" + filename[9:-3] + "png"
        plt.savefig(pltName)

def main():
    parser = argparse.ArgumentParser(
        description="Generate and plot mismatch contours for varying mcz and td."
    )
    parser.add_argument('-NP', action='store_true', help='Use non-precessing mode (NP)')
    # parser.add_argument('-graph', action="store_true")
    parser.add_argument('-pbar', action="store_true", help='Show progress bar during computation')
    parser.add_argument('-slice', nargs=2, metavar=('MCZ', 'TD'), help='Plot a slice at given mcz and td values')
    parser.add_argument('-I', type=float, default=0.6, help='Flux Ratio I (default: 0.6)')
    parser.add_argument('-m-bounds', type=float, nargs=2, default=[0.01, 90], metavar=('MIN', 'MAX'), help='Bounds for mcz (default: 0.01 90)')
    parser.add_argument('-td-bounds', type=float, nargs=2, default=[0.01, 0.1], metavar=('MIN', 'MAX'), help='Bounds for td (default: 0.01 0.1)')
    parser.add_argument('-resolution', type=int, nargs=2, default=[100, 100], metavar=('MCZ_RES', 'TD_RES'), help='Resolution for mcz and td grid (default: 100 100)')

    parser.add_argument('-t-S', default='pi/3', type=str, help='Source theta_S (default: pi/3)')
    parser.add_argument('-p-S', default='pi/4', type=str, help='Source phi_S (default: pi/4)')
    parser.add_argument('-t-J', default='pi/6', type=str, help='Source theta_J (default: pi/6)')
    parser.add_argument('-p-J', default='pi/3', type=str, help='Source phi_J (default: pi/3)')

    args = parser.parse_args()
    # calc_mcz_vs_td.py -pbar -I 0.3 -resolution 50 50

    # calc_mcz_vs_td.py -pbar -I 0.6 -slice mcz 20
    # calc_mcz_vs_td.py -pbar -I 0.6 -slice mcz 40

    # calc_mcz_vs_td.py -pbar -I 0.3 -slice mcz 40
    # calc_mcz_vs_td.py -pbar -I 0.3 -slice mcz 20

    # calc_mcz_vs_td.py -pbar -I 0.6 -slice td 0.02
    # calc_mcz_vs_td.py -pbar -I 0.6 -slice td 0.04
    # calc_mcz_vs_td.py -pbar -I 0.6 -slice td 0.08

    # calc_mcz_vs_td.py -pbar -I 0.3 -slice td 0.02
    # calc_mcz_vs_td.py -pbar -I 0.3 -slice td 0.04
    # calc_mcz_vs_td.py -pbar -I 0.3 -slice td 0.08

    # calc_mcz_vs_td.py -pbar -m-bounds 10 90 -td-bounds 0.02 0.07

    sky_location = parse_sky_location(args.t_S, args.p_S, args.t_J, args.p_J)


    lens_params, NP_params, RP_params = set_to_location_class(
        sky_location, lens_params_1, NP_params_1, RP_params_1
    )

    num_slice_points = 5000 if args.NP else 150

    lens_params["I"] = args.I
    lens_params["td"] = 0.02

    td_lb, td_ub = args.td_bounds
    m_lb, m_ub = args.m_bounds
    num_td, num_mcz = args.resolution

    # set_method('fork')

    def mismatch_helper_NP(lens_params, t_params):
        return (mismatch(t_params, lens_params)["mismatch"], None, None)

    def calc_function_across_points(eval_fn, m_bounds, t_bounds, res):
        m_lb, m_ub = m_bounds
        td_lb, td_ub = t_bounds
        num_td, num_mcz = res
        td_list = np.linspace(td_lb, td_ub, num_td)
        mcz_list = np.linspace(m_lb, m_ub, num_mcz)

        eval_parameters = ("td", "mcz")
        m_lb *= solar_mass
        m_ub *= solar_mass
        bounds=((td_lb, td_ub), (m_lb, m_ub))
        resolution = (num_td, num_mcz)
        
        results = evaluate_function_2D(eval_fn, lens_params, RP_params, eval_parameters, bounds, resolution, show_pbar=args.pbar)["results"]

        results = np.array(results) #converts 2d array of tuples to 3d array
        mismatch_list, theta_best_list, omega_best_list = np.transpose(results, axes=(2, 0, 1))

        return td_list, mcz_list, mismatch_list, theta_best_list, omega_best_list
    
    def calc_function_across_points_1D(eval_fn, param: str, bounds: tuple[float, float], resolution: int, show_pbar: bool = True):
        eval_list = np.transpose(np.asarray([np.linspace(bounds[0], bounds[1], resolution)]), (1, 0))
        eval_fn = functools.partial(evaluate_function_with_parameters, eval_fn, lens_params, RP_params, (param,))

        results = evaluate_multithread(eval_fn, eval_list, show_pbar=show_pbar)
        results = np.array(results) #converts 1d array of tuples to 2d array
        mismatch_list, theta_best_list, omega_best_list = np.transpose(results, axes=(1, 0))

        return eval_list, mismatch_list, theta_best_list, omega_best_list

    if(args.NP):
        eval_fn = mismatch_helper_NP
        filename = "NP_"
    else:
        eval_fn = find_optimal_RP_mismatch
        filename = ""

    if(args.slice):
        slice_type, slice_val = args.slice
        mismatch_list = []
        x_list = []
        x_label = ""
        filename = ""
        label = ""
        value = 0
        units = ""
        one_cycle = 0
        two_cycle = 0
        if(slice_type == "mcz"):
            mass = float(slice_val)
            num_td = num_slice_points
            one_cycle = fsolve(lambda t: get_calculated_mcz(1, t) - mass, 1)[0]
            two_cycle = fsolve(lambda t: get_calculated_mcz(2, t) - mass, 1)[0]
            td_lb = one_cycle - 0.01
            td_ub = two_cycle + 0.01

            x_label = r"$\Delta t_d$"
            label = r"$\mathcal{M}_{\text{s}}$"
            value = mass
            units = r"$[M_{\odot}]$"
            filename = filename + "mismatch_slice_m{mcz}".format(mcz=mass)

            x_list, mismatch_list, theta_best_list, omega_best_list = calc_function_across_points_1D(eval_fn, 'td', (td_lb, td_ub), num_td)

            mismatch_list, theta_best_list, omega_best_list = np.squeeze(mismatch_list), np.squeeze(theta_best_list), np.squeeze(omega_best_list)

            # print(mismatch_list)

                
        elif(slice_type == "td"):
            td = float(slice_val)
            num_mcz = num_slice_points
            one_cycle = get_calculated_mcz(1, td)
            two_cycle = get_calculated_mcz(2, td)
            print("One cycle mass:", one_cycle)
            print("Two cycle mass:", two_cycle)
            m_lb = (two_cycle - 10) * solar_mass
            m_ub = (one_cycle + 10) * solar_mass
            x_label = r"$\mathcal{M}_{\text{s}}$ $[M_{\odot}]$"
            label = r"$\Delta t_d$"
            value = td * 1000
            units = r" $[ms]$"
            filename = filename + "mismatch_slice_td{td}".format(td=td)


            x_list, mismatch_list, theta_best_list, omega_best_list = calc_function_across_points_1D(eval_fn, 'mcz', (m_lb, m_ub), num_mcz)
            x_list = x_list / solar_mass  # convert to solar masses for plotting

            mismatch_list, theta_best_list, omega_best_list = np.squeeze(mismatch_list), np.squeeze(theta_best_list), np.squeeze(omega_best_list)
        
        plt.figure()
        plt.plot(x_list, mismatch_list, color='k')
        plt.axvline(x=one_cycle, color="magenta", linestyle="--")
        plt.axvline(x=two_cycle, color="red", linestyle="--")

        plt.title("Mismatch With a Lensed Source, {label} = {value} {units}".format(label=label, value=value, units=units))
        plt.xlabel(x_label)
        plt.ylabel(r"$\epsilon(\~h_{\rm L}, \~h_{\rm NP})$" if args.NP else r"$\min_{\~\theta, \~\Omega, \gamma_P} \epsilon(\~h_{\rm L}, \~h_{\rm RP})$")
        plt.savefig("./Figures/" + filename + ".png")

        if(not args.NP):
            plt.figure()
            plt.plot(x_list, theta_best_list, color='k')
            plt.axvline(x=one_cycle, color="magenta", linestyle="--")
            plt.axvline(x=two_cycle, color="red", linestyle="--")
            plt.xlabel(x_label)
            plt.ylabel(r"$\~\theta_{best}$")
            plt.savefig("./Figures/theta_" + filename + ".png")

            plt.figure()
            plt.plot(x_list, omega_best_list, color='k')
            plt.axvline(x=one_cycle, color="magenta", linestyle="--")
            plt.axvline(x=two_cycle, color="red", linestyle="--")
            plt.xlabel(x_label)
            plt.ylabel(r"$\~\Omega_{best}$")
            plt.savefig("./Figures/omega_" + filename + ".png")
        
    else:
        folder = './output/'
        filename = 'mcz_vs_td_results_' + str(td_lb) + "-" + str(td_ub) + "_" + str(m_lb) + "-" + str(m_ub) + "_" + str(num_td) + "x" + str(num_mcz) + '.pkl'
        if(args.NP):
            filename = 'NP_' + filename
            eval_fn = mismatch_helper_NP
        else:
            eval_fn = find_optimal_RP_mismatch
        filename = folder + filename
        td_list, mcz_list, mismatch_list, theta_best_list, omega_best_list = calc_function_across_points(eval_fn, args.m_bounds, args.td_bounds, args.resolution)
        data = [td_list, mcz_list, mismatch_list]
        if(not args.NP): data += [theta_best_list, omega_best_list]
        with open(filename, "wb") as file:
            pickle.dump(data, file)
        if(args.NP):
            create_contour(filename, plt, zlabel=r"$\epsilon(\~h_{\rm L}, \~h_{\rm NP})$")
        else:
            create_contour(filename, plt)

if __name__ == "__main__":
    main()