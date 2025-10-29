import sys
sys.path.insert(0, "./lensing_and_precession/")

from modules.Classes_ver2 import *
from modules.functions_ver2 import *
from modules.functions_Precessing import *
from modules.default_params_ver2 import solar_mass

from helper_classes import *

from multiprocessing import Pool, cpu_count
from simpleeval import simple_eval
from scipy.optimize import Bounds, shgo
from tqdm import tqdm
import time

import functools

import numpy as np
import matplotlib.pyplot as plt

def parse_sky_location(ts, ps, tj, pj):
    sky_location = {}
    sky_location["theta_S"] = simple_eval(ts, names={"pi": np.pi})
    sky_location["phi_S"] = simple_eval(ps, names={"pi": np.pi})
    sky_location["theta_J"] = simple_eval(tj, names={"pi": np.pi})
    sky_location["phi_J"] = simple_eval(pj, names={"pi": np.pi})
    return sky_location
    

def evaluate_function_with_parameters(fun,
                                      t_params: dict[str, float],
                                      s_params: dict[str, float],
                                      coord_names: tuple[str, ...],
                                      *coord_vals: tuple[float, ...]):
    return fun(update_dict(t_params, coord_names, coord_vals), 
            s_params)

def evaluate_multithread(eval_fn, eval_list, show_pbar = True):
    num_cpu = cpu_count()
    temp_results = []
    if(show_pbar):
        pbar = tqdm(total=len(eval_list))
        callback = lambda result: pbar.update(1)
    else: callback = lambda result: None
    with Pool(num_cpu) as pool:
        pool_results_arr = []
        for arg in eval_list:
            pool_results_arr.append(
                pool.apply_async(eval_fn, args=arg, callback=callback))
        
        for res in pool_results_arr:
            temp_results.append(res.get())
    return temp_results

def get_list_of_coords(coord_1_array, coord_2_array):
    tempList = np.transpose(np.meshgrid(coord_1_array, coord_2_array), (1, 2, 0))
    return tempList.reshape(-1, tempList.shape[-1])

def evaluate_function_2D(fun,
    t_params: dict[str, float],
    s_params: dict[str, float],
    eval_parameters: tuple[str, str],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    resolution: tuple[int, int],
    multithread: bool = True,
    show_pbar: bool = True
    ) -> dict[str, np.ndarray]:

    """
    Evaluates a function across a specified 2D parameter space
    """

    param_1 = eval_parameters[0]
    param_2 = eval_parameters[1]

    param_1_bounds = bounds[0]
    param_2_bounds = bounds[1]

    param_1_resolution = resolution[0]
    param_2_resolution = resolution[1]

    coord_1_array = np.linspace(param_1_bounds[0], param_1_bounds[1], param_1_resolution)
    coord_2_array = np.linspace(param_2_bounds[0], param_2_bounds[1], param_2_resolution)

    list_of_param_tuples = get_list_of_coords(coord_1_array, coord_2_array)

    total = param_1_resolution * param_2_resolution
    
    # if(multithread):

    temp_results = []

    # if multithread:
    eval_function = functools.partial(evaluate_function_with_parameters, fun, t_params, s_params, (param_1, param_2))

    temp_results = evaluate_multithread(eval_function, list_of_param_tuples, show_pbar=show_pbar)

    results = []
    for r in range(param_1_resolution):
        sublist = []
        for c in range(param_2_resolution):
            index = param_2_resolution * r + c
            res = temp_results[index]
            sublist.append(res)
        results.append(sublist)
    # else:
    #     results = []
    #     for coord_1 in coord_1_array:
    #         sublist = []
    #         for coord_2 in coord_2_array:
    #             sublist.append(fun(Parameters({**t_params_copy,
    #                                 param_1: coord_1,
    #                                 param_2: coord_2
    #                                 }), 
    #                                 s_params_copy))
    #             callback_update(None)
    #         results.append(sublist)

    results = np.asarray(results)
    
    return {param_1: coord_1_array, param_2: coord_2_array, "results": results}
    
def evaluate_function_with_args_2D(
    fun,
    t_params: dict[str, float],
    s_params: dict[str, float],
    eval_parameters: tuple[str, str],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    resolution: tuple[int, int],
    multithread: bool = True,
    pbar: bool = False,
    **kwargs) -> dict[str, np.ndarray]:

    partial = functools.partial(fun, **kwargs)
    return evaluate_function_2D(partial, t_params, s_params, eval_parameters, bounds, resolution, multithread=multithread, show_pbar=pbar)

def evaluate_mismatch(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
) -> dict:
    """
    Finds the mismatch between template and source waveforms

    Parameters
    ----------
    t_params : dict
        The parameters for the template waveform.
    s_params : dict
        The parameters for the source waveform.
    eval_parameters : tuple[string]
        A tuple containing the parameter keys to evaluate over. Ordering should be (t_eval_param, s_eval_param)
    bounds : tuple[Bounds]
        A tuple containing a bounds object representing the upper and lower bounds of the parameters to evaluate over. Ordering should be (t_bounds, s_bounds)
    resolution : tuple[float, float]
        A tuple containing the resolution for the evaluation parameters. Ordering should be (t_resolution, s_resolution)
    f_min : float, optional
        The minimum frequency for the waveform. Default is 20 Hz.
    delta_f : float, optional
        The frequency spacing between samples. Default is 0.25 Hz.
    psd : FrequencySeries, optional
        The power spectral density of the detector noise. If not provided, it will be calculated based on the aLIGO noise curve from arXiv:0903.0338, as a function of the source waveform's frequency range. Default is None.
    lens_Class : class, optional
        A class representing the lensed waveform. Default is LensingGeo.
    prec_Class : class, optional
        A class representing the precessing waveform. Default is Precessing.
    use_opt_match : bool, optional
        If True, uses the optimized_match function from pycbc.filter. Default is True.
    get_updated_mismatch_results : bool, optional
        If True, gets the updated mismatch results dictionary after the correct t_c and phi_c are updated in the template parameters. The t_c (index) and phi_c in the updated mismatch results should be ~0. This is useful for debugging but also slows down the function. Default is False.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "t" (list[float]): A list of t coordinate parameters
        - "s" (list[float]): A list of s coordinate parameters
        - "mismatch_results" (list[float]): The updated mismatch results.
    """
    mismatch_val = None
    if("gamma_P" in t_params):
        mismatch_val = optimize_mismatch_gammaP(t_params, s_params)["ep_min"]
    else:
        mismatch_val = mismatch(
        t_params, 
        s_params, 
        f_min,
        delta_f,
        psd,
        lens_Class,
        prec_Class,
        use_opt_match)["mismatch"]
    return mismatch_val

def evaluate_mismatch_2D(
        t_params: dict[str, float],
        s_params: dict[str, float],
        eval_parameters: tuple[str, str],
        bounds: tuple[tuple[float, float], tuple[float, float]],
        resolution: tuple[int, int],
        **kwargs) -> dict[str, np.ndarray]:
    
    return evaluate_function_with_args_2D(evaluate_mismatch, t_params, s_params, eval_parameters, bounds, resolution, pbar=True, multithread=True, **kwargs)


def waveform_graph(NP_params=None, RP_params=None, lens_params=None, filename="strain_from_NP_RP_lensed_sources"):
    if(not(NP_params or RP_params or lens_params)):
        return
    
    fcut = np.inf
    if(NP_params):        
        NP_Model = Precessing(NP_params)
        fcut = NP_Model.f_cut()

    if(RP_params):
        RP_Model = Precessing(RP_params)
        fcut = min(fcut, RP_Model.f_cut())

    if(lens_params):
        L_Model = LensingGeo(lens_params)
        fcut = min(fcut, L_Model.f_cut())

    plt.figure()

    frequencyList = np.array([f for f in np.arange(20, fcut, 0.25)])
    if(NP_params):
        NP_strainArray = np.absolute(NP_Model.strain(frequencyList))
        plt.loglog(frequencyList, NP_strainArray, label="Unlensed, Not Precessing")
    
    if(RP_params):
        RP_strainArray = np.absolute(RP_Model.strain(frequencyList))
        plt.loglog(frequencyList, RP_strainArray, label="Regular Precession")

    if(lens_params):
        L_strainArray = np.absolute(L_Model.strain(frequencyList))
        plt.loglog(frequencyList, L_strainArray, label="Lensed")

    plt.legend()
    plt.title("Strain of Waveforms from Various Sources")
    plt.xlabel(r"$f$ $[Hz]$")
    plt.ylabel(r"$ | \tilde{h} |$")

    plt.savefig("./Figures/" + filename + ".png")

def set_to_location_class(loc_dict: dict, *args):
    args_classes = []
    new_args = set_to_location(loc_dict, *args)
    for args in new_args:
        if(("MLz" in args and "y" in args) or ("td" in args and "I" in args)):
            args_classes.append(LensParameters(args))
        else:
            args_classes.append(Parameters(args))
    return tuple(args_classes)

def update_dict(updated_dict, keys, x):
    updated_params_dict = dict(zip(keys, x))
    updated_dict.update(updated_params_dict)
    return updated_dict


def find_optimal_RP_mismatch(lens_params, RP_params, omega_bounds = (0, 5), theta_bounds= (0, 15), resolution = 0.1):
    # omega_arr = np.arange(omega_min, omega_max + resolution/2, resolution)
    # theta_arr = np.arange(theta_min, theta_max + resolution/2, resolution)

    # coord_arr = get_list_of_coords(omega_arr, theta_arr)

    gamma_bounds = (0, 2*np.pi)

    map_fn = functools.partial(evaluate_mismatch_helper, lens_params, RP_params)
    results = shgo(map_fn, (omega_bounds, theta_bounds), iters=5)

    if(not(results.success)): print("Warning: Minimization failed for t parameters: ", RP_params, "\ns parameters: ", lens_params) 

    min_eps = results.fun
    omega_best, theta_best = results.x

    RP_params = update_dict(RP_params, ["omega_tilde", "theta_tilde"], [omega_best, theta_best])

    return (min_eps, omega_best, theta_best)

def evaluate_mismatch_helper(lens_params, RP_params, x):
    updated_params_dict = update_dict(RP_params, ["omega_tilde", "theta_tilde"], x)
    # print("evaluating omega: {o}, theta: {t}, gamma: {g}".format(o=x[0], t=x[1], g=x[2]))
    # print("new SHGO")
    return optimize_mismatch_gammaP(updated_params_dict, lens_params)["ep_min"]

def get_max_N(mcz, td, eta=0.25, fmin=20):
    fcut = get_fcut_from_mcz(mcz, eta)
    N = np.floor(2*td*(fcut - fmin))
    return N

def get_calculated_mcz(N, td, eta=0.25, fmin=20):
    m = np.pow(eta, 3/5) / (np.pow(6, 3/2) * np.pi * (N/(td) + fmin))
    return m / solar_mass

def calc_minima(n, td, eta=0.25):
    return (np.pow(eta, 3/5) * td / (np.pow(6, 3/2) * np.pi * (n+0.5))) / solar_mass

def calc_maxima(n, td, eta=0.25):
    return (np.pow(eta, 3/5) * td / (np.pow(6, 3/2) * np.pi * n)) / solar_mass

def filter_list(x, y_arr, bounds):
    y1 = list(filter(lambda y: y <= bounds[1], y_arr)) # upper limit
    x_filtered = x[:len(y1)]
    y1 = list(filter(lambda y: y >= bounds[0], y1)) # lower limit
    x_filtered = x_filtered[len(x_filtered) - len(y1):]
    return x_filtered, y1

def plot_nested_data(plt, y_data_nested, x_list, color, m_bounds, linestyle="--"):
    y_data = np.transpose(y_data_nested)

    for y_list in y_data:
        x_list_filtered, y_list_filtered = filter_list(x_list, y_list, m_bounds)
        plt.plot(x_list_filtered, y_list_filtered, color=color, linestyle=linestyle, alpha=0.5)
    return plt

def plot_boundary_curve(plt, td_bounds):

    y1 = []
    y2 = []
    m_limits = plt.ylim()
    td_list = np.linspace(td_bounds.lb, td_bounds.ub, 1000)
    for td in td_list:
        m1 = get_calculated_mcz(1, td)
        m2 = get_calculated_mcz(2, td)
        y1.append(m1)
        y2.append(m2)

    td_filtered_1, y1 = filter_list(td_list, y1, m_limits)
    td_filtered_2, y2 = filter_list(td_list, y2, m_limits)
    plt.plot(td_filtered_1, y1, color='magenta', label="One orientation cycle")
    plt.plot(td_filtered_2, y2, color='r', label="Two orientation cycles")
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    return plt

def plot_max_and_min(plt, td_bounds, numN=10):
    td_list = np.linspace(td_bounds.lb, td_bounds.ub, 1000)
    m_limits = plt.ylim()
    minima = []
    maxima = []
    for td in td_list:
        n = np.asarray(range(1, numN))
        minima.append(calc_minima(n, td))
        maxima.append(calc_maxima(n, td))
    
    plt = plot_nested_data(plt, minima, td_list, "purple", m_limits, linestyle="dotted")
    plt = plot_nested_data(plt, maxima, td_list, "red", m_limits, linestyle="dotted")
    return plt
    
    