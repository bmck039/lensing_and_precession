import sys
sys.path.insert(0, "./lensing_and_precession/")

from modules.Classes_ver2 import *
from modules.functions_ver2 import *
from modules.functions_Precessing import *
from modules.default_params_ver2 import solar_mass
from modules.multithreading import evaluate_multithread

from helper_classes import *

import copy
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

def get_list_of_coords(coord_1_array, coord_2_array):
    """Return a deterministic row-major list of (coord1, coord2) pairs.

    We avoid relying on meshgrid/reshape/transposes which can be easy to
    misalign with later indexing. This explicit construction guarantees the
    ordering matches index = j + n2 * i used in results assembly.
    """
    pairs = []
    for c1 in coord_1_array:           # i index (rows)
        for c2 in coord_2_array:       # j index (cols)
            pairs.append((float(c1), float(c2)))
    return pairs

def _call_fun_by_name(module_name: str, fun_name: str,
                      t_params: dict, s_params: dict,
                      coord_names: tuple[str, str], coord_vals: tuple[float, float],
                      extra_kwargs: dict | None = None):
    """Resolve a function by module/name and invoke on updated params.
    This stays pickle-friendly for spawn/forkserver contexts.
    """
    import importlib
    mod = importlib.import_module(module_name)
    fn = getattr(mod, fun_name)
    kwargs = extra_kwargs or {}
    updated = update_dict(t_params, coord_names, coord_vals)
    return fn(updated, s_params, **kwargs)


def evaluate_function_2D(fun,
    t_params: dict[str, float],
    s_params: dict[str, float],
    eval_parameters: tuple[str, str],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    resolution: tuple[int, int],
    multithread: bool = True,
    show_pbar: bool = True,
    extra_kwargs: dict | None = None
    ) -> dict[str, np.ndarray]:

    """
    Evaluates a function across a specified 2D parameter space
    """

    param_1, param_2 = eval_parameters
    param_1_bounds, param_2_bounds = bounds
    param_1_resolution, param_2_resolution = resolution

    coord_1_array = np.linspace(param_1_bounds[0], param_1_bounds[1], param_1_resolution)
    coord_2_array = np.linspace(param_2_bounds[0], param_2_bounds[1], param_2_resolution)

    list_of_param_tuples = get_list_of_coords(coord_1_array, coord_2_array)
    total = param_1_resolution * param_2_resolution

    temp_results: list = []
    kwargs = extra_kwargs or {}

    if multithread:
        # Use spawn + function-by-name to avoid fork/OpenMP issues
        if callable(fun):
            module_name = fun.__module__
            fun_name = fun.__name__
        else:
            module_name = __name__
            fun_name = str(fun)

        eval_list = [
            (module_name, fun_name, t_params, s_params, (param_1, param_2), coords, kwargs)
            for coords in list_of_param_tuples
        ]

        temp_results = evaluate_multithread(
            _call_fun_by_name,
            eval_list,
            show_pbar=show_pbar,
            max_workers=None,
            chunksize=8,
            start_method='spawn',
            set_single_thread_blas=True,
        )
    else:
        iterator = list_of_param_tuples
        if show_pbar:
            iterator = tqdm(iterator, total=len(list_of_param_tuples))
        for c1, c2 in iterator:
            updated = update_dict(t_params, (param_1, param_2), (c1, c2))
            if callable(fun):
                temp_results.append(fun(updated, s_params, **kwargs))
            else:
                temp_results.append(_call_fun_by_name(__name__, fun, updated, s_params, (param_1, param_2), (c1, c2), kwargs))

    # Assemble row-major results matrix
    results = []
    for r in range(param_1_resolution):
        row = []
        for c in range(param_2_resolution):
            idx = r * param_2_resolution + c
            row.append(temp_results[idx])
        results.append(row)

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

    # Avoid functools.partial to keep multiprocessing pickle-friendly;
    # pass kwargs down explicitly.
    return evaluate_function_2D(
        fun,
        t_params,
        s_params,
        eval_parameters,
        bounds,
        resolution,
        multithread=multithread,
        show_pbar=pbar,
        extra_kwargs=kwargs,
    )

def evaluate_mismatch(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=Precessing,
    use_opt_match=True,
    gamma_points: int | None = None,
    optimize_gamma: bool = True,
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
    if ("gamma_P" in t_params) and optimize_gamma:
        # Optimize over gamma grid
        npts = 51 if gamma_points is None else int(gamma_points)
        mismatch_val = optimize_mismatch_gammaP(
            t_params,
            s_params,
            num_points=npts,
            lens_Class=lens_Class,
            prec_Class=prec_Class,
            use_opt_match=use_opt_match,
        )["ep_min"]
    else:
        # Respect fixed gamma_P if provided (no optimization)
        mismatch_val = mismatch(
            t_params,
            s_params,
            f_min,
            delta_f,
            psd,
            lens_Class,
            prec_Class,
            use_opt_match,
        )["mismatch"]
    return mismatch_val

def evaluate_mismatch_2D(
        t_params: dict[str, float],
        s_params: dict[str, float],
        eval_parameters: tuple[str, str],
        bounds: tuple[tuple[float, float], tuple[float, float]],
        resolution: tuple[int, int],
        multithread: bool = True,
        pbar: bool = True,
        **kwargs) -> dict[str, np.ndarray]:
    
    return evaluate_function_with_args_2D(
        evaluate_mismatch,
        t_params,
        s_params,
        eval_parameters,
        bounds,
        resolution,
        pbar=pbar,
        multithread=multithread,
        **kwargs,
    )


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
    """Return a NEW dict with the given keys updated, without mutating input.

    This avoids cross-task state bleed when running many evaluations in a
    multiprocessing pool where the same dict object can be reused.
    """
    updated_params_dict = dict(zip(keys, x))
    # Create a shallow copy to avoid mutating the original
    new_dict = {**updated_dict, **updated_params_dict}
    return new_dict


def find_optimal_RP_mismatch(lens_params, RP_params, omega_bounds = (0, 5), theta_bounds= (0, 15)):
    """Find optimal omega_tilde and theta_tilde to minimize mismatch.
    
    Args:
        lens_params: Lensing parameters
        RP_params: Regular precessing parameters
        omega_bounds: (min, max) bounds for omega_tilde
        theta_bounds: (min, max) bounds for theta_tilde
        resolution: Not used (kept for backward compatibility)
    
    Returns:
        (min_eps, omega_best, theta_best): Minimum mismatch and optimal parameters
    """
    map_fn = functools.partial(evaluate_mismatch_helper, lens_params, RP_params)
    
    # Reduce iterations for faster convergence (iters=5 -> iters=2)
    # SHGO samples the space and converges to local minima; fewer iterations
    # trades minor accuracy for major speed improvement
    gamma_bounds = (0, 2*np.pi)
    results = shgo(map_fn, (omega_bounds, theta_bounds, gamma_bounds), iters=5, options={'ftol': 1e-4})

    if(not(results.success)): 
        print("Warning: Minimization failed for t parameters: ", RP_params, "\ns parameters: ", lens_params) 

    min_eps = results.fun
    print(results.x)
    omega_best, theta_best, _ = results.x

    RP_params = update_dict(RP_params, ["omega_tilde", "theta_tilde"], [omega_best, theta_best])

    return (min_eps, omega_best, theta_best)

def evaluate_mismatch_helper(lens_params, RP_params, x):
    updated_params_dict = update_dict(RP_params, ["omega_tilde", "theta_tilde", "gamma_P"], x)
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
    
    