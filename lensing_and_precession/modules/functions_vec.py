#############################
# Section 1: Import Modules #
#############################


# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

# import py scripts
from lensing_and_precession.modules.functions_Lensing import *
from lensing_and_precession.modules.functions_Precessing import *
from lensing_and_precession.modules.Classes_ver2 import *
from lensing_and_precession.modules.functions_ver2 import set_to_params, get_gw, Sn



from pycbc.filter import match, optimized_match

# import modules
import numpy as np

error_handler = np.seterr(invalid="raise")

######################################
# Section 2: Shortcuts & Convenience #
######################################


def get_gw_vec(f_min=20, delta_f=0.25, **kwargs):
    if "MLz" in kwargs and "y" in kwargs:  # lensing parameters
        f_cut_arr = np.array([L_f_cut(**kwargs)])
        f_arr_arr = np.array([np.arange(f_min, f_cut, delta_f) for f_cut in f_cut_arr])
        strain_arr = L_strain(f_arr_arr, **kwargs)
    elif "omega_tilde" in kwargs and "theta_tilde" in kwargs:  # precessing parameters
        f_cut_arr = np.array([P_f_cut(**kwargs)])
        f_arr_arr = np.array([np.arange(f_min, f_cut, delta_f) for f_cut in f_cut_arr])
        strain_arr = P_strain(f_arr_arr, **kwargs)
    return {"f_array": f_arr_arr, "strain": strain_arr}


def mismatch_vec(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=PrecessingV2,
    use_opt_match=True,
) -> list:
    """
    Calculates the mismatch between two waveforms using the given parameters.

    Parameters
    ----------
    t_params : dict
        The parameters for the template waveform.
    s_params : dict
        The parameters for the source waveform.
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

    Returns
    -------
    dict
        A dictionary containing the Mismatchfollowing keys:
        - "mismatch" (float): The mismatch between the two waveforms.
        - "index" (int): The number of samples to shift the source waveform to match with the template.
        - "phi" (float): The phase to rotate the complex source waveform to match with the template.
    """

    t_gw = get_gw(t_params, f_min, delta_f, lens_Class, prec_Class)
    t_h = t_gw["strain"]
    s_gw = get_gw(s_params, f_min, delta_f, lens_Class, prec_Class)
    s_h = s_gw["strain"]
    t_h.resize(len(s_h))

    if psd is None:
        f_arr = s_gw["f_array"]
        psd = Sn(f_arr)

    match_func = optimized_match if use_opt_match else match
    match_val, index, phi = match_func(t_h, s_h, psd, return_phase=True)  # type: ignore
    match_val_coarse, index_coarse, phi_coarse = match(t_h, s_h, psd, return_phase=True)

    if(match_val_coarse > match_val):
        index = index_coarse
        phi = phi_coarse
        match_val = match_val_coarse

    mismatch = 1 - match_val

    return [mismatch, index, phi]


def optimize_mismatch_gammaP_vec(
    t_params: dict,  # template parameters
    s_params: dict,  # source parameters
    f_min=20,
    delta_f=0.25,
    psd=None,
    lens_Class=LensingGeo,
    prec_Class=PrecessingV2,
    use_opt_match=True,
    num_points=51
) -> dict:
    """
    Optimizes the mismatch between the precessing template and the signal by varying the initial precessing phase gamma_P of the template.

    Parameters
    ----------
    t_params : dict
        The parameters for the template waveform.
    s_params : dict
        The parameters for the source waveform.
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

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "ep_min" (float): The minimum mismatch value.
        - "ep_min_gammaP" (float): The gamma_P value corresponding to the minimum mismatch.
        - "ep_min_idx" (int): The number of samples to shift to get the minimum mismatch at ep_min_gammaP.
        - "ep_min_phi" (float): The phase to rotate the complex waveform to get the minimum mismatch at ep_min_gammaP.
        - "ep_max" (float): The maximum mismatch value.
        - "ep_max_gammaP" (float): The gamma_P value corresponding to the maximum mismatch.
        - "ep_max_idx" (int): The number of samples to shift to get the maximum mismatch at ep_max_gammaP.
        - "ep_max_phi" (float): The phase to rotate the complex waveform to get the maximum mismatch at ep_max_gammaP.
        - "ep_0" (float): The mismatch value at gamma_P = 0.
        - "ep_0_idx" (int): The number of samples to shift to get the mismatch at gamma_P = 0.
        - "ep_0_phi" (float): The phase to rotate the complex waveform to get the mismatch at gamma_P = 0.
    """

    t_params_copy, s_params_copy = set_to_params(t_params, s_params)

    # condition that t_params must be precessing parameters and already contain gamma_P
    if "gamma_P" not in t_params_copy:
        raise ValueError("t_params must be precessing parameters")

    gamma_arr = np.linspace(0, 2 * np.pi, num_points)

    result = [mismatch_vec(
            {**t_params_copy, "gamma_P": gamma_P},
            s_params_copy,
            f_min,
            delta_f,
            psd,
            lens_Class,
            prec_Class,
            use_opt_match,
        ) for gamma_P in gamma_arr]
    
    result = np.asarray(result)
    
    ep_arr, idx_arr, phi_arr = result.T

    # ep_arr = np.array([mismatch_dict[gamma_P]["mismatch"] for gamma_P in gamma_arr])
    # idx_arr = np.array([mismatch_dict[gamma_P]["index"] for gamma_P in gamma_arr])
    # phi_arr = np.array([mismatch_dict[gamma_P]["phi"] for gamma_P in gamma_arr])

    ep_min_idx = np.argmin(ep_arr)
    ep_max_idx = np.argmax(ep_arr)

    results = {
        "ep_min": ep_arr[ep_min_idx],
        "ep_min_gammaP": gamma_arr[ep_min_idx],
        "ep_min_idx": idx_arr[ep_min_idx],
        "ep_min_phi": phi_arr[ep_min_idx],
        "ep_max": ep_arr[ep_max_idx],
        "ep_max_gammaP": gamma_arr[ep_max_idx],
        "ep_max_idx": idx_arr[ep_max_idx],
        "ep_max_phi": phi_arr[ep_max_idx],
        "ep_0": ep_arr[0],
        "ep_0_idx": idx_arr[0],
        "ep_0_phi": phi_arr[0],
    }

    return results
