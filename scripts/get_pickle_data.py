from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import argparse
import pickle


from lensing_and_precession.modules.Classes_ver2 import *
from lensing_and_precession.modules.functions_ver2 import *
from lensing_and_precession.modules.functions_Precessing import *

from scripts.helper_classes import *
from scripts.helper_functions import *

total = 5000

def Sn(f_arr, f_min=20, delta_f=0.25, frequencySeries=True):
    """
    Calculates the power spectral density of the aLIGO noise curve based on arXiv:0903.0338.

    Parameters
    ----------
    f_arr : np.ndarray
        The frequency array.
    f_min : float, optional
        The minimum frequency. Defaults to 20 Hz.
    delta_f : float, optional
        The frequency step size. Defaults to 0.25 Hz.
    frequencySeries : bool, optional
        If True, returns a FrequencySeries object. Defaults to True.

    Returns
    -------
    np.ndarray or FrequencySeries
        The power spectral density of the aLIGO noise curve.
    """

    Sn_val = np.zeros_like(f_arr)
    for i in range(len(f_arr)):
        if f_arr[i] < f_min:
            Sn_val[i] = np.inf
        else:
            S0 = 1e-49
            f0 = 215
            Sn_temp = (
                np.power(f_arr[i] / f0, -4.14)
                - 5 * np.power(f_arr[i] / f0, -2)
                + 111
                * (
                    (1 - np.power(f_arr[i] / f0, 2) + 0.5 * np.power(f_arr[i] / f0, 4))
                    / (1 + 0.5 * np.power(f_arr[i] / f0, 2))
                )
            )
            Sn_val[i] = Sn_temp * S0

    if frequencySeries:
        return FrequencySeries(Sn_val, delta_f=delta_f)
    return Sn_val

def waveform_helper(lens_params, t_params, td, mcz):
    lens_params["td"] = td
    lens_params["mcz"] = mcz
    t_params["mcz"] = mcz

    l_s = get_gw(lens_params)["strain"]
    ul_s = get_gw(t_params)["strain"]
    l_s.resize(len(ul_s))

    f = get_gw(lens_params)["f_array"]
    psd = Sn(f)

    return ul_s, l_s, psd


sky_location = {
    "theta_S": np.pi / 3,
    "phi_S": np.pi / 4,
    "theta_J": np.pi / 6,
    "phi_J": np.pi / 3
}

lens_params, NP_params, RP_params = set_to_location_class(
    sky_location, lens_params_1, NP_params_1, RP_params_1
)

lens_params["I"] = 0.6
td = 0.019811962392478497

data_unlensed, data_lensed, psd = waveform_helper(lens_params, NP_params, td, 15 * solar_mass)
print(list(data_unlensed))
print(list(data_lensed))
print(list(psd))
# data = (td_list, data_lensed, data_unlensed)

# with open('data.pkl', 'wb') as file:
#     pickle.dump(data, file)