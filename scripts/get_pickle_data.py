from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import argparse
import pickle


from modules.Classes_ver2 import *
from modules.functions_ver2 import *
from modules.functions_Precessing import *

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

    return ul_s, l_s, psd


lens_params, NP_params, RP_params = set_to_location_class(
    sky_location, lens_params_1, NP_params_1, RP_params_1
)
lens_params["I"] = 0.6
data_unlensed, data_lensed, psd = waveform_helper(lens_params, NP_params, td, 15 * solar_mass)
def main():
    parser = argparse.ArgumentParser(
        description="Extract and print waveform and PSD data for given parameters."
    )
    parser.add_argument('-I', type=float, default=0.6, help='Inclination angle I (default: 0.6)')
    parser.add_argument('-td', type=float, default=0.0198, help='Time delay td (default: 0.0198)')
    parser.add_argument('-m', type=float, default=15, help='Chirp mass mcz (default: 15)')
    args = parser.parse_args()

    sky_location = {
        "theta_S": np.pi / 3,
        "phi_S": np.pi / 4,
        "theta_J": np.pi / 6,
        "phi_J": np.pi / 3
    }

    lens_params, NP_params, RP_params = set_to_location_class(
        sky_location, lens_params_1, NP_params_1, RP_params_1
    )

    lens_params["I"] = args.I
    td = args.td
    mcz = args.m * solar_mass

    data_unlensed, data_lensed, psd = waveform_helper(lens_params, NP_params, td, mcz)
    print(list(data_unlensed))
    print(list(data_lensed))
    print(list(psd))


if __name__ == "__main__":
    main()