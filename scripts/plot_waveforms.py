import matplotlib.pyplot as plt
import argparse
import numpy as np

from lensing_and_precession.modules.Classes_ver2 import *
from lensing_and_precession.modules.functions_ver2 import *
from lensing_and_precession.modules.functions_Precessing import *
from lensing_and_precession.modules.plot_utils import *

from scripts.helper_classes import *
from scripts.helper_functions import *

def main():
    # Get lens_params
    sky_location = {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": np.pi / 2,
        "phi_J": np.pi / 2
    }

    lens_params, NP_params, RP_params = set_to_location_class(
        sky_location, lens_params_1, NP_params_1, RP_params_1
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', type=float, default=0.6)
    parser.add_argument('-m', type=float, default=20)
    parser.add_argument('-td', type=float, default=0.02)
    parser.add_argument('-omega', type=float, default=0)
    parser.add_argument('-theta', type=float, default=0)

    # parser.add_argument('-optimize', action='store_true')

    args = parser.parse_args()

    lens_params["mcz"] = args.m * solar_mass
    RP_params["mcz"] = args.m * solar_mass

    lens_params["I"] = args.I
    lens_params["td"] = args.td

    RP_params["omega_tilde"] = args.omega
    RP_params["theta_tilde"] = args.theta

    plot_waveform_comparison(RP_params, lens_params)

if __name__ == "__main__":
    main()

# plot_waveforms.py -m 20 -td 0.022 -omega 3.8 -theta 8