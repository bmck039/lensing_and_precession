#!/usr/bin/env python3
"""
Quick test to verify the parallel execution fix works.
"""
import sys
sys.path.insert(0, "./lensing_and_precession/")

from modules.default_params_ver2 import *
from helper_classes import *
from helper_functions import *
import numpy as np
import time


def main():
    print("=" * 60)
    print("Testing parallel mismatch evaluation")
    print("=" * 60)
    
    # Set up minimal parameters
    sky_location = {
        "theta_S": np.pi / 4,
        "phi_S": 0,
        "theta_J": 8 * np.pi / 9,
        "phi_J": np.pi / 4,
    }
    
    lens_params, NP_params, RP_params = set_to_location_class(
        sky_location, lens_params_1, NP_params_1, RP_params_1
    )
    
    lens_params["mcz"] = 40 * solar_mass
    RP_params["mcz"] = 40 * solar_mass
    lens_params["I"] = 0.5
    lens_params["td"] = 0.03
    
    print("\n1. Testing single point evaluation...")
    test_params = RP_params.copy()
    test_params["omega_tilde"] = 1.5
    test_params["theta_tilde"] = 7.0
    start = time.time()
    result = evaluate_mismatch(test_params, lens_params, gamma_points=11)
    elapsed = time.time() - start
    print(f"   Result: {result}")
    print(f"   Time: {elapsed:.2f}s")
    
    print("\n2. Testing small 2D grid with multiprocessing (3x3)...")
    start = time.time()
    try:
        mismatch_dict = evaluate_mismatch_2D(
            RP_params,
            lens_params,
            ("omega_tilde", "theta_tilde"),
            ((1.0, 2.0), (6.0, 8.0)),
            (3, 3),
            multithread=True,
            pbar=True,
            gamma_points=11,
        )
        elapsed = time.time() - start
        print(f"   ✓ Completed successfully in {elapsed:.2f}s")
        print(f"   Result shape: {mismatch_dict['results'].shape}")
        print(f"   Min epsilon: {np.nanmin(mismatch_dict['results']):.6f}")
        print(f"   Max epsilon: {np.nanmax(mismatch_dict['results']):.6f}")
    except Exception as e:
        import traceback
        print(f"   ✗ Failed: {e}")
        print(traceback.format_exc())
    
    print("\n3. Testing slightly larger grid with multiprocessing (5x5)...")
    start = time.time()
    try:
        mismatch_dict = evaluate_mismatch_2D(
            RP_params,
            lens_params,
            ("omega_tilde", "theta_tilde"),
            ((0.5, 2.5), (5.0, 10.0)),
            (5, 5),
            multithread=True,
            pbar=True,
            gamma_points=11,
        )
        elapsed = time.time() - start
        print(f"   ✓ Completed successfully in {elapsed:.2f}s")
        print(f"   Result shape: {mismatch_dict['results'].shape}")
        print(f"   Min epsilon: {np.nanmin(mismatch_dict['results']):.6f}")
        print(f"   Max epsilon: {np.nanmax(mismatch_dict['results']):.6f}")
    except Exception as e:
        import traceback
        print(f"   ✗ Failed: {e}")
        print(traceback.format_exc())
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
