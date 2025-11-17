#!/usr/bin/env python3
"""
Diagnostic test to check if multiprocessing works correctly.
This will help identify pickling or deadlock issues.
"""
from lensing_and_precession.modules.default_params_ver2 import *
from scripts.helper_classes import *
from scripts.helper_functions import *
import numpy as np
import concurrent.futures

def simple_test_function(x, y):
    """A simple test function that should always work."""
    return x + y

def test_with_params(t_params, s_params, omega, theta):
    """Test function that uses parameter dicts."""
    # Just return a simple value to test pickling
    return omega * theta + t_params.get("mcz", 1.0)

def test_mismatch_single_point(t_params, s_params, omega, theta):
    """Test a single mismatch calculation."""
    try:
        from scripts.helper_functions import evaluate_mismatch
        t_copy = t_params.copy()
        t_copy["omega_tilde"] = omega
        t_copy["theta_tilde"] = theta
        result = evaluate_mismatch(t_copy, s_params)
        return result
    except Exception as e:
        return f"ERROR: {e}"

if __name__ == "__main__":
    print("=" * 60)
    print("DIAGNOSTIC TEST 1: Simple function with ProcessPoolExecutor")
    print("=" * 60)
    
    # Test 1: Simple function
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simple_test_function, i, i*2) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        print(f"✓ Test 1 passed: {results}")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST 2: Function with dict parameters")
    print("=" * 60)
    
    # Test 2: Function with dicts
    try:
        test_dict1 = {"mcz": 40.0, "I": 0.5}
        test_dict2 = {"td": 0.03}
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(test_with_params, test_dict1, test_dict2, i*0.5, i*1.0) 
                      for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        print(f"✓ Test 2 passed: {results}")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST 3: Actual mismatch calculation")
    print("=" * 60)
    
    # Test 3: Real mismatch calculation
    try:
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
        
        print("Attempting to pickle parameters...")
        import pickle
        pickle.dumps(RP_params)
        pickle.dumps(lens_params)
        print("✓ Parameters are pickleable")
        
        print("\nAttempting single sequential mismatch calculation...")
        result = test_mismatch_single_point(RP_params, lens_params, 1.5, 7.0)
        print(f"✓ Sequential calculation result: {result}")
        
        print("\nAttempting parallel mismatch calculations (4 workers, 4 points)...")
        test_points = [(1.0, 5.0), (1.5, 7.0), (2.0, 9.0), (2.5, 11.0)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(test_mismatch_single_point, RP_params, lens_params, omega, theta)
                      for omega, theta in test_points]
            results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures, timeout=60)):
                result = future.result()
                results.append(result)
                print(f"  Completed {i+1}/{len(test_points)}: {result}")
        
        print(f"✓ Test 3 passed: All parallel calculations completed")
        print(f"  Results: {results}")
        
    except concurrent.futures.TimeoutError:
        print("✗ Test 3 timed out after 60 seconds (likely deadlock)")
    except Exception as e:
        import traceback
        print(f"✗ Test 3 failed: {e}")
        print(traceback.format_exc())
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
