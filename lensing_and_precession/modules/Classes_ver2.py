#############################
# Section 1: Import Modules #
#############################

# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

import numpy as np
import os

error_handler = np.seterr(invalid="raise")
from warnings import warn
from scipy.integrate import cumulative_trapezoid as integrate

from scipy.integrate import odeint

import scipy.special as sc
import mpmath as mp
from pycbc.types import FrequencySeries

from scipy.integrate import solve_ivp
from typing import Union, Sequence, Dict

NEAR_ZERO_THRESHOLD = 1e-8


def get_numerical_derivative(func, x0, dx = 0.01):
    x1 = x0 + dx
    y0 = func(x0)
    y1 = func(x1)
    return (y1 - y0) / dx

############################
# Section 2: Lensing Class #
############################

class LensingBase:
    def __init__(self, params):
        self.params = params

        assert type(self.params == dict), "Parameters should be a dictionary"

        # unlensed parameters
        self.theta_S = params["theta_S"]
        self.phi_S = params["phi_S"]
        self.theta_J = params["theta_J"]  # J == L (no precession)
        self.phi_J = params["phi_J"]  # J == L (no precession)
        self.mcz = params["mcz"]
        self.dist = params["dist"]
        self.eta = params["eta"]
        self.t_c = params["t_c"]
        self.phi_c = params["phi_c"]

        # lensed parameters
        self.M_Lz = params["MLz"]
        self.y = params["y"]

        self.cache = {}

    def total_mass(self):
        """Total mass from chirp mass [seconds]"""
        return self.mcz / (self.eta ** (3 / 5))

    def f_cut(self):
        """f_cut"""
        return 1 / (6 ** (3 / 2) * np.pi * self.total_mass())

    def LdotN(self):
        """(cosine angle between l and n)"""
        cos_term = np.cos(self.theta_S) * np.cos(self.theta_J)
        sin_term = (
            np.sin(self.theta_S)
            * np.sin(self.theta_J)
            * np.cos(self.phi_S - self.phi_J)
        )
        inner_prod = cos_term + sin_term
        return inner_prod

    def amp(self):
        """A for h(f)"""
        amplitude = (
            np.sqrt(5 / 96) * np.pi ** (-2 / 3) * self.mcz ** (5 / 6) / (self.dist)
        )
        return amplitude

    def Psi(self, f):
        """eqn 3.13 in Cutler-Flanaghan 1994"""
        x = (np.pi * self.total_mass() * f) ** (2 / 3)
        term1 = 2 * np.pi * f * self.t_c - self.phi_c - np.pi / 4
        prefactor = (3 / 4) * (8 * np.pi * self.mcz * f) ** (-5 / 3)
        term2 = (
            1
            + (20 / 9) * (743 / 336 + (11 / 4) * self.eta) * x
            - 16 * np.pi * x ** (3 / 2)
        )
        Psi = term1 + prefactor * term2
        return Psi

    def psi_s(self):
        """psi_s that goes into F_plus and F_cross"""

        numerator = np.cos(self.theta_J) - np.cos(self.theta_S) * (self.LdotN())
        denominator = (
            np.sin(self.theta_S)
            * np.sin(self.theta_J)
            * np.sin(self.phi_J - self.phi_S)
        )

        psi_s_val = np.arctan2(numerator, denominator)
        return psi_s_val

    def fIp(self):
        """F_plus"""

        term_1 = (
            1
            / 2
            * (1 + np.power(np.cos(self.theta_S), 2))
            * np.cos(2 * self.phi_S)
            * np.cos(2 * self.psi_s())
        )
        term_2 = (
            np.cos(self.theta_S) * np.sin(2 * self.phi_S) * np.sin(2 * self.psi_s())
        )

        fIp_val = term_1 - term_2
        return fIp_val

    def fIc(self):
        """F_cross"""

        term_1 = (
            1
            / 2
            * (1 + np.power(np.cos(self.theta_S), 2))
            * np.cos(2 * self.phi_S)
            * np.sin(2 * self.psi_s())
        )
        term_2 = (
            np.cos(self.theta_S) * np.sin(2 * self.phi_S) * np.cos(2 * self.psi_s())
        )

        fIc_val = term_1 + term_2
        return fIc_val

    def lambdaI(self):
        """|F_plus (1+L.N**2) - i (2*F_cross*L.N)|"""

        term_1 = np.power(2 * self.LdotN() * self.fIc(), 2)
        term_2 = np.power((1 + np.power(self.LdotN(), 2)) * self.fIp(), 2)
        lambdaI_val = np.sqrt(term_1 + term_2)
        return lambdaI_val

    def phi_pI(self):
        """tan-1((2*F_cross*L.N)/F_plus (1+L.N**2))"""

        numerator = 2 * self.LdotN() * self.fIc()
        denominator = (1 + np.power(self.LdotN(), 2)) * self.fIp()

        phi_pI_val = np.arctan2(numerator, denominator)
        return phi_pI_val

    def hI(self, f):
        """Unlensed Waveform"""

        term_1 = self.lambdaI()
        term_2 = np.exp(-1j * self.phi_pI())
        term_3 = self.amp() * f ** (-7 / 6)
        term_4 = np.exp(1j * self.Psi(f))

        signal_I = term_1 * term_2 * term_3 * term_4

        return signal_I
    
    def F(self, f) -> np.complex128:
        warn("Warning: amplification factor not implemented in LensingBase!")
        return np.complex128(0)

    def strain(self, f, delta_f=0.25, frequencySeries=True):
        """lensed strain = unlensed strain * amplification factor
        Args:
            f (numpy array): frequency range
            delta_f (float): interval length of frequency. Default at 0.25 Hz.
            frequencySeries (bool): True for FrequencySeries. False otherwise.

        Returns:
            hL (numpy array): lensed strain
        """
        # cacheName = "strain" + str(f) + str(delta_f) + str(frequencySeries)
        # if cacheName in self.cache: return self.cache[cacheName]

        hL = self.hI(f) * self.F(f)

        if frequencySeries:
            # self.cache[cacheName] = FrequencySeries(hL, delta_f)
            # return self.cache[cacheName]
            return FrequencySeries(hL, delta_f)

        # self.cache[cacheName] = hL
        return hL

class LensingPM(LensingBase):
    def __init__(self, params):
        super().__init__(params)
    
    def F(self, f) -> np.complex128:
        """PM amplification factor in exact form, equation 17 in Takahashi & Nakamura 2003"""
        self.w = 8 * np.pi * self.M_Lz * f
        x_m = 0.5 * (self.y + np.sqrt(self.y**2 + 4))
        phi_m = np.power((x_m - self.y), 2) / 2 - np.log(x_m)

        term1 = np.exp(
            np.pi * self.w / 4 + 1j * (self.w / 2) * (np.log(self.w / 2) - 2 * phi_m)
        )
        term2 = sc.gamma(1 - 1j * (self.w / 2))

        # broadcasting mp hyp1f1 function to NumPy ufunc
        hyp1f1_np = np.frompyfunc(mp.hyp1f1, 3, 1)

        term3 = hyp1f1_np(1j * self.w / 2, 1, 1j * (self.w / 2) * (self.y**2))

        F_val = np.complex128(term1 * term2 * term3)

        return F_val

class LensingGeo_PM(LensingBase):
    def __init__(self, params):
        super().__init__(params)

    def mu_plus(self):
        """plus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023"""
        mu_plus_val = (
            1 / 2 + (self.y**2 + 2) / (2 * self.y * np.sqrt(self.y**2 + 4)) + 0j
        )
        return mu_plus_val

    def mu_minus(self):
        """minus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023"""
        mu_minus_val = (
            1 / 2 - (self.y**2 + 2) / (2 * self.y * np.sqrt(self.y**2 + 4)) + 0j
        )
        return mu_minus_val

    def I(self):
        """flux ratio, equation 17a in Saif et al. 2023"""
        I_val = np.abs(self.mu_minus()) / np.abs(self.mu_plus())
        return I_val

    def td(self):
        """time delay, equation 16b in Saif et al. 2023"""
        td_val = (
            2
            * self.M_Lz
            * (
                self.y * np.sqrt(self.y**2 + 4)
                + 2
                * np.log(
                    (np.sqrt(self.y**2 + 4) + self.y)
                    / (np.sqrt(self.y**2 + 4) - self.y)
                )
            )
        )
        return td_val

    def F(self, f):
        """PM amplification factor in geometric optics limit, equation 18 in Takahashi & Nakamura 2003"""
        F_val = np.sqrt(np.abs(self.mu_plus())) - 1j * np.sqrt(
            np.abs(self.mu_minus())
        ) * np.exp(2j * np.pi * f * self.td())
        return F_val

class LensingGeo(LensingBase):
    def __init__(self, params):
        self.td_val = params["td"]
        self.I_val = params["I"]
        super().__init__(params)
    
    def F(self, f) -> np.complex128:
        F_val = 1 - 1j * np.sqrt(self.I_val) * np.exp(2j * np.pi * f * self.td_val)
        return F_val
    
    def td(self): return self.td_val

    def I(self): return self.I_val


###############################
# Section 3: Precessing Class #
###############################


class PrecessingV2:

    cutoff_threshold = 0

    def __init__(self, params):
        self.params = params

        # self.solver = Kvaerno5()

        assert type(self.params == dict), "Parameters should be a dictionary"

        # non-precession/unlensed parameters
        self.theta_S = params["theta_S"]
        self.phi_S = params["phi_S"]
        self.theta_J = params["theta_J"]
        self.phi_J = params["phi_J"]
        self.mcz = params["mcz"]
        self.dist = params["dist"]
        self.eta = params["eta"]
        self.t_c = params["t_c"]
        self.phi_c = params["phi_c"]

        # regular precession parameters
        self.theta_tilde = params["theta_tilde"]
        self.omega_tilde = params["omega_tilde"]
        self.gamma_P = params["gamma_P"]

        # some converters/constants

        self.cache = {}

        self.LdotN_threshold = 0.0625 # found with convergence study. see convergence_test_for_delta_phi.py
        self.SOLMASS2SEC = 4.92624076 * 1e-6  # solar mass -> seconds
        self.GIGAPC2SEC = 1.02927125 * 1e17  # gigaparsec -> seconds
        self.FMIN = 20  # lower frequency of the detector sensitivity band [Hz]

        # self.angle_diff = 0.01


    def total_mass(self):
        """Total mass from chirp mass [seconds]"""
        return self.mcz / (self.eta ** (3 / 5))

    def f_cut(self):
        """f_cut"""
        return 1 / (6 ** (3 / 2) * np.pi * self.total_mass())

    def theta_LJ(self, f):
        """theta_LJ_new"""
        return (0.1 / (4 * self.eta)) * self.theta_tilde * (f / self.f_cut()) ** (1 / 3)
    
    def Omega_LJ(self, f):
        return 1000 * self.omega_tilde * (f / self.f_cut())**(5/3) / (self.total_mass() / self.SOLMASS2SEC)

    def phi_LJ(self, f):
        """phi_LJ"""
        num = (5000 / 96) * self.omega_tilde
        deno = (
            (self.total_mass() / self.SOLMASS2SEC)
            * (np.pi ** (8 / 3))
            * (self.mcz ** (5 / 3))
            * (self.f_cut() ** (5 / 3))
        )
        phi_LJ_amp = num / deno
        return phi_LJ_amp * (1 / self.FMIN - 1 / f) + self.gamma_P

    def amp_prefactor(self) -> float:
        """amplitude prefactor calculated using chirp mass and distance"""
        amp_prefactor = (
            np.sqrt(5 / 96) * (np.pi ** (-2 / 3)) * (self.mcz ** (5 / 6)) / self.dist
        )
        return amp_prefactor

    def precession_angles(self):
        """some angles"""

        if self.phi_J == self.phi_S:
            if self.theta_J == self.theta_S:
                cos_i_JN = 1
            else:
                cos_i_JN = np.cos(self.theta_J - self.theta_S)

        else:
            cos_i_JN = np.sin(self.theta_J) * np.sin(self.theta_S) * np.cos(
                self.phi_J - self.phi_S
            ) + np.cos(self.theta_J) * np.cos(self.theta_S)

        sin_i_JN = np.sqrt(1 - cos_i_JN**2.0)

        if np.abs(sin_i_JN) < NEAR_ZERO_THRESHOLD:
            cos_o_XH = 1
            sin_o_XH = 0
        else:
            cos_o_XH = (
                np.cos(self.theta_S)
                * np.sin(self.theta_J)
                * np.cos(self.phi_J - self.phi_S)
                - np.sin(self.theta_S) * np.cos(self.theta_J)
            ) / (
                sin_i_JN
            )  # seems to be cos Omega_{XH}
            sin_o_XH = (np.sin(self.theta_J) * np.sin(self.phi_J - self.phi_S)) / (
                sin_i_JN
            )
        return cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH

    def LdotN(self, f):
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        LdotN = (
            np.sin(self.theta_LJ(f)) * sin_i_JN * np.sin(self.phi_LJ(f))
            + np.cos(self.theta_LJ(f)) * cos_i_JN
        )
        return LdotN

    def polarization_amplitude_and_phase(self, f):
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        # for C
        C_amp = np.sqrt(
            0.25
            * (1 + (np.cos(self.theta_S)) ** 2) ** 2
            * ((np.cos(2 * self.phi_S)) ** 2)
            + ((np.cos(self.theta_S)) ** 2 * (np.sin(2 * self.phi_S)) ** 2)
        )

        # define alpha
        sin_alpha = np.cos(self.theta_S) * np.sin(2 * self.phi_S) / C_amp
        cos_alpha = (
            (1 + np.cos(self.theta_S) ** 2) * np.cos(2 * self.phi_S) / (2 * C_amp)
        )
        alpha = np.arctan2(sin_alpha, cos_alpha)

        # compute psi robustly with atan2 to avoid singular tan when denominator ~ 0
        num_psi = (
            np.sin(self.theta_LJ(f))
            * (
                np.cos(self.phi_LJ(f)) * sin_o_XH
                + np.sin(self.phi_LJ(f)) * cos_i_JN * cos_o_XH
            )
            - np.cos(self.theta_LJ(f)) * sin_i_JN * cos_o_XH
        )
        den_psi = (
            np.sin(self.theta_LJ(f))
            * (
                np.cos(self.phi_LJ(f)) * cos_o_XH
                - np.sin(self.phi_LJ(f)) * cos_i_JN * sin_o_XH
            )
            + np.cos(self.theta_LJ(f)) * sin_i_JN * sin_o_XH
        )
        if self.phi_S == self.phi_J and self.theta_S == self.theta_J:
            psi = self.phi_LJ(f)
        else:
            psi = np.arctan2(num_psi, den_psi)

        # if den_psi.all() == 0:  # True for face-on and theta_tilde = 0
        #     if self.theta_tilde == 0:  # WRONG!!! Refer to Eq A14 in Taman's paper!
        #         return C_amp, 0, -1

        # define  2 * Psi + alpha using direct sin/cos for numerical stability
        ang = 2.0 * psi + alpha
        sin_2pa = np.sin(ang)
        cos_2pa = np.cos(ang)

        return C_amp, sin_2pa, cos_2pa

    ### get the amplitude
    def amplitude(self, f) -> np.ndarray:
        """NP/Unlensed amplitude"""
        LdotN = self.LdotN(f)
        C_amp, sin_2pa, cos_2pa = self.polarization_amplitude_and_phase(f)

        amp = (
            self.amp_prefactor()
            * C_amp
            * f ** (-7 / 6)
            * np.sqrt(4 * LdotN**2 * sin_2pa**2 + cos_2pa**2 * (1 + LdotN**2) ** 2)
        )
        return amp
    
    def freq_from_angle(self, mu):
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        i_JN = np.atan2(sin_i_JN, cos_i_JN)
        return self.f_cut() * np.pow((i_JN + mu) * 4 * self.eta / (0.1 * self.theta_tilde), 3) if self.theta_tilde != 0 else 0

    ### get the phase phi_P
    def phase_phi_P(self, f):
        """phi_p"""
        LdotN = self.LdotN(f)
        C_amp, sin_2pa, cos_2pa = self.polarization_amplitude_and_phase(f)

        phi_p_temp = np.arctan2(2 * LdotN * sin_2pa, (1 + LdotN**2) * cos_2pa)
        phi_p = np.unwrap(phi_p_temp, discont=np.pi)
        return phi_p

    def f_dot(self, f):
        """df/dt from Cutler Flanagan 1994"""
        prefactor = (96 / 5) * np.pi ** (8 / 3) * self.mcz ** (5 / 3) * f ** (11 / 3)
        return prefactor # * (1 - (743/336 + (11/4) * self.eta) * (np.pi * self.total_mass() * f)**(2/3) + 4 * np.pi * (np.pi * self.total_mass() * f))

    def get_outer_integrand(self, f):
        LdotN = self.LdotN(f)
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        f_dot = self.f_dot(f)
        return ((LdotN / (1 - LdotN**2))
                * (self.Omega_LJ(f)
                * np.sin(self.theta_LJ(f))
                * (
                    np.cos(self.theta_LJ(f)) * sin_i_JN * np.sin(self.phi_LJ(f))
                    - np.sin(self.theta_LJ(f)) * cos_i_JN
                ) / f_dot 
                - (self.theta_LJ(f) / (3 * f)) * np.cos(self.phi_LJ(f)) * sin_i_JN))
 
    def beta(self, f):
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        f_dot = self.f_dot(f)

        return self.theta_LJ(f) * f_dot / (3 * f * sin_i_JN * self.Omega_LJ(f)) if self.omega_tilde != 0 else 0

    def get_const_inner_term(self, f):
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        f_dot = self.f_dot(f)
        beta = self.beta(f)
        return -0.5 * self.Omega_LJ(f) * cos_i_JN * ((1 + 2*np.pow(beta, 2))/(1 + np.pow(beta, 2))) / f_dot
    
    def is_small_angle(self, f):
        LdotN = self.LdotN(f)
        return np.abs(np.abs(LdotN) - 1) < NEAR_ZERO_THRESHOLD
    
    def get_coeff_matrix(self, x0, x1, x2):
        return [[np.pow(x0, 4), np.pow(x0, 3), np.pow(x0, 2), x0, 1],
                [np.pow(x1, 4), np.pow(x1, 3), np.pow(x1, 2), x1, 1],
                [np.pow(x2, 4), np.pow(x2, 3), np.pow(x2, 2), x2, 1],
                [4*np.pow(x0, 3), 3*np.pow(x0, 2), 2*x0, 1, 0],
                [4*np.pow(x2, 3), 3*np.pow(x2, 2), 2*x2, 1, 0],
                ]

    def get_small_angle_approx(self, f):
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()

        f = np.asarray(f)

        f_al = self.freq_from_angle(0)
        # delta_f = f - f_al
        # f_dot = self.f_dot(f_al)
        # beta = self.beta(f_al)
        # theta_LJ = self.theta_LJ(f_al)
        # cot_i_JN = cos_i_JN / sin_i_JN
        # csc_i_JN = 1 / sin_i_JN
        # tan_i_JN = sin_i_JN / cos_i_JN
        y1 = self.get_const_inner_term(f_al)

        if(f.size == 1):
            if(self.is_small_angle(f)): return y1
            else: return 0


        small_f_indices = np.nonzero(np.where(self.is_small_angle(f), f, 0))[0]

        if(len(small_f_indices) == 0): return 0
        
        if(len(small_f_indices) == 1): return y1

        f_lower = f[small_f_indices[0]]
        f_upper = f[small_f_indices[-1]]

        y0 = self.get_outer_integrand(f_lower)
        y2 = self.get_outer_integrand(f_upper)

        deriv_0 = get_numerical_derivative(self.get_outer_integrand, f_lower)
        deriv_2 = get_numerical_derivative(self.get_outer_integrand, f_upper)

        coeff = self.get_coeff_matrix(f_lower, f_al, f_upper)

        result_arr = [y0, y1, y2, deriv_0, deriv_2]

        polynomial_coeff = np.linalg.solve(coeff, result_arr)
        f_list = np.asarray([np.pow(f, 4), np.pow(f, 3), np.pow(f, 2), f, np.ones_like(f)])
        f_list = np.transpose(f_list)
        return np.dot(f_list, polynomial_coeff)


    ### get the delta phi_P
    def integrand_delta_phi(self, f):
        """integrand for delta phi p (equations in Apostolatos 1994, and appendix of Evangelos in prep)"""

        # if self.theta_tilde == 0:  # non-precessing
        #     integrand_delta_phi = 0
        #     # not necessary to include this case, but just in case, check equations 17, 18a, A18 in Evangelos

        # if (
        #     np.abs(1 - cos_i_JN) < NEAR_ZERO_THRESHOLD
        # ):  # face-on (precessing & non-precessing)
        #     integrand_delta_phi = -Omega_LJ * np.cos(self.theta_LJ(f)) / f_dot

        # elif 1 - LdotN < self.cutoff_threshold: # verified with convergence study
        #     integrand_delta_phi = 0

        # else:
        #     integrand_delta_phi = (
        #         (LdotN / (1 - LdotN**2))
        #         * Omega_LJ
        #         * np.sin(self.theta_LJ(f))
        #         * (
        #             np.cos(self.theta_LJ(f)) * sin_i_JN * np.sin(self.phi_LJ(f))
        #             - np.sin(self.theta_LJ(f)) * cos_i_JN
        #         )
        #         / f_dot
        #     )

        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        f_dot = self.f_dot(f)
        
        return np.where(
            np.abs(1 - cos_i_JN) < NEAR_ZERO_THRESHOLD,
            -self.Omega_LJ(f) * np.cos(self.theta_LJ(f)) / f_dot,
            np.where(
                self.is_small_angle(f),
                self.get_const_inner_term(f),
                self.get_outer_integrand(f)
            )
        )

    def phase_delta_phi(self, f):
        #v2:
        integral = odeint(self.integrand_delta_phi_v2, 0, f)
        return np.squeeze(integral)

        #v3:
        # integrand = self.integrand_delta_phi(f)

        # I = integrate(integrand, x=f, initial=0)
        # return I

        #v4:
        # jax_integral = phase_delta_phi_jit(f, self.omega_tilde, self.theta_tilde, self.gamma_P, self.mcz, self.eta, self.phi_J, self.phi_S, self.theta_J, self.theta_S, self.t_c, self.phi_c, 0.01, solver=self.solver)
        # return jax.numpy.asarray(jax_integral)

    def integrand_delta_phi_v2(self, y, f):
        """integrand for delta phi p (equations in Apostolatos 1994, and appendix of Evangelos in prep)"""
        # Precompute reused quantities
        LdotN = self.LdotN(f)
        cos_i_JN, sin_i_JN, *_ = self.precession_angles()
        theta_LJ = self.theta_LJ(f)
        phi_LJ = self.phi_LJ(f)
        f_dot = self.f_dot(f)

        Omega_LJ = (
            1000
            * self.omega_tilde
            * (f / self.f_cut()) ** (5 / 3)
            / (self.total_mass() / self.SOLMASS2SEC)
        )

        if self.theta_tilde == 0:  # non-precessing
            return 0
        # not necessary to include this case, but just in case, check equations 17, 18a, A18 in Evangelos

        # Face-on case (precessing & non-precessing)
        if np.abs(1 - np.abs(cos_i_JN)) < NEAR_ZERO_THRESHOLD:
            return -Omega_LJ * np.cos(theta_LJ) / f_dot

        # L and N aligned case (coordinate singularity)
        # EXPANDED THRESHOLD: catch near-singularities before they cause spikes
        # When |LdotN| is close to 1, the denominator (1 - LdotN²) → 0
        # causing numerical explosions. Increase threshold to 1e-3 to catch earlier.
        if np.abs(np.abs(LdotN) - 1) < NEAR_ZERO_THRESHOLD:
            # NOT face-on & STILL precessing, when L and N are aligned at some point in the precession cycle
            # very rare, L aligns with N only ONCE as it spirals out --> blows up???
            # a coordinate singularity!!!
            return 0
        
        # Generic (non face-on) expression
        denominator = 1 - LdotN**2
        
        # Base term (Apostolatos 1994, Eq. A18)
        base = (
            (LdotN / denominator)
            * Omega_LJ
            * np.sin(theta_LJ)
            * (
                np.cos(theta_LJ) * sin_i_JN * np.sin(phi_LJ)
                - np.sin(theta_LJ) * cos_i_JN
            )
            / f_dot
        )
        
        # Correction term (Taman et al. 2025, Eq. A19)
        corr = (LdotN / denominator) * (
            -(theta_LJ / (3.0 * f)) * np.cos(phi_LJ) * sin_i_JN
        )
        
        return base + corr

    def Psi(self, f):
        """GW phase"""
        x = (np.pi * self.total_mass() * f) ** (2 / 3)
        Psi = (
            (2 * np.pi * f * self.t_c)
            - self.phi_c
            - np.pi / 4
            + ((3 / 4) * (8 * np.pi * self.mcz * f) ** (-5 / 3))
            * (
                1
                + (20 / 9) * (743 / 336 + (11 / 4) * self.eta) * x
                - 16 * np.pi * x ** (3 / 2)
            )
        )
        return Psi

    def cos_theta_L(self, f):
        """for figure 2 in Evangelos"""
        # from equation A8
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        # L_H = np.sin(self.theta_LJ(f)) * (np.cos(self.phi_LJ(f)) * cos_o_XH - np.sin(self.phi_LJ(f)) * cos_i_JN * sin_o_XH) + sin_i_JN * sin_o_XH * np.cos(self.theta_LJ(f))
        # L_V = np.sin(self.theta_LJ(f)) * (np.cos(self.phi_LJ(f)) * sin_o_XH + np.sin(self.phi_LJ(f)) * cos_i_JN * cos_o_XH) - sin_i_JN * cos_o_XH * np.cos(self.theta_LJ(f))
        # L_N = np.sin(self.theta_LJ(f)) * np.sin(self.phi_LJ(f)) * sin_i_JN + np.cos(self.theta_LJ(f)) * cos_i_JN

        L_z = (
            np.sin(self.theta_LJ(f))
            * (
                np.cos(self.phi_LJ(f)) * sin_o_XH
                + np.sin(self.phi_LJ(f)) * cos_i_JN * cos_o_XH
            )
            - sin_i_JN * cos_o_XH * np.cos(self.theta_LJ(f))
        ) * np.sin(self.theta_S) + (
            np.sin(self.theta_LJ(f)) * np.sin(self.phi_LJ(f)) * sin_i_JN
            + np.cos(self.theta_LJ(f)) * cos_i_JN
        ) * np.cos(
            self.theta_S
        )
        return L_z

    def phi_L(self, f):
        """for figure 2 in Evangelos"""
        # from equation A8
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        L_H = np.sin(self.theta_LJ(f)) * (
            np.cos(self.phi_LJ(f)) * cos_o_XH
            - np.sin(self.phi_LJ(f)) * cos_i_JN * sin_o_XH
        ) + sin_i_JN * sin_o_XH * np.cos(self.theta_LJ(f))
        L_V = np.sin(self.theta_LJ(f)) * (
            np.cos(self.phi_LJ(f)) * sin_o_XH
            + np.sin(self.phi_LJ(f)) * cos_i_JN * cos_o_XH
        ) - sin_i_JN * cos_o_XH * np.cos(self.theta_LJ(f))
        L_N = (
            np.sin(self.theta_LJ(f)) * np.sin(self.phi_LJ(f)) * sin_i_JN
            + np.cos(self.theta_LJ(f)) * cos_i_JN
        )

        L_x = (
            -np.sin(self.phi_S) * L_H
            - np.cos(self.theta_S) * np.cos(self.phi_S) * L_V
            + np.sin(self.theta_S) * np.cos(self.phi_S) * L_N
        )
        L_y = (
            np.cos(self.phi_S) * L_H
            - np.cos(self.theta_S) * np.sin(self.phi_S) * L_V
            + np.sin(self.theta_S) * np.sin(self.phi_S) * L_N
        )
        Phi_L = np.arctan2(L_y, L_x)
        # Phi_L_ur = np.unwrap(Phi_L, discont = np.pi)
        return Phi_L  # _ur

    def strain(self, f, delta_f=0.25, frequencySeries=True):
        """precessing GW"""
        
        # Compute phase with validation
        delta_phi = self.phase_delta_phi(f)
        
        strain = self.amplitude(f) * np.exp(
            1j * (self.Psi(f) - self.phase_phi_P(f) - 2 * delta_phi)
        )

        if frequencySeries:
            return FrequencySeries(strain, delta_f)
        
        return strain
    
class PrecessingV3(PrecessingV2):
    def phase_delta_phi(self, f):
        integrand = self.integrand_delta_phi(f)

        I = integrate(integrand, x=f, initial=0)
        return I
    
class Precessing(PrecessingV2):
    def integrand_delta_phi(self, y, f):
        """Wrapper to match solve_ivp signature - delegates to integrand_delta_phi_v2."""
        return self.integrand_delta_phi_v2(y, f)

    def phase_delta_phi(
        self,
        f,
        ivp_method: Union[str, Sequence[str]] = "LSODA",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: float = np.inf,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Compute delta phi_P using solve_ivp.

        Args:
            f (array-like): Strictly increasing frequency array.
            ivp_method (str | Sequence[str]): One or more solve_ivp methods
                (e.g., "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA").
            rtol (float): Relative tolerance for solve_ivp. Default 1e-3.
            atol (float): Absolute tolerance for solve_ivp. Default 1e-6.
            max_step (float): Maximum allowed step size.

        Returns:
            If ivp_method is a string, returns np.ndarray (len(f),).
            If ivp_method is a sequence, returns dict method -> np.ndarray.
        """
        f = np.asarray(f)
        if f.ndim != 1:
            f = f.ravel()
        if not np.all(np.diff(f) > 0):
            raise ValueError("Frequency array f must be strictly increasing.")

        def _solve_with_method(method_name: str) -> np.ndarray:
            # Use solve_ivp
            def rhs(freq, y):
                # dy/df = integrand_delta_phi(y, f)
                # Support both y.shape == (1,) and vectorized calls with y.shape == (1, m)
                try:
                    # Compute scalar integrand value (independent of y in our model)
                    val = float(self.integrand_delta_phi(0.0, freq))
                except Exception:
                    val = float(self.integrand_delta_phi(y, freq))

                y_arr = np.asarray(y)
                if y_arr.ndim == 0:
                    return np.asarray([val], dtype=float)
                # Broadcast to match shape of y
                return np.full_like(y_arr, fill_value=val, dtype=float)

            sol = solve_ivp(
                rhs,
                (float(f[0]), float(f[-1])),
                y0=[0.0],
                t_eval=f,
                method=method_name,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            if not sol.success:
                raise RuntimeError(
                    f"solve_ivp failed with method '{method_name}': {sol.message}"
                )
            return sol.y[0]

        if isinstance(ivp_method, str):
            return _solve_with_method(ivp_method)

        results: Dict[str, np.ndarray] = {}
        failures = []
        for method_name in ivp_method:
            try:
                results[str(method_name)] = _solve_with_method(str(method_name))
            except Exception as exc:
                failures.append((str(method_name), str(exc)))

        if results:
            return results
        raise RuntimeError(f"All solve_ivp methods failed: {failures}")