import numpy as np
import pandas as pd
import pickle
from scipy import interpolate
from scipy import integrate
from scipy.optimize import fmin_l_bfgs_b
from scipy import special

import os
import sys

from functools import partial
import math
import cmath

import pdf
import eff_area
import smear_mat
import dataset

class Flux:

    def __init__(self):
        self.E0 = 1000

    def dflux_dE(self, E, gamma):
        return (E/self.E0)**-gamma

    def dflux_dlog10E(self, log10E, gamma):    

        E = 10**log10E
        dflux = np.log(10)*E*self.dflux_dE(E, gamma)

        return dflux

class Decay_31_Flux(Flux):


    def __init__(self, coupling, m3_eV, distance_Mpc):
        self.E0 = 1000
        distance_m = distance_Mpc*3.086*10**22        
        self.L = distance_m/(1.97*10**-16)
        self.g = coupling

        self.m3 = m3_eV*10**-9 #Convert from eV to GeV

        self.calc_mixing_matrix()
        self.get_initial_normalisation()


    def calc_mixing_matrix(self):

        #Define neutrino flavour mixing parameters
        #Values taken from NuFit Nov 2022
        sin2_theta12 = 0.303
        sin2_theta23 = 0.572
        sin2_theta13 = 0.02203

        sin_theta_12  = math.sqrt(sin2_theta12) 
        cos_theta_12  = math.sqrt(1-sin2_theta12) 

        sin_theta_23  = math.sqrt(sin2_theta23) 
        cos_theta_23  = math.sqrt(1-sin2_theta23) 

        sin_theta_13  = math.sqrt(sin2_theta13) 
        cos_theta_13  = math.sqrt(1-sin2_theta13) 

        deltaCP = np.radians(197)
        deltaCP_exp = complex(math.cos(deltaCP), -math.sin(deltaCP))

        U_e1 = cos_theta_12*cos_theta_13
        self.U_e1_sq = abs(U_e1)**2

        U_mu1 = -sin_theta_12*cos_theta_23 - cos_theta_12*sin_theta_23*sin_theta_13*deltaCP_exp
        self.U_mu1_sq = abs(U_mu1)**2

        U_e2=  (sin_theta_12*cos_theta_13)**2
        self.U_e2_sq = abs(U_e2)**2
        U_mu2 = cos_theta_12*cos_theta_23 - sin_theta_12*sin_theta_23*sin_theta_13*deltaCP_exp
        self.U_mu2_sq = abs(U_mu2)**2

        U_e3 = sin_theta_13
        self.U_e3_sq = abs(U_e3)**2
        U_mu3 = sin_theta_23*cos_theta_13
        self.U_mu3_sq = abs(U_mu3)**2

    def get_initial_normalisation(self):

        self.flavour_ratio = np.array([1/3, 2/3, 0])

        self.Phi0_1 = self.flavour_ratio[0]*self.U_e1_sq + self.flavour_ratio[1]*self.U_mu1_sq
        
        self.Phi0_2 = self.flavour_ratio[0]*self.U_e2_sq + self.flavour_ratio[1]*self.U_mu2_sq

        self.Phi0_3 = self.flavour_ratio[0]*self.U_e3_sq + self.flavour_ratio[1]*self.U_mu3_sq

    def dflux1_dE(self, E, gamma):

        dflux_at_source = self.Phi0_1*(E/self.E0)**-gamma

        A = self.g**2*self.m3**2/(64*np.pi)
        AL_E = A*self.L/E
        dflux_from_decay = 2*self.Phi0_3*(E/self.E0)**-gamma/(1+gamma) 
        dflux_from_decay *= 1-np.exp(-AL_E) - AL_E**-gamma * special.gammainc(2+gamma, AL_E)*special.gamma(2+gamma)
    
        return dflux_at_source + dflux_from_decay

    def dflux2_dE(self, E, gamma):

        dflux_at_source = self.Phi0_2*(E/self.E0)**-gamma
        return dflux_at_source 

    def dflux3_dE(self, E, gamma):

        dflux_at_source = self.Phi0_3*(E/self.E0)**-gamma

        decay_rate = self.g**2 * self.m3**2 / (64*np.pi*E)

        return np.exp(-decay_rate*self.L)*dflux_at_source

    def dflux_dE(self, E, gamma):
        
        return self.U_mu1_sq * self.dflux1_dE(E, gamma) + self.U_mu2_sq * self.dflux2_dE(E, gamma) + self.U_mu3_sq * self.dflux3_dE(E, gamma)

class Invisible_Decay_31_Flux(Decay_31_Flux):

    def dflux1_dE(self, E, gamma):

        dflux_at_source = self.Phi0_1*(E/self.E0)**-gamma
        return dflux_at_source 


class Majorana_Decay_31_Flux(Decay_31_Flux):

    def dflux1_dE(self, E, gamma):

        dflux_at_source = self.Phi0_1*(E/self.E0)**-gamma

        A = self.g**2*self.m3**2/(32*np.pi)
        AL_E = A*self.L/E

        dflux_from_decay = 2*self.Phi0_3*(E/1000)**-gamma / gamma
        dflux_from_decay *= 1-np.exp(-AL_E) - AL_E**-gamma * special.gammainc(1+gamma, AL_E)*special.gamma(1+gamma)

        return dflux_at_source + dflux_from_decay

class Invisible_Decay_31_21_Flux(Decay_31_Flux):

    def __init__(self, coupling, m2_eV, m3_eV, distance_Mpc): 
        self.E0 = 1000
        distance_m = distance_Mpc*3.086*10**22        
        self.L = distance_m/(1.97*10**-16)
        self.g = coupling

        self.m2 = m2_eV*10**-9 #Convert from eV to GeV
        self.m3 = m3_eV*10**-9 #Convert from eV to GeV

        self.calc_mixing_matrix()
        self.get_initial_normalisation()

    def dflux1_dE(self, E, gamma):

        dflux_at_source = self.Phi0_1*(E/self.E0)**-gamma
        return dflux_at_source 

    def dflux2_dE(self, E, gamma):

        dflux_at_source = self.Phi0_2*(E/self.E0)**-gamma

        decay_rate = self.g**2 * self.m2**2 / (64*np.pi*E)

        return np.exp(-decay_rate*self.L)*dflux_at_source

class Dirac_Decay_31_21_Flux(Invisible_Decay_31_21_Flux):


    def dflux1_dE(self, E, gamma):

        dflux_at_source = self.Phi0_1*(E/self.E0)**-gamma

        A3 = self.g**2*self.m3**2/(64*np.pi)
        A3L_E = A3*self.L/E
        dflux_from_3decay = 2*self.Phi0_3*(E/self.E0)**-gamma/(1+gamma) 
        dflux_from_3decay *= 1-np.exp(-A3L_E) - A3L_E**-gamma * special.gammainc(2+gamma, A3L_E)*special.gamma(2+gamma)

        A2 = self.g**2*self.m2**2/(64*np.pi)
        A2L_E = A2*self.L/E
        dflux_from_2decay = 2*self.Phi0_3*(E/self.E0)**-gamma/(1+gamma) 
        dflux_from_2decay *= 1-np.exp(-A2L_E) - A2L_E**-gamma * special.gammainc(2+gamma, A2L_E)*special.gamma(2+gamma)
    
        return dflux_at_source + dflux_from_3decay + dflux_from_2decay
