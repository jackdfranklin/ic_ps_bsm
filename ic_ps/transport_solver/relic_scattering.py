import numpy as np
from scipy import interpolate
import math

import ic_ps.core.flux as flux
import ic_ps.transport_solver as transport_solver

class Relic_Scattered_Flux(flux.Flux):

    def __init__(self, relic_density_cm_3, neutrino_masses_GeV, distance_Mpc):
        self.relic_density_cm_3 = relic_density_cm_3
        self.neutrino_masses_GeV = neutrino_masses_GeV
        self.flux_dict = {}

        self.distance_Mpc = distance_Mpc

    def dflux_dE(self, E, gamma):

        if gamma in self.flux_dict:
            return self.flux_dict[gamma](E)
        else:

            flux = transport_solver.transport_flux(np.logspace(2,10,300), gamma, self.distance_Mpc, self.neutrino_masses_GeV, self.relic_density_cm_3, 1000)
            self.flux_dict[gamma] = interpolate.PchipInterpolator(np.logspace(2,10,300), flux, extrapolate=False)
            return self.flux_dict[gamma](E)

