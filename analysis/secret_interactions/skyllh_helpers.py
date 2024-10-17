import numpy as np
from scipy import interpolate
from scipy import integrate
import math

import ic_ps.core.flux as flux
import ic_ps.bsm.transport_solver as transport_solver

import abc
from astropy import (
    units,
)
import numpy as np
from scipy.integrate import (
    quad,
)
import scipy.special
import scipy.stats

from skyllh.core import (
    tool,
)
from skyllh.core.config import (
    Config,
    HasConfig,
)
from skyllh.core.math import (
    MathFunction,
)
from skyllh.core.model import (
    Model,
)
from skyllh.core.py import (
    classname,
    float_cast,
)
from skyllh.core.source_model import (
    IsPointlike,
)

from skyllh.core.flux_model import (
    EnergyFluxProfile,
)

def three_point_quad(a, b, f):

    s1 = -np.sqrt(3/5)
    s2 = 0
    s3 = np.sqrt(3/5)

    x1 = 0.5*(b-a)*s1 + 0.5*(a+b)
    x2 = 0.5*(b-a)*s2 + 0.5*(a+b)
    x3 = 0.5*(b-a)*s3 + 0.5*(a+b)

    w1 = 5/9
    w2 = 8/9
    w3 = 5/9
    
    return 0.5 * (b-a) * ( w1 * f(x1) + w2 * f(x2) + w3 * f(x3) )

class SIInteractionsFluxProfile(
        EnergyFluxProfile,
):
    def __init__(
            self,
            E0,
            gamma_grid,
            distance_Mpc,
            g,
            m_phi_GeV,
            neutrino_masses_GeV,
            relic_density_cm_3,
            steps,
            energy_bins=500,
            energy_unit=None,
            **kwargs):
        """
        Parameters
        ----------
        E0 : castable to float
            The reference energy.
        gamma_grid:
            numpy ndarray of values of gamma
        gamma : castable to float
            The spectral index.
        distance_Mpc : The distance to the source in Mpc
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(
            energy_unit=energy_unit,
            **kwargs)

        self._E0 = E0
        self._gamma = 2.0 
        self._neutrino_masses_GeV = neutrino_masses_GeV 
        self._relic_density_cm_3 = relic_density_cm_3 
        self._distance_Mpc = distance_Mpc
        self._g = g
        self._m_phi_GeV = m_phi_GeV

        self._E_vals = np.logspace(1,10,energy_bins)

        self._interpolated_flux_dict = {}
        self._integrated_flux_dict = {}

        fluxes = transport_solver.transport_flux_SI(
                        self._g, self._m_phi_GeV, 
                        self._E_vals, 
                        gamma_grid, 
                        self._distance_Mpc,
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3,
                        100) 

        for i in range(gamma_grid.size):
            gamma = gamma_grid[i]
            self._interpolated_flux_dict[gamma] = interpolate.PchipInterpolator(
                    self._E_vals, fluxes[:,i], 
                    extrapolate=False)

            self._integrated_flux_dict[gamma] = self._interpolated_flux_dict[gamma].antiderivative()

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('gamma')

    @property
    def E0(self):
        """The reference energy in the set energy unit of this EnergyFluxProfile
        instance.
        """
        return self._E0

    @property
    def gamma(self):
        """The spectral index.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, v):
        v = float_cast(
            v,
            'Property gamma must be castable to type float!')
        self._gamma = v

        if (self._gamma in self._interpolated_flux_dict) == False:

            flux = transport_solver.transport_flux_SI_redshift(
                        self._g, self._m_phi_GeV, 
                        self._E_vals, 
                        np.expand_dims(np.array([self._gamma]), axis=1), 
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3,
                        100) 

            self._interpolated_flux_dict[self._gamma] = interpolate.PchipInterpolator(
                    self._E_vals, np.squeeze(flux), 
                    extrapolate=False)

            self._integrated_flux_dict[self._gamma] = self._interpolated_flux_dict[self._gamma].antiderivative()

    def __call__(
            self,
            E,
            unit=None):

        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        value = self._interpolated_flux_dict[self._gamma](E)

        return value

    def get_integral(
            self,
            E1,
            E2,
            unit=None):

        if (self._gamma in self._interpolated_flux_dict) == False:

            flux = transport_solver.transport_flux_SI_redshift(
                        self._g, self._m_phi_GeV, 
                        self._E_vals, 
                        np.expand_dims(np.array([self._gamma]), axis=1), 
                        self._z, 
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3)

            self._interpolated_flux_dict[self._gamma] = interpolate.PchipInterpolator(
                    self._E_vals, np.squeeze(flux), 
                    extrapolate=False)

            self._integrated_flux_dict[self._gamma] = self._interpolated_flux_dict[self._gamma].antiderivative()

        integral = three_point_quad(E1, E2, self._interpolated_flux_dict[self._gamma])

        return integral 

    @property
    def math_function_str(self):
        return f'Solution to transport PDE for g = {self._g}, Mphi {self._m_phi_GeV*1e3} MeV'


class SIInteractionsRedshiftedFluxProfile(
        EnergyFluxProfile,
):
    def __init__(
            self,
            E0,
            gamma_grid,
            z,
            g,
            m_phi_GeV,
            neutrino_masses_GeV,
            relic_density_cm_3,
            energy_bins = 500,
            energy_unit=None,
            **kwargs):
        """
        Parameters
        ----------
        E0 : castable to float
            The reference energy.
        gamma_grid:
            numpy ndarray of values of gamma
        gamma : castable to float
            The spectral index.
        z : The redshift of the source of interest
        energy_unit : instance of astropy.units.UnitBase | None
            The used unit for energy.
            If set to ``None``, the configured default energy unit for fluxes is
            used.
        """
        super().__init__(
            energy_unit=energy_unit,
            **kwargs)

        self._E0 = E0
        self._gamma = 2.0 
        self._z = z
        self._neutrino_masses_GeV = neutrino_masses_GeV 
        self._relic_density_cm_3 = relic_density_cm_3 
        self._g = g
        self._m_phi_GeV = m_phi_GeV

        self._E_vals = np.logspace(1,10,energy_bins)

        self._interpolated_flux_dict = {}
        self._integrated_flux_dict = {}

        fluxes = transport_solver.transport_flux_SI_redshift(
                        self._g, self._m_phi_GeV, 
                        self._E_vals, 
                        gamma_grid, self._z, 
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3) 

        for i in range(gamma_grid.size):
            gamma = gamma_grid[i]
            self._interpolated_flux_dict[gamma] = interpolate.PchipInterpolator(
                    self._E_vals, fluxes[:,i], 
                    extrapolate=False)

            self._integrated_flux_dict[gamma] = self._interpolated_flux_dict[gamma].antiderivative()

        # Define the parameters which can be set via the `set_params`
        # method.
        self.param_names = ('gamma')

    @property
    def E0(self):
        """The reference energy in the set energy unit of this EnergyFluxProfile
        instance.
        """
        return self._E0

    @property
    def gamma(self):
        """The spectral index.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, v):
        v = float_cast(
            v,
            'Property gamma must be castable to type float!')
        self._gamma = v

        if (self._gamma in self._interpolated_flux_dict) == False:

            flux = transport_solver.transport_flux_SI_redshift(
                        self._g, self._m_phi_GeV, 
                        self._E_vals, 
                        np.expand_dims(np.array([self._gamma]), axis=1), 
                        self._z, 
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3) 

            self._interpolated_flux_dict[self._gamma] = interpolate.PchipInterpolator(
                    self._E_vals, np.squeeze(flux), 
                    extrapolate=False)

            self._integrated_flux_dict[self._gamma] = self._interpolated_flux_dict[self._gamma].antiderivative()

    def __call__(
            self,
            E,
            unit=None):

        E = np.atleast_1d(E)

        if (unit is not None) and (unit != self._energy_unit):
            E = E * unit.to(self._energy_unit)

        value = self._interpolated_flux_dict[self._gamma](E)

        return value

    def get_integral(
            self,
            E1,
            E2,
            unit=None):

        if (self._gamma in self._interpolated_flux_dict) == False:

            flux = transport_solver.transport_flux_SI_redshift(
                        self._g, self._m_phi_GeV, 
                        self._E_vals, 
                        np.expand_dims(np.array([self._gamma]), axis=1), 
                        self._z, 
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3)

            flux[flux == 0.0] = np.max(flux) * 1e-30

            self._interpolated_flux_dict[self._gamma] = interpolate.PchipInterpolator(
                    self._E_vals, np.squeeze(flux), 
                    extrapolate=False)

            self._integrated_flux_dict[self._gamma] = self._interpolated_flux_dict[self._gamma].antiderivative()

        integral = three_point_quad(E1, E2, self._interpolated_flux_dict[self._gamma])
            
        return integral 

    @property
    def math_function_str(self):
        return f'Solution to transport PDE for g = {self._g}, Mphi {self._m_phi_GeV*1e3} MeV'

