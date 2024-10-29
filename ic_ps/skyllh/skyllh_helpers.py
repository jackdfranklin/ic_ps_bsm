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

def bin_integral(a, b, f_bins, x_bin_edges):
    # Integral between a and b from bins

    a = np.where(a < x_bin_edges[0], x_bin_edges[0], a)
    b = np.where(b > x_bin_edges[-1], x_bin_edges[-1], b)

    f_int_bins = f_bins * np.diff(x_bin_edges)
    # Index of bin ABOVE a
    a_ind = np.searchsorted(x_bin_edges, a)
    # Index of bin BELOW b
    b_ind = np.searchsorted(x_bin_edges, b) - 1

    # Integrate in reverse to preserve integral of small bins
    rev_cumsum = np.cumsum(f_int_bins[::-1])[::-1]

    integral = np.where(b < x_bin_edges[-1],
                        rev_cumsum[a_ind] - rev_cumsum[b_ind],
                        rev_cumsum[a_ind] - rev_cumsum[-1])

    # Add residual integral when a is not aligned with bin edges
    integral += f_bins[a_ind] * ( x_bin_edges[a_ind] - a )

    mask = b < x_bin_edges[-1]
    integral[mask] += f_bins[b_ind[mask]] * ( b[mask] - x_bin_edges[b_ind[mask]] )

    return integral

class ScatteredPLFluxProfile(
        EnergyFluxProfile,
):
    def __init__(
            self,
            E0,
            gamma_grid,
            neutrino_masses_GeV,
            relic_density_cm_3,
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

        self._l10Emin = 1
        self._l10Emax = 10
        self._Nbins = energy_bins
        self._deltalog10E = (self._l10Emax - self._l10Emin)/self._Nbins

        self._E_vals = np.logspace(1.5,10.5,energy_bins)

        self._E_lower = np.power(10, np.log10(self._E_vals) - 0.5*self._deltalog10E)
        self._E_upper = np.power(10, np.log10(self._E_vals) + 0.5*self._deltalog10E)

        self._E_bin_edges = np.append(self._E_lower, self._E_upper[-1])

        self._interpolated_flux_dict = {}
        self._flux_dict = {}

        fluxes = self.solve_for_flux(gamma_grid) 

        for i in range(gamma_grid.size):
            gamma = gamma_grid[i]
            self._interpolated_flux_dict[gamma] = interpolate.PchipInterpolator(
                    self._E_vals, fluxes[:,i], 
                    extrapolate=False)

            self._flux_dict[gamma] = fluxes[:,i]

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

            fluxes = self.solve_for_flux(
                            np.expand_dims(np.array([self._gamma]), axis=1)) 

            self._flux_dict[self._gamma] = fluxes[:,i]

            self._interpolated_flux_dict[self._gamma] = interpolate.PchipInterpolator(
                    self._E_vals, np.squeeze(flux), 
                    extrapolate=False)

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

            fluxes = self.solve_for_flux(
                            np.expand_dims(np.array([self._gamma]), axis=1)) 

            self._flux_dict[self._gamma] = fluxes[:,i]

            self._interpolated_flux_dict[self._gamma] = interpolate.PchipInterpolator(
                    self._E_vals, np.squeeze(flux), 
                    extrapolate=False)

        integral = bin_integral(E1, E2, self._flux_dict[self._gamma], self._E_bin_edges)

        return integral 

    @property
    def math_function_str(self):
        return f'Solution to transport PDE for g = {self._g}, Mphi {self._m_phi_GeV*1e3} MeV'

    def solve_for_flux(self, gamma_grid):

        return np.zeros((gamma_grid.size, self._E_vals.size))

class SIInteractionsFluxProfile(
        ScatteredPLFluxProfile,
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
        distance_Mpc : The distance to the source in Mpc
        g            : The coupling between tau neutrinos to be probed
        m_phi_GeV    : The mass of scalar mediator to be probed
        """

        self._distance_Mpc = distance_Mpc
        self._g = g
        self._m_phi_GeV = m_phi_GeV
        self._steps = steps

        super().__init__(
                E0 = E0,
                gamma_grid = gamma_grid,
                neutrino_masses_GeV = neutrino_masses_GeV,
                relic_density_cm_3 = relic_density_cm_3,
                energy_bins = energy_bins,
                energy_unit=energy_unit,
                **kwargs)


    def solve_for_flux(self, gamma_grid):

        fluxes = transport_solver.transport_flux_SI(
                        self._g, self._m_phi_GeV, 
                        self._E_vals, 
                        gamma_grid, 
                        self._distance_Mpc,
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3,
                        self._steps) 

        return fluxes

class SIInteractionsRedshiftedFluxProfile(
        ScatteredPLFluxProfile,
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
        z : The redshift of the source of interest
        """

        self._z = z
        self._g = g
        self._m_phi_GeV = m_phi_GeV

        super().__init__(
                E0 = E0,
                gamma_grid = gamma_grid,
                neutrino_masses_GeV = neutrino_masses_GeV,
                relic_density_cm_3 = relic_density_cm_3,
                energy_bins = energy_bins,
                energy_unit=energy_unit,
                **kwargs)

    def solve_for_flux(self, gamma_grid):

        fluxes = transport_solver.transport_flux_SI_redshift(
                        self._g, self._m_phi_GeV, 
                        self._E_vals, 
                        gamma_grid, self._z, 
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3) 

        return fluxes

class SMInteractionsFluxProfile(
        ScatteredPLFluxProfile,
):
    def __init__(
            self,
            E0,
            gamma_grid,
            distance_Mpc,
            neutrino_masses_GeV,
            relic_density_cm_3,
            steps,
            energy_bins=500,
            energy_unit=None,
            **kwargs):
        """
        Parameters
        ----------
        distance_Mpc : The distance to the source in Mpc
        """

        self._distance_Mpc = distance_Mpc
        self._steps = steps

        super().__init__(
                E0 = E0,
                gamma_grid = gamma_grid,
                neutrino_masses_GeV = neutrino_masses_GeV,
                relic_density_cm_3 = relic_density_cm_3,
                energy_bins = energy_bins,
                energy_unit=energy_unit,
                **kwargs)

    def solve_for_flux(self, gamma_grid):
        fluxes = transport_solver.transport_flux_SM(
                        self._E_vals, 
                        gamma_grid, 
                        self._distance_Mpc,
                        self._neutrino_masses_GeV, 
                        self._relic_density_cm_3,
                        self._steps) 
        return fluxes

class SMInteractionsRedshiftedFluxProfile(
        ScatteredPLFluxProfile,
):
    def __init__(
            self,
            E0,
            gamma_grid,
            z,
            neutrino_masses_GeV,
            relic_density_cm_3,
            energy_bins=500,
            energy_unit=None,
            **kwargs):
        """
        Parameters
        ----------
        distance_Mpc : The distance to the source in Mpc
        """
        self._z = z

        super().__init__(
                E0 = E0,
                gamma_grid = gamma_grid,
                neutrino_masses_GeV = neutrino_masses_GeV,
                relic_density_cm_3 = relic_density_cm_3,
                energy_bins = energy_bins,
                energy_unit=energy_unit,
                **kwargs)

    def solve_for_flux(self, gamma_grid):

        fluxes = transport_solver.transport_flux_SM_redshift(
                    self._E_vals, 
                    gamma_grid,
                    self._z, 
                    self._neutrino_masses_GeV, 
                    self._relic_density_cm_3)

        return fluxes
