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

import ic_ps.core.pdf as pdf
import ic_ps.core.convolve as convolve
import ic_ps.core.eff_area as eff_area
import ic_ps.core.smear_mat as smear_mat
import ic_ps.core.dataset as dataset

class Likelihood:

    def __init__(self, src_ra, src_dec, directory, bkg_vals_file, flux_obj):
        self.src_ra = src_ra
        self.src_dec = src_dec
        
        self.effA = eff_area.effective_area(directory+"/irfs/IC86_II_effectiveArea.csv")
        self.smear = smear_mat.smear_matrix(directory+"/irfs/IC86_II_smearing.csv")
        self.rebin = convolve.rebin(self.effA, self.smear)

        self.flux_obj = flux_obj

        self.dataset = dataset.DataSet(directory, "IC86_II-VII")
        self.dataset.select_and_prepare_data(src_dec, src_ra)
        self.runtime = self.dataset.total_runtime
        self.events_log10Ereco = self.dataset.events_log10Ereco
        self.events_psi = self.dataset.events_psi
        self.events_ang_err = self.dataset.events_ang_err
        self.N = self.dataset.N
        self.N_tot = self.dataset.N_tot
        self.deltaN = self.N_tot-self.N

        self.log10E_min = np.min(self.dataset.events_log10Ereco)
        self.log10E_max = np.max(self.dataset.events_log10Ereco)

        #If pre-calculated background pd values don't exist, calculate them
        if(os.path.isfile(bkg_vals_file)):
        #Load pre-calculated background pd values from pickle file
            with open(bkg_vals_file, 'rb') as file:
                self.bkg_pdf_vals = pickle.load(file)
        else:

            with open(directory+"/bkg_pdf_2d_kde.pkl", "rb") as bkg_file:
                bkg_pdf = pickle.load(bkg_file)

                #Evaluate bkg pdf for events once and store
                print("Evaluating background pd values")
                self.bkg_pdf_vals = bkg_pdf([self.dataset.events_log10Ereco,np.sin(np.radians(self.dataset.events_dec))])/(2*np.pi)
                #Add small tolerance to avoid dividing by zero
                self.bkg_pdf_vals = np.where(self.bkg_pdf_vals<1e-20, 1e-20, self.bkg_pdf_vals)

                with open(bkg_vals_file, 'wb') as file:
                    pickle.dump(self.bkg_pdf_vals, file)

        #source pdfs 
        self.energy_pdf = self.smear.get_energy_pdf(src_dec) 
        self.source_psi_pdf = pdf.RayleighPDF()
        #print("Evaluating source energy pdf")
        self.source_energy_pdf = self.get_source_energy_pdf(self.energy_pdf) 

    def dflux_dlog10E(self, log10E, gamma):

        return self.flux_obj.dflux_dlog10E(log10E, gamma)

    def get_model_pdf(self, gamma):
        
        dflux = lambda log10E: self.dflux_dlog10E(log10E, gamma)
        model_pdf = convolve.convolve_point_source(dflux, self.src_dec, self.effA)
        model_pdf[model_pdf < 1e-20] = 1e-20
        pdf_spline = interpolate.PchipInterpolator(self.effA.log10Enu_points, model_pdf, extrapolate=False)

        return pdf_spline

    def get_source_energy_pdf(self, energy_pdf):

        def spline_1d_safe(spline, vals):
            #Evaluates a spline and replaces out of bounds values (which are NaN) with zero

            pdf_vals = spline(vals)
            pdf_vals = np.where(np.isnan(pdf_vals), 0.0, pdf_vals)
        
            return pdf_vals

        #Get minimum and maximum neutrino energies
        l10Enu_min = energy_pdf.log10Enu_bin_edges[0]
        l10Enu_max = energy_pdf.log10Enu_bin_edges[-1]

        n_Enu_bins = energy_pdf.n_Enu_bins
        Enu_bin_edges = energy_pdf.log10Enu_bin_edges

        #Define axes for final pdf spline
        l10Es = np.linspace(self.log10E_min, self.log10E_max, 150)
        gammas = np.linspace(1.5,4.5,80)

        source_pdf = np.zeros((150,80))

        for i,g in enumerate(gammas):
            #Get pdf of Enu given gamma
            model_pdf_spline = self.get_model_pdf(g)
            model_pdf = partial(spline_1d_safe, model_pdf_spline)
            for j in range(n_Enu_bins):
    
                energy_range = [np.maximum(Enu_bin_edges[j], l10Enu_min), np.minimum(Enu_bin_edges[j+1], l10Enu_max)]
                #Integrate model pdf over energy bin
                weight = integrate.quad(model_pdf, energy_range[0], energy_range[1])[0]
                log10Enu = self.energy_pdf.log10Enu_mids[j]
                pdf_vals = self.energy_pdf(l10Es, log10Enu)
                pdf_vals = np.where(np.isnan(pdf_vals), 0.0, pdf_vals)
                source_pdf[:,i] += pdf_vals*weight

        log10pdf = np.empty(source_pdf.shape, dtype=np.float64)
        pdf_mask = source_pdf > 0 #Only want values greater than zero
        log10pdf[pdf_mask] = np.log10(source_pdf[pdf_mask])
        log10pdf[np.invert(pdf_mask)] = 1e-20 #np.min(log10pdf[pdf_mask]) - 3 

        source_pdf_spline = interpolate.RectBivariateSpline(l10Es, gammas, log10pdf)

        return source_pdf_spline

    def _dspdf_dgamma(self, psi, ang_err, log10Emu, gamma):
        if(isinstance(self.source_psi_pdf, pdf.RayleighPDF)):
            return 0.0

    def _depdf_dgamma(self, log10Ereco, gamma, epdf_vals):
        deriv = self.source_energy_pdf.ev(log10Ereco, np.full(np.shape(log10Ereco), gamma), 0, 1)
        deriv *= np.log(10)*epdf_vals
        return deriv

    def logl_vals(self, ns, gamma):

        spatial_pdf_vals = self.source_psi_pdf.pdf(self.events_psi, self.events_ang_err)/(2*np.pi*np.sin(np.radians(self.events_psi)))

        energy_pdf_vals = 10**self.source_energy_pdf.ev(self.events_log10Ereco, np.full(self.N, gamma))
        #norm = integrate.quad(lambda l10E: 10**self.source_energy_pdf.ev(l10E, gamma), self.log10E_min, self.log10E_max)[0]
        #Replace nan values (out of bounds) with very small values 
        energy_pdf_vals = np.where(np.isnan(energy_pdf_vals), 1e-20, energy_pdf_vals)
        energy_pdf_vals = np.where(energy_pdf_vals<1e-20, 1e-20, energy_pdf_vals)
        #energy_pdf_vals /= norm

        signal_background_ratio = spatial_pdf_vals*energy_pdf_vals/self.bkg_pdf_vals

        n_r = ns/self.N
        #n_r = ns/self.N_tot

        likelihood_ratio_vals = n_r*(signal_background_ratio - 1) + 1
        likelihood_ratio_vals = np.where(likelihood_ratio_vals<1e-20, 1e-20, likelihood_ratio_vals)

        alpha = n_r*(signal_background_ratio-1)
        alpha_ = (alpha-1e-6)/(1+1e-6)
        log_likelihood_vals = np.where(np.abs(alpha)<1e-6, np.log(1+1e-6) + alpha_ - 0.5*alpha_**2, np.log(likelihood_ratio_vals))

        return log_likelihood_vals

    def events_by_logl(self, ns, gamma):

        log_likelihood_vals = self.logl_vals(ns, gamma)

        new_df = pd.DataFrame({
            'logl value': log_likelihood_vals,
            'log10Ereco': self.events_log10Ereco,
            'psi':        self.events_psi,
        })

        return new_df.sort_values(by=['logl value'], ascending=False)

    def logl(self, params, return_grad=False):
        ns = params[0]
        gamma = params[1]

        if(ns < 0 or gamma < 0):
            if(return_grad):
                return -1./1e-20, np.asarray([-0.5, -0.5])
            else:
                return -1./1e-20

        spatial_pdf_vals = self.source_psi_pdf.pdf(self.events_psi, self.events_ang_err)/(2*np.pi*np.sin(np.radians(self.events_psi)))

        energy_pdf_vals = 10**self.source_energy_pdf.ev(self.events_log10Ereco, np.full(self.N, gamma))
        #norm = integrate.quad(lambda l10E: 10**self.source_energy_pdf.ev(l10E, gamma), self.log10E_min, self.log10E_max)[0]
        #Replace nan values (out of bounds) with very small values 
        energy_pdf_vals = np.where(np.isnan(energy_pdf_vals), 1e-20, energy_pdf_vals)
        energy_pdf_vals = np.where(energy_pdf_vals<1e-20, 1e-20, energy_pdf_vals)
        #energy_pdf_vals /= norm

        signal_background_ratio = spatial_pdf_vals*energy_pdf_vals/self.bkg_pdf_vals

        n_r = ns/self.N
        #n_r = ns/self.N_tot

        likelihood_ratio_vals = n_r*(signal_background_ratio - 1) + 1
        likelihood_ratio_vals = np.where(likelihood_ratio_vals<1e-20, 1e-20, likelihood_ratio_vals)

        alpha = n_r*(signal_background_ratio-1)
        alpha_ = (alpha-1e-6)/(1+1e-6)
        log_likelihood_vals = np.where(np.abs(alpha)<1e-6, np.log(1+1e-6) + alpha_ - 0.5*alpha_**2, np.log(likelihood_ratio_vals))

        logl_ratio = np.sum(log_likelihood_vals)
        #logl_ratio += self.deltaN * np.log(1-n_r) #Add contribution from data outside of box with S~0
        
        if(ns == 0.0):
            dlogl_dns = (np.sum(signal_background_ratio-1)-self.deltaN)/self.N_tot
            d2logl_dns2 = -(np.sum((signal_background_ratio-1)**2) + self.deltaN)/self.N_tot**2
            logl_ratio = dlogl_dns**2/(4*d2logl_dns2) 

        #logl_ratio = ns*dlogl_dns + 0.5*d2logl_dns2*ns**2
        
        if(return_grad):
            dlogl_dn = self._dlogl_dns(ns, likelihood_ratio_vals, signal_background_ratio)
            spatial_deriv = self._dspdf_dgamma(0,0,0,0) #Replace 0's with values if using actual spatial pdf
            energy_deriv = self._depdf_dgamma(self.events_log10Ereco, gamma, energy_pdf_vals)
            dlogl_dg = self._dlogl_dgamma(ns, likelihood_ratio_vals, spatial_pdf_vals, spatial_deriv, energy_pdf_vals, energy_deriv)

            #print(params, dlogl_dn, dlogl_dg)

            return (logl_ratio, np.asarray([dlogl_dn, dlogl_dg]))
        else:
            return logl_ratio 

    def _dlogl_dns(self, ns, l_vals, sbr_vals):

        n_r = ns/self.N
        #n_r = ns/self.N_tot
        dlogl_dns = np.sum((sbr_vals-1.)/l_vals)
        #dlogl_dns += self.deltaN*-1./(1.-n_r)
        dlogl_dns /= self.N
        #dlogl_dns /= self.N_tot
        return dlogl_dns

    def _d2logl_dns2(self, ns, l_vals, sbr_vals):

        n_r = ns/self.N_tot
        d2logl_dns2 = np.sum(((sbr_vals-1)/l_vals)**2)
        d2logl_dns2 += self.deltaN/(1-n_r)**2
        d2logl_dns2 *= -1./self.N_tot**2 
        return d2logl_dns2


    def _dlogl_dgamma(self, ns, l_vals, s_vals, s_deriv, e_vals, e_deriv):

        n_r = ns/self.N
        #n_r = ns/self.N_tot
        dlogl_dgamma = 1/(l_vals*self.bkg_pdf_vals)
        product_rule = s_deriv*e_vals + s_vals*e_deriv
        dlogl_dgamma *= product_rule
        dlogl_dgamma = n_r* np.sum(dlogl_dgamma)

        return dlogl_dgamma

    def fit(self):
        ftol = 1.e-9
        pgtol = 1.e-9
        factr = ftol / np.finfo(float).eps

        #Use scipy l_bfgs_b minimiser
        def func(x):
            params = x
            f, grad = self.logl(params, return_grad=True)
            f *= -2.0
            grad *= -2.0

            return f, grad
   
        N_tries = 10

        fmin = None
        final_result = None
        bounds = np.asarray([[0, 1000], [1.5, 4.0]])

        for i in range(N_tries):

            r = np.random.rand(2)
            initial_guess = [r[0]*bounds[0,1]+(1-r[0])*bounds[0,0], r[1]*bounds[1,1]+(1-r[1])*bounds[1,0]]
            #initial_guess = [10, 2.5]

            result, fval, warning = fmin_l_bfgs_b(func, initial_guess, approx_grad=False, bounds=bounds, pgtol=pgtol, factr=factr, iprint=-1, maxls=40)
            #print(result, -fval)
            #print(warning)
            #print(self.logl(result, return_grad=True))

            if fmin==None or fval<fmin:
                fmin = fval
                final_result = result
                
         
        return final_result, -fmin

    def null_h(self):

        return self.bkg_pdf_vals

    def ns_to_Phi0(self, ns, gamma):
        #Calulate binned num events
        integral=0.0
        effA_vals = self.effA.point_source(self.src_dec)
        dflux_dlE = lambda log10Enu: self.dflux_dlog10E(log10Enu, gamma)
        for i, effA_val in enumerate(effA_vals):
            l10Enu_min = self.effA.log10Enu_bin_edges[i]
            l10Enu_max = self.effA.log10Enu_bin_edges[i+1]
            integral += integrate.quad(dflux_dlE, l10Enu_min, l10Enu_max)[0]*effA_val

        Phi0 = ns/(integral*self.runtime) 
        return Phi0

class New_Likelihood(Likelihood):

    def __init__(self, src_ra, src_dec, directory, flux_obj, sigma):

        self.src_ra = src_ra
        self.src_dec = src_dec
        
        self.effA = eff_area.effective_area(directory+"/irfs/IC86_II_effectiveArea.csv")

        self.flux_obj = flux_obj

        self.dataset = dataset.DataSet(directory, "new")
        self.dataset.select_and_prepare_data(src_dec, src_ra)
        self.runtime = self.dataset.total_runtime
        self.events_log10Ereco = self.dataset.events_log10Ereco
        self.events_psi = self.dataset.events_psi
        self.events_ang_err = self.dataset.events_ang_err
        self.N = self.dataset.N
        self.N_tot = self.dataset.N_tot
        self.deltaN = self.N_tot-self.N
        #print("Total number of events = "+str(self.N))

        self.log10E_min = np.min(self.dataset.events_log10Ereco)
        self.log10E_max = np.max(self.dataset.events_log10Ereco)

        #Use this when utilising a new background pdf for the first time 

        with open(directory+"/new_bkg_pdf_2d_kde.pkl", "rb") as bkg_file:
            self.bkg_pdf = pickle.load(bkg_file)

        #Evaluate bkg pdf for events once and store

        print("Evaluating background pd values")
        self.bkg_pdf_vals = self.bkg_pdf([self.dataset.events_log10Ereco,np.sin(np.radians(self.dataset.events_dec))])/(2*np.pi)
        #Add small tolerance to avoid dividing by zero
        self.bkg_pdf_vals = np.where(self.bkg_pdf_vals<1e-20, 1e-20, self.bkg_pdf_vals)
        
        with open('new_bkg_pd_vals.pkl', 'wb') as file:
            pickle.dump(self.bkg_pdf_vals, file)
        #Load pre-calculated background pd values from pickle file

        #with open(directory+'/new_bkg_pd_vals.pkl', 'rb') as file:
        #    self.bkg_pdf_vals = pickle.load(file)

        #source pdfs 
        #print("Evaluating smearing pdfs")
        self.energy_pdf = pdf.GaussianPDF(sigma)
        self.source_psi_pdf = pdf.RayleighPDF()
        #print("Evaluating source energy pdf")
        self.source_energy_pdf = self.get_source_energy_pdf(self.energy_pdf) 

    def get_source_energy_pdf(self, energy_pdf):

        def spline_1d_safe(spline, vals):
            #Evaluates a spline and replaces out of bounds values (which are NaN) with zero

            pdf_vals = spline(vals)
            pdf_vals = np.where(np.isnan(pdf_vals), 0.0, pdf_vals)
        
            return pdf_vals

        #Define axes for final pdf spline
        l10Es = np.linspace(self.log10E_min, self.log10E_max, 150)
        gammas = np.linspace(0.5,4.5,80)

        source_pdf = np.zeros((150,80))

        for i,g in enumerate(gammas):
            #Get pdf of Enu given gamma
            model_pdf_spline = self.get_model_pdf(g)
            model_pdf = partial(spline_1d_safe, model_pdf_spline)
            for j, l10E in enumerate(l10Es):
    
                source_pdf[j,i]= integrate.quad(lambda x: model_pdf(x)*self.energy_pdf(l10E, x), 2, 10)[0]

        source_pdf = np.where(np.isnan(pdf_vals), 0.0, pdf_vals)

        log10pdf = np.empty(source_pdf.shape, dtype=np.float64)
        pdf_mask = source_pdf > 0 #Only want values greater than zero
        log10pdf[pdf_mask] = np.log10(source_pdf[pdf_mask])
        log10pdf[np.invert(pdf_mask)] = 1e-20 #np.min(log10pdf[pdf_mask]) - 3 

        source_pdf_spline = interpolate.RectBivariateSpline(l10Es, gammas, log10pdf)

        return source_pdf_spline
