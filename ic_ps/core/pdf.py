import numpy as np
import pandas as pd

from scipy import interpolate
from scipy import integrate

class pdf:

    def __init__(self, pdf_array, edges):
        self.pdf_array = pdf_array
        self.bin_edges = edges
        self.D = np.shape(edges)[0]    
        self.logE_max_val = edges[0,-1]        
        self.logE_min_val = edges[0,0]
        self.dec_max_val = edges[1,-1]        
        self.dec_min_val = edges[1,0]
        self.ang_err_max_val = edges[2,-1]        
        self.ang_err_min_val = edges[2,0]

        print(np.shape(self.pdf_array))


    def indices_in_range(self, logE, dec, ang_err):
        #Checks if point is outside of bin range
        logE_test = (logE<self.logE_max_val) & (logE>=self.logE_min_val)
        dec_test = (dec<self.dec_max_val) & (dec>=self.dec_min_val)
        ang_err_test = (ang_err<self.ang_err_max_val) & (ang_err>=self.ang_err_min_val)
        test =  logE_test & dec_test & ang_err_test
        return np.nonzero(test) 

    def get_val(self, logE, dec, ang_err):
        #Indices of points contained by bins
        c_idxs = self.indices_in_range(logE, dec, ang_err)
        
        #Find indices in pdf array corresponding to points that are contained by the bins
        #This resolves the issue of looking for pdf values that are outside of the bins
        id0 = np.searchsorted(self.bin_edges[0],logE[c_idxs])-1 
        id1 = np.searchsorted(self.bin_edges[1],dec[c_idxs])-1 
        id2 = np.searchsorted(self.bin_edges[2],ang_err[c_idxs])-1 
 
        pdf_vals = np.zeros(np.shape(logE))
        pdf_vals[c_idxs] = self.pdf_array[id0,id1,id2]
        return pdf_vals

class RayleighPDF:

    def pdf(self, psi, sigma):

        return (psi/sigma**2)*np.exp(-0.5*(psi/sigma)**2)

class GaussianPDF:

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, log10Ereco, log10Emu):
        sigma = self.sigma * log10Emu 
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(x/sigma)**2)

class EnergyPDF:

    def __init__(self, log10Enu_bin_edges, log10Ereco_bin_edges, pdf_vals):

        self.n_Enu_bins = np.shape(log10Enu_bin_edges)[0]-1
        self.n_Ereco_bins = np.shape(log10Ereco_bin_edges)[1]-1

        if not np.shape(log10Ereco_bin_edges) == (self.n_Enu_bins, self.n_Ereco_bins+1):
            raise RuntimeError('reconstructed energy binning does not match nu energy and/or nu declination binning') 

        if not np.shape(pdf_vals) == (self.n_Enu_bins, self.n_Ereco_bins):
            raise RuntimeError('pdf values different shape to bins')
        
        self.log10Enu_bin_edges = log10Enu_bin_edges
        self.log10Enu_mids = 0.5*(log10Enu_bin_edges[:-1]+log10Enu_bin_edges[1:])
        self.log10Ereco_bin_edges = log10Ereco_bin_edges
        self.pdf_vals = pdf_vals
         
    def __call__(self, log10Ereco, log10Enu):

        #Calculate pdf values for given neutrino and reco energy.
        #Use get_pdf to get values of pdf at energy and reco bins, then interpolate using pchip, then eval at reco energies requested
        if not np.isscalar(log10Enu):
            raise RuntimeError('log10Enu must be a scalar')

        pdf_slice, log10Ereco_bin_edges_slice = self.get_pdf_slice(log10Enu) 

        log10Ereco_mids = 0.5*(log10Ereco_bin_edges_slice[:-1]+log10Ereco_bin_edges_slice[1:])
        log10Ereco_min = log10Ereco_bin_edges_slice[0]
        log10Ereco_max = log10Ereco_bin_edges_slice[-1]

        pdf_spline = interpolate.PchipInterpolator(log10Ereco_mids, pdf_slice, extrapolate=False)

        def pdf_func(log10Ereco):
            f = pdf_spline(log10Ereco)
            f = np.where(np.isnan(f), 0.0, f)
            return f

        norm = integrate.quad(pdf_func , log10Ereco_min, log10Ereco_max, limit=200, full_output=1)[0]

        pds = pdf_spline(log10Ereco)/norm

        return pds

    def get_pdf_slice(self, log10Enu):

        log10Enu_idx = np.searchsorted(self.log10Enu_bin_edges, log10Enu)-1
        return self.pdf_vals[log10Enu_idx], self.log10Ereco_bin_edges[log10Enu_idx]

    def get_pdf(self, log10Ereco, log10Enu):

        if not (np.isscalar(log10Enu) or np.shape(log10Enu)==np.shape(log10Ereco)):
            raise RuntimeError('log10Enu must either be a scalar or have the same dimensions as log10Ereco')

        log10Enu_idx = np.searchsorted(self.log10Enu_bin_edges, log10Enu)-1
        if(log10Enu_idx<0 or log10Enu>self.n_Enu_bins-1):
            raise RuntimeError('log10Enu is not in valid range')
        
        log10Ereco_idxs = np.searchsorted(self.log10Ereco_bin_edges[log10Enu_idx], log10Ereco)-1
        log10Ereco_mask = (log10Ereco_idxs>=0)&(log10Ereco_idxs<=self.n_Ereco_bins-1)

        pdf_vals = np.zeros(np.size(log10Ereco))
        pdf_vals[log10Ereco_mask] = self.pdf_vals[log10Enu_idx, log10Ereco_idxs[log10Ereco_mask]]

        return pdf_vals

class DetectionPDF:

    def __init__(self, pdf_spline, Enu_lims, dec_lims):

        self.pdf_spline = pdf_spline

        self.Enu_min = Enu_lims[0]
        self.Enu_max = Enu_lims[1]
        self.dec_min = dec_lims[0]
        self.dec_max = dec_lims[1]

    def is_inside_lims(self, Enu, dec):
        Enu_test = (self.Enu_min<=Enu)&(Enu<self.Enu_max)
        dec_test = (self.dec_min<=dec)&(dec<self.dec_max)

        return Enu_test&dec_test
    def pdf(self, Enu, dec):

        if(self.is_inside_lims(Enu, dec)==True):
            return self.pdf_spline(Enu, dec)
        else:
            return 0.0


