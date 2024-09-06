import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import integrate

#Calculates the rebinning matrix 
def rebin(effA, smear):

    rebin_arr = np.zeros((len(smear.log10Enu_bin_edges)-1, len(effA.log10Enu_bin_edges)-1))
    for y_idx in range(len(smear.log10Enu_bin_edges)-1):
        iyps = np.flatnonzero(effA.log10Enu_bin_edges > smear.log10Enu_bin_edges[y_idx])-1

        for iyp in iyps:
            ly = 0

            if(effA.log10Enu_bin_edges[iyp]<=smear.log10Enu_bin_edges[y_idx]): #upper part of old bin
                ly = effA.log10Enu_bin_edges[iyp+1]-smear.log10Enu_bin_edges[y_idx]

            elif(effA.log10Enu_bin_edges[iyp]<smear.log10Enu_bin_edges[y_idx+1]):

                if(effA.log10Enu_bin_edges[iyp+1]<=smear.log10Enu_bin_edges[y_idx+1]): #bin is between bin edges
                    ly = effA.log10Enu_bin_edges[iyp+1]-effA.log10Enu_bin_edges[iyp]

                elif(effA.log10Enu_bin_edges[iyp+1]>smear.log10Enu_bin_edges[y_idx+1]): #lower part of old bin
                    ly = smear.log10Enu_bin_edges[y_idx+1]-effA.log10Enu_bin_edges[iyp] 

            L = effA.log10Enu_bin_edges[iyp+1]-effA.log10Enu_bin_edges[iyp]
            rebin_arr[y_idx, iyp] = ly/L

    return rebin_arr

#Calculates the number of events in each effA bin given a flux object and runtime
def binned_1D(dflux_dlog10E, effA, runtime):
    
    mu = np.zeros((np.size(effA.dec_bin_edges)-1, np.size(effA.log10Enu_bin_edges)-1))
    
    for i in range(len(effA.log10Enu_bin_edges)-1):
        l10Enu_min = effA.log10Enu_bin_edges[i]
        l10Enu_max = effA.log10Enu_bin_edges[i+1]
        
        loc = (l10Enu_min, l10Enu_max, dec_min, dec_max)
            
        if(loc in effA.mindex):
            
            mu[j,i] = integrate.dblquad(flux.dflux_dlog10E, l10Enu_min, l10Enu_max, np.sin(np.radians(dec_min)), np.sin(np.radians(dec_max)))[0]
            mu[j,i] *= effA.effA.loc[loc]['Aeff']*runtime
            
    return mu

#Calculates the number of events in each effA bin given a flux object and runtime
def binned_2D(flux, effA, runtime):
    
    mu = np.zeros((np.size(effA.dec_bin_edges)-1, np.size(effA.log10Enu_bin_edges)-1))
    
    for i in range(len(effA.log10Enu_bin_edges)-1):
        l10Enu_min = effA.log10Enu_bin_edges[i]
        l10Enu_max = effA.log10Enu_bin_edges[i+1]
        
        for j in range(len(effA.dec_bin_edges)-1):
            dec_min = effA.dec_bin_edges[j]
            dec_max = effA.dec_bin_edges[j+1]
            
            loc = (l10Enu_min, l10Enu_max, dec_min, dec_max)
            
            if(loc in effA.mindex):
            
                mu[j,i] = integrate.dblquad(flux.dflux_dlog10E, l10Enu_min, l10Enu_max, np.sin(np.radians(dec_min)), np.sin(np.radians(dec_max)))[0]
                mu[j,i] *= effA.effA.loc[loc]['Aeff']*runtime
            
    return mu

#Convolves the point source flux and effective area and returns the PDF in measured values as a numpy ndarray
def convolve_point_source(dflux_dlE, dec, effA):
    
    #Calulate binned num events
    mu = np.zeros(np.size(effA.log10Enu_bin_edges)-1) 
    effA_vals = effA.point_source(dec)
    dlE = np.zeros(np.shape(effA_vals))
    for i, effA_val in enumerate(effA_vals):
        l10Enu_min = effA.log10Enu_bin_edges[i]
        l10Enu_max = effA.log10Enu_bin_edges[i+1]
        mu[i] = integrate.quad(dflux_dlE, l10Enu_min, l10Enu_max)[0]*effA_val
        dlE[i] = l10Enu_max-l10Enu_min
         
    mu_tot = np.sum(mu)
     
    if(mu_tot == 0):
        pdf = np.zeros(mu.shape)
    else:
        pdf = mu/(mu_tot*dlE)

    return pdf


def convolve_point_source_MC(flux, dec, effA, smear, N_samples=100):

    #Calulate binned num events
    mu = np.zeros(np.size(effA.log10Enu_bin_edges)-1) 
    for i in range(len(effA.log10Enu_bin_edges)-1):
        l10Enu_min = effA.log10Enu_bin_edges[i]
        l10Enu_max = effA.log10Enu_bin_edges[i+1]
        effA_val = np.array(effA.point_source(dec, [l10Enu_min, l10Enu_max])).flatten()
        mu[i] = integrate.quad(flux, 10**l10Enu_min, 10**l10Enu_max)[0]*effA_val
        
    mu_tot = np.sum(mu)
    
    mu_rebin = np.zeros(np.size(smear.l10E_bin_edges)-1)

    #Create weighting from number of events for effective area binning
    
    weights = mu/mu_tot
    
    #Need bin edges of effective area
    
    l10E_bin_edges = effA.log10Enu_bin_edges
    
    points = np.random.rand(N_samples)
    
    cdf = np.cumsum(weights)
    cdf = cdf/cdf[-1]
    
    #Distribute points according to proportion of events in bin
    point_bins = np.searchsorted(cdf, points)
    
    measured_points = []
    
    for i in range(N_samples):
        
        #Find index for weight
        iy = point_bins[i]
       
        r = np.random.rand(1)
    
        #Get values of energy and declination
        l10Enu = l10E_bin_edges[iy] + (l10E_bin_edges[iy+1]-l10E_bin_edges[iy])*r[0]
        
        #Find distribution in measured values for point
        table = smear.point_like(dec, l10Enu)
        weights = np.array(table['frac_counts'].copy())
        
        #Distribute points according to cumulative dsitribution
        cdf_m = np.cumsum(weights)
        cdf_m = cdf_m/cdf_m[-1]
        
        p = np.random.rand(N_samples)
        
        p_bins = np.searchsorted(cdf_m, p)
        
        for j in range(N_samples):
        
            p_bin = p_bins[j]
            #Generate random measured value
            r_m = np.random.rand(3)
        
            l10Emin = table['log10E_min'].iloc[p_bin]
            l10Emax = table['log10E_max'].iloc[p_bin]
        
            l10E = l10Emin + (l10Emax-l10Emin)*r_m[0]
        
            psi_min = table['psi_min'].iloc[p_bin]
            psi_max = table['psi_max'].iloc[p_bin]
        
            psi = psi_min + (psi_max-psi_min)*r_m[1]
        
            ang_err_min = table['ang_err_min'].iloc[p_bin]
            ang_err_max = table['ang_err_max'].iloc[p_bin]
        
            ang_err = ang_err_min + (ang_err_max-ang_err_min)*r_m[2]
        
            measured_points.append([l10E,psi,ang_err])
            
    measured_points = np.array(measured_points)
    
    measured_hist, edges = np.histogramdd(measured_points, bins=15, density=True)
    
    return np.array(measured_hist), np.array(edges)

def convolve_diffuse_source(flux_source, effA, smear, num_bins=50):
    #Calculate number of events in the effective area binning
    
    runtime = 1.0
    mu = binned(flux_source, effA, runtime)
    mu_tot = np.sum(mu)
    
    #Create weighting from number of events for effective area binning
    
    effA_weights = mu/mu_tot
    
    #Need bin edges of effective area
    
    log10Enu_bin_edges = effA.log10Enu_bin_edges
    dec_bin_edges = effA.dec_bin_edges
    
    N_points = 100000

    points = np.random.rand(N_points)
    
    cdf = np.cumsum(np.ravel(effA_weights))
    cdf = cdf/cdf[-1]
    
    #Distribute points according to proportion of events in bin
    point_bins = np.searchsorted(cdf, points)
    ixs, iys = np.unravel_index(point_bins, np.shape(effA_weights))
    
    measured_points = []
    
    for i in range(N_points):
        
        #Find index for weight
        ix = ixs[i]
        iy = iys[i]
       
        r = np.random.rand(2)
    
        #Get values of energy and declination
        log10Enu = log10Enu_bin_edges[iy] + (log10Enu_bin_edges[iy+1]-log10Enu_bin_edges[iy])*r[0]
        dec = dec_bin_edges[ix] + (dec_bin_edges[ix+1]-dec_bin_edges[ix])*r[1]
        
        #Find distribution in measured values for point
        table = smear.point_like(dec, log10Enu)
        log10E_min = table['log10E_min'].to_numpy()
        log10E_max = table['log10E_max'].to_numpy()
        psi_min = table['psi_min'].to_numpy()
        psi_max = table['psi_max'].to_numpy()
        
        weights = table['frac_counts'].to_numpy()
        
        #Distribute points according to cumulative dsitribution
        cdf_m = np.cumsum(weights)
        cdf_m = cdf_m/cdf_m[-1]
        
        N_points_m = 10
        p = np.random.rand(N_points_m)
        
        p_bins = np.searchsorted(cdf_m, p)
        
        for j in range(N_points_m):
        
            p_bin = p_bins[j]
            #Generate random measured value
            r_m = np.random.rand(3)
        
            log10E_min_val = log10E_min[p_bin]
            log10E_max_val = log10E_max[p_bin]
        
            log10E = log10E_min_val+ (log10E_max_val-log10E_max_val)*r_m[0]
        
            psi_min_val = psi_min[p_bin]
            psi_max_val = psi_max[p_bin]
        
            psi = psi_min_val + (psi_max_val-psi_min_val)*r_m[1]

            theta = 2*np.pi*r_m[2]
            dec_m = dec + psi*np.sin(theta)
            sindec_m = np.sin(np.radians(dec_m))
        
            #ang_err_min = table['ang_err_min'].iloc[p_bin]
            #ang_err_max = table['ang_err_max'].iloc[p_bin]
        
            #ang_err = ang_err_min + (ang_err_max-ang_err_min)*r_m[3]
        
            measured_points.append([log10E,sindec_m])
            
    measured_points = np.array(measured_points)

    return measured_points

