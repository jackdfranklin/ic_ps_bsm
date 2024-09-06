import numpy as np
import scipy.interpolate as interpolate

#Class that contains MC atmospheric neutrino fluxes, interpolates them and then returns them via a wrapper function
class atmData():
    def __init__(self,infile):
        data = np.loadtxt(infile)
        x_axis = np.unique(data[:,0])
        y_axis = np.unique(data[:,1])
        total_flux = np.sum(data[:,2:], axis=1) + 1e-20
        self.total_flux = np.reshape(total_flux, (np.size(x_axis), np.size(y_axis)))
        self.total_flux = np.flip(self.total_flux, axis=0)
        self.sindec_axis = np.flip(-1*x_axis)
        self.e_true_axis = y_axis     

        self.log10Enu_min = 2.0
        
        self.bbox = [np.min(x_axis), np.max(x_axis), self.log10Enu_min, np.log10(np.max(y_axis))]
        
        self.flux_spline = interpolate.RectBivariateSpline(self.sindec_axis, np.log10(self.e_true_axis), np.log10(self.total_flux), kx=1, ky=1)

    def dflux_dE(self,x,y):
        #Wrapper around flux interpolation
        #return self.flux_interp((x,y))
        return 10**self.flux_spline(x,y)
    def dflux_dlog10E(self,x,y):
        #Wrapper around flux interpolation
        #return self.flux_interp((x,y))
        return self.dflux_dE(x,y)*10**y*np.log(10)
        
#Class that contains diffuse astrophysical neutrino flux function
class astro_bkg():
    def __init__(self, Phi0, gamma):
        self.Phi0 = Phi0
        self.gamma = gamma
        self.e_true_axis = [10**2,10**9]
        self.sindec_axis = [-1.,1.]
    def flux(self,x,y):
        return self.Phi0*(10**y/10**5)**-self.gamma 


#Class that combines the atmospheric and astrophysical fluxes
class background():
    def __init__(self, filename):
        self.Phi0 = 1.44*10**-18
        self.gamma = 2.28
        self.atmdata = atmData(filename)        

        self.sindec_min = self.atmdata.bbox[0]
        self.sindec_max = self.atmdata.bbox[1]
        self.log10Enu_min = self.atmdata.bbox[2]
        self.log10Enu_max = self.atmdata.bbox[3]

    def dflux_dlog10E(self,x,y):

        astro_flux = np.log(10)*10**y*self.Phi0*(10**y/10**5)**-self.gamma

        outside_box = (y < self.log10Enu_min or y > self.log10Enu_max) or (np.sin(np.radians(x)) < self.sindec_min or np.sin(np.radians(x)) > self.sindec_max)

        if(outside_box == False):

            atmo_flux = self.atmdata.dflux_dlog10E(x,y) 
        else:
            atmo_flux = 0.0

        return astro_flux + atmo_flux

    def get_true_pdf(self, effA):
        
        events = convolve.binned(self.dflux_dlog10E, effA, 1)
        prob = events/np.sum(events)
        
        log10Enu_bw = np.diff(effA.l10E_bin_edges)
        decnu_bw = np.diff(effA.dec_bin_edges)

        volumes = log10Enu_bw[:,new_axis]*decnu_bw[new_axis,:]#Calculate bin volumes to then get pdf

        pdf = prob/volumes

        spline = interpolate.RectBivariateSpline
