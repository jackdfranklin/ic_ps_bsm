import pandas as pd
import numpy as np

import pdf

#Class that contains smear matrix probabilities as a pandas DataFrame
class smear_matrix:
    
    def __init__(self, input_file, skiprows=1):
        
        headers = ['log10Enu_min', 'log10Enu_max', 'dec_min', 'dec_max', 'log10E_min', 'log10E_max', 'psi_min', 'psi_max', 'ang_err_min', 'ang_err_max', 'frac_counts']
        self.smear_table = pd.read_table(input_file, sep="\s+", header=None, skiprows=skiprows, names=headers)
        self.log10Enu_min = self.smear_table['log10Enu_min'].to_numpy()
        self.log10Enu_max = self.smear_table['log10Enu_max'].to_numpy()
        self.dec_min = self.smear_table['dec_min'].to_numpy()
        self.dec_max = self.smear_table['dec_max'].to_numpy()
        
        #Get true energy bin edges
        self.log10Enu_bin_edges = np.union1d(self.log10Enu_min, self.log10Enu_max)
        self.n_Enu_bins = len(self.log10Enu_bin_edges)-1
        #Get true declination bin edges
        self.dec_true_bin_edges = np.union1d(self.dec_min, self.dec_max)
        self.n_dec_true_bins = len(self.dec_true_bin_edges)-1

        def get_nbins(lower_edges, upper_edges):
            #Helper function to calculate number of bins of a certain value
    
            #Select only valid bin edges
            mask = upper_edges-lower_edges > 0
            
            #Calculate number of times lower bin edge changes value
            steps = np.diff(lower_edges[mask])
            #Find where first binning stops
            end_bin = np.argwhere(steps<0).item(0)
            steps = steps[:end_bin]
            n_steps = np.sum(np.where(steps>0, 1, 0))
            n_bins = 1 + n_steps #Number of bins = number of changes in bin edge value + 1 for original value 
            
            return n_bins

        self.n_Ereco_bins = get_nbins(self.smear_table['log10E_min'], self.smear_table['log10E_max'])
        self.n_psi_bins = get_nbins(self.smear_table['psi_min'], self.smear_table['psi_max'])
        self.n_ang_err_bins = get_nbins(self.smear_table['ang_err_min'], self.smear_table['ang_err_max'])

        #Find mask of unique reco energy
        Ereco_mask = (np.arange(len(self.smear_table)) % (self.n_psi_bins * self.n_ang_err_bins)) == 0
        log10E_min = self.smear_table.loc[Ereco_mask]['log10E_min'].to_numpy() 
        log10E_max = self.smear_table.loc[Ereco_mask]['log10E_max'].to_numpy() 

        #Get reco energy bins as 3d numpy array
        self.log10E_min = np.reshape(log10E_min, (self.n_Enu_bins, self.n_dec_true_bins, self.n_Ereco_bins))
        self.log10E_max = np.reshape(log10E_max, (self.n_Enu_bins, self.n_dec_true_bins, self.n_Ereco_bins))

        s = np.array(self.log10E_min.shape)
        s[-1]+=1
        self.log10E_bin_edges = np.empty(s, dtype=np.double)
        self.log10E_bin_edges[..., :-1] = self.log10E_min
        self.log10E_bin_edges[..., -1] = self.log10E_max[...,-1] 

        psi_mask = (np.arange(len(self.smear_table)) % self.n_ang_err_bins) ==0

        psi_min = self.smear_table.loc[psi_mask]['psi_min'].to_numpy()
        psi_max = self.smear_table.loc[psi_mask]['psi_max'].to_numpy()

        #Get psi bins as 4d numpy array
        self.psi_min = np.reshape(psi_min, (self.n_Enu_bins, self.n_dec_true_bins, self.n_Ereco_bins, self.n_psi_bins))
        self.psi_max = np.reshape(psi_max, (self.n_Enu_bins, self.n_dec_true_bins, self.n_Ereco_bins, self.n_psi_bins))

        #Get psi bin edges
        s = np.array(self.psi_min.shape)
        s[-1]+=1
        self.psi_bin_edges = np.empty(s, dtype=np.double)
        self.psi_bin_edges[..., :-1] = self.psi_min
        self.psi_bin_edges[..., -1] = self.psi_max[...,-1] 

        ang_err_min = self.smear_table['ang_err_min'].to_numpy()
        ang_err_max = self.smear_table['ang_err_max'].to_numpy()

        #Get angular error bins as 5d numpy array
        self.ang_err_min = np.reshape(ang_err_min, (self.n_Enu_bins, self.n_dec_true_bins, self.n_Ereco_bins, self.n_psi_bins, self.n_ang_err_bins))
        self.ang_err_max = np.reshape(ang_err_max, (self.n_Enu_bins, self.n_dec_true_bins, self.n_Ereco_bins, self.n_psi_bins, self.n_ang_err_bins))

        s = np.array(self.ang_err_min.shape)
        s[-1]+=1
        self.ang_err_bin_edges = np.empty(s, dtype=np.double)
        self.ang_err_bin_edges[..., :-1] = self.ang_err_min
        self.ang_err_bin_edges[..., -1] = self.ang_err_max[...,-1] 
        


        #Get pdf values as 5d numpy array
        self.frac_counts = np.reshape(self.smear_table['frac_counts'].to_numpy(), (self.n_Enu_bins, self.n_dec_true_bins, self.n_Ereco_bins, self.n_psi_bins, self.n_ang_err_bins))
        
        self.log10E_bw = self.log10E_max - self.log10E_min

        self.psi_bw = self.psi_max - self.psi_min

        self.ang_err_bw = self.ang_err_max- self.ang_err_min

        bin_vols = (self.log10E_bw[:,:,:,np.newaxis,np.newaxis]*self.psi_bw[:,:,:,:,np.newaxis]*self.ang_err_bw[:,:,:,:,:])

        self.pdf_vals = np.copy(self.frac_counts)
        pdf_mask = self.pdf_vals != 0

        self.pdf_vals[pdf_mask] /= bin_vols[pdf_mask]


    def point_like(self, dec, log10_Enu):
    #Find smearing matrix for given energy bin with specific declination (i.e. point-like source)
    
        ind_E = np.searchsorted(self.log10Enu_bin_edges, log10_Enu)
        ind_dec = np.searchsorted(self.dec_true_bin_edges, dec)
            
        return self.smear_table.loc[(self.log10Enu_min==self.log10Enu_bin_edges[ind_E-1])&(self.dec_min==self.dec_true_bin_edges[ind_dec-1])].copy()

    def get_energy_pdf(self, src_dec):

        dec_idx = np.searchsorted(self.dec_true_bin_edges, src_dec)-1

        f_counts = self.frac_counts.copy()[:,dec_idx,:,:,:]
        pdf_e = np.sum(f_counts, axis=(-2,-1))/self.log10E_bw[:,dec_idx,:]  
 
        energypdf = pdf.EnergyPDF(self.log10Enu_bin_edges, self.log10E_bin_edges[:,dec_idx,:], pdf_e)

        return energypdf
