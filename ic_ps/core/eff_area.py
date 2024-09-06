import pandas as pd
import numpy as np

#Class to contain effective area data in panda DataFrame
class effective_area:
    
    def __init__(self, input_file, skiprows=1):

        headers = ['log10Enu_min', 'log10Enu_max', 'dec_min', 'dec_max', 'Aeff']
        effA = pd.read_table(input_file, sep="\s+", header=None, skiprows=skiprows, names=headers, index_col=[0,1,2,3])
        self.effA = effA.loc[effA['Aeff']>0]
        self.mindex = self.effA.index
        self.log10Enu_min = self.mindex.get_level_values('log10Enu_min').copy()
        self.log10Enu_max = self.mindex.get_level_values('log10Enu_max').copy()
        self.dec_min = self.mindex.get_level_values('dec_min').copy()
        self.dec_max = self.mindex.get_level_values('dec_max').copy()
        
        self.log10Enu_bin_edges = np.unique(np.array(self.log10Enu_min))
        self.log10Enu_bin_edges = np.append(self.log10Enu_bin_edges, np.max(self.log10Enu_max))
        self.dec_bin_edges = np.unique(np.array(self.dec_min))
        self.dec_bin_edges = np.append(self.dec_bin_edges, np.max(self.dec_max))
        
        self.log10Enu_points = (self.log10Enu_bin_edges[1:]+self.log10Enu_bin_edges[:-1])/2
        self.dec_points = (self.dec_bin_edges[1:]+self.dec_bin_edges[:-1])/2

    def point_source(self, dec):
        #Find effective area for given energy bin with specific declination (i.e. point-like source)
    
        dec_idx = np.searchsorted(self.dec_bin_edges, dec)-1
        dec_min = self.dec_bin_edges[dec_idx]
        dec_max = self.dec_bin_edges[dec_idx+1]
    
        return self.effA.loc[(slice(None), slice(None), dec_min, dec_max)]["Aeff"]
