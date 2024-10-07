import pandas as pd
import numpy as np

events_dict = {
    "IC86_II-VII":["IC86_II_exp.csv","IC86_III_exp.csv","IC86_IV_exp.csv","IC86_V_exp.csv","IC86_VI_exp.csv","IC86_VII_exp.csv"],
    "new":["event_list.txt"]
}

class DataSet:

    def __init__(self, location, dataset_name):

        self.dataset_name = dataset_name

        self.location = location
        
        self.events_files = events_dict[dataset_name]


    def select_and_prepare_data(self, src_dec, src_ra):

        events_log10Ereco = [] 
        events_dec = []
        events_ra = []
        events_psi = []
        events_ang_err = []

        self.N = 0
        self.N_tot = 0

        for file in self.events_files:

            events = np.loadtxt(self.location+"/events/"+file, skiprows=1)

            temp_events_log10Ereco = events[:,1]
            temp_events_ang_err = events[:,2]
            temp_events_ra = events[:,3]
            temp_events_dec = events[:,4]

            temp_events_psi = angular_distance(np.radians(temp_events_ra), np.radians(temp_events_dec), np.radians(src_ra), np.radians(src_dec))
            temp_events_psi = np.degrees(temp_events_psi)

            events_log10Ereco = np.append(events_log10Ereco, temp_events_log10Ereco)
            events_dec = np.append(events_dec, temp_events_dec)
            events_ra = np.append(events_ra, temp_events_ra)
            events_psi = np.append(events_psi, temp_events_psi)
            events_ang_err = np.append(events_ang_err, temp_events_ang_err)
            
            self.N_tot += events.shape[0]


        psi_cut_mask = events_psi <= 15 #Choose only events within 15 degrees of source

        self.events_log10Ereco = events_log10Ereco[psi_cut_mask]
        self.events_dec = events_dec[psi_cut_mask]
        self.events_ra = events_ra[psi_cut_mask]
        self.events_psi = events_psi[psi_cut_mask]
        self.events_ang_err = events_ang_err[psi_cut_mask]
        self.N = self.events_log10Ereco.shape[0]
        #print(self.N_tot, self.N)

        self.total_runtime = self.get_runtime()
        #print("Total runtime = "+str(self.total_runtime))

    def get_runtime(self):
        total_runtime = 0.0
 
        if self.dataset_name == "IC86_II-VII":
            for file in self.events_files:
                mjd = np.loadtxt(self.location+"/uptime/"+file, skiprows=1)
                runtime = np.diff(mjd, axis=1)
                total_runtime += np.sum(runtime)
        elif self.dataset_name == "new":
            total_runtime = 3186.105475562

        return total_runtime*(24*3600)

def angular_distance(ra_1, dec_1, ra_2, dec_2):
    #Compute the great circle distance between two events (all units must be radians)
    delta_dec = np.abs(dec_1-dec_2)
    delta_ra = np.abs(ra_1-ra_2)
    x = np.sin(delta_dec/2)**2 + np.cos(dec_1)*np.cos(dec_2)*np.sin(delta_ra / 2)**2
    return 2*np.arcsin(np.sqrt(x))
