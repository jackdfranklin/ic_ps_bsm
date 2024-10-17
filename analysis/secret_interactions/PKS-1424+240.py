from multiprocessing import Pool
import time

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import pandas as pd

from skyllh.core.config import Config

cfg = Config()

from skyllh.datasets.i3.PublicData_10y_ps import create_dataset_collection

dsc = create_dataset_collection(cfg=cfg, base_path="/home/jfranklin/Documents/ic_ps_bsm/datasets/")
datasets = dsc.get_datasets(['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II-VII'])
#datasets = dsc.get_datasets(['IC86_II-VII'])

from analysis import (
        create_analysis,
        create_si_z_analysis
)

from skyllh.core.source_model import PointLikeSource
from skyllh.core.flux_model import (
    PowerLawEnergyFluxProfile,
)

alpha_PKS = 216.76
dec_PKS = 23.8
z_PKS = 0.6

source = PointLikeSource(ra=np.deg2rad(alpha_PKS), dec=np.deg2rad(dec_PKS))

pl_profile = PowerLawEnergyFluxProfile(
                            E0=1e3,
                            gamma=2.0,
                            cfg=cfg,
                            )

pl_analysis = create_analysis(cfg=cfg, datasets=datasets, source=source, energy_profile=pl_profile)

events_list = [data.exp for data in pl_analysis.data_list]
pl_analysis.initialize_trial(events_list)

from skyllh.core.random import RandomStateService

rss = RandomStateService(seed=1)

(pl_log_lambda_max, pl_fitparam_values, pl_status) = pl_analysis.llhratio.maximize(rss)

print(f'PL log_lambda_max = {pl_log_lambda_max}')
print(f'PL fitparam_values = {pl_fitparam_values}')
#print(f'status = {status}')

#Assume NO and try to saturate the cosmological bound mtot = 0.15 eV
Delta21_eV = 7.42e-5
Delta31_eV = 2.514e-3

m1_eV =  0.0218369 #0.33204 
m1_GeV = m1_eV*1e-9
m2_GeV = np.sqrt(m1_eV**2 + Delta21_eV)*1e-9
m3_GeV = np.sqrt(m1_eV**2 + Delta31_eV)*1e-9

neutrino_masses_GeV = [m1_GeV, m2_GeV, m3_GeV]

relic_density_cm_3 = 56

m_phi_GeV_vals = np.logspace(-1, 2, 20)*1e-3
g_vals = np.logspace(-2, np.log10(0.9), 20)

M, G = np.meshgrid(m_phi_GeV_vals, g_vals)

M = M.flatten()
G = G.flatten()

def model_logllh(index):

    m_phi_GeV = M[index]
    g = G[index]

    #print("g = "+str(g))
    #print("m_phi = "+str(m_phi_GeV*1e3)+" MeV")

    si_analysis = create_si_z_analysis(cfg=cfg, datasets=datasets, source=source,
                                       E0 = 1e3,
                                       z = z_PKS,
                                       g = g,
                                       m_phi_GeV = m_phi_GeV,
                                       neutrino_masses_GeV = neutrino_masses_GeV,
                                       relic_density_cm_3 = relic_density_cm_3 
                                       )

    new_events_list = [data.exp for data in si_analysis.data_list]
    si_analysis.initialize_trial(new_events_list)

    new_rss = RandomStateService(seed=1)

    (log_lambda_max, fitparam_values, status) = si_analysis.llhratio.maximize(new_rss)
    #print(fitparam_values)
    return log_lambda_max

si_logllh = []

start = time.perf_counter()

with Pool(10) as p:
    si_logllh = p.map(model_logllh, range(M.size))

end = time.perf_counter()
print("Time taken = " + str(end - start) + " s")

TS = -2 * (np.array(si_logllh) - pl_log_lambda_max)
#print(TS)

result_df = pd.DataFrame(data={"M_phi": M, "g":G, "-2 logllh": TS})
result_df.to_csv("/tmp/PKS_logllh.csv")
