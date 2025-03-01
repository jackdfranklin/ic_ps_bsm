import argparse
import os
from multiprocessing import Pool
import time
from copy import deepcopy
from functools import partial

import numpy as np
import scipy.stats
import pandas as pd
import h5py

from skyllh.core.random import RandomStateService
from skyllh.core.config import Config
from skyllh.core.flux_model import PowerLawEnergyFluxProfile
from skyllh.core.source_model import PointLikeSource
from skyllh.datasets.i3.PublicData_10y_ps import create_dataset_collection

from ic_ps.skyllh.analyses import (
        create_analysis,
        create_sm_analysis,
        create_sm_z_analysis,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line \
                                                    arguments.')
    parser.add_argument('-d', '--dataset_dir', type=dir_path, 
                        default=os.path.abspath("../../datasets/"),
                        help="Directory of public datasets")
    parser.add_argument("-o", "--out_dir", type=dir_path, 
                        default=os.path.abspath("results"), 
                        help="Directory to write output files to")
    parser.add_argument("--n_cpus", type=int, default=1, 
                        help="Number of cpu threads to use")
    parser.add_argument("--seed", type=int, default=1, 
                        help="Seed used to initialise skyllh \
                                         RandomStateService")
    parser.add_argument("--n_iter", type=int, default=1000, 
                        help="Number of iterations of the future \
                                        analysis to perform")
    parser.add_argument("--n_gen", type=int, default=2, 
                        help="Number of 5 year datasets to generate\
                                        for future analysis")

    return parser.parse_args()


def dir_path(path):
    if os.path.isdir(path):
        return os.path.abspath(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} \
                                           is not a valid path")

def add_datasets(ana, data_list):

    pseudo_ds = ana.dataset_list[-1]
    pseudo_pdfratio = ana._pdfratio_list[-1]
    pseudo_tdm = ana.tdm_list[-1]
    pseudo_es = ana._event_selection_method_list[-1]
    pseudo_bkg_gen = ana.bkg_generator_list[-1]
    pseudo_sig_gen = ana.sig_generator_list[-1]

    for data in data_list:
        ana.add_dataset(
            dataset=pseudo_ds,
            data=data,
            pdfratio=pseudo_pdfratio,
            tdm=pseudo_tdm,
            event_selection_method=pseudo_es,
            bkg_generator=pseudo_bkg_gen,
            sig_generator=pseudo_sig_gen)

    ana.construct_services()
    ana.llhratio = ana.construct_llhratio(minimizer=ana.llhratio.minimizer)

def gen_n_new_datasets(ana, n_s, n, rss):

    data_list = []

    for i in range(n):

        data = ana.generate_pseudo_data(rss=rss, mean_n_sig=n_s)[2][-1]
        pseudo_dataset_data = deepcopy(ana.data_list[-1])
        pseudo_dataset_data.exp = data
        data_list.append(pseudo_dataset_data)

    return data_list


def perform_pl_analysis(source, dataset_collection, args, cfg):

    pl_profile = PowerLawEnergyFluxProfile(E0=1e3,
                                           gamma=2.0,
                                           cfg=cfg,
                                           )

    pl_analysis = create_analysis(cfg=cfg, 
                                  datasets=dataset_collection, 
                                  source=source, 
                                  energy_profile=pl_profile)

    events_list = [data.exp for data in pl_analysis.data_list]
    pl_analysis.initialize_trial(events_list)

    rss = RandomStateService(seed=args.seed)

    (pl_TS, pl_fitparam_values, pl_status) = pl_analysis.unblind(rss)

    pl_profile.gamma = pl_fitparam_values['gamma']

    print(f'PL TS = {pl_TS}')
    print(f'PL fitparam_values = {pl_fitparam_values}')

    pl_analysis = create_analysis(cfg=cfg, 
                                  datasets=dataset_collection, 
                                  source=source, 
                                  energy_profile=pl_profile)

    n_s = pl_fitparam_values['ns']

    new_rss = RandomStateService(seed=args.seed)

    ts_vals = []
    gamma_vals = []
    ns_vals = []

    return (pl_analysis, pl_fitparam_values, pl_TS)

if __name__ == "__main__":

    args = parse_arguments()

    cfg = Config()

    dsc = create_dataset_collection(cfg=cfg, base_path=args.dataset_dir)
    datasets = dsc.get_datasets(['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II-VII'])
     
    #Assume NO and try to saturate the cosmological bound mtot = 0.15 eV
    Delta21_eV = 7.42e-5
    Delta31_eV = 2.514e-3

    relic_density_cm_3 = 56

    m1_vals = [0.001, 0.034] #, 0.8]

    r_md = (19, 20, 10)
    relic_dens_vals = 56*np.logspace(r_md[0], r_md[1], r_md[2])

    M, R = np.meshgrid(m1_vals, relic_dens_vals)

    M = M.flatten()
    R = R.flatten()

    filename = args.out_dir+'/SM_analysis_full_'+str(r_md[0])+'_'+str(r_md[1])+'.h5'

    # Store metadata for generation of results
    with pd.HDFStore(filename) as hdf_store:

        metadata = pd.Series(data = {'N_m': 3,
                                     'l10r_min': r_md[0],
                                     'l10r_max': r_md[1],
                                     'N_r': r_md[2],
                                     'seed': args.seed}) 
        hdf_store.put('metadata', metadata)

############################### NGC ################################
        alpha_NGC = 40.667
        dec_NGC = -0.0069
        NGCsource = PointLikeSource(ra=np.deg2rad(alpha_NGC), dec=np.deg2rad(dec_NGC))
        distance_NGC = 14 #Mpc
        source_name = "NGC"

        pl_analysis, pl_fitparam_values, pl_ts_current = perform_pl_analysis(NGCsource, 
                                                                    datasets, 
                                                                    args, 
                                                                    cfg)

        n_s = pl_fitparam_values['ns']

        def model_NGC_TS(index):

            m1_eV = M[index]
            relic_density_cm_3 = R[index]

            m1_GeV = m1_eV*1e-9
            m2_GeV = np.sqrt(m1_eV**2 + Delta21_eV)*1e-9
            m3_GeV = np.sqrt(m1_eV**2 + Delta31_eV)*1e-9

            neutrino_masses_GeV = [m1_GeV, m2_GeV, m3_GeV]

            sm_analysis = create_sm_analysis(cfg=cfg, datasets=datasets, source=NGCsource,
                                               distance_Mpc = distance_NGC,
                                               neutrino_masses_GeV = neutrino_masses_GeV,
                                               relic_density_cm_3 = relic_density_cm_3,
                                               steps = 20,
                                               )

            new_events_list = [data.exp for data in sm_analysis.data_list]
            sm_analysis.initialize_trial(new_events_list)

            new_rss = RandomStateService(seed=args.seed)

            (SM_TS, SM_fitparam_values, SM_status) = sm_analysis.unblind(new_rss)

            return SM_TS

        start = time.perf_counter()

        with Pool(args.n_cpus) as p:

            sm_TS_list = p.map(model_NGC_TS, range(M.size))

        end = time.perf_counter()
        print("Time taken = " + str(end - start) + " s")

        result_df = pd.DataFrame(data={"m_nu": M, 
                                       "relic_dens":R, 
                                       "-2 logllh":(pl_ts_current
                                                    - np.array(sm_TS_list))})
        hdf_store.put(source_name, result_df)
