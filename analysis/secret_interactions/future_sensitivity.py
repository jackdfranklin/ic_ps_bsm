import argparse
import os
from multiprocessing import Pool
import time
from copy import deepcopy
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
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
        create_si_analysis,
        create_si_z_analysis,
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

    n_gen = pl_fitparam_values['ns']

    ts_vals = []
    gamma_vals = []
    ns_vals = []

    for i in range(args.n_iter):

        future_analysis = deepcopy(pl_analysis)
        data_list = gen_n_new_datasets(future_analysis, n_gen, 2, rss)
        add_datasets(future_analysis, data_list)

        (NGC_TS, NGC_fitparam_values, NGC_status) = future_analysis.unblind(rss)

        ts_vals.append(NGC_TS)
        ns_vals.append(NGC_fitparam_values['ns'])
        gamma_vals.append(NGC_fitparam_values['gamma'])

    ts_exp = np.mean(ts_vals)

    print(np.mean(gamma_vals))

    return (pl_analysis, pl_fitparam_values, pl_TS, ts_exp)

def create_si_analysis_wrapper(
        g,
        m_phi_GeV,
        cfg,
        datasets,
        source,
        distance_Mpc,
        neutrino_masses_GeV,
        relic_density_cm_3, 
        steps):

    return create_si_analysis(cfg= cfg,
                              datasets= datasets,
                              source= source,
                              distance_Mpc= distance_Mpc,
                              g= g,
                              m_phi_GeV= m_phi_GeV,
                              neutrino_masses_GeV= neutrino_masses_GeV,
                              relic_density_cm_3= relic_density_cm_3, 
                              steps= steps)

if __name__ == "__main__":

    args = parse_arguments()

    cfg = Config()

    dsc = create_dataset_collection(cfg=cfg, base_path=args.dataset_dir)
    datasets = dsc.get_datasets(['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II-VII'])
     
    #Assume NO and try to saturate the cosmological bound mtot = 0.15 eV
    Delta21_eV = 7.42e-5
    Delta31_eV = 2.514e-3

    m1_eV =  0.0218369 #0.33204 
    m1_GeV = m1_eV*1e-9
    m2_GeV = np.sqrt(m1_eV**2 + Delta21_eV)*1e-9
    m3_GeV = np.sqrt(m1_eV**2 + Delta31_eV)*1e-9

    neutrino_masses_GeV = [m1_GeV, m2_GeV, m3_GeV]

    relic_density_cm_3 = 56

    m_md = (-1, 2, 20)
    m_phi_GeV_vals = np.logspace(m_md[0],m_md[1],m_md[2])*1e-3

    g_md = (-3, 1, 20)
    g_vals = np.logspace(g_md[0], g_md[1], g_md[2])

    M, G = np.meshgrid(m_phi_GeV_vals, g_vals)

    M = M.flatten()
    G = G.flatten()

    alpha_NGC = 40.667
    dec_NGC = -0.0069
    NGCsource = PointLikeSource(ra=np.deg2rad(alpha_NGC), dec=np.deg2rad(dec_NGC))
    distance_NGC = 14 #Mpc

    NGC_partial = partial(create_si_analysis, cfg = cfg, 
                          datasets=datasets, source=NGCsource, 
                          distance_Mpc = distance_NGC, 
                          neutrino_masses_GeV = neutrino_masses_GeV, 
                          relic_density_cm_3 = relic_density_cm_3, steps=20)

    alpha_PKS = 216.76
    dec_PKS = 23.8
    PKSsource = PointLikeSource(ra=np.deg2rad(alpha_PKS), dec=np.deg2rad(dec_PKS))
    z_PKS = 0.6

    PKS_partial = partial(create_si_z_analysis, cfg = cfg, 
                          datasets=datasets, source=PKSsource, z = z_PKS,
                          neutrino_masses_GeV = neutrino_masses_GeV, 
                          relic_density_cm_3 = relic_density_cm_3)

    alpha_TXS = 77.3582
    dec_TXS = 5.69314
    z_TXS = 0.45

    TXSsource = PointLikeSource(ra=np.deg2rad(alpha_TXS), dec=np.deg2rad(dec_TXS))

    TXS_partial = partial(create_si_z_analysis, cfg = cfg, 
                          datasets=datasets, source=TXSsource, z = z_TXS,
                          neutrino_masses_GeV = neutrino_masses_GeV, 
                          relic_density_cm_3 = relic_density_cm_3)

    def model_TS(index, n_s):

        m_phi_GeV = M[index]
        g = G[index]

        #print("g = "+str(g))
        #print("m_phi = "+str(m_phi_GeV*1e3)+" MeV")

        si_analysis = create_si_analysis(cfg=cfg, datasets=datasets, source=NGCsource,
                                           distance_Mpc = distance_NGC,
                                           g = g,
                                           m_phi_GeV = m_phi_GeV,
                                           neutrino_masses_GeV = neutrino_masses_GeV,
                                           relic_density_cm_3 = relic_density_cm_3,
                                           steps = 20,
                                           )

        new_events_list = [data.exp for data in si_analysis.data_list]
        si_analysis.initialize_trial(new_events_list)

        new_rss = RandomStateService(seed=args.seed)

        ts_vals = []
        gamma_vals = []
        ns_vals = []

        for i in range(args.n_iter):

            future_analysis = deepcopy(si_analysis)
            data_list = gen_n_new_datasets(pl_analysis, n_s, 2, new_rss)
            add_datasets(future_analysis, data_list)

            (SI_TS, SI_fitparam_values, SI_status) = future_analysis.unblind(new_rss)

            ts_vals.append(SI_TS)
            ns_vals.append(SI_fitparam_values['ns'])
            gamma_vals.append(SI_fitparam_values['gamma'])

        ts_exp = np.mean(ts_vals)

        return (SI_TS, ts_exp)

    with pd.HDFStore('SI_analysis.h5') as hdf_store:

        metadata = pd.Series(data = {'l10m_min': m_md[0],
                                     'l10m_max': m_md[1],
                                     'N_m': m_md[2],
                                     'l10g_min': g_md[0],
                                     'l10g_max': g_md[1],
                                     'N_g': g_md[2],
                                     'seed': args.seed}) 
        hdf_store.put('metadata', metadata)

        #perform_SI_analysis(NGCsource, 
        #                    "NGC", 
        #                    datasets, 
        #                    distance_NGC,
        #                    M, 
        #                    G,
        #                    neutrino_masses_GeV,
        #                    relic_density_cm_3,
        #                    hdf_store, 
        #                    args, 
        #                    cfg)

#def perform_SI_analysis(
#        source, 
#        source_name, 
#        datasets,
#        distance_Mpc,
#        M, 
#        G, 
#        neutrino_masses_GeV,
#        relic_density_cm_3,
#        hdf_store, 
#        args,
#        cfg):


        source = NGCsource
        source_name = "NGC"

        pl_analysis, pl_fitparam_values, pl_ts_current, pl_ts_exp = perform_pl_analysis(NGCsource, 
                                                                    datasets, 
                                                                    args, 
                                                                    cfg)

        n_s = pl_fitparam_values['ns']

        start = time.perf_counter()

        with Pool(args.n_cpus) as p:

            si_TS_list = p.map(partial(model_TS, 
                                           n_s=n_s), 
                                   range(M.size))

        end = time.perf_counter()
        print("Time taken = " + str(end - start) + " s")

        print(si_TS_list)

        result_df = pd.DataFrame(data={"m_phi": M, 
                                       "g":G, 
                                       "-2 logllh":(pl_ts_current
                                                    - np.array(si_TS_list)[:,0])})
        future_df = pd.DataFrame(data={"m_phi": M, 
                                       "g":G, 
                                       "-2 logllh":(pl_ts_current 
                                                    - np.array(si_TS_list)[:,1])})

        hdf_store.put(source_name, result_df)
        hdf_store.put(source_name+str("_future"), future_df)

        #perform_SI_analysis(PKSsource, "PKS", PKS_partial, M, G, 
        #                    hdf_store, args, cfg)

        #perform_SI_analysis(TXSsource, "TXS", TSX_partial, M, G,
        #                    hdf_store, args, cfg)
