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
                                        for the analysis")

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


    return (pl_analysis, pl_fitparam_values, pl_TS)

if __name__ == "__main__":

    args = parse_arguments()

    cfg = Config()

    dsc = create_dataset_collection(cfg=cfg, base_path=args.dataset_dir)
    datasets = dsc.get_datasets(['IC40', 'IC59', 'IC79', 'IC86_I', 'IC86_II-VII'])
     
    with h5py.File(args.out_dir+'/pl_analysis.h5', 'w') as f:

############################### NGC ################################
        alpha_NGC = 40.667
        dec_NGC = -0.0069
        NGCsource = PointLikeSource(ra=np.deg2rad(alpha_NGC), dec=np.deg2rad(dec_NGC))

        pl_analysis, pl_fitparam_values, pl_ts_current = perform_pl_analysis(NGCsource, 
                                                                             datasets, 
                                                                             args, 
                                                                             cfg)

        n_s = pl_fitparam_values['ns']
        def future_pl(index):
            rss = RandomStateService(seed=index)
            future_analysis = deepcopy(pl_analysis)
            data_list = gen_n_new_datasets(future_analysis, n_s, args.n_gen, rss)
            add_datasets(future_analysis, data_list)

            (NGC_TS, NGC_fitparam_values, NGC_status) = future_analysis.unblind(rss)

            return (NGC_TS, NGC_fitparam_values['ns'], NGC_fitparam_values['gamma']) 

        start = time.perf_counter()

        with Pool(args.n_cpus) as p:

            results = np.array(p.map(future_pl, range(args.n_iter)))

        end = time.perf_counter()
        
        ts_future = results[:,0]
        ns_future = results[:,1]
        gamma_future = results[:,2]

        ts_exp = np.mean(ts_future)
        print(f'80 years PL TS = {ts_exp}')
        print("Time taken = " + str(end - start) + " s")

        NGCgrp = f.create_group("NGC1068")
        ts_dset = NGCgrp.create_dataset("ts future", data=ts_future)
        ns_dset = NGCgrp.create_dataset("ns future", data=ns_future)
        gamma_dset = NGCgrp.create_dataset("gamma future", data=gamma_future)

############################### PKS ################################

        alpha_PKS = 216.76
        dec_PKS = 23.8
        PKSsource = PointLikeSource(ra=np.deg2rad(alpha_PKS), dec=np.deg2rad(dec_PKS))

        pl_analysis, pl_fitparam_values, pl_ts_current = perform_pl_analysis(PKSsource, 
                                                                             datasets, 
                                                                             args, 
                                                                             cfg)

        n_s = pl_fitparam_values['ns']
        def future_pl(index):
            rss = RandomStateService(seed=index)
            future_analysis = deepcopy(pl_analysis)
            data_list = gen_n_new_datasets(future_analysis, n_s, args.n_gen, rss)
            add_datasets(future_analysis, data_list)

            (PKS_TS, PKS_fitparam_values, PKS_status) = future_analysis.unblind(rss)

            return (PKS_TS, PKS_fitparam_values['ns'], PKS_fitparam_values['gamma']) 

        start = time.perf_counter()

        with Pool(args.n_cpus) as p:

            results = np.array(p.map(future_pl, range(args.n_iter)))

        end = time.perf_counter()
        
        ts_future = results[:,0]
        ns_future = results[:,1]
        gamma_future = results[:,2]

        ts_exp = np.mean(ts_future)
        print(f'80 years PL TS = {ts_exp}')
        print("Time taken = " + str(end - start) + " s")

        PKSgrp = f.create_group("PKS1424240")
        ts_dset = PKSgrp.create_dataset("ts future", data=ts_future)
        ns_dset = PKSgrp.create_dataset("ns future", data=ns_future)
        gamma_dset = PKSgrp.create_dataset("gamma future", data=gamma_future)

############################### TXS ################################

        alpha_TXS = 77.3582
        dec_TXS = 5.69314
        TXSsource = PointLikeSource(ra=np.deg2rad(alpha_TXS), dec=np.deg2rad(dec_TXS))

        pl_analysis, pl_fitparam_values, pl_ts_current = perform_pl_analysis(TXSsource, 
                                                                             datasets, 
                                                                             args, 
                                                                             cfg)

        n_s = pl_fitparam_values['ns']
        def future_pl(index):
            rss = RandomStateService(seed=index)
            future_analysis = deepcopy(pl_analysis)
            data_list = gen_n_new_datasets(future_analysis, n_s, args.n_gen, rss)
            add_datasets(future_analysis, data_list)

            (TXS_TS, TXS_fitparam_values, TXS_status) = future_analysis.unblind(rss)

            return (TXS_TS, TXS_fitparam_values['ns'], TXS_fitparam_values['gamma']) 

        start = time.perf_counter()

        with Pool(args.n_cpus) as p:

            results = np.array(p.map(future_pl, range(args.n_iter)))

        end = time.perf_counter()
        
        ts_future = results[:,0]
        ns_future = results[:,1]
        gamma_future = results[:,2]

        ts_exp = np.mean(ts_future)
        print(f'80 years PL TS = {ts_exp}')
        print("Time taken = " + str(end - start) + " s")

        TXSgrp = f.create_group("TXS0506056")
        ts_dset = TXSgrp.create_dataset("ts future", data=ts_future)
        ns_dset = TXSgrp.create_dataset("ns future", data=ns_future)
        gamma_dset = TXSgrp.create_dataset("gamma future", data=gamma_future)
