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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line \
                                                    arguments.')
    parser.add_argument('-d', '--in_dir', type=dir_path, 
                        default=os.path.abspath("./"),
                        help="Directory of data")
    parser.add_argument("-o", "--out_dir", type=dir_path, 
                        default=os.path.abspath("results"), 
                        help="Directory to write output files to")

    return parser.parse_args()


def dir_path(path):
    if os.path.isdir(path):
        return os.path.abspath(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} \
                                           is not a valid path")

if __name__ == "__main__":

    args = parse_arguments()

    with h5py.File(args.in_dir+'/pl_analysis.h5', 'r') as f:
    
        for src in f.keys():
            ts_dset = f[src+"/ts future"]
            ns_dset = f[src+"/ns future"]
            gamma_dset = f[src+"/gamma future"]

            ts_exp = np.mean(ts_dset[:])
            ts_median = np.median(ts_dset[:])

            print(src)
            print(f'80 years PL mean TS = {ts_exp}')
            print(f'80 years PL median TS = {ts_median}')
