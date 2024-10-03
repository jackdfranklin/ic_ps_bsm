import sys
import os
import numpy as np
import pandas as pd
import pickle

import ic_ps.core.eff_area as eff_area
import ic_ps.core.convolve as convolve
import ic_ps.core.smear_mat as smear_mat
import ic_ps.core.pdf as pdf
import ic_ps.core.flux as flux
import ic_ps.core.likelihood as llh 

parameters = ['ns', 'gamma']

pl_flux = flux.Flux()

llh_pl = llh.Likelihood(77.3582, 5.69314, "../../datasets/icecube_10year_ps", "TXS_bkg.pkl", pl_flux)

result, fval = llh_pl.fit()

events_df = llh_pl.events_by_logl(result[0], result[1])

print(result, fval)
