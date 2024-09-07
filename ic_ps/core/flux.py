import numpy as np
import pandas as pd
import pickle
from scipy import interpolate
from scipy import integrate
from scipy.optimize import fmin_l_bfgs_b
from scipy import special

import os
import sys

from functools import partial
import math
import cmath

class Flux:

    def __init__(self):
        self.E0 = 1000

    def dflux_dE(self, E, gamma):
        return (E/self.E0)**-gamma

    def dflux_dlog10E(self, log10E, gamma):    

        E = 10**log10E
        dflux = np.log(10)*E*self.dflux_dE(E, gamma)

        return dflux
