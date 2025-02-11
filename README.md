# IC_PS_BSM

Using point sources at IceCube to constrain neutrino physics beyond the standard model.

## Overview

The project is split into two main components, the ic_ps library and analyses code in analysis. The
library provides analysis builders for SkyLLH analyses objects, which can be found in ic_ps/skyllh/
analysis.py. This directory also contains some other SkyLLH files, with modifications to adapt the 
analyis to the new physics cases. 

The C++ code in ic_ps/bsm solves transport equations for neutrino fluxes originating from point-like
astrophysical sources e.g. AGNs. It is possible to solve over distance or redshift, depending on the
distance of the source to Earth, and for either Standard Model or scalar-mediated interactions.

## Requirements

- numpy
- skyllh
- pybind11
- mesonpy

## Installation

The ic_ps library can be installed using pip by running the following command in the root directory:

```
python -m pip install .
```

It is recommended to do this in a python or conda virtual environment to avoid possible dependecy clashes. 

## Use

The analysis directory contains all of the python scripts (and some extras) used to produce results for 
[this paper](https://arxiv.org/abs/2404.02202). I have no intention of developing this repository into
a fully fledged library, and only host it publicly to allow for scrutiny of our methodology.
