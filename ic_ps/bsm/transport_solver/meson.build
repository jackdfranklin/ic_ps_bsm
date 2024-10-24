eigen_dep = dependency('eigen3', version : '>=3.2.7', fallback: ['eigen', 'eigen_dep'])

gsl_dep = dependency('gsl', version : '>=2.6')

py.extension_module('transport_solver', 
    [
    'transport_solver.cpp',
    'transport_flux_redshift.cpp',
    'utils.cpp',
    'constants.hpp',
    'secret_interactions.cpp',
    'sm_interactions.cpp',
    ],
    install: true,
    subdir: 'ic_ps/bsm',
    dependencies : [pybind11_dep, eigen_dep, gsl_dep],
)
