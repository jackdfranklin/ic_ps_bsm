eigen_dep = dependency('eigen3', version : '>=3.2.7')

py.extension_module('transport_solver',
    ['transport_solver.cpp',
    'utils.cpp',
    'constants.hpp',
    'secret_interactions.cpp',
    'sm_interactions.cpp',
    ],
    install: true,
    subdir: 'ic_ps/bsm',
    dependencies : [pybind11_dep, eigen_dep],
)
