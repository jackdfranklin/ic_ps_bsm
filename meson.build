project('IC_PC_BSM', 'cpp', version : '0.0.1', license : 'MIT')

py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')

eigen_dep = dependency('eigen3', version : '>=3.2.7')

py.extension_module('transport_solver',
    'transport_solver/transport_solver.cpp',
    install: false,
    dependencies : [pybind11_dep, eigen_dep],
)
