eigen_dep = dependency('eigen3', version : '>=3.2.7')

py.extension_module('transport_solver',
    'transport_solver.cpp',
    install: true,
    subdir: 'ic_ps/transport_solver',
    dependencies : [pybind11_dep, eigen_dep],
)

py.install_sources(
  ['relic_scattering.py'],
  subdir: 'ic_ps/transport_solver',
)
