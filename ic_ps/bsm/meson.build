eigen_dep = dependency('eigen3', version : '>=3.2.7')

py.extension_module('transport_solver',
    ['transport_solver.cpp',
    'temp.cpp',
    ],
    install: true,
    subdir: 'ic_ps/bsm',
    dependencies : [pybind11_dep, eigen_dep],
)

python_sources = [
  '__init__.py',
]

py.install_sources(
  python_sources,
  subdir: 'ic_ps/bsm'
)
