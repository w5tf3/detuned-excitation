from setuptools import find_packages
from numpy.distutils.core import setup, Extension


tls_fortran = Extension(name='two_level_system/tls.f90',
                 sources=['detuned_excitation/two_level_system/tls_.f90'],
                 f2py_options=['--quiet'],
                )

sixls_fortran = Extension(name='two_level_system/sixls.f90',
                 sources=['detuned_excitation/two_level_system/sixls_.f90'],
                 f2py_options=['--quiet'],
                )

biexciton_fortran = Extension(name='two_level_system/biexciton.f90',
                 sources=['detuned_excitation/two_level_system/biexciton_.f90'],
                 f2py_options=['--quiet'],
                )

setup(
      name='detuned_excitation',
      version='0.0.1',
      package_dir={"": "detuned_excitation"},
      packages=find_packages(where="detuned_excitation"),
      ext_modules=[tls_fortran,sixls_fortran,biexciton_fortran],
      install_requires=[
                        'numpy',
                        'matplotlib',
                        'tqdm',
                        'torch',
                        'scipy'
                        ]
      )
