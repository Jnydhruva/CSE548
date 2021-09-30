#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-cc=mpicc',
  '--with-fc=mpif90',
  '--with-cxx=mpicxx',

  '--download-cmake=1',
  '--download-sprng=1',
  '--download-random123=1',
  '--download-saws=1',
  '--download-yaml=1',
  '--download-scalapack=1',
  '--download-strumpack=1',
  '--download-mumps=1',
  '--download-hypre=1',
  '--download-ctetgen=1',
  '--download-triangle=1',
  '--download-p4est=1',
  '--download-ml=1',
  '--download-hpddm=1',
  '--download-spai=1',
  '--download-radau5=1',
  #'--download-sundials2=1',

  '--download-make=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-superlu_dist=1',
  '--download-hdf5=1',
  '--download-netcdf=1',
  '--download-pnetcdf=1',
  '--download-zlib=1',
  '--download-exodusii=1',
  '--download-kokkos=1',
  '--download-kokkos-kernels=1',
  '--download-tchem=1',
  '--download-revolve=1',
  '--download-pragmatic=1',
  '--download-parmmg=1',
  '--download-parms=1',
  '--download-muparser=1',
  '--download-mstk=1',
  '--download-moose=1',
  '--download-mmg=1',
  '--download-med=1',
  '--download-eigen=1',
  '--download-ptscotch=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
