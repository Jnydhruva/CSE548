#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

opts = [
  'LDFLAGS=-L/opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/lib -lquadmath',
   '--with-mpi-dir=/scratch/soft/mpich',
   '--download-fblaslapack',
   '--download-hypre',
   '--with-hipc=/opt/rocm/bin/hipcc',
   '--with-hip-dir=/opt/rocm',
]

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = opts + [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--download-cmake',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    'HIPOPTFLAGS=-g -O',
    '--with-cuda=0',
    '--with-hip=1',
    '--with-precision=double',
    '--with-clanguage=c',
    '--download-kokkos',
    '--download-kokkos-kernels',
    '--download-hypre-configure-arguments=--enable-unified-memory',
    '--download-magma',
    '--with-magma-fortran-bindings=0',
  ]

  configure.petsc_configure(configure_options)
