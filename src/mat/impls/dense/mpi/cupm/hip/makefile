-include ../../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_HIP'

SOURCEHIP = matmpidensehip.hip.cpp
LIBBASE   = libpetscmat
MANSEC    = Mat
LOCDIR    = src/mat/impls/dense/mpi/cupm/hip/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
