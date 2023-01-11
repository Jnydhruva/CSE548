-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_CUPM'

DIRS    = cuda hip
SOURCEH = matmpidensecupm.hpp
LIBBASE = libpetscmat
MANSEC  = Mat
LOCDIR  = src/mat/impls/dense/mpi/cupm/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
