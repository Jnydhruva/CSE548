#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC   =
SOURCECU  = mpisellcuda.cu
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/impls/sell/mpi/mpicuda/
MANSEC    = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
