#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC   =
SOURCECU  = sellcuda.cu
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/impls/sell/seq/cuda/
MANSEC    = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
