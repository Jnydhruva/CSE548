-include ../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_H2OPUS'

ALL: lib

FFLAGS    =
SOURCECXX = pch2opuscpu.cxx
SOURCECU  = pch2opus.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = PC
LOCDIR    = src/ksp/pc/impls/h2opus/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
