#ifndef PETSC_FINCLUDE_PETSCDEVICE_H
#define PETSC_FINCLUDE_PETSCDEVICE_H
!
! Include file for Fortran use of the PetscDevice package in PETSc
!

#include "petsc/finclude/petscsys.h"

! PetscDevice
#define PetscDevice type(tPetscDevice)

#define PetscDeviceType PetscEnum
#define PetscDeviceInitType PetscEnum

! PetscDeviceContext
#define PetscDeviceContext type(tPetscDeviceContext)

#define PetscStreamType PetscEnum
#define PetscDeviceContextJoinMode PetscEnum

#endif /* PETSC_FINCLUDE_PETSCDEVICE_H */
