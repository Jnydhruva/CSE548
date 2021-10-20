!
! Used by petscsysmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdevice.h"

!
! PetscDevice
!
      type tPetscDevice
        PetscFortranAddr:: d PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscDevice

!
! PetscDeviceTypes
!
      PetscEnum, parameter :: PETSC_DEVICE_INVALID = 0
      PetscEnum, parameter :: PETSC_DEVICE_CUDA    = 1
      PetscEnum, parameter :: PETSC_DEVICE_HIP     = 2
      PetscEnum, parameter :: PETSC_DEVICE_MAX     = 3
#if defined(PETSC_HAVE_HIP)
#  define PETSC_DEVICE_DEFAULT PETSC_DEVICE_HIP
#elif defined(PETSC_HAVE_CUDA)
#  define PETSC_DEVICE_DEFAULT PETSC_DEVICE_CUDA
#else
#  define PETSC_DEVICE_DEFAULT PETSC_DEVICE_INVALID
#endif

!
!     PetscDeviceInitTypes
!
      PetscEnum, parameter :: PETSC_DEVICE_INIT_NONE = 0
      PetscEnum, parameter :: PETSC_DEVICE_LAZY      = 1
      PetscEnum, parameter :: PETSC_DEVICE_EAGER     = 2

!
! PetscDeviceContext
!
      type tPetscDeviceContext
        PetscFortranAddr:: d PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscDeviceContext

!
! PetscStreamTypes
!
      PetscEnum, parameter :: PETSC_STREAM_GLOBAL_BLOCKING    = 0
      PetscEnum, parameter :: PETSC_STREAM_DEFAULT_BLOCKING   = 1
      PetscEnum, parameter :: PETSC_STREAM_GLOBAL_NONBLOCKING = 2
      PetscEnum, parameter :: PETSC_STREAM_MAX                = 3

!
! PetscDeviceContextJoinModes
!
      PetscEnum, parameter :: PETSC_DEVICE_CONTEXT_JOIN_DESTROY = 0
      PetscEnum, parameter :: PETSC_DEVICE_CONTEXT_JOIN_SYNC    = 1
      PetscEnum, parameter :: PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC = 2
