program ex96f90
#include "petsc/finclude/petsc.h"
  use petsc
  implicit none

  ! Get the fortran kind associated with PetscInt and PetscReal so that we can use literal constants.
  PetscInt                           :: dummyPetscInt
  PetscReal                          :: dummyPetscreal
  integer,parameter                  :: kPI = kind(dummyPetscInt)
  integer,parameter                  :: kPR = kind(dummyPetscReal)
  character(len=256)                 :: iobuffer
  PetscErrorCode                     :: ierr

  Type(tDM)                          :: dm
  PetscInt                           :: n
  Type(tIS)                          :: is
  PetscInt,Dimension(:),Pointer      :: nindices

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER,ierr))
  PetscCallA(DMPlexCreateBoxMesh(PETSC_COMM_WORLD,2_kPI,PETSC_TRUE,PETSC_NULL_INTEGER,PETSC_NULL_REAL,PETSC_NULL_REAL,PETSC_NULL_INTEGER,PETSC_TRUE,dm,ierr))
  PetscCallA(PetscObjectSetName(dm,"ex96f90",ierr))
  PetscCallA(DMView(dm,PETSC_VIEWER_STDOUT_WORLD,ierr))
  PetscCallA(DMGetStratumSize(dm,"depth",1,n,ierr))
  Write(IOBuffer,'("Size of stratum depth: ",I3,"\n")') n
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr))
  PetscCallA(DMGetStratumIS(dm,"depth",1,is,ierr))
  if (is == PETSC_NULL_IS) then
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"IS is null\n",ierr))
  else
    PetscCallA(ISGetIndicesF90(is,nindices,ierr))
    PetscCallA(ISView(is,PETSC_VIEWER_STDOUT_SELF,ierr))
    PetscCallA(ISRestoreIndicesF90(is,nindices,ierr))
  end if
  PetscCallA(ISDestroy(is,ierr))
  PetscCallA(DMGetStratumSize(dm,"zorglub",1,n,ierr))
  Write(IOBuffer,'("Size of stratum zorglub: ",I3,"\n")') n
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,iobuffer,ierr))
  PetscCallA(DMGetStratumIS(dm,"zorglub",1,is,ierr))
  if (is == PETSC_NULL_IS) then
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"IS is null\n",ierr))
  else
    PetscCallA(ISGetIndicesF90(is,nindices,ierr))
    PetscCallA(ISView(is,PETSC_VIEWER_STDOUT_SELF,ierr))
    PetscCallA(ISRestoreIndicesF90(is,nindices,ierr))
  end if
  PetscCallA(ISDestroy(is,ierr))
  PetscCallA(DMDestroy(dm,ierr))
  PetscCallA(PetscFinalize(ierr))
end program ex96f90
! /*TEST
!   build:
!     requires: triangle
!   test:
!     suffix: 0
! TEST*/

