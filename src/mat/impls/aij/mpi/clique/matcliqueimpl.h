#ifndef _matcliqueimpl_h
#define _matcliqueimpl_h

#include <clique.hpp>
#include <petsc-private/matimpl.h>

#if defined (PETSC_USE_COMPLEX)
typedef cliq::Complex<PetscReal> PetscCliqScalar;
#else
typedef PetscScalar PetscCliqScalar;
#endif

typedef struct {
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat;  /* Clique sparse matrix */
  MatStructure   matstruc;
  PetscBool      CleanUpClique;  /* Boolean indicating if we call Clique clean step */
  MPI_Comm       cliq_comm;      /* Clique MPI communicator                         */
  cliq::DistSymmInfo info;
  cliq::DistSymmFrontTree<PetscCliqScalar> frontTree;
  PetscErrorCode (*Destroy)(Mat);
} Mat_Clique;

#endif
