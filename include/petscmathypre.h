#ifndef PETSCMATHYPRE_H
#define PETSCMATHYPRE_H

#include <petscmat.h>

typedef struct hypre_ParCSRMatrix_struct hypre_ParCSRMatrix;
PETSC_EXTERN PetscErrorCode MatCreateFromParCSR(hypre_ParCSRMatrix*,MatType,PetscCopyMode,Mat*);
PETSC_EXTERN PetscErrorCode MatHYPREGetParCSR(Mat,hypre_ParCSRMatrix**);

#endif
