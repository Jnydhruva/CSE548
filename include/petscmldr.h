#ifndef PETSCMLDR_H
#define PETSCMLDR_H

typedef struct _p_MLDR* MLDR;

/*J
    MLDRType - String with the name of a PETSc dimension reduction method.

   Level: beginner

.seealso: MLDRSetType(), MLDR, MLDRRegister(), MLDRCreate(), MLDRSetFromOptions()
J*/
typedef const char *MLDRType;
#define MLDRPCA "pca"

PETSC_EXTERN PetscClassId MLDR_CLASSID;

PETSC_EXTERN PetscErrorCode MLDRInitializePackage(void);
PETSC_EXTERN PetscErrorCode MLDRCreate(MPI_Comm,MLDR*);
PETSC_EXTERN PetscErrorCode MLDRReset(MLDR);
PETSC_EXTERN PetscErrorCode MLDRDestroy(MLDR*);
