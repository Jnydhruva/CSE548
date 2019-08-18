#if !defined(_PETSCHYPRE_H)
#define _PETSCHYPRE_H

#include <petscsys.h>
#include <petscpkg_version.h>
/* workaround for C++/CUDA code in public header */
#if !defined(HYPRE_SKIP_CUDA_PUBLIC)
#define HYPRE_SKIP_CUDA_PUBLIC 1
#endif
#include <HYPRE_config.h>
#include <HYPRE_utilities.h>

/* from version 2.16 on, HYPRE_BigInt is 64 bit for 64bit installations
   and 32 bit for 32bit installations -> not the best name for a variable */
#if PETSC_PKG_HYPRE_VERSION_LT(2,16,0)
typedef PetscInt HYPRE_BigInt;
#endif

#if defined(HYPRE_USING_CUDA)
typedef HYPRE_Int hypre_Index[3];
#endif

/*
  With scalar type == real, HYPRE_Complex == PetscScalar;
  With scalar type == complex,  HYPRE_Complex is double __complex__ while PetscScalar may be std::complex<double>
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHYPREScalarCast(PetscScalar a, HYPRE_Complex *b)
{
  PetscFunctionBegin;
#if defined(HYPRE_COMPLEX)
  ((PetscReal*)b)[0] = PetscRealPart(a);
  ((PetscReal*)b)[1] = PetscImaginaryPart(a);
#else
  *b = a;
#endif
  PetscFunctionReturn(0);
}
#endif

