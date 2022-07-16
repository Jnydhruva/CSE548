/*
     Provides the interface functions for vector operations that have PetscScalar/PetscReal in the signature
   These are the vector functions the user calls.
*/
#include "petsc/private/sfimpl.h"
#include "petscsystypes.h"
#include <petsc/private/vecimpl.h> /*I  "petscvec.h"   I*/
#if PetscDefined(HAVE_CUPM)
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/veccupmimpl.h>
#endif

PetscInt VecGetSubVectorSavedStateId = -1;

PETSC_EXTERN PetscErrorCode VecValidValues(Vec vec, PetscInt argnum, PetscBool begin) {
  PetscFunctionBegin;
  if (!PetscDefined(USE_DEBUG)) PetscFunctionReturn(0);
  if ((vec->petscnative || vec->ops->getarray) && (PetscDefined(HAVE_DEVICE) ? vec->offloadmask & PETSC_OFFLOAD_CPU : 1)) {
    PetscInt           n;
    const PetscScalar *x;

    PetscCall(VecGetLocalSize(vec, &n));
    PetscCall(VecGetArrayRead(vec, &x));
    for (PetscInt i = 0; i < n; i++) {
      if (begin) {
        PetscCheck(!PetscIsInfOrNanScalar(x[i]), PETSC_COMM_SELF, PETSC_ERR_FP, "Vec entry at local location %" PetscInt_FMT " is not-a-number or infinite at beginning of function: Parameter number %" PetscInt_FMT, i, argnum);
      } else {
        PetscCheck(!PetscIsInfOrNanScalar(x[i]), PETSC_COMM_SELF, PETSC_ERR_FP, "Vec entry at local location %" PetscInt_FMT " is not-a-number or infinite at end of function: Parameter number %" PetscInt_FMT, i, argnum);
      }
    }
    PetscCall(VecRestoreArrayRead(vec, &x));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMaxPointwiseDivideAsync(Vec x, Vec y, PetscManagedReal max, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscUseTypeMethod(x, maxpointwisedivide, y, max, dctx);
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(0);
}

/*@
   VecMaxPointwiseDivide - Computes the maximum of the componentwise division max = max_i abs(x_i/y_i).

   Logically Collective on Vec, Synchronous

   Input Parameters:
.  x, y  - the vectors

   Output Parameter:
.  max - the result

   Level: advanced

   Notes:
    x and y may be the same vector
          if a particular y_i is zero, it is treated as 1 in the above formula

.seealso: `VecPointwiseDivide()`, `VecPointwiseMult()`, `VecPointwiseMax()`, `VecPointwiseMin()`, `VecPointwiseMaxAbs()`
@*/
PetscErrorCode VecMaxPointwiseDivide(Vec x, Vec y, PetscReal *max) {
  PetscManagedReal tmp;

  PetscFunctionBegin;
  PetscValidRealPointer(max, 3);
  PetscCall(PetscManageHostReal(NULL, max, 1, &tmp));
  PetscCall(VecMaxPointwiseDivideAsync(x, y, tmp, NULL));
  PetscCall(PetscManagedHostRealDestroy(NULL, &tmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotAsync(Vec x, Vec y, PetscManagedScalar val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscLogEventBegin(VEC_Dot, x, y, 0, 0));
  PetscUseTypeMethod(x, dot, y, val, dctx);
  PetscCall(PetscLogEventEnd(VEC_Dot, x, y, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(0);
}

/*@
   VecDot - Computes the vector dot product.

   Collective on Vec, Synchronous

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  val - the dot product

   Performance Issues:
$    per-processor memory bandwidth
$    interprocessor latency
$    work load imbalance that causes certain processes to arrive much earlier than others

   Notes for Users of Complex Numbers:
   For complex vectors, VecDot() computes
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y. Note that this corresponds to the usual "mathematicians" complex
   inner product where the SECOND argument gets the complex conjugate. Since the BLASdot() complex conjugates the first
   first argument we call the BLASdot() with the arguments reversed.

   Use VecTDot() for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Level: intermediate

.seealso: `VecMDot()`, `VecTDot()`, `VecNorm()`, `VecDotBegin()`, `VecDotEnd()`, `VecDotRealPart()`
@*/
PetscErrorCode VecDot(Vec x, Vec y, PetscScalar *val) {
  PetscManagedScalar tmp;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidScalarPointer(val, 3);
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscCall(PetscManageHostScalar(dctx, val, 1, &tmp));
  PetscCall(VecDotAsync(x, y, tmp, dctx));
  PetscCall(PetscManagedHostScalarDestroy(dctx, &tmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotRealPartAsync(Vec x, Vec y, PetscManagedReal val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidPointer(val, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (PetscDefined(USE_COMPLEX)) {
    const PetscInt     size = 1;
    PetscScalar       *ptr;
    PetscMemType       mtype;
    PetscManagedScalar tmp;

    PetscCall(PetscManagedScalarCreateDefault(dctx, size, &tmp));
    PetscCall(VecDotAsync(x, y, tmp, dctx));
    PetscCall(PetscManagedScalarGetPointerAndMemType(dctx, tmp, PETSC_MEMORY_ACCESS_READ, &ptr, &mtype));
    PetscCall(PetscManagedRealSetValues(dctx, val, mtype, (PetscReal *)ptr, size));
    PetscCall(PetscManagedScalarDestroy(dctx, &tmp));
  } else {
    // PetscReal is PetscScalar
    PetscCall(VecDotAsync(x, y, (PetscManagedScalar)val, dctx));
  }
  PetscFunctionReturn(0);
}

/*@
   VecDotRealPart - Computes the real part of the vector dot product.

   Collective on Vec, Synchronous

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  val - the real part of the dot product;

   Performance Issues:
$    per-processor memory bandwidth
$    interprocessor latency
$    work load imbalance that causes certain processes to arrive much earlier than others

   Notes for Users of Complex Numbers:
     See VecDot() for more details on the definition of the dot product for complex numbers

     For real numbers this returns the same value as VecDot()

     For complex numbers in C^n (that is a vector of n components with a complex number for each component) this is equal to the usual real dot product on the
     the space R^{2n} (that is a vector of 2n components with the real or imaginary part of the complex numbers for components)

   Developer Note: This is not currently optimized to compute only the real part of the dot product.

   Level: intermediate

.seealso: `VecMDot()`, `VecTDot()`, `VecNorm()`, `VecDotBegin()`, `VecDotEnd()`, `VecDot()`, `VecDotNorm2()`
@*/
PetscErrorCode VecDotRealPart(Vec x, Vec y, PetscReal *val) {
  PetscManagedReal tmp;

  PetscFunctionBegin;
  PetscValidRealPointer(val, 3);
  PetscCall(PetscManageHostReal(NULL, val, 1, &tmp));
  PetscCall(VecDotRealPartAsync(x, y, tmp, NULL));
  PetscCall(PetscManagedHostRealDestroy(NULL, &tmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecNormAsync(Vec x, NormType type, PetscManagedReal scal, PetscDeviceContext dctx) {
  PetscBool flg;
  PetscReal rval;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidPointer(scal, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  /* Cached data? */
  PetscCall(VecNormAvailable(x, type, &flg, &rval));
  if (flg) {
    PetscCall(PetscManagedRealSetValues(dctx, scal, PETSC_MEMTYPE_HOST, &rval, 1));
    PetscFunctionReturn(0);
  }

  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_Norm, x, 0, 0, 0));
  PetscUseTypeMethod(x, norm, type, scal, dctx);
  PetscCall(PetscLogEventEnd(VEC_Norm, x, 0, 0, 0));
  PetscCall(VecLockReadPop(x));

  if (type != NORM_1_AND_2) {
    PetscReal *values;

    PetscCall(PetscManagedRealGetValuesAvailable(dctx, scal, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, &values, &flg));
    if (flg) PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[type], *values));
  }
  PetscFunctionReturn(0);
}

/*@
   VecNorm  - Computes the vector norm.

   Collective on Vec, Synchronous

   Input Parameters:
+  x - the vector
-  type - the type of the norm requested

   Output Parameter:
.  val - the norm

   Values of NormType:
+     NORM_1 - sum_i |x_i|
.     NORM_2 - sqrt(sum_i |x_i|^2)
.     NORM_INFINITY - max_i |x_i|
-     NORM_1_AND_2 - computes efficiently both  NORM_1 and NORM_2 and stores them each in an output array

   Notes:
      For complex numbers NORM_1 will return the traditional 1 norm of the 2 norm of the complex numbers; that is the 1
      norm of the absolute values of the complex entries. In PETSc 3.6 and earlier releases it returned the 1 norm of
      the 1 norm of the complex entries (what is returned by the BLAS routine asum()). Both are valid norms but most
      people expect the former.

      This routine stashes the computed norm value, repeated calls before the vector entries are changed are then rapid since the
      precomputed value is immediately available. Certain vector operations such as VecSet() store the norms so the value is
      immediately available and does not need to be explicitly computed. VecScale() updates any stashed norm values, thus calls after VecScale()
      do not need to explicitly recompute the norm.

   Level: intermediate

   Performance Issues:
+    per-processor memory bandwidth - limits the speed of the computation of local portion of the norm
.    interprocessor latency - limits the accumulation of the result across ranks, .i.e. MPI_Allreduce() time
.    number of ranks - the time for the result will grow with the log base 2 of the number of ranks sharing the vector
-    work load imbalance - the rank with the largest number of vector entries will limit the speed up

.seealso: `VecDot()`, `VecTDot()`, `VecNorm()`, `VecDotBegin()`, `VecDotEnd()`, `VecNormAvailable()`,
          `VecNormBegin()`, `VecNormEnd()`, `NormType()`

@*/
PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val) {
  PetscManagedReal   tmp;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidRealPointer(val, 3);
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscCall(PetscManageHostReal(dctx, val, 1 + (type == NORM_1_AND_2), &tmp));
  PetscCall(VecNormAsync(x, type, tmp, dctx));
  PetscCall(PetscManagedHostRealDestroy(dctx, &tmp));
  if (type != NORM_1_AND_2) PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[type], *val));
  PetscFunctionReturn(0);
}

/*@
   VecNormAvailable  - Returns the vector norm if it is already known.

   Not Collective, Synchronous

   Input Parameters:
+  x - the vector
-  type - one of NORM_1, NORM_2, NORM_INFINITY.  Also available
          NORM_1_AND_2, which computes both norms and stores them
          in a two element array.

   Output Parameters:
+  available - PETSC_TRUE if the val returned is valid
-  val - the norm

   Notes:
$     NORM_1 denotes sum_i |x_i|
$     NORM_2 denotes sqrt(sum_i (x_i)^2)
$     NORM_INFINITY denotes max_i |x_i|

   Level: intermediate

   Performance Issues:
$    per-processor memory bandwidth
$    interprocessor latency
$    work load imbalance that causes certain processes to arrive much earlier than others

   Compile Option:
   PETSC_HAVE_SLOW_BLAS_NORM2 will cause a C (loop unrolled) version of the norm to be used, rather
 than the BLAS. This should probably only be used when one is using the FORTRAN BLAS routines
 (as opposed to vendor provided) because the FORTRAN BLAS NRM2() routine is very slow.

.seealso: `VecDot()`, `VecTDot()`, `VecNorm()`, `VecDotBegin()`, `VecDotEnd()`, `VecNorm()`
          `VecNormBegin()`, `VecNormEnd()`

@*/
PetscErrorCode VecNormAvailable(Vec x, NormType type, PetscBool *available, PetscReal *val) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidBoolPointer(available, 3);
  PetscValidRealPointer(val, 4);

  if (type == NORM_1_AND_2) {
    *available = PETSC_FALSE;
  } else {
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[type], *val, *available));
  }
  PetscFunctionReturn(0);
}

/*@
   VecNormalize - Normalizes a vector by 2-norm.

   Collective on Vec, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameter:
.  val - the vector norm before normalization. May be `NULL` if the value is not needed.

   Level: intermediate

@*/
PetscErrorCode VecNormalize(Vec x, PetscReal *val) {
  PetscReal norm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (val) PetscValidRealPointer(val, 2);
  PetscCall(PetscLogEventBegin(VEC_Normalize, x, 0, 0, 0));
  PetscCall(VecNorm(x, NORM_2, &norm));
  if (norm == 0.0) {
    PetscCall(PetscInfo(x, "Vector of zero norm can not be normalized; Returning only the zero norm\n"));
  } else if (norm != 1.0) {
    PetscCall(VecScale(x, 1.0 / norm));
  }
  PetscCall(PetscLogEventEnd(VEC_Normalize, x, 0, 0, 0));
  if (val) *val = norm;
  PetscFunctionReturn(0);
}

// called by VecMinMax_Private
static PetscErrorCode VecMinMaxAsync_Private(Vec x, PetscManagedInt p, PetscManagedReal val, PetscDeviceContext dctx, PetscLogEvent event, PetscErrorCode (*const minmax_op)(Vec, PetscManagedInt, PetscManagedReal, PetscDeviceContext)) {
  PetscFunctionBegin;
  PetscValidType(x, 1);
  if (p) PetscValidManagedType(p, 2);
  PetscValidManagedType(val, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (p) PetscCheckManagedTypeCompatibleDeviceContext(dctx, 4, p, 2);
  PetscCheckManagedTypeCompatibleDeviceContext(dctx, 4, val, 3);
  PetscValidFunction(minmax_op, 6);
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(event, x, 0, 0, 0));
  PetscCall((*minmax_op)(x, p, val, dctx));
  PetscCall(PetscLogEventEnd(event, x, 0, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(0);
}

// sets up the managed values, then calls the respective min/max function
static PetscErrorCode VecMinMax_Private(Vec x, PetscInt *p, PetscReal *val, PetscErrorCode (*const MinMaxAsyncFunc)(Vec, PetscManagedInt, PetscManagedReal, PetscDeviceContext)) {
  PetscManagedInt    tmpp = NULL; // critical, as tmpp is checked for null to check internally
  PetscManagedReal   tmpv;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  if (p) {
    PetscValidIntPointer(p, 2);
    PetscCall(PetscManageHostInt(dctx, p, 1, &tmpp));
  }
  PetscValidRealPointer(val, 3);
  PetscValidFunction(MinMaxAsyncFunc, 4);
  PetscCall(PetscManageHostReal(dctx, val, 1, &tmpv));
  PetscCall((*MinMaxAsyncFunc)(x, tmpp, tmpv, dctx));
  PetscCall(PetscManagedHostRealDestroy(dctx, &tmpv));
  if (p) PetscCall(PetscManagedHostIntDestroy(dctx, &tmpp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMaxAsync(Vec x, PetscManagedInt p, PetscManagedReal val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  // need to do this first to catch x being a bad pointer before we derefence the ops
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMinMaxAsync_Private(x, p, val, dctx, VEC_Max, x->ops->max));
  PetscFunctionReturn(0);
}

/*@C
   VecMax - Determines the vector component with maximum real part and its location.

   Collective on Vec, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameters:
+  p - the location of val (pass NULL if you don't want this)
-  val - the maximum component

   Notes:
   Returns the value PETSC_MIN_REAL and negative p if the vector is of length 0.

   Returns the smallest index with the maximum value
   Level: intermediate

.seealso: `VecNorm()`, `VecMin()`
@*/
PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val) {
  PetscFunctionBegin;
  PetscCall(VecMinMax_Private(x, p, val, VecMaxAsync));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMinAsync(Vec x, PetscManagedInt p, PetscManagedReal val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  // need to do this first to catch x being a bad pointer before we derefence the ops
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMinMaxAsync_Private(x, p, val, dctx, VEC_Min, x->ops->min));
  PetscFunctionReturn(0);
}

/*@C
   VecMin - Determines the vector component with minimum real part and its location.

   Collective on Vec, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameters:
+  p - the location of val (pass NULL if you don't want this location)
-  val - the minimum component

   Level: intermediate

   Notes:
   Returns the value PETSC_MAX_REAL and negative p if the vector is of length 0.

   This returns the smallest index with the minumum value

.seealso: `VecMax()`
@*/
PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val) {
  PetscFunctionBegin;
  PetscCall(VecMinMax_Private(x, p, val, VecMinAsync));
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDotAsync(Vec x, Vec y, PetscManagedScalar val, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  PetscValidPointer(val, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscLogEventBegin(VEC_TDot, x, y, 0, 0));
  PetscUseTypeMethod(x, tdot, y, val, dctx);
  PetscCall(PetscLogEventEnd(VEC_TDot, x, y, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(0);
}

/*@
   VecTDot - Computes an indefinite vector dot product. That is, this
   routine does NOT use the complex conjugate.

   Collective on Vec, Synchronous

   Input Parameters:
.  x, y - the vectors

   Output Parameter:
.  val - the dot product

   Notes for Users of Complex Numbers:
   For complex vectors, VecTDot() computes the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecDot() for the inner product
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Level: intermediate

.seealso: `VecDot()`, `VecMTDot()`
@*/
PetscErrorCode VecTDot(Vec x, Vec y, PetscScalar *val) {
  PetscManagedScalar tmp;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidScalarPointer(val, 3);
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscCall(PetscManageHostScalar(dctx, val, 1, &tmp));
  PetscCall(VecTDotAsync(x, y, tmp, dctx));
  PetscCall(PetscManagedHostScalarDestroy(dctx, &tmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecScaleAsync(Vec x, PetscManagedScalar alpha, PetscDeviceContext dctx) {
  const PetscObject xobj = (PetscObject)x;
  PetscScalar      *alpha_ptr;
  PetscReal         norms[4];
  PetscBool         flags[4];
  PetscBool         avail;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCheck(x->stash.insertmode == NOT_SET_VALUES, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled vector");
  PetscValidPointer(alpha, 2);
  if (PetscManagedScalarKnownAndEqual(alpha, 1.0)) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSetErrorIfLocked(x, 1));
  /* get current stashed norms */
  for (PetscInt i = 0; i < 4; ++i) { PetscCall(PetscObjectComposedDataGetReal(xobj, NormIds[i], norms[i], flags[i])); }
  PetscCall(PetscManagedScalarGetValuesAvailable(dctx, alpha, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, &alpha_ptr, &avail));

  PetscCall(PetscLogEventBegin(VEC_Scale, x, 0, 0, 0));
  PetscUseTypeMethod(x, scale, alpha, dctx);
  PetscCall(PetscLogEventEnd(VEC_Scale, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease(xobj));

  if (avail) {
    const PetscReal alpha_abs = PetscAbsScalar(*alpha_ptr);

    for (PetscInt i = 0; i < 4; ++i) {
      if (flags[i]) PetscCall(PetscObjectComposedDataSetReal(xobj, NormIds[i], alpha_abs * norms[i]));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   VecScale - Scales a vector.

   Not collective on Vec, Synchronous

   Input Parameters:
+  x - the vector
-  alpha - the scalar

   Note:
   For a vector with n components, VecScale() computes
$      x[i] = alpha * x[i], for i=1,...,n.

   Level: intermediate

@*/
PetscErrorCode VecScale(Vec x, PetscScalar alpha) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (alpha != (PetscScalar)1.0) {
    const PetscObject  xobj = (PetscObject)x;
    PetscReal          norms[4];
    PetscBool          flags[4];
    PetscManagedScalar tmp;

    /* get current stashed norms */
    for (PetscInt i = 0; i < 4; ++i) { PetscCall(PetscObjectComposedDataGetReal(xobj, NormIds[i], norms[i], flags[i])); }
    PetscCall(PetscManageHostScalar(NULL, &alpha, 1, &tmp));
    PetscCall(VecScaleAsync(x, tmp, NULL));
    PetscCall(PetscManagedHostScalarDestroy(NULL, &tmp));
    for (PetscInt i = 0; i < 4; ++i) { PetscCall(PetscObjectComposedDataGetReal(xobj, NormIds[i], norms[i], flags[i])); }
    PetscCall(PetscCopyHostScalar(NULL, &alpha, 1, &tmp));
    PetscCall(VecScaleAsync(x, tmp, NULL));
    PetscCall(PetscManagedScalarDestroy(NULL, &tmp));
    for (PetscInt i = 0; i < 4; ++i) {
      if (flags[i]) { PetscCall(PetscObjectComposedDataSetReal(xobj, NormIds[i], PetscAbsScalar(alpha) * norms[i])); }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetAsync(Vec x, PetscManagedScalar alpha, PetscDeviceContext dctx) {
  const PetscObject obj = (PetscObject)x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCheck(x->stash.insertmode == NOT_SET_VALUES, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not for unassembled vector");
  PetscValidPointer(alpha, 2);
  PetscCall(VecSetErrorIfLocked(x, 1));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(PetscLogEventBegin(VEC_Set, x, 0, 0, 0));
  PetscUseTypeMethod(x, set, alpha, dctx);
  PetscCall(PetscLogEventEnd(VEC_Set, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease(obj));
  {
    const PetscInt N = x->map->N;

    if (PetscUnlikely(N == 0)) {
      PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_1], 0.0l));
      PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_2], 0.0));
      PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_FROBENIUS], 0.0));
      PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_INFINITY], 0.0));
    } else {
      PetscScalar *alpha_ptr;
      PetscBool    avail;

      PetscCall(PetscManagedScalarGetValuesAvailable(dctx, alpha, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, &alpha_ptr, &avail));
      if (avail) {
        const PetscReal nreal = (PetscReal)N;
        PetscReal       areal = PetscAbsScalar(*alpha_ptr);

        if (areal > (PETSC_MAX_REAL / nreal)) {
          PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_INFINITY], areal));
        } else {
          PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_1], N * areal));
          PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_INFINITY], areal));
          areal *= PetscSqrtReal(nreal);
          PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_2], areal));
          PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_FROBENIUS], areal));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@
   VecSet - Sets all components of a vector to a single scalar value.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  x  - the vector
-  alpha - the scalar

   Output Parameter:
.  x  - the vector

   Note:
   For a vector of dimension n, VecSet() computes
$     x[i] = alpha, for i=1,...,n,
   so that all vector entries then equal the identical
   scalar value, alpha.  Use the more general routine
   VecSetValues() to set different vector entries.

   You CANNOT call this after you have called VecSetValues() but before you call
   VecAssemblyBegin/End().

   Level: beginner

.seealso `VecSetValues()`, `VecSetValuesBlocked()`, `VecSetRandom()`

@*/
PetscErrorCode VecSet(Vec x, PetscScalar alpha) {
  PetscReal          areal = PetscAbsScalar(alpha);
  PetscManagedScalar tmp;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveScalar(x, alpha, 2);
  PetscCall(PetscCopyHostScalar(NULL, &alpha, 1, &tmp));
  PetscCall(VecSetAsync(x, tmp, NULL));
  PetscCall(PetscManagedScalarDestroy(NULL, &tmp));
  {
    const PetscInt N = x->map->N;

    if (PetscLikely(N != 0)) {
      const PetscObject obj   = (PetscObject)x;
      const PetscReal   nreal = (PetscReal)N;

      if (areal > (PETSC_MAX_REAL / nreal)) {
        PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_INFINITY], areal));
      } else {
        PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_1], N * areal));
        PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_INFINITY], areal));
        areal *= PetscSqrtReal(nreal);
        PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_2], areal));
        PetscCall(PetscObjectComposedDataSetReal(obj, NormIds[NORM_FROBENIUS], areal));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecXPYAsync_Private(Vec y, PetscManagedScalar alpha, Vec x, PetscDeviceContext dctx, PetscLogEvent VEC_Event, PetscErrorCode (*const xpy_op)(Vec, PetscManagedScalar, Vec, PetscDeviceContext)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidType(y, 1);
  PetscValidType(x, 3);
  PetscCheckSameTypeAndComm(y, 1, x, 3);
  VecCheckSameSize(y, 1, x, 3);
  PetscCheck(x != y, PetscObjectComm((PetscObject)y), PETSC_ERR_ARG_IDN, "x and y cannot be the same vector");
  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscValidPointer(alpha, 2);
  PetscValidFunction(xpy_op, 6);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_Event, y, x, 0, 0));
  PetscCall((*xpy_op)(y, alpha, x, dctx));
  PetscCall(PetscLogEventEnd(VEC_Event, y, x, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPYAsync(Vec y, PetscManagedScalar alpha, Vec x, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1); // check before ops dereference
  if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) PetscFunctionReturn(0);
  PetscCall(VecXPYAsync_Private(y, alpha, x, dctx, VEC_AXPY, y->ops->axpy));
  PetscFunctionReturn(0);
}

/*@
   VecAXPY - Computes y = alpha x + y.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes:
    x and y MUST be different vectors
    This routine is optimized for alpha of 0.0, otherwise it calls the BLAS routine

$    VecAXPY(y,alpha,x)                   y = alpha x           +      y
$    VecAYPX(y,beta,x)                    y =       x           + beta y
$    VecAXPBY(y,alpha,beta,x)             y = alpha x           + beta y
$    VecWAXPY(w,alpha,x,y)                w = alpha x           +      y
$    VecAXPBYPCZ(w,alpha,beta,gamma,x,y)  z = alpha x           + beta y + gamma z
$    VecMAXPY(y,nv,alpha[],x[])           y = sum alpha[i] x[i] +      y

.seealso: `VecAYPX()`, `VecMAXPY()`, `VecWAXPY()`, `VecAXPBYPCZ()`, `VecAXPBY()`
@*/
PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x) {
  PetscManagedScalar tmp;

  PetscFunctionBegin;
  // check valid header for y so we can use logical collective below, rest is checked in the
  // async version
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  PetscCall(PetscCopyHostScalar(NULL, &alpha, 1, &tmp));
  PetscCall(VecAXPYAsync(y, tmp, x, NULL));
  PetscCall(PetscManagedScalarDestroy(NULL, &tmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAYPXAsync(Vec y, PetscManagedScalar beta, Vec x, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  if (PetscManagedScalarKnownAndEqual(beta, 0.0)) {
    PetscCall(VecCopyAsync(x, y, dctx));
  } else {
    PetscValidHeaderSpecific(y, VEC_CLASSID, 1); // check before ops dereference
    PetscCall(VecXPYAsync_Private(y, beta, x, dctx, VEC_AYPX, y->ops->aypx));
  }
  PetscFunctionReturn(0);
}

/*@
   VecAYPX - Computes y = x + beta y.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  beta - the scalar
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes:
    x and y MUST be different vectors
    The implementation is optimized for beta of -1.0, 0.0, and 1.0

.seealso: `VecMAXPY()`, `VecWAXPY()`, `VecAXPY()`, `VecAXPBYPCZ()`, `VecAXPBY()`
@*/
PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x) {
  PetscManagedScalar btmp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveScalar(y, beta, 2);
  PetscCall(PetscCopyHostScalar(NULL, &beta, 1, &btmp));
  PetscCall(VecAYPXAsync(y, btmp, x, NULL));
  PetscCall(PetscManagedScalarDestroy(NULL, &btmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYAsync(Vec y, PetscManagedScalar alpha, PetscManagedScalar beta, Vec x, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  PetscValidType(y, 1);
  PetscValidType(x, 4);
  PetscCheckSameTypeAndComm(x, 4, y, 1);
  VecCheckSameSize(y, 1, x, 4);
  PetscCheck(x != y, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_IDN, "x and y cannot be the same vector");
  PetscValidPointer(alpha, 2);
  PetscValidPointer(beta, 3);
  if (PetscManagedScalarKnownAndEqual(alpha, 0.0) && PetscManagedScalarKnownAndEqual(beta, 1.0)) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_AXPY, x, y, 0, 0));
  PetscUseTypeMethod(y, axpby, alpha, beta, x, dctx);
  PetscCall(PetscLogEventEnd(VEC_AXPY, x, y, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(0);
}

/*@
   VecAXPBY - Computes y = alpha x + beta y.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  alpha,beta - the scalars
-  x, y  - the vectors

   Output Parameter:
.  y - output vector

   Level: intermediate

   Notes:
    x and y MUST be different vectors
    The implementation is optimized for alpha and/or beta values of 0.0 and 1.0

.seealso: `VecAYPX()`, `VecMAXPY()`, `VecWAXPY()`, `VecAXPY()`, `VecAXPBYPCZ()`
@*/
PetscErrorCode VecAXPBY(Vec y, PetscScalar alpha, PetscScalar beta, Vec x) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  PetscValidLogicalCollectiveScalar(y, beta, 3);
  if ((alpha != (PetscScalar)0.0) || (beta != (PetscScalar)1.0)) {
    PetscManagedScalar atmp, btmp;
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
    PetscCall(PetscCopyHostScalar(NULL, &alpha, 1, &atmp));
    PetscCall(PetscCopyHostScalar(NULL, &beta, 1, &btmp));
    PetscCall(VecAXPBYAsync(y, atmp, btmp, x, dctx));
    PetscCall(PetscManagedScalarDestroy(dctx, &atmp));
    PetscCall(PetscManagedScalarDestroy(dctx, &btmp));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZAsync(Vec z, PetscManagedScalar alpha, PetscManagedScalar beta, PetscManagedScalar gamma, Vec x, Vec y, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(z, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 6);
  PetscValidType(z, 1);
  PetscValidType(x, 5);
  PetscValidType(y, 6);
  PetscCheckSameTypeAndComm(x, 5, z, 1);
  PetscCheckSameTypeAndComm(x, 5, y, 6);
  VecCheckSameSize(x, 1, y, 5);
  VecCheckSameSize(x, 1, z, 6);
  PetscCheck((x != y) && (x != z), PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_IDN, "x, y, and z must be different vectors");
  PetscCheck(y != z, PetscObjectComm((PetscObject)y), PETSC_ERR_ARG_IDN, "x, y, and z must be different vectors");
  PetscCall(VecSetErrorIfLocked(z, 1));
  if (PetscManagedScalarKnownAndEqual(alpha, 0.0) && PetscManagedScalarKnownAndEqual(beta, 0.0) && PetscManagedScalarKnownAndEqual(gamma, 1.0)) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscLogEventBegin(VEC_AXPBYPCZ, x, y, z, 0));
  PetscUseTypeMethod(z, axpbypcz, alpha, beta, gamma, x, y, dctx);
  PetscCall(PetscLogEventEnd(VEC_AXPBYPCZ, x, y, z, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)z));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(0);
}
/*@
   VecAXPBYPCZ - Computes z = alpha x + beta y + gamma z

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  alpha,beta, gamma - the scalars
-  x, y, z  - the vectors

   Output Parameter:
.  z - output vector

   Level: intermediate

   Notes:
    x, y and z must be different vectors
    The implementation is optimized for alpha of 1.0 and gamma of 1.0 or 0.0

.seealso:  `VecAYPX()`, `VecMAXPY()`, `VecWAXPY()`, `VecAXPY()`, `VecAXPBY()`
@*/
PetscErrorCode VecAXPBYPCZ(Vec z, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec x, Vec y) {
  PetscManagedScalar atmp, btmp, gtmp;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(z, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveScalar(z, alpha, 2);
  PetscValidLogicalCollectiveScalar(z, beta, 3);
  PetscValidLogicalCollectiveScalar(z, gamma, 4);
  if (alpha == (PetscScalar)0.0 && beta == (PetscScalar)0.0 && gamma == (PetscScalar)1.0) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  PetscCall(PetscCopyHostScalar(NULL, &alpha, 1, &atmp));
  PetscCall(PetscCopyHostScalar(NULL, &beta, 1, &btmp));
  PetscCall(PetscCopyHostScalar(NULL, &gamma, 1, &gtmp));
  PetscCall(VecAXPBYPCZAsync(z, atmp, btmp, gtmp, x, y, dctx));
  PetscCall(PetscManagedScalarDestroy(dctx, &atmp));
  PetscCall(PetscManagedScalarDestroy(dctx, &btmp));
  PetscCall(PetscManagedScalarDestroy(dctx, &gtmp));
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPYAsync(Vec w, PetscManagedScalar alpha, Vec x, Vec y, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidType(w, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 4);
  PetscCheckSameTypeAndComm(x, 3, y, 4);
  PetscCheckSameTypeAndComm(y, 4, w, 1);
  VecCheckSameSize(x, 3, y, 4);
  VecCheckSameSize(x, 3, w, 1);
  PetscCheck(w != y, PETSC_COMM_SELF, PETSC_ERR_SUP, "Result vector w cannot be same as input vector y, suggest VecAXPY()");
  PetscCheck(w != x, PETSC_COMM_SELF, PETSC_ERR_SUP, "Result vector w cannot be same as input vector x, suggest VecAYPX()");
  PetscCall(VecSetErrorIfLocked(w, 1));
  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  if (PetscManagedScalarKnownAndEqual(alpha, 0.0)) {
    PetscCall(VecCopyAsync(y, w, dctx));
  } else {
    PetscCall(PetscLogEventBegin(VEC_WAXPY, x, y, w, 0));
    PetscUseTypeMethod(w, waxpy, alpha, x, y, dctx);
    PetscCall(PetscLogEventEnd(VEC_WAXPY, x, y, w, 0));
    PetscCall(PetscObjectStateIncrease((PetscObject)w));
  }
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(0);
}

/*@
   VecWAXPY - Computes w = alpha x + y.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  alpha - the scalar
-  x, y  - the vectors

   Output Parameter:
.  w - the result

   Level: intermediate

   Notes:
    w cannot be either x or y, but x and y can be the same
    The implementation is optimzed for alpha of -1.0, 0.0, and 1.0

.seealso: `VecAXPY()`, `VecAYPX()`, `VecAXPBY()`, `VecMAXPY()`, `VecAXPBYPCZ()`
@*/
PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y) {
  PetscManagedScalar atmp;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  PetscCall(PetscCopyHostScalar(NULL, &alpha, 1, &atmp));
  PetscCall(VecWAXPYAsync(w, atmp, x, y, NULL));
  PetscCall(PetscManagedScalarDestroy(NULL, &atmp));
  PetscFunctionReturn(0);
}

/*@C
   VecSetValues - Inserts or adds values into certain locations of a vector.

   Not Collective, Synchronous

   Input Parameters:
+  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   VecSetValues() sets x[ix[i]] = y[i], for i=0,...,ni-1.

   Calls to VecSetValues() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd()
   MUST be called after all calls to VecSetValues() have been completed.

   VecSetValues() uses 0-based indices in Fortran as well as in C.

   If you call VecSetOption(x, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE),
   negative indices may be passed in ix. These rows are
   simply ignored. This allows easily inserting element load matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the vector.

   Level: beginner

.seealso: `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValuesLocal()`,
          `VecSetValue()`, `VecSetValuesBlocked()`, `InsertMode`, `INSERT_VALUES`, `ADD_VALUES`, `VecGetValues()`
@*/
PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora) {
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);

  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  PetscUseTypeMethod(x, setvalues, ni, ix, y, iora);
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@C
   VecGetValues - Gets values from certain locations of a vector. Currently
          can only get values on the same processor

    Not Collective, Synchronous

   Input Parameters:
+  x - vector to get values from
.  ni - number of elements to get
-  ix - indices where to get them from (in global 1d numbering)

   Output Parameter:
.   y - array of values

   Notes:
   The user provides the allocated array y; it is NOT allocated in this routine

   VecGetValues() gets y[i] = x[ix[i]], for i=0,...,ni-1.

   VecAssemblyBegin() and VecAssemblyEnd()  MUST be called before calling this

   VecGetValues() uses 0-based indices in Fortran as well as in C.

   If you call VecSetOption(x, VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE),
   negative indices may be passed in ix. These rows are
   simply ignored.

   Level: beginner

.seealso: `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValues()`
@*/
PetscErrorCode VecGetValues(Vec x, PetscInt ni, const PetscInt ix[], PetscScalar y[]) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);
  PetscUseTypeMethod(x, getvalues, ni, ix, y);
  PetscFunctionReturn(0);
}

/*@C
   VecSetValuesBlocked - Inserts or adds blocks of values into certain locations of a vector.

   Not Collective, Synchronous

   Input Parameters:
+  x - vector to insert in
.  ni - number of blocks to add
.  ix - indices where to add in block count, rather than element count
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   VecSetValuesBlocked() sets x[bs*ix[i]+j] = y[bs*i+j],
   for j=0,...,bs-1, for i=0,...,ni-1. where bs was set with VecSetBlockSize().

   Calls to VecSetValuesBlocked() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd()
   MUST be called after all calls to VecSetValuesBlocked() have been completed.

   VecSetValuesBlocked() uses 0-based indices in Fortran as well as in C.

   Negative indices may be passed in ix, these rows are
   simply ignored. This allows easily inserting element load matrices
   with homogeneous Dirchlet boundary conditions that you don't want represented
   in the vector.

   Level: intermediate

.seealso: `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValuesBlockedLocal()`,
          `VecSetValues()`
@*/
PetscErrorCode VecSetValuesBlocked(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora) {
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);

  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  PetscUseTypeMethod(x, setvaluesblocked, ni, ix, y, iora);
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@C
   VecSetValuesLocal - Inserts or adds values into certain locations of a vector,
   using a local ordering of the nodes.

   Not Collective, Synchronous

   Input Parameters:
+  x - vector to insert in
.  ni - number of elements to add
.  ix - indices where to add
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: intermediate

   Notes:
   VecSetValuesLocal() sets x[ix[i]] = y[i], for i=0,...,ni-1.

   Calls to VecSetValues() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd()
   MUST be called after all calls to VecSetValuesLocal() have been completed.

   VecSetValuesLocal() uses 0-based indices in Fortran as well as in C.

.seealso: `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValues()`, `VecSetLocalToGlobalMapping()`,
          `VecSetValuesBlockedLocal()`
@*/
PetscErrorCode VecSetValuesLocal(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora) {
  PetscInt lixp[128], *lix = lixp;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);

  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  if (!x->ops->setvalueslocal) {
    if (x->map->mapping) {
      if (ni > 128) PetscCall(PetscMalloc1(ni, &lix));
      PetscCall(ISLocalToGlobalMappingApply(x->map->mapping, ni, (PetscInt *)ix, lix));
      PetscUseTypeMethod(x, setvalues, ni, lix, y, iora);
      if (ni > 128) PetscCall(PetscFree(lix));
    } else PetscUseTypeMethod(x, setvalues, ni, ix, y, iora);
  } else PetscUseTypeMethod(x, setvalueslocal, ni, ix, y, iora);
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@
   VecSetValuesBlockedLocal - Inserts or adds values into certain locations of a vector,
   using a local ordering of the nodes.

   Not Collective, Synchronous

   Input Parameters:
+  x - vector to insert in
.  ni - number of blocks to add
.  ix - indices where to add in block count, not element count
.  y - array of values
-  iora - either INSERT_VALUES or ADD_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: intermediate

   Notes:
   VecSetValuesBlockedLocal() sets x[bs*ix[i]+j] = y[bs*i+j],
   for j=0,..bs-1, for i=0,...,ni-1, where bs has been set with VecSetBlockSize().

   Calls to VecSetValuesBlockedLocal() with the INSERT_VALUES and ADD_VALUES
   options cannot be mixed without intervening calls to the assembly
   routines.

   These values may be cached, so VecAssemblyBegin() and VecAssemblyEnd()
   MUST be called after all calls to VecSetValuesBlockedLocal() have been completed.

   VecSetValuesBlockedLocal() uses 0-based indices in Fortran as well as in C.

.seealso: `VecAssemblyBegin()`, `VecAssemblyEnd()`, `VecSetValues()`, `VecSetValuesBlocked()`,
          `VecSetLocalToGlobalMapping()`
@*/
PetscErrorCode VecSetValuesBlockedLocal(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora) {
  PetscInt lixp[128], *lix = lixp;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni) PetscFunctionReturn(0);
  PetscValidIntPointer(ix, 3);
  PetscValidScalarPointer(y, 4);
  PetscValidType(x, 1);
  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  if (x->map->mapping) {
    if (ni > 128) PetscCall(PetscMalloc1(ni, &lix));
    PetscCall(ISLocalToGlobalMappingApplyBlock(x->map->mapping, ni, (PetscInt *)ix, lix));
    PetscUseTypeMethod(x, setvaluesblocked, ni, lix, y, iora);
    if (ni > 128) PetscCall(PetscFree(lix));
  } else {
    PetscUseTypeMethod(x, setvaluesblocked, ni, ix, y, iora);
  }
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMXDotAsync_Private(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar vals, PetscDeviceContext dctx, PetscLogEvent VEC_MXDot, PetscErrorCode (*const mxdot_op)(Vec, PetscManagedInt, const Vec *, PetscManagedScalar, PetscDeviceContext)) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidPointer(y, 3);
  PetscValidHeaderSpecific(*y, VEC_CLASSID, 3);
  PetscValidType(*y, 3);
  PetscCheckSameTypeAndComm(x, 1, *y, 3);
  VecCheckSameSize(x, 1, *y, 3);
  PetscValidFunction(mxdot_op, 7);
  if (PetscManagedIntKnownAndEqual(nv, 0)) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_MXDot, x, *y, 0, 0));
  PetscCall((*mxdot_op)(x, nv, y, vals, dctx));
  PetscCall(PetscLogEventEnd(VEC_MXDot, x, *y, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDotAsync(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar vals, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMXDotAsync_Private(x, nv, y, vals, dctx, VEC_MDot, x->ops->mdot));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDotAsync(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar vals, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMXDotAsync_Private(x, nv, y, vals, dctx, VEC_MTDot, x->ops->mtdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMXDot_Private(Vec x, PetscInt nv, const Vec y[], PetscScalar vals[], PetscErrorCode (*const VecMXDotAsyncFunction)(Vec, PetscManagedInt, const Vec *, PetscManagedScalar, PetscDeviceContext)) {
  PetscFunctionBegin;
  // some of these checks are repeated in the final dispatcher, but need to be done here to
  // safely use x as a PetscObject
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(x, nv, 2);
  PetscCheck(nv >= 0, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_OUTOFRANGE, "Number of vectors (nv = %" PetscInt_FMT ") cannot be negative", nv);
  if (nv) {
    PetscManagedInt    nvtmp;
    PetscManagedScalar valtmp;
    PetscDeviceContext dctx;

    PetscValidPointer(y, 3);
    for (PetscInt i = 0; i < nv; ++i) {
      // do these checks here since the async version may not safely know the size of nv
      // without a sync
      PetscValidHeaderSpecific(y[i], VEC_CLASSID, 3);
      PetscValidType(y[i], 3);
      PetscCheckSameTypeAndComm(x, 1, y[i], 3);
      VecCheckSameSize(x, 1, y[i], 3);
      PetscCall(VecLockReadPush(y[i]));
    }
    PetscCall(VecLockReadPush(x));
    PetscValidScalarPointer(vals, 4);
    PetscValidFunction(VecMXDotAsyncFunction, 5);
    PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));

    PetscCall(PetscCopyHostInt(dctx, &nv, 1, &nvtmp));
    PetscCall(PetscManageHostScalar(dctx, vals, nv, &valtmp));
    PetscCall(VecMXDotAsyncFunction(x, nvtmp, y, valtmp, dctx));
    PetscCall(PetscManagedHostScalarDestroy(dctx, &valtmp));
    PetscCall(PetscManagedIntDestroy(dctx, &nvtmp));
    PetscCall(VecLockReadPop(x));

    for (PetscInt i = 0; i < nv; ++i) PetscCall(VecLockReadPop(y[i]));
  }
  PetscFunctionReturn(0);
}

/*@
   VecMTDot - Computes indefinite vector multiple dot products.
   That is, it does NOT use the complex conjugate.

   Collective on Vec, Synchronous

   Input Parameters:
+  x - one vector
.  nv - number of vectors
-  y - array of vectors.  Note that vectors are pointers

   Output Parameter:
.  val - array of the dot products

   Notes for Users of Complex Numbers:
   For complex vectors, VecMTDot() computes the indefinite form
$      val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Use VecMDot() for the inner product
$      val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Level: intermediate

.seealso: `VecMDot()`, `VecTDot()`
@*/
PetscErrorCode VecMTDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
  PetscFunctionBegin;
  PetscCall(VecMXDot_Private(x, nv, y, val, VecMTDotAsync));
  PetscFunctionReturn(0);
}

/*@
   VecMDot - Computes vector multiple dot products.

   Collective on Vec, Synchronous

   Input Parameters:
+  x - one vector
.  nv - number of vectors
-  y - array of vectors.

   Output Parameter:
.  val - array of the dot products (does not allocate the array)

   Notes for Users of Complex Numbers:
   For complex vectors, VecMDot() computes
$     val = (x,y) = y^H x,
   where y^H denotes the conjugate transpose of y.

   Use VecMTDot() for the indefinite form
$     val = (x,y) = y^T x,
   where y^T denotes the transpose of y.

   Level: intermediate

.seealso: `VecMTDot()`, `VecDot()`
@*/
PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
  PetscFunctionBegin;
  PetscCall(VecMXDot_Private(x, nv, y, val, VecMDotAsync));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPYAsync(Vec y, PetscManagedInt nv, PetscManagedScalar alpha, Vec x[], PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(y, 1);
  PetscValidPointer(x, 4);
  PetscValidHeaderSpecific(*x, VEC_CLASSID, 4);
  PetscValidType(*x, 4);
  PetscCheckSameTypeAndComm(y, 1, *x, 4);
  VecCheckSameSize(y, 1, *x, 4);
  PetscCall(VecSetErrorIfLocked(y, 1));
  if (PetscManagedIntKnownAndEqual(nv, 0)) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  PetscCall(PetscLogEventBegin(VEC_MAXPY, *x, y, 0, 0));
  PetscUseTypeMethod(y, maxpy, nv, alpha, x, dctx);
  PetscCall(PetscLogEventEnd(VEC_MAXPY, *x, y, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(0);
}

/*@
   VecMAXPY - Computes y = y + sum alpha[i] x[i]

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  nv - number of scalars and x-vectors
.  alpha - array of scalars
.  y - one vector
-  x - array of vectors

   Level: intermediate

   Notes:
    y cannot be any of the x vectors

.seealso: `VecAYPX()`, `VecWAXPY()`, `VecAXPY()`, `VecAXPBYPCZ()`, `VecAXPBY()`
@*/
PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[], Vec x[]) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(y, nv, 2);
  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscCheck(nv >= 0, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_OUTOFRANGE, "Number of vectors (nv = %" PetscInt_FMT ") cannot be negative", nv);
  if (nv) {
    PetscInt zeros = 0;

    PetscValidScalarPointer(alpha, 3);
    PetscValidPointer(x, 4);
    // do these checks on x here since we may not know the size of nv in the async version
    for (PetscInt i = 0; i < nv; ++i) {
      PetscValidLogicalCollectiveScalar(y, alpha[i], 3);
      PetscValidHeaderSpecific(x[i], VEC_CLASSID, 4);
      PetscValidType(x[i], 4);
      PetscCheckSameTypeAndComm(y, 1, x[i], 4);
      VecCheckSameSize(y, 1, x[i], 4);
      PetscCheck(y != x[i], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Array of vectors 'x' cannot contain y, found x[%" PetscInt_FMT "] == y", i);
      PetscCall(VecLockReadPush(x[i]));
      zeros += alpha[i] == (PetscScalar)0.0;
    }

    if (zeros < nv) {
      PetscManagedInt    nvtmp;
      PetscManagedScalar alphatmp;
      PetscDeviceContext dctx;

      // at least 1 nonzero
      PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
      PetscCall(PetscCopyHostScalar(dctx, (PetscScalar *)alpha, nv, &alphatmp));
      PetscCall(PetscCopyHostInt(dctx, &nv, 1, &nvtmp));
      PetscCall(VecMAXPYAsync(y, nvtmp, alphatmp, x, dctx));
      PetscCall(PetscManagedScalarDestroy(dctx, &alphatmp));
      PetscCall(PetscManagedIntDestroy(dctx, &nvtmp));
    }

    for (PetscInt i = 0; i < nv; ++i) PetscCall(VecLockReadPop(x[i]));
  }
  PetscFunctionReturn(0);
}

/*@
   VecConcatenate - Creates a new vector that is a vertical concatenation of all the given array of vectors
                    in the order they appear in the array. The concatenated vector resides on the same
                    communicator and is the same type as the source vectors.

   Collective on X, Synchronous

   Input Parameters:
+  nx   - number of vectors to be concatenated
-  X    - array containing the vectors to be concatenated in the order of concatenation

   Output Parameters:
+  Y    - concatenated vector
-  x_is - array of index sets corresponding to the concatenated components of Y (NULL if not needed)

   Notes:
   Concatenation is similar to the functionality of a VecNest object; they both represent combination of
   different vector spaces. However, concatenated vectors do not store any information about their
   sub-vectors and own their own data. Consequently, this function provides index sets to enable the
   manipulation of data in the concatenated vector that corresponds to the original components at creation.

   This is a useful tool for outer loop algorithms, particularly constrained optimizers, where the solver
   has to operate on combined vector spaces and cannot utilize VecNest objects due to incompatibility with
   bound projections.

   Level: advanced

.seealso: `VECNEST`, `VECSCATTER`, `VecScatterCreate()`
@*/
PetscErrorCode VecConcatenate(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[]) {
  MPI_Comm comm;
  VecType  vec_type;
  Vec      Ytmp, Xtmp;
  IS      *is_tmp;
  PetscInt i, shift = 0, Xnl, Xng, Xbegin;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(*X, nx, 1);
  PetscValidHeaderSpecific(*X, VEC_CLASSID, 2);
  PetscValidType(*X, 2);
  PetscValidPointer(Y, 3);

  if ((*X)->ops->concatenate) {
    /* use the dedicated concatenation function if available */
    PetscCall((*(*X)->ops->concatenate)(nx, X, Y, x_is));
  } else {
    /* loop over vectors and start creating IS */
    comm = PetscObjectComm((PetscObject)(*X));
    PetscCall(VecGetType(*X, &vec_type));
    PetscCall(PetscMalloc1(nx, &is_tmp));
    for (i = 0; i < nx; i++) {
      PetscCall(VecGetSize(X[i], &Xng));
      PetscCall(VecGetLocalSize(X[i], &Xnl));
      PetscCall(VecGetOwnershipRange(X[i], &Xbegin, NULL));
      PetscCall(ISCreateStride(comm, Xnl, shift + Xbegin, 1, &is_tmp[i]));
      shift += Xng;
    }
    /* create the concatenated vector */
    PetscCall(VecCreate(comm, &Ytmp));
    PetscCall(VecSetType(Ytmp, vec_type));
    PetscCall(VecSetSizes(Ytmp, PETSC_DECIDE, shift));
    PetscCall(VecSetUp(Ytmp));
    /* copy data from X array to Y and return */
    for (i = 0; i < nx; i++) {
      PetscCall(VecGetSubVector(Ytmp, is_tmp[i], &Xtmp));
      PetscCall(VecCopy(X[i], Xtmp));
      PetscCall(VecRestoreSubVector(Ytmp, is_tmp[i], &Xtmp));
    }
    *Y = Ytmp;
    if (x_is) {
      *x_is = is_tmp;
    } else {
      for (i = 0; i < nx; i++) PetscCall(ISDestroy(&is_tmp[i]));
      PetscCall(PetscFree(is_tmp));
    }
  }
  PetscFunctionReturn(0);
}

/* A helper function for VecGetSubVector to check if we can implement it with no-copy (i.e. the subvector shares
   memory with the original vector), and the block size of the subvector.

    Input Parameters:
+   X - the original vector
-   is - the index set of the subvector

    Output Parameters:
+   contig - PETSC_TRUE if the index set refers to contiguous entries on this process, else PETSC_FALSE
.   start  - start of contiguous block, as an offset from the start of the ownership range of the original vector
-   blocksize - the block size of the subvector

*/
PetscErrorCode VecGetSubVectorContiguityAndBS_Private(Vec X, IS is, PetscBool *contig, PetscInt *start, PetscInt *blocksize) {
  PetscInt  gstart, gend, lstart;
  PetscBool red[2] = {PETSC_TRUE /*contiguous*/, PETSC_TRUE /*validVBS*/};
  PetscInt  n, N, ibs, vbs, bs = -1;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetSize(is, &N));
  PetscCall(ISGetBlockSize(is, &ibs));
  PetscCall(VecGetBlockSize(X, &vbs));
  PetscCall(VecGetOwnershipRange(X, &gstart, &gend));
  PetscCall(ISContiguousLocal(is, gstart, gend, &lstart, &red[0]));
  /* block size is given by IS if ibs > 1; otherwise, check the vector */
  if (ibs > 1) {
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, red, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)is)));
    bs = ibs;
  } else {
    if (n % vbs || vbs == 1) red[1] = PETSC_FALSE; /* this process invalidate the collectiveness of block size */
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, red, 2, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)is)));
    if (red[0] && red[1]) bs = vbs; /* all processes have a valid block size and the access will be contiguous */
  }

  *contig    = red[0];
  *start     = lstart;
  *blocksize = bs;
  PetscFunctionReturn(0);
}

/* A helper function for VecGetSubVector, to be used when we have to build a standalone subvector through VecScatter

    Input Parameters:
+   X - the original vector
.   is - the index set of the subvector
-   bs - the block size of the subvector, gotten from VecGetSubVectorContiguityAndBS_Private()

    Output Parameters:
.   Z  - the subvector, which will compose the VecScatter context on output
*/
PetscErrorCode VecGetSubVectorThroughVecScatter_Private(Vec X, IS is, PetscInt bs, Vec *Z) {
  PetscInt   n, N;
  VecScatter vscat;
  Vec        Y;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetSize(is, &N));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)is), &Y));
  PetscCall(VecSetSizes(Y, n, N));
  PetscCall(VecSetBlockSize(Y, bs));
  PetscCall(VecSetType(Y, ((PetscObject)X)->type_name));
  PetscCall(VecScatterCreate(X, is, Y, NULL, &vscat));
  PetscCall(VecScatterBegin(vscat, X, Y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat, X, Y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(PetscObjectCompose((PetscObject)Y, "VecGetSubVector_Scatter", (PetscObject)vscat));
  PetscCall(VecScatterDestroy(&vscat));
  *Z = Y;
  PetscFunctionReturn(0);
}

/*@
   VecGetSubVector - Gets a vector representing part of another vector

   Collective on X and IS, Synchronous

   Input Parameters:
+ X - vector from which to extract a subvector
- is - index set representing portion of X to extract

   Output Parameter:
. Y - subvector corresponding to is

   Level: advanced

   Notes:
   The subvector Y should be returned with VecRestoreSubVector().
   X and is must be defined on the same communicator

   This function may return a subvector without making a copy, therefore it is not safe to use the original vector while
   modifying the subvector.  Other non-overlapping subvectors can still be obtained from X using this function.
   The resulting subvector inherits the block size from the IS if greater than one. Otherwise, the block size is guessed from the block size of the original vec.

.seealso: `MatCreateSubMatrix()`
@*/
PetscErrorCode VecGetSubVector(Vec X, IS is, Vec *Y) {
  Vec Z;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscCheckSameComm(X, 1, is, 2);
  PetscValidPointer(Y, 3);
  if (X->ops->getsubvector) {
    PetscUseTypeMethod(X, getsubvector, is, &Z);
  } else { /* Default implementation currently does no caching */
    PetscBool contig;
    PetscInt  n, N, start, bs;

    PetscCall(ISGetLocalSize(is, &n));
    PetscCall(ISGetSize(is, &N));
    PetscCall(VecGetSubVectorContiguityAndBS_Private(X, is, &contig, &start, &bs));
    if (contig) { /* We can do a no-copy implementation */
      const PetscScalar *x;
      PetscMPIInt        size;
      PetscInt           state = 0;
      PetscBool          isstd, iscuda, iship;
      PetscDeviceContext dctx;

      PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &isstd, VECSEQ, VECMPI, VECSTANDARD, ""));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iscuda, VECSEQCUDA, VECMPICUDA, ""));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iship, VECSEQHIP, VECMPIHIP, ""));
      if (iscuda || iship || isstd) {
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)X), &size));
        if (iscuda || iship) PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
      }
      if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
        const PetscScalar *x_d;
        PetscOffloadMask   flg;

        PetscCall(VecCUDAGetArrays_Private(X, &x, &x_d, &flg, dctx));
        PetscCheck(flg != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for PETSC_OFFLOAD_UNALLOCATED");
        PetscCheck(!n || x || x_d, PETSC_COMM_SELF, PETSC_ERR_SUP, "Missing vector data");
        if (x) x += start;
        if (x_d) x_d += start;
        if (size == 1) {
          PetscCall(VecCreateSeqCUDAWithArrays(PetscObjectComm((PetscObject)X), bs, n, x, x_d, &Z));
        } else {
          PetscCall(VecCreateMPICUDAWithArrays(PetscObjectComm((PetscObject)X), bs, n, N, x, x_d, &Z));
        }
        Z->offloadmask = flg;
#endif
      } else if (iship) {
#if defined(PETSC_HAVE_HIP)
        const PetscScalar *x_d;
        PetscOffloadMask   flg;

        PetscCall(VecHIPGetArrays_Private(X, &x, &x_d, &flg, dctx));
        PetscCheck(flg != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for PETSC_OFFLOAD_UNALLOCATED");
        PetscCheck(!n || x || x_d, PETSC_COMM_SELF, PETSC_ERR_SUP, "Missing vector data");
        if (x) x += start;
        if (x_d) x_d += start;
        if (size == 1) {
          PetscCall(VecCreateSeqHIPWithArrays(PetscObjectComm((PetscObject)X), bs, n, x, x_d, &Z));
        } else {
          PetscCall(VecCreateMPIHIPWithArrays(PetscObjectComm((PetscObject)X), bs, n, N, x, x_d, &Z));
        }
        Z->offloadmask = flg;
#endif
      } else if (isstd) {
        PetscCall(VecGetArrayRead(X, &x));
        if (x) x += start;
        if (size == 1) {
          PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)X), bs, n, x, &Z));
        } else {
          PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)X), bs, n, N, x, &Z));
        }
        PetscCall(VecRestoreArrayRead(X, &x));
      } else { /* default implementation: use place array */
        PetscCall(VecGetArrayRead(X, &x));
        PetscCall(VecCreate(PetscObjectComm((PetscObject)X), &Z));
        PetscCall(VecSetType(Z, ((PetscObject)X)->type_name));
        PetscCall(VecSetSizes(Z, n, N));
        PetscCall(VecSetBlockSize(Z, bs));
        PetscCall(VecPlaceArray(Z, x ? x + start : NULL));
        PetscCall(VecRestoreArrayRead(X, &x));
      }

      /* this is relevant only in debug mode */
      PetscCall(VecLockGet(X, &state));
      if (state) PetscCall(VecLockReadPush(Z));
      Z->ops->placearray   = NULL;
      Z->ops->replacearray = NULL;
    } else { /* Have to create a scatter and do a copy */
      PetscCall(VecGetSubVectorThroughVecScatter_Private(X, is, bs, &Z));
    }
  }
  /* Record the state when the subvector was gotten so we know whether its values need to be put back */
  if (VecGetSubVectorSavedStateId < 0) PetscCall(PetscObjectComposedDataRegister(&VecGetSubVectorSavedStateId));
  PetscCall(PetscObjectComposedDataSetInt((PetscObject)Z, VecGetSubVectorSavedStateId, 1));
  *Y = Z;
  PetscFunctionReturn(0);
}

/*@
   VecRestoreSubVector - Restores a subvector extracted using VecGetSubVector()

   Collective on IS, Synchronous

   Input Parameters:
+ X - vector from which subvector was obtained
. is - index set representing the subset of X
- Y - subvector being restored

   Level: advanced

.seealso: `VecGetSubVector()`
@*/
PetscErrorCode VecRestoreSubVector(Vec X, IS is, Vec *Y) {
  PETSC_UNUSED PetscObjectState dummystate = 0;
  PetscBool                     unchanged;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscCheckSameComm(X, 1, is, 2);
  PetscValidPointer(Y, 3);
  PetscValidHeaderSpecific(*Y, VEC_CLASSID, 3);

  if (X->ops->restoresubvector) PetscUseTypeMethod(X, restoresubvector, is, Y);
  else {
    PetscCall(PetscObjectComposedDataGetInt((PetscObject)*Y, VecGetSubVectorSavedStateId, dummystate, unchanged));
    if (!unchanged) { /* If Y's state has not changed since VecGetSubVector(), we only need to destroy Y */
      VecScatter scatter;
      PetscInt   state;

      PetscCall(VecLockGet(X, &state));
      PetscCheck(state == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vec X is locked for read-only or read/write access");

      PetscCall(PetscObjectQuery((PetscObject)*Y, "VecGetSubVector_Scatter", (PetscObject *)&scatter));
      if (scatter) {
        PetscCall(VecScatterBegin(scatter, *Y, X, INSERT_VALUES, SCATTER_REVERSE));
        PetscCall(VecScatterEnd(scatter, *Y, X, INSERT_VALUES, SCATTER_REVERSE));
      } else {
        PetscBool iscuda, iship;
        PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iscuda, VECSEQCUDA, VECMPICUDA, ""));
        PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iship, VECSEQHIP, VECMPIHIP, ""));

        if (iscuda) {
#if defined(PETSC_HAVE_CUDA)
          PetscOffloadMask ymask = (*Y)->offloadmask;

          /* The offloadmask of X dictates where to move memory
              If X GPU data is valid, then move Y data on GPU if needed
              Otherwise, move back to the CPU */
          switch (X->offloadmask) {
          case PETSC_OFFLOAD_BOTH:
            if (ymask == PETSC_OFFLOAD_CPU) {
              PetscCall(VecCUDAResetArray(*Y));
            } else if (ymask == PETSC_OFFLOAD_GPU) {
              X->offloadmask = PETSC_OFFLOAD_GPU;
            }
            break;
          case PETSC_OFFLOAD_GPU:
            if (ymask == PETSC_OFFLOAD_CPU) PetscCall(VecCUDAResetArray(*Y));
            break;
          case PETSC_OFFLOAD_CPU:
            if (ymask == PETSC_OFFLOAD_GPU) PetscCall(VecResetArray(*Y));
            break;
          case PETSC_OFFLOAD_UNALLOCATED:
          case PETSC_OFFLOAD_KOKKOS: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This should not happen");
          }
#endif
        } else if (iship) {
#if defined(PETSC_HAVE_HIP)
          PetscOffloadMask ymask = (*Y)->offloadmask;

          /* The offloadmask of X dictates where to move memory
              If X GPU data is valid, then move Y data on GPU if needed
              Otherwise, move back to the CPU */
          switch (X->offloadmask) {
          case PETSC_OFFLOAD_BOTH:
            if (ymask == PETSC_OFFLOAD_CPU) {
              PetscCall(VecHIPResetArray(*Y));
            } else if (ymask == PETSC_OFFLOAD_GPU) {
              X->offloadmask = PETSC_OFFLOAD_GPU;
            }
            break;
          case PETSC_OFFLOAD_GPU:
            if (ymask == PETSC_OFFLOAD_CPU) PetscCall(VecHIPResetArray(*Y));
            break;
          case PETSC_OFFLOAD_CPU:
            if (ymask == PETSC_OFFLOAD_GPU) PetscCall(VecResetArray(*Y));
            break;
          case PETSC_OFFLOAD_UNALLOCATED:
          case PETSC_OFFLOAD_KOKKOS: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This should not happen");
          }
#endif
        } else {
          /* If OpenCL vecs updated the device memory, this triggers a copy on the CPU */
          PetscCall(VecResetArray(*Y));
        }
        PetscCall(PetscObjectStateIncrease((PetscObject)X));
      }
    }
  }
  PetscCall(VecDestroy(Y));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVectorReadAsync(Vec v, Vec w, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 2);
  VecCheckSameLocalSize(v, 1, w, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (v->ops->getlocalvectorread) {
    PetscUseTypeMethod(v, getlocalvectorread, w, dctx);
  } else {
    PetscScalar *a;

    PetscCall(VecGetArrayReadAsync(v, (const PetscScalar **)&a, dctx));
    PetscCall(VecPlaceArrayAsync(w, a, dctx));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscCall(VecLockReadPush(v));
  PetscCall(VecLockReadPush(w));
  PetscFunctionReturn(0);
}

/*@
   VecGetLocalVectorRead - Maps the local portion of a vector into a
   vector.  You must call VecRestoreLocalVectorRead() when the local
   vector is no longer needed.

   Not collective, Synchronous

   Input parameter:
.  v - The vector for which the local vector is desired.

   Output parameter:
.  w - Upon exit this contains the local vector.

   Level: beginner

   Notes:
   This function is similar to VecGetArrayRead() which maps the local
   portion into a raw pointer.  VecGetLocalVectorRead() is usually
   almost as efficient as VecGetArrayRead() but in certain circumstances
   VecGetLocalVectorRead() can be much more efficient than
   VecGetArrayRead().  This is because the construction of a contiguous
   array representing the vector data required by VecGetArrayRead() can
   be an expensive operation for certain vector types.  For example, for
   GPU vectors VecGetArrayRead() requires that the data between device
   and host is synchronized.

   Unlike VecGetLocalVector(), this routine is not collective and
   preserves cached information.

.seealso: `VecRestoreLocalVectorRead()`, `VecGetLocalVector()`, `VecGetArrayRead()`, `VecGetArray()`
@*/
PetscErrorCode VecGetLocalVectorRead(Vec v, Vec w) {
  PetscFunctionBegin;
  PetscCall(VecGetLocalVectorReadAsync(v, w, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVectorReadAsync(Vec v, Vec w, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 2);
  VecCheckSameLocalSize(v, 1, w, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (v->ops->restorelocalvectorread) {
    PetscUseTypeMethod(v, restorelocalvectorread, w, dctx);
  } else {
    PetscScalar *a;

    PetscCall(VecGetArrayReadAsync(w, (const PetscScalar **)&a, dctx));
    PetscCall(VecRestoreArrayReadAsync(v, (const PetscScalar **)&a, dctx));
    PetscCall(VecResetArrayAsync(w, dctx));
  }
  PetscCall(VecLockReadPop(v));
  PetscCall(VecLockReadPop(w));
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(0);
}

/*@
   VecRestoreLocalVectorRead - Unmaps the local portion of a vector
   previously mapped into a vector using VecGetLocalVectorRead().

   Not collective, Synchronous

   Input parameter:
+  v - The local portion of this vector was previously mapped into w using VecGetLocalVectorRead().
-  w - The vector into which the local portion of v was mapped.

   Level: beginner

.seealso: `VecGetLocalVectorRead()`, `VecGetLocalVector()`, `VecGetArrayRead()`, `VecGetArray()`
@*/
PetscErrorCode VecRestoreLocalVectorRead(Vec v, Vec w) {
  PetscFunctionBegin;
  PetscCall(VecRestoreLocalVectorReadAsync(v, w, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVectorAsync(Vec v, Vec w, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 2);
  VecCheckSameLocalSize(v, 1, w, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (v->ops->getlocalvector) {
    PetscUseTypeMethod(v, getlocalvector, w, dctx);
  } else {
    PetscScalar *a;

    PetscCall(VecGetArrayAsync(v, &a, dctx));
    PetscCall(VecPlaceArrayAsync(w, a, dctx));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(0);
}

/*@
   VecGetLocalVector - Maps the local portion of a vector into a
   vector.

   Collective on v, not collective on w, Synchronous

   Input parameter:
.  v - The vector for which the local vector is desired.

   Output parameter:
.  w - Upon exit this contains the local vector.

   Level: beginner

   Notes:
   This function is similar to VecGetArray() which maps the local
   portion into a raw pointer.  VecGetLocalVector() is usually about as
   efficient as VecGetArray() but in certain circumstances
   VecGetLocalVector() can be much more efficient than VecGetArray().
   This is because the construction of a contiguous array representing
   the vector data required by VecGetArray() can be an expensive
   operation for certain vector types.  For example, for GPU vectors
   VecGetArray() requires that the data between device and host is
   synchronized.

.seealso: `VecRestoreLocalVector()`, `VecGetLocalVectorRead()`, `VecGetArrayRead()`, `VecGetArray()`
@*/
PetscErrorCode VecGetLocalVector(Vec v, Vec w) {
  PetscFunctionBegin;
  PetscCall(VecGetLocalVectorAsync(v, w, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVectorAsync(Vec v, Vec w, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 2);
  VecCheckSameLocalSize(v, 1, w, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (v->ops->restorelocalvector) {
    PetscUseTypeMethod(v, restorelocalvector, w, dctx);
  } else {
    PetscScalar *a;

    PetscCall(VecGetArrayAsync(w, &a, dctx));
    PetscCall(VecRestoreArrayAsync(v, &a, dctx));
    PetscCall(VecResetArrayAsync(w, dctx));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(0);
}

/*@
   VecRestoreLocalVector - Unmaps the local portion of a vector
   previously mapped into a vector using VecGetLocalVector().

   Logically collective, Synchronous

   Input parameter:
+  v - The local portion of this vector was previously mapped into w using VecGetLocalVector().
-  w - The vector into which the local portion of v was mapped.

   Level: beginner

.seealso: `VecGetLocalVector()`, `VecGetLocalVectorRead()`, `VecRestoreLocalVectorRead()`, `LocalVectorRead()`, `VecGetArrayRead()`, `VecGetArray()`
@*/
PetscErrorCode VecRestoreLocalVector(Vec v, Vec w) {
  PetscFunctionBegin;
  PetscCall(VecRestoreLocalVectorAsync(v, w, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayAsync(Vec x, PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCall(VecSetErrorIfLocked(x, 1));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, ((PetscObject)x)->id, PETSC_MEMORY_ACCESS_READ_WRITE, ((PetscObject)x)->name));
  if (x->ops->getarray) {
    /* The if-else order matters! VECNEST, VECCUDA etc should have ops->getarray while VECCUDA
       etc are petscnative */
    PetscUseTypeMethod(x, getarray, a, dctx);
  } else if (x->petscnative) { /* VECSTANDARD */
    *a = *((PetscScalar **)x->data);
  } else SETERRQ(PetscObjectComm((PetscObject)x), PETSC_ERR_SUP, "Cannot get array for vector type \"%s\"", ((PetscObject)x)->type_name);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray - Returns a pointer to a contiguous array that contains this
   processor's portion of the vector data. For the standard PETSc
   vectors, VecGetArray() returns a pointer to the local data array and
   does not use any copies. If the underlying vector data is not stored
   in a contiguous array this routine will copy the data to a contiguous
   array and return a pointer to that. You MUST call VecRestoreArray()
   when you no longer need access to the array.

   Logically Collective on Vec, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   This routine is used differently from Fortran 77
$    Vec         x
$    PetscScalar x_array(1)
$    PetscOffset i_x
$    PetscErrorCode ierr
$       call VecGetArray(x,x_array,i_x,ierr)
$
$   Access first local entry in vector with
$      value = x_array(i_x + 1)
$
$      ...... other code
$       call VecRestoreArray(x,x_array,i_x,ierr)
   For Fortran 90 see VecGetArrayF90()

   See the Fortran chapter of the users manual and
   petsc/src/snes/tutorials/ex5f.F for details.

   Level: beginner

.seealso: `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`, `VecPlaceArray()`, `VecGetArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`, `VecRestoreArrayWrite()`
@*/
PetscErrorCode VecGetArray(Vec x, PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecGetArrayAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArrayAsync(Vec x, PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (a) PetscValidPointer(a, 2);
  if (x->ops->restorearray) { /* VECNEST, VECCUDA etc */
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    PetscUseTypeMethod(x, restorearray, a, dctx);
  } else if (x->petscnative) { /* VECSTANDARD */
    /* nothing */
  } else SETERRQ(PetscObjectComm((PetscObject)x), PETSC_ERR_SUP, "Cannot restore array for vector type \"%s\"", ((PetscObject)x)->type_name);
  if (a) *a = NULL;
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray - Restores a vector after VecGetArray() has been called.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from VecGetArray()

   Level: beginner

.seealso: `VecGetArray()`, `VecRestoreArrayRead()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecRestoreArrayReadF90()`, `VecPlaceArray()`, `VecRestoreArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArray(Vec x, PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayReadAsync(Vec x, const PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, ((PetscObject)x)->id, PETSC_MEMORY_ACCESS_READ, ((PetscObject)x)->name));
  if (x->ops->getarrayread) {
    PetscUseTypeMethod(x, getarrayread, a, dctx);
  } else if (x->ops->getarray) {
    /* VECNEST, VECCUDA, VECKOKKOS etc */
    PetscUseTypeMethod(x, getarray, (PetscScalar **)a, dctx);
  } else if (x->petscnative) {
    /* VECSTANDARD */
    *a = *((PetscScalar **)x->data);
  } else SETERRQ(PetscObjectComm((PetscObject)x), PETSC_ERR_SUP, "Cannot get array read for vector type \"%s\"", ((PetscObject)x)->type_name);
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrayRead - Get read-only pointer to contiguous array containing this processor's portion of the vector data.

   Not Collective, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - the array

   Level: beginner

   Notes:
   The array must be returned using a matching call to VecRestoreArrayRead().

   Unlike VecGetArray(), this routine is not collective and preserves cached information like vector norms.

   Standard PETSc vectors use contiguous storage so that this routine does not perform a copy.  Other vector
   implementations may require a copy, but must such implementations should cache the contiguous representation so that
   only one copy is performed when this routine is called multiple times in sequence.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecGetArrayReadAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArrayReadAsync(Vec x, const PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (a) PetscValidPointer(a, 2);
  if (x->petscnative) { /* VECSTANDARD, VECCUDA, VECKOKKOS etc */
    /* nothing */
  } else if (x->ops->restorearrayread) { /* VECNEST */
    PetscUseTypeMethod(x, restorearrayread, a, dctx);
  } else { /* No one? */
    PetscUseTypeMethod(x, restorearray, (PetscScalar **)a, dctx);
  }
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayRead - Restore array obtained with VecGetArrayRead()

   Not Collective, Synchronous

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: beginner

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArrayRead(Vec x, const PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayReadAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayWriteAsync(Vec x, PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 2);
  PetscCall(VecSetErrorIfLocked(x, 1));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (x->ops->getarraywrite) {
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx, ((PetscObject)x)->id, PETSC_MEMORY_ACCESS_WRITE, ((PetscObject)x)->name));
    PetscUseTypeMethod(x, getarraywrite, a, dctx);
  } else {
    PetscCall(VecGetArrayAsync(x, a, dctx));
  }
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrayWrite - Returns a pointer to a contiguous array that WILL contains this
   processor's portion of the vector data. The values in this array are NOT valid, the routine calling this
   routine is responsible for putting values into the array; any values it does not set will be invalid

   Logically Collective on Vec, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameter:
.  a - location to put pointer to the array

   Level: intermediate

   This is for vectors associate with GPUs, the vector is not copied up before giving access. If you need correct
   values in the array use VecGetArray()

.seealso: `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`, `VecPlaceArray()`, `VecGetArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArray()`, `VecRestoreArrayWrite()`
@*/
PetscErrorCode VecGetArrayWrite(Vec x, PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecGetArrayWriteAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArrayWriteAsync(Vec x, PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (a) PetscValidPointer(a, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (x->ops->restorearraywrite) {
    PetscUseTypeMethod(x, restorearraywrite, a, dctx);
  } else if (x->ops->restorearray) {
    PetscUseTypeMethod(x, restorearray, a, dctx);
  }
  if (a) *a = NULL;
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayWrite - Restores a vector after VecGetArrayWrite() has been called.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from VecGetArray()

   Level: beginner

.seealso: `VecGetArray()`, `VecRestoreArrayRead()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecRestoreArrayReadF90()`, `VecPlaceArray()`, `VecRestoreArray2d()`,
          `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`
@*/
PetscErrorCode VecRestoreArrayWrite(Vec x, PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayWriteAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrays - Returns a pointer to the arrays in a set of vectors
   that were created by a call to VecDuplicateVecs().  You MUST call
   VecRestoreArrays() when you no longer need access to the array.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  x - the vectors
-  n - the number of vectors

   Output Parameter:
.  a - location to put pointer to the array

   Fortran Note:
   This routine is not supported in Fortran.

   Level: intermediate

.seealso: `VecGetArray()`, `VecRestoreArrays()`
@*/
PetscErrorCode VecGetArrays(const Vec x[], PetscInt n, PetscScalar **a[]) {
  PetscInt      i;
  PetscScalar **q;

  PetscFunctionBegin;
  PetscValidPointer(x, 1);
  PetscValidHeaderSpecific(*x, VEC_CLASSID, 1);
  PetscValidPointer(a, 3);
  PetscCheck(n > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must get at least one array n = %" PetscInt_FMT, n);
  PetscCall(PetscMalloc1(n, &q));
  for (i = 0; i < n; ++i) PetscCall(VecGetArray(x[i], &q[i]));
  *a = q;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrays - Restores a group of vectors after VecGetArrays()
   has been called.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  x - the vector
.  n - the number of vectors
-  a - location of pointer to arrays obtained from VecGetArrays()

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the arrays obtained with VecGetArrays().

   Fortran Note:
   This routine is not supported in Fortran.

   Level: intermediate

.seealso: `VecGetArrays()`, `VecRestoreArray()`
@*/
PetscErrorCode VecRestoreArrays(const Vec x[], PetscInt n, PetscScalar **a[]) {
  PetscInt      i;
  PetscScalar **q = *a;

  PetscFunctionBegin;
  PetscValidPointer(x, 1);
  PetscValidHeaderSpecific(*x, VEC_CLASSID, 1);
  PetscValidPointer(a, 3);

  for (i = 0; i < n; ++i) PetscCall(VecRestoreArray(x[i], &q[i]));
  PetscCall(PetscFree(q));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayAndMemTypeAsync(Vec x, PetscScalar **a, PetscMemType *mtype, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidPointer(a, 2);
  if (mtype) PetscValidPointer(mtype, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (x->ops->getarrayandmemtype) { /* VECCUDA, VECKOKKOS etc */
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx, ((PetscObject)x)->id, PETSC_MEMORY_ACCESS_READ_WRITE, ((PetscObject)x)->name));
    PetscUseTypeMethod(x, getarrayandmemtype, a, mtype, dctx);
  } else { /* VECSTANDARD, VECNEST, VECVIENNACL */
    PetscCall(VecGetArrayAsync(x, a, dctx));
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrayAndMemType - Like VecGetArray(), but if this is a standard device vector (e.g., VECCUDA), the returned pointer will be a device
   pointer to the device memory that contains this processor's portion of the vector data. Device data is guaranteed to have the latest value.
   Otherwise, when this is a host vector (e.g., VECMPI), this routine functions the same as VecGetArray() and returns a host pointer.

   For VECKOKKOS, if Kokkos is configured without device (e.g., use serial or openmp), per this function, the vector works like VECSEQ/VECMPI;
   otherwise, it works like VECCUDA or VECHIP etc.

   Logically Collective on Vec, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameters:
+  a - location to put pointer to the array
-  mtype - memory type of the array

   Level: beginner

.seealso: `VecRestoreArrayAndMemType()`, `VecGetArrayReadAndMemType()`, `VecGetArrayWriteAndMemType()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecGetArrayReadF90()`,
          `VecPlaceArray()`, `VecGetArray2d()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayWrite()`, `VecRestoreArrayWrite()`
@*/
PetscErrorCode VecGetArrayAndMemType(Vec x, PetscScalar **a, PetscMemType *mtype) {
  PetscFunctionBegin;
  PetscCall(VecGetArrayAndMemTypeAsync(x, a, mtype, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArrayAndMemTypeAsync(Vec x, PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  if (a) PetscValidPointer(a, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (x->ops->restorearrayandmemtype) {
    /* VECCUDA, VECKOKKOS etc */
    PetscUseTypeMethod(x, restorearrayandmemtype, a, dctx);
  } else {
    /* VECNEST, VECVIENNACL */
    PetscCall(VecRestoreArrayAsync(x, a, dctx));
  } /* VECSTANDARD does nothing */
  if (a) *a = NULL;
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayAndMemType - Restores a vector after VecGetArrayAndMemType() has been called.

   Logically Collective on Vec, Synchronous

   Input Parameters:
+  x - the vector
-  a - location of pointer to array obtained from VecGetArrayAndMemType()

   Level: beginner

.seealso: `VecGetArrayAndMemType()`, `VecGetArray()`, `VecRestoreArrayRead()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecRestoreArrayReadF90()`,
          `VecPlaceArray()`, `VecRestoreArray2d()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArrayAndMemType(Vec x, PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayAndMemTypeAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayReadAndMemTypeAsync(Vec x, const PetscScalar **a, PetscMemType *mtype, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscValidPointer(a, 2);
  if (mtype) PetscValidPointer(mtype, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, ((PetscObject)x)->id, PETSC_MEMORY_ACCESS_READ, ((PetscObject)x)->name));
  if (x->ops->getarrayreadandmemtype) {
    /* VECCUDA/VECHIP though they are also petscnative */
    PetscUseTypeMethod(x, getarrayreadandmemtype, a, mtype, dctx);
  } else if (x->ops->getarrayandmemtype) {
    /* VECKOKKOS */
    PetscUseTypeMethod(x, getarrayandmemtype, (PetscScalar **)a, mtype, dctx);
  } else {
    PetscCall(VecGetArrayReadAsync(x, a, dctx));
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrayReadAndMemType - Like VecGetArrayRead(), but if the input vector is a device vector, it will return a read-only device pointer. The returned pointer is guarenteed to point to up-to-date data. For host vectors, it functions as VecGetArrayRead().

   Not Collective, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameters:
+  a - the array
-  mtype - memory type of the array

   Level: beginner

   Notes:
   The array must be returned using a matching call to VecRestoreArrayReadAndMemType().

.seealso: `VecRestoreArrayReadAndMemType()`, `VecGetArrayAndMemType()`, `VecGetArrayWriteAndMemType()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`, `VecGetArrayAndMemType()`
@*/
PetscErrorCode VecGetArrayReadAndMemType(Vec x, const PetscScalar **a, PetscMemType *mtype) {
  PetscFunctionBegin;
  PetscCall(VecGetArrayReadAndMemTypeAsync(x, a, mtype, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArrayReadAndMemTypeAsync(Vec x, const PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  if (a) PetscValidPointer(a, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (x->ops->restorearrayreadandmemtype) {
    /* VECCUDA/VECHIP */
    PetscUseTypeMethod(x, restorearrayreadandmemtype, a, dctx);
  } else if (x->petscnative) {
    /* VECSTANDARD, VECKOKKOS, VECVIENNACL etc */
    /* nothing */
  } else {
    /* VECNEST */
    PetscCall(VecRestoreArrayReadAsync(x, a, dctx));
  }
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayReadAndMemType - Restore array obtained with VecGetArrayReadAndMemType()

   Not Collective, Synchronous

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: beginner

.seealso: `VecGetArrayReadAndMemType()`, `VecRestoreArrayAndMemType()`, `VecRestoreArrayWriteAndMemType()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArrayReadAndMemType(Vec x, const PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayReadAndMemTypeAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayWriteAndMemTypeAsync(Vec x, PetscScalar **a, PetscMemType *mtype, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  PetscValidPointer(a, 2);
  if (mtype) PetscValidPointer(mtype, 3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (x->ops->getarraywriteandmemtype) {
    /* VECCUDA, VECHIP, VECKOKKOS etc, though they are also petscnative */
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx, ((PetscObject)x)->id, PETSC_MEMORY_ACCESS_WRITE, ((PetscObject)x)->name));
    PetscUseTypeMethod(x, getarraywriteandmemtype, a, mtype, dctx);
  } else if (x->ops->getarrayandmemtype) {
    PetscCall(VecGetArrayAndMemTypeAsync(x, a, mtype, dctx));
  } else {
    /* VECNEST, VECVIENNACL */
    PetscCall(VecGetArrayWriteAsync(x, a, dctx));
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(0);
}

/*@C
   VecGetArrayWriteAndMemType - Like VecGetArrayWrite(), but if this is a device vector it will aways return
    a device pointer to the device memory that contains this processor's portion of the vector data.

   Not Collective, Synchronous

   Input Parameter:
.  x - the vector

   Output Parameters:
+  a - the array
-  mtype - memory type of the array

   Level: beginner

   Notes:
   The array must be returned using a matching call to VecRestoreArrayWriteAndMemType(), where it will label the device memory as most recent.

.seealso: `VecRestoreArrayWriteAndMemType()`, `VecGetArrayReadAndMemType()`, `VecGetArrayAndMemType()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`,
@*/
PetscErrorCode VecGetArrayWriteAndMemType(Vec x, PetscScalar **a, PetscMemType *mtype) {
  PetscFunctionBegin;
  PetscCall(VecGetArrayWriteAndMemTypeAsync(x, a, mtype, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArrayWriteAndMemTypeAsync(Vec x, PetscScalar **a, PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (a) PetscValidPointer(a, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (x->ops->restorearraywriteandmemtype) {
    /* VECCUDA/VECHIP */
    PetscMemType PETSC_UNUSED mtype; // since this function doesn't accept a memtype?
    PetscUseTypeMethod(x, restorearraywriteandmemtype, a, &mtype, dctx);
  } else if (x->ops->restorearrayandmemtype) {
    PetscCall(VecRestoreArrayAndMemTypeAsync(x, a, dctx));
  } else {
    PetscCall(VecRestoreArrayAsync(x, a, dctx));
  }
  if (a) *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArrayWriteAndMemType - Restore array obtained with VecGetArrayWriteAndMemType()

   Not Collective, Synchronous

   Input Parameters:
+  vec - the vector
-  array - the array

   Level: beginner

.seealso: `VecGetArrayWriteAndMemType()`, `VecRestoreArrayAndMemType()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayPair()`, `VecRestoreArrayPair()`
@*/
PetscErrorCode VecRestoreArrayWriteAndMemType(Vec x, PetscScalar **a) {
  PetscFunctionBegin;
  PetscCall(VecRestoreArrayWriteAndMemTypeAsync(x, a, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArrayAsync(Vec vec, const PetscScalar array[], PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidType(vec, 1);
  if (array) PetscValidScalarPointer(array, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscUseTypeMethod(vec, placearray, array, dctx);
  PetscCall(PetscObjectStateIncrease((PetscObject)vec));
  PetscFunctionReturn(0);
}

/*@
   VecPlaceArray - Allows one to replace the array in a vector with an
   array provided by the user. This is useful to avoid copying an array
   into a vector.

   Not Collective, Synchronous

   Input Parameters:
+  vec - the vector
-  array - the array

   Notes:
   You can return to the original array with a call to `VecResetArray()`. `vec` does not take
   ownership of `array` in any way. The user must free `array` themselves but be careful not to
   do so before the vector has either been destroyed, had its original array restored with
   `VecResetArray()` or permanently replaced with `VecReplaceArray()`.

   Level: developer

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecReplaceArray()`, `VecResetArray()`

@*/
PetscErrorCode VecPlaceArray(Vec vec, const PetscScalar array[]) {
  PetscFunctionBegin;
  PetscCall(VecPlaceArrayAsync(vec, array, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArrayAsync(Vec vec, const PetscScalar array[], PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidType(vec, 1);
  PetscValidScalarPointer(array, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscUseTypeMethod(vec, replacearray, array, dctx);
  PetscCall(PetscObjectStateIncrease((PetscObject)vec));
  PetscFunctionReturn(0);
}

/*@C
   VecReplaceArray - Allows one to replace the array in a vector with an
   array provided by the user. This is useful to avoid copying an array
   into a vector.

   Not Collective, Synchronous

   Input Parameters:
+  vec - the vector
-  array - the array

   Notes:
   This permanently replaces the array and frees the memory associated
   with the old array.

   The memory passed in MUST be obtained with PetscMalloc() and CANNOT be
   freed by the user. It will be freed when the vector is destroyed.

   Not supported from Fortran

   Level: developer

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecPlaceArray()`, `VecResetArray()`

@*/
PetscErrorCode VecReplaceArray(Vec vec, const PetscScalar array[]) {
  PetscFunctionBegin;
  PetscCall(VecReplaceArrayAsync(vec, array, NULL));
  PetscFunctionReturn(0);
}

/*MC
    VecDuplicateVecsF90 - Creates several vectors of the same type as an existing vector
    and makes them accessible via a Fortran90 pointer.

    Synopsis:
    VecDuplicateVecsF90(Vec x,PetscInt n,{Vec, pointer :: y(:)},integer ierr)

    Collective on Vec

    Input Parameters:
+   x - a vector to mimic
-   n - the number of vectors to obtain

    Output Parameters:
+   y - Fortran90 pointer to the array of vectors
-   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    Vec x
    Vec, pointer :: y(:)
    ....
    call VecDuplicateVecsF90(x,2,y,ierr)
    call VecSet(y(2),alpha,ierr)
    call VecSet(y(2),alpha,ierr)
    ....
    call VecDestroyVecsF90(2,y,ierr)
.ve

    Notes:
    Not yet supported for all F90 compilers

    Use VecDestroyVecsF90() to free the space.

    Level: beginner

.seealso: `VecDestroyVecsF90()`, `VecDuplicateVecs()`

M*/

/*MC
    VecRestoreArrayF90 - Restores a vector to a usable state after a call to
    VecGetArrayF90().

    Synopsis:
    VecRestoreArrayF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective on Vec

    Input Parameters:
+   x - vector
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayF90(x,xx_v,ierr)
    xx_v(3) = a
    call VecRestoreArrayF90(x,xx_v,ierr)
.ve

    Level: beginner

.seealso: `VecGetArrayF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrayReadF90()`

M*/

/*MC
    VecDestroyVecsF90 - Frees a block of vectors obtained with VecDuplicateVecsF90().

    Synopsis:
    VecDestroyVecsF90(PetscInt n,{Vec, pointer :: x(:)},PetscErrorCode ierr)

    Collective on Vec

    Input Parameters:
+   n - the number of vectors previously obtained
-   x - pointer to array of vector pointers

    Output Parameter:
.   ierr - error code

    Notes:
    Not yet supported for all F90 compilers

    Level: beginner

.seealso: `VecDestroyVecs()`, `VecDuplicateVecsF90()`

M*/

/*MC
    VecGetArrayF90 - Accesses a vector array from Fortran90. For default PETSc
    vectors, VecGetArrayF90() returns a pointer to the local data array. Otherwise,
    this routine is implementation dependent. You MUST call VecRestoreArrayF90()
    when you no longer need access to the array.

    Synopsis:
    VecGetArrayF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective on Vec

    Input Parameter:
.   x - vector

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayF90(x,xx_v,ierr)
    xx_v(3) = a
    call VecRestoreArrayF90(x,xx_v,ierr)
.ve

    If you ONLY intend to read entries from the array and not change any entries you should use VecGetArrayReadF90().

    Level: beginner

.seealso: `VecRestoreArrayF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayReadF90()`

M*/

/*MC
    VecGetArrayReadF90 - Accesses a read only array from Fortran90. For default PETSc
    vectors, VecGetArrayF90() returns a pointer to the local data array. Otherwise,
    this routine is implementation dependent. You MUST call VecRestoreArrayReadF90()
    when you no longer need access to the array.

    Synopsis:
    VecGetArrayReadF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective on Vec

    Input Parameter:
.   x - vector

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayReadF90(x,xx_v,ierr)
    a = xx_v(3)
    call VecRestoreArrayReadF90(x,xx_v,ierr)
.ve

    If you intend to write entries into the array you must use VecGetArrayF90().

    Level: beginner

.seealso: `VecRestoreArrayReadF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecRestoreArrayRead()`, `VecGetArrayF90()`

M*/

/*MC
    VecRestoreArrayReadF90 - Restores a readonly vector to a usable state after a call to
    VecGetArrayReadF90().

    Synopsis:
    VecRestoreArrayReadF90(Vec x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Logically Collective on Vec

    Input Parameters:
+   x - vector
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage:
.vb
#include <petsc/finclude/petscvec.h>
    use petscvec

    PetscScalar, pointer :: xx_v(:)
    ....
    call VecGetArrayReadF90(x,xx_v,ierr)
    a = xx_v(3)
    call VecRestoreArrayReadF90(x,xx_v,ierr)
.ve

    Level: beginner

.seealso: `VecGetArrayReadF90()`, `VecGetArray()`, `VecRestoreArray()`, `VecGetArrayRead()`, `VecRestoreArrayRead()`, `VecRestoreArrayF90()`

M*/

/*@C
   VecGetArray2d - Returns a pointer to a 2d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray2d()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart and nstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray2d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray2d(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[]) {
  PetscInt     i, N;
  PetscScalar *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 2d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n);
  PetscCall(VecGetArray(x, &aa));

  PetscCall(PetscMalloc1(m, a));
  for (i = 0; i < m; i++) (*a)[i] = aa + i * n - nstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray2dWrite - Returns a pointer to a 2d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call VecRestoreArray2dWrite()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart and nstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray2d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray2dWrite(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[]) {
  PetscInt     i, N;
  PetscScalar *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 2d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n);
  PetscCall(VecGetArrayWrite(x, &aa));

  PetscCall(PetscMalloc1(m, a));
  for (i = 0; i < m; i++) (*a)[i] = aa + i * n - nstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray2d - Restores a vector after VecGetArray2d() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of the two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray2d()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray2d(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArray(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray2dWrite - Restores a vector after VecGetArray2dWrite() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of the two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray2d()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray2dWrite(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayWrite(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray1d - Returns a pointer to a 1d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray1d()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray2d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray1d(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[]) {
  PetscInt N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 4);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m == N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local array size %" PetscInt_FMT " does not match 1d array dimensions %" PetscInt_FMT, N, m);
  PetscCall(VecGetArray(x, a));
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray1dWrite - Returns a pointer to a 1d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call VecRestoreArray1dWrite()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray2d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray1dWrite(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[]) {
  PetscInt N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 4);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m == N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local array size %" PetscInt_FMT " does not match 1d array dimensions %" PetscInt_FMT, N, m);
  PetscCall(VecGetArrayWrite(x, a));
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray1d - Restores a vector after VecGetArray1d() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray21()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray1d().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray2d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray1d(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[]) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecRestoreArray(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray1dWrite - Restores a vector after VecGetArray1dWrite() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray21()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray1d().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray2d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray1dWrite(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[]) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecRestoreArrayWrite(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray3d - Returns a pointer to a 3d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray3d()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of three dimensional array
.  p - third dimension of three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  pstart - first index in the third coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray3d(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[]) {
  PetscInt     i, N, j;
  PetscScalar *aa, **b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 3d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p);
  PetscCall(VecGetArray(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar **) + m * n * sizeof(PetscScalar *), a));
  b = (PetscScalar **)((*a) + m);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = aa + i * n * p + j * p - pstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray3dWrite - Returns a pointer to a 3d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call VecRestoreArray3dWrite()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of three dimensional array
.  p - third dimension of three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  pstart - first index in the third coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray3dWrite(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[]) {
  PetscInt     i, N, j;
  PetscScalar *aa, **b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 3d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p);
  PetscCall(VecGetArrayWrite(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar **) + m * n * sizeof(PetscScalar *), a));
  b = (PetscScalar **)((*a) + m);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = aa + i * n * p + j * p - pstart;

  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray3d - Restores a vector after VecGetArray3d() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of the three dimensional array
.  p - third dimension of the three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray3d()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray3d(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArray(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray3dWrite - Restores a vector after VecGetArray3dWrite() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of the three dimensional array
.  p - third dimension of the three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray3d()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray3dWrite(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayWrite(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray4d - Returns a pointer to a 4d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray4d()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of four dimensional array
.  p - third dimension of four dimensional array
.  q - fourth dimension of four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  qstart - first index in the fourth coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: beginner

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray4d(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[]) {
  PetscInt     i, N, j, k;
  PetscScalar *aa, ***b, **c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p * q == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 4d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p, q);
  PetscCall(VecGetArray(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar ***) + m * n * sizeof(PetscScalar **) + m * n * p * sizeof(PetscScalar *), a));
  b = (PetscScalar ***)((*a) + m);
  c = (PetscScalar **)(b + m * n);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = c + i * n * p + j * p - pstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < p; k++) c[i * n * p + j * p + k] = aa + i * n * p * q + j * p * q + k * q - qstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray4dWrite - Returns a pointer to a 4d contiguous array that will contain this
   processor's portion of the vector data.  You MUST call VecRestoreArray4dWrite()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of four dimensional array
.  p - third dimension of four dimensional array
.  q - fourth dimension of four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  qstart - first index in the fourth coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: beginner

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray4dWrite(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[]) {
  PetscInt     i, N, j, k;
  PetscScalar *aa, ***b, **c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p * q == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 4d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p, q);
  PetscCall(VecGetArrayWrite(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar ***) + m * n * sizeof(PetscScalar **) + m * n * p * sizeof(PetscScalar *), a));
  b = (PetscScalar ***)((*a) + m);
  c = (PetscScalar **)(b + m * n);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = c + i * n * p + j * p - pstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < p; k++) c[i * n * p + j * p + k] = aa + i * n * p * q + j * p * q + k * q - qstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray4d - Restores a vector after VecGetArray3d() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of the four dimensional array
.  p - third dimension of the four dimensional array
.  q - fourth dimension of the four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
.  qstart - first index in the fourth coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray4d()

   Level: beginner

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray4d(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArray(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray4dWrite - Restores a vector after VecGetArray3dWrite() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of the four dimensional array
.  p - third dimension of the four dimensional array
.  q - fourth dimension of the four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
.  qstart - first index in the fourth coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray4d()

   Level: beginner

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray4dWrite(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayWrite(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray2dRead - Returns a pointer to a 2d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray2dRead()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  nstart - first index in the second coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart and nstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray2d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray2dRead(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[]) {
  PetscInt           i, N;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 2d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n);
  PetscCall(VecGetArrayRead(x, &aa));

  PetscCall(PetscMalloc1(m, a));
  for (i = 0; i < m; i++) (*a)[i] = (PetscScalar *)aa + i * n - nstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray2dRead - Restores a vector after VecGetArray2dRead() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  n - second dimension of the two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray2d()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray2dRead(Vec x, PetscInt m, PetscInt n, PetscInt mstart, PetscInt nstart, PetscScalar **a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 6);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayRead(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray1dRead - Returns a pointer to a 1d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray1dRead()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
-  mstart - first index you will use in first coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray2d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray1dRead(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[]) {
  PetscInt N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 4);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m == N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local array size %" PetscInt_FMT " does not match 1d array dimensions %" PetscInt_FMT, N, m);
  PetscCall(VecGetArrayRead(x, (const PetscScalar **)a));
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray1dRead - Restores a vector after VecGetArray1dRead() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of two dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray21()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray1dRead().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray2d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecRestoreArray1dRead(Vec x, PetscInt m, PetscInt mstart, PetscScalar *a[]) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecRestoreArrayRead(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray3dRead - Returns a pointer to a 3d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray3dRead()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of three dimensional array
.  p - third dimension of three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
-  pstart - first index in the third coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: developer

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3dRead().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray3dRead(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[]) {
  PetscInt           i, N, j;
  const PetscScalar *aa;
  PetscScalar      **b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 3d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p);
  PetscCall(VecGetArrayRead(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar **) + m * n * sizeof(PetscScalar *), a));
  b = (PetscScalar **)((*a) + m);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = (PetscScalar *)aa + i * n * p + j * p - pstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray3dRead - Restores a vector after VecGetArray3dRead() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of three dimensional array
.  n - second dimension of the three dimensional array
.  p - third dimension of the three dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray3dRead()

   Level: developer

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray3dRead(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscScalar ***a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 8);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayRead(x, NULL));
  PetscFunctionReturn(0);
}

/*@C
   VecGetArray4dRead - Returns a pointer to a 4d contiguous array that contains this
   processor's portion of the vector data.  You MUST call VecRestoreArray4dRead()
   when you no longer need access to the array.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of four dimensional array
.  p - third dimension of four dimensional array
.  q - fourth dimension of four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
-  qstart - first index in the fourth coordinate direction (often 0)

   Output Parameter:
.  a - location to put pointer to the array

   Level: beginner

  Notes:
   For a vector obtained from DMCreateLocalVector() mstart, nstart, and pstart are likely
   obtained from the corner indices obtained from DMDAGetGhostCorners() while for
   DMCreateGlobalVector() they are the corner indices from DMDAGetCorners(). In both cases
   the arguments from DMDAGet[Ghost]Corners() are reversed in the call to VecGetArray3d().

   For standard PETSc vectors this is an inexpensive call; it does not copy the vector values.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecGetArrays()`, `VecGetArrayF90()`, `VecPlaceArray()`,
          `VecRestoreArray2d()`, `DMDAVecGetarray()`, `DMDAVecRestoreArray()`, `VecGetArray3d()`, `VecRestoreArray3d()`,
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`
@*/
PetscErrorCode VecGetArray4dRead(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[]) {
  PetscInt           i, N, j, k;
  const PetscScalar *aa;
  PetscScalar     ***b, **c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  PetscCall(VecGetLocalSize(x, &N));
  PetscCheck(m * n * p * q == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Local array size %" PetscInt_FMT " does not match 4d array dimensions %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT " by %" PetscInt_FMT, N, m, n, p, q);
  PetscCall(VecGetArrayRead(x, &aa));

  PetscCall(PetscMalloc(m * sizeof(PetscScalar ***) + m * n * sizeof(PetscScalar **) + m * n * p * sizeof(PetscScalar *), a));
  b = (PetscScalar ***)((*a) + m);
  c = (PetscScalar **)(b + m * n);
  for (i = 0; i < m; i++) (*a)[i] = b + i * n - nstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) b[i * n + j] = c + i * n * p + j * p - pstart;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < p; k++) c[i * n * p + j * p + k] = (PetscScalar *)aa + i * n * p * q + j * p * q + k * q - qstart;
  *a -= mstart;
  PetscFunctionReturn(0);
}

/*@C
   VecRestoreArray4dRead - Restores a vector after VecGetArray3d() has been called.

   Logically Collective, Synchronous

   Input Parameters:
+  x - the vector
.  m - first dimension of four dimensional array
.  n - second dimension of the four dimensional array
.  p - third dimension of the four dimensional array
.  q - fourth dimension of the four dimensional array
.  mstart - first index you will use in first coordinate direction (often 0)
.  nstart - first index in the second coordinate direction (often 0)
.  pstart - first index in the third coordinate direction (often 0)
.  qstart - first index in the fourth coordinate direction (often 0)
-  a - location of pointer to array obtained from VecGetArray4dRead()

   Level: beginner

   Notes:
   For regular PETSc vectors this routine does not involve any copies. For
   any special vectors that do not store local vector data in a contiguous
   array, this routine will copy the data back into the underlying
   vector data structure from the array obtained with VecGetArray().

   This routine actually zeros out the a pointer.

.seealso: `VecGetArray()`, `VecRestoreArray()`, `VecRestoreArrays()`, `VecRestoreArrayF90()`, `VecPlaceArray()`,
          `VecGetArray2d()`, `VecGetArray3d()`, `VecRestoreArray3d()`, `DMDAVecGetArray()`, `DMDAVecRestoreArray()`
          `VecGetArray1d()`, `VecRestoreArray1d()`, `VecGetArray4d()`, `VecRestoreArray4d()`, `VecGet`
@*/
PetscErrorCode VecRestoreArray4dRead(Vec x, PetscInt m, PetscInt n, PetscInt p, PetscInt q, PetscInt mstart, PetscInt nstart, PetscInt pstart, PetscInt qstart, PetscScalar ****a[]) {
  void *dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(a, 10);
  PetscValidType(x, 1);
  dummy = (void *)(*a + mstart);
  PetscCall(PetscFree(dummy));
  PetscCall(VecRestoreArrayRead(x, NULL));
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)

/*@
   VecLockGet  - Gets the current lock status of a vector

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Output Parameter:
.  state - greater than zero indicates the vector is locked for read; less then zero indicates the vector is
           locked for write; equal to zero means the vector is unlocked, that is, it is free to read or write.

   Level: beginner

.seealso: `VecRestoreArray()`, `VecGetArrayRead()`, `VecLockReadPush()`, `VecLockReadPop()`
@*/
PetscErrorCode VecLockGet(Vec x, PetscInt *state) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidIntPointer(state, 2);
  *state = x->lock;
  PetscFunctionReturn(0);
}

PetscErrorCode VecLockGetLocation(Vec x, const char *file[], const char *func[], int *line) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidPointer(file, 2);
  PetscValidPointer(func, 3);
  PetscValidIntPointer(line, 4);
  {
    const int index = x->lockstack.currentsize - 1;

    PetscCheck(index >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Corrupted vec lock stack, have negative index %d", index);
    *file = x->lockstack.file[index];
    *func = x->lockstack.function[index];
    *line = x->lockstack.line[index];
  }
  PetscFunctionReturn(0);
}

/*@
   VecLockReadPush  - Pushes a read-only lock on a vector to prevent it from writing

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Notes:
    If this is set then calls to VecGetArray() or VecSetValues() or any other routines that change the vectors values will fail.

    The call can be nested, i.e., called multiple times on the same vector, but each VecLockReadPush(x) has to have one matching
    VecLockReadPop(x), which removes the latest read-only lock.

   Level: beginner

.seealso: `VecRestoreArray()`, `VecGetArrayRead()`, `VecLockReadPop()`, `VecLockGet()`
@*/
PetscErrorCode VecLockReadPush(Vec x) {
  const char *file, *func;
  int         index, line;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCheck(x->lock++ >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector is already locked for exclusive write access but you want to read it");
  if ((index = petscstack.currentsize - 2) == -1) {
    // vec was locked "outside" of petsc, either in user-land or main. the error message will
    // now show this function as the culprit, but it will include the stacktrace
    file = "unknown user-file";
    func = "unknown_user_function";
    line = 0;
  } else {
    PetscCheck(index >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected petscstack, have negative index %d", index);
    file = petscstack.file[index];
    func = petscstack.function[index];
    line = petscstack.line[index];
  }
  PetscStackPush_Private(x->lockstack, file, func, line, petscstack.petscroutine[index], PETSC_FALSE);
  PetscFunctionReturn(0);
}

/*@
   VecLockReadPop  - Pops a read-only lock from a vector

   Logically Collective on Vec

   Input Parameter:
.  x - the vector

   Level: beginner

.seealso: `VecRestoreArray()`, `VecGetArrayRead()`, `VecLockReadPush()`, `VecLockGet()`
@*/
PetscErrorCode VecLockReadPop(Vec x) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCheck(--x->lock >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector has been unlocked from read-only access too many times");
  {
    const char *previous = x->lockstack.function[x->lockstack.currentsize - 1];

    PetscStackPop_Private(x->lockstack, previous);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecLockWriteSet  - Lock or unlock a vector for exclusive read/write access

   Logically Collective on Vec

   Input Parameters:
+  x   - the vector
-  flg - PETSC_TRUE to lock the vector for exclusive read/write access; PETSC_FALSE to unlock it.

   Notes:
    The function is usefull in split-phase computations, which usually have a begin phase and an end phase.
    One can call VecLockWriteSet(x,PETSC_TRUE) in the begin phase to lock a vector for exclusive
    access, and call VecLockWriteSet(x,PETSC_FALSE) in the end phase to unlock the vector from exclusive
    access. In this way, one is ensured no other operations can access the vector in between. The code may like

       VecGetArray(x,&xdata); // begin phase
       VecLockWriteSet(v,PETSC_TRUE);

       Other operations, which can not acceess x anymore (they can access xdata, of course)

       VecRestoreArray(x,&vdata); // end phase
       VecLockWriteSet(v,PETSC_FALSE);

    The call can not be nested on the same vector, in other words, one can not call VecLockWriteSet(x,PETSC_TRUE)
    again before calling VecLockWriteSet(v,PETSC_FALSE).

   Level: beginner

.seealso: `VecRestoreArray()`, `VecGetArrayRead()`, `VecLockReadPush()`, `VecLockReadPop()`, `VecLockGet()`
@*/
PetscErrorCode VecLockWriteSet(Vec x, PetscBool flg) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (flg) {
    PetscCheck(x->lock <= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector is already locked for read-only access but you want to write it");
    PetscCheck(x->lock >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector is already locked for exclusive write access but you want to write it");
    x->lock = -1;
  } else {
    PetscCheck(x->lock == -1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector is not locked for exclusive write access but you want to unlock it from that");
    x->lock = 0;
  }
  PetscFunctionReturn(0);
}

/*@
   VecLockPush  - Pushes a read-only lock on a vector to prevent it from writing

   Level: deprecated

.seealso: `VecLockReadPush()`
@*/
PetscErrorCode VecLockPush(Vec x) {
  PetscFunctionBegin;
  PetscCall(VecLockReadPush(x));
  PetscFunctionReturn(0);
}

/*@
   VecLockPop  - Pops a read-only lock from a vector

   Level: deprecated

.seealso: `VecLockReadPop()`
@*/
PetscErrorCode VecLockPop(Vec x) {
  PetscFunctionBegin;
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(0);
}

#endif
