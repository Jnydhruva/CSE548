#include <../src/sys/classes/random/randomimpl.h>
#include <curand.h>

#define CHKERRCURAND(stat) \
do { \
   if (PetscUnlikely(stat != CURAND_STATUS_SUCCESS)) { \
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuRand error %d",(int)stat); \
   } \
} while (0)

typedef struct {
  curandGenerator_t gen;
} PetscRandom_CURAND;

PetscErrorCode PetscRandomSeed_CURAND(PetscRandom r)
{
  curandStatus_t     cerr;
  PetscRandom_CURAND *curand = (PetscRandom_CURAND*)r->data;

  PetscFunctionBegin;
  cerr = curandSetPseudoRandomGeneratorSeed(curand->gen,r->seed);CHKERRCURAND(cerr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscRandomGetValuesReal_CURAND(PetscRandom r, PetscInt n, PetscReal *val)
{
  curandStatus_t     cerr;
  PetscScalar        zero = 0.0, one = 1.0;
  PetscRandom_CURAND *curand = (PetscRandom_CURAND*)r->data;

  PetscFunctionBegin;
  if (r->low != zero || r->width != one) SETERRQ(PetscObjectComm((PetscObject)r),PETSC_ERR_SUP,"Only for numbers in [0,1)");
#if defined(PETSC_USE_REAL_SINGLE)
  cerr = curandGenerateUniform(curand->gen,val,n);CHKERRCURAND(cerr);
#else
  cerr = curandGenerateUniformDouble(curand->gen,val,n);CHKERRCURAND(cerr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRandomGetValues_CURAND(PetscRandom r, PetscInt n, PetscScalar *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscRandomGetValuesReal_CURAND(r,2*n,(PetscReal*)val);CHKERRQ(ierr);
#else
  ierr = PetscRandomGetValuesReal_CURAND(r,n,val);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  PetscRandomSeed_CURAND,
  NULL,
  NULL,
  PetscRandomGetValues_CURAND,
  PetscRandomGetValuesReal_CURAND,
  NULL,
  NULL
};

/*MC
   PETSCCURAND - access to the CUDA random number generator

  Level: beginner

.seealso: PetscRandomCreate(), PetscRandomSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscRandomCreate_CURAND(PetscRandom r)
{
  PetscErrorCode     ierr;
  curandStatus_t     cerr;
  PetscRandom_CURAND *curand;

  PetscFunctionBegin;

  PetscFunctionBegin;
  ierr = PetscNewLog(r,&curand);CHKERRQ(ierr);
  r->data = curand;
  cerr = curandCreateGenerator(&curand->gen,CURAND_RNG_PSEUDO_DEFAULT);CHKERRCURAND(cerr);
  /* https://docs.nvidia.com/cuda/curand/host-api-overview.html#performance-notes2 */
  cerr = curandSetGeneratorOrdering(curand->gen,CURAND_ORDERING_PSEUDO_SEEDED);CHKERRCURAND(cerr);
  ierr = PetscMemcpy(r->ops,&PetscRandomOps_Values,sizeof(PetscRandomOps_Values));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)r,PETSCCURAND);CHKERRQ(ierr);
  r->seed = 1234ULL; /* taken from example */
  ierr = PetscRandomSeed_CURAND(r);CHKERRCURAND(cerr);
  PetscFunctionReturn(0);
}
