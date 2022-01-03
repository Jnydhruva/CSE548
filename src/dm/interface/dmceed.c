#include <petsc/private/dmimpl.h>           /*I      "petscdm.h"          I*/

#ifdef PETSC_HAVE_LIBCEED
#include <petscdmceed.h>

/*@C
  DMGetCeed - Get the LibCEED context associated with this DM

  Not collective

  Input Parameter:
. DM   - The DM

  Output Parameter:
. ceed - The LibCEED context

  Level: intermediate

.seealso: DMCreate()
@*/
PetscErrorCode DMGetCeed(DM dm, Ceed *ceed)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ceed, 2);
  if (!dm->ceed) {
    char        ceedresource[PETSC_MAX_PATH_LEN]; /* libCEED resource specifier */
    const char *prefix;

    ierr = PetscStrcpy(ceedresource, "/cpu/self");CHKERRQ(ierr);
    ierr = PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, prefix, "-dm_ceed", ceedresource, sizeof(ceedresource), NULL);CHKERRQ(ierr);
    ierr = CeedInit(ceedresource, &dm->ceed);CHKERRQ(ierr);
  }
  *ceed = dm->ceed;
  PetscFunctionReturn(0);
}

static CeedMemType PetscMemType2Ceed(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

PetscErrorCode VecGetCeedVector(Vec X, Ceed ceed, CeedVector *cx)
{
  PetscMemType   memtype;
  PetscScalar   *x;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(X, &x, &memtype);CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, n, cx);CHKERRQ_CEED(ierr);
  ierr = CeedVectorSetArray(*cx, PetscMemType2Ceed(memtype), CEED_USE_POINTER, x);CHKERRQ_CEED(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreCeedVector(Vec X, CeedVector *cx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecRestoreArrayAndMemType(X, NULL);CHKERRQ(ierr);
  ierr = CeedVectorDestroy(cx);CHKERRQ_CEED(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetCeedVectorRead(Vec X, Ceed ceed, CeedVector *cx)
{
  PetscMemType       memtype;
  const PetscScalar *x;
  PetscInt           n;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(X, &x, &memtype);CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, n, cx);CHKERRQ_CEED(ierr);
  ierr = CeedVectorSetArray(*cx, PetscMemType2Ceed(memtype), CEED_USE_POINTER, (PetscScalar*)x);CHKERRQ_CEED(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreCeedVectorRead(Vec X, CeedVector *cx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecRestoreArrayReadAndMemType(X, NULL);CHKERRQ(ierr);
  ierr = CeedVectorDestroy(cx);CHKERRQ_CEED(ierr);
  PetscFunctionReturn(0);
}

#endif
