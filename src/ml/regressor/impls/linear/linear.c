#include <../src/ml/regressor/impls/linear/linearimpl.h> /*I "petscmlregressor.h" I*/

PetscErrorCode MLRegressorSetUp_Linear(MLRegressor mlregressor)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorReset_Linear(MLRegressor mlregressor)
{
  PetscErrorCode ierr;
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;

  PetscFunctionBegin;
  /* Destroy the PETSc objects associated with the linear regressor implementation. */
  ierr = MatDestroy(&linear->X);CHKERRQ(ierr);
  ierr = KSPDestroy(&linear->ksp);CHKERRQ(ierr);

  /* Reset options/parameters to the setupcalled = 0 state. */
  /* TODO: Add the reset code once the linear regressor is fleshed out enough to need resetting! */
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorDestroy_Linear(MLRegressor mlregressor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MLRegressorReset_Linear(mlregressor);CHKERRQ(ierr);
  ierr = PetscFree(mlregressor->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorSetFromOptions_Linear(PetscOptionItems *PetscOptionsObject, MLRegressor mlregressor)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorView_Linear(MLRegressor mlregressor, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorFit_Linear(MLRegressor mlregressor)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorPredict_Linear(MLRegressor mlregressor, Mat X, Vec y)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MLRegressorLinearGetKSP(MLRegressor mlregressor,KSP *ksp)
{
  MLREGRESSOR_LINEAR *linear = (MLREGRESSOR_LINEAR*)mlregressor->data;
  PetscErrorCode ierr;

  /* Analogous to how SNESGetKSP() operates, this routine should create the KSP if it doesn't exist.
   * TODO: Follow what SNESGetKSP() does when setting this up. */
  if (!linear->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)mlregressor),&linear->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)linear->ksp,(PetscObject)mlregressor,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mlregressor,(PetscObject)linear->ksp);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)linear->ksp,((PetscObject)mlregressor)->options);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MLRegressorCreate_Linear(MLRegressor mlregressor)
{
  MLREGRESSOR_LINEAR *linear;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(mlregressor,&linear);CHKERRQ(ierr);
  mlregressor->data = (void*)linear;

  mlregressor->ops->setup          = MLRegressorSetUp_Linear;
  mlregressor->ops->reset          = MLRegressorReset_Linear;
  mlregressor->ops->destroy        = MLRegressorDestroy_Linear;
  mlregressor->ops->setfromoptions = MLRegressorSetFromOptions_Linear;
  mlregressor->ops->view           = MLRegressorView_Linear;
  mlregressor->ops->fit            = MLRegressorFit_Linear;
  mlregressor->ops->predict        = MLRegressorPredict_Linear;

  linear->intercept = 0.0;
  linear->fit_intercept = PETSC_FALSE;  /* TODO: This should probably default to true; but using false for now so I can initially implement less! */
  PetscFunctionReturn(0);
}
