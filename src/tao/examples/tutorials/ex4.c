static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>
#include <petsctao.h>
#include <petscvec.h>
#include <petscmath.h>

#define NWORKLEFT 4
#define NWORKRIGHT 16

typedef struct _UserCtx
{
  PetscInt m;      /* The row dimension of F */
  PetscInt n;      /* The column dimension of F */
  PetscReal hStart; /* Starting point for Taylor test */
  PetscReal hFactor;/* Taylor test step factor */
  PetscReal hMin;   /* Taylor test end goal */
  Mat F;           /* matrix in least squares component $(1/2) * || F x - d ||_2^2$ */
  Mat W;           /* Workspace matrix. ATA */
  Mat W1;           /* Workspace matrix. AAT */
  Mat Id;           /* Workspace matrix. Dense. Identity */
  Mat Fp;           /* Workspace matrix.  FFTinv  */
  Mat Fpinv;           /* Workspace matrix.   F*FFTinv */
  Mat P;           /* I - FT*((FFT)^-1 * F) */
  Mat temp;
  Mat Hm;           /* Hessian Misfit*/
  Mat Hr;           /* Hessian Reg*/
  Vec d;           /* RHS in least squares component $(1/2) * || F x - d ||_2^2$ */
  Vec workLeft[NWORKLEFT];       /* Workspace for temporary vec */
  Vec workRight[NWORKRIGHT];       /* Workspace for temporary vec */
  PetscReal alpha; /* regularization constant applied to || x ||_p */
  PetscReal relax; /* Overrelaxation parameter  */
  PetscReal eps; /* small constant for approximating gradient of || x ||_1 */
  PetscReal mu;  /* the augmented Lagrangian term in ADMM */
  PetscInt matops;
  PetscInt iter;
  NormType p;
  PetscRandom    rctx;
  PetscBool taylor; /*Flag to determine whether to run Taylor test or not */
  PetscBool use_admm; /*Flag to determine whether to run Taylor test or not */
} * UserCtx;

PetscErrorCode CreateRHS(UserCtx ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* build the rhs d in ctx */
  ierr = VecCreate(PETSC_COMM_WORLD,&(ctx->d)); CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->d,PETSC_DECIDE,ctx->m); CHKERRQ(ierr);
  ierr = VecSetFromOptions(ctx->d); CHKERRQ(ierr);
  ierr = VecSetRandom(ctx->d,ctx->rctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMatrix(UserCtx ctx)
{
  PetscInt       Istart,Iend,i,j,Ii;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* build the matrix F in ctx */
  ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->F)); CHKERRQ(ierr);
  ierr = MatSetSizes(ctx->F,PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n);CHKERRQ(ierr);
  ierr = MatSetType(ctx->F,MATAIJ); CHKERRQ(ierr); /* TODO: Decide specific SetType other than dummy*/
  ierr = MatMPIAIJSetPreallocation(ctx->F, 5, NULL, 5, NULL); CHKERRQ(ierr); /*TODO: some number other than 5?*/
  ierr = MatSeqAIJSetPreallocation(ctx->F, 5, NULL); CHKERRQ(ierr);
  ierr = MatSetUp(ctx->F); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(ctx->F,&Istart,&Iend); CHKERRQ(ierr);  

  ierr = PetscLogStageRegister("Assembly", &stage); CHKERRQ(ierr);
  ierr= PetscLogStagePush(stage); CHKERRQ(ierr);

  /* Set matrix elements in  2-D fiveopoint stencil format. */
  if (!(ctx->matops)){
    PetscInt gridN;
    if (ctx->m != ctx->n) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Stencil matrix must be square");

    gridN = (PetscInt) PetscSqrtReal((PetscReal) ctx->m);
    if (gridN * gridN != ctx->m) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of rows must be square");
    for (Ii=Istart; Ii<Iend; Ii++) {
      PetscInt I_n, I_s, I_e, I_w;

      i = Ii / gridN; j = Ii % gridN;

      I_n = i * gridN + j + 1;
      if (j + 1 >= gridN) I_n = -1;
      I_s = i * gridN + j - 1;
      if (j - 1 < 0) I_s = -1;
      I_e = (i + 1) * gridN + j;
      if (i + 1 >= gridN) I_e = -1;
      I_w = (i - 1) * gridN + j;
      if (i - 1 < 0) I_w = -1;

      ierr = MatSetValue(ctx->F, Ii, Ii, 4., INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(ctx->F, Ii, I_n, -1., INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(ctx->F, Ii, I_s, -1., INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(ctx->F, Ii, I_e, -1., INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(ctx->F, Ii, I_w, -1., INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  else {
    ierr = MatSetRandom(ctx->F, ctx->rctx); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(ctx->F, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->F, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscLogStagePop(); CHKERRQ(ierr);

  //TODO if condition for running ADMM?
  ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n, NULL, &(ctx->Id)); CHKERRQ(ierr);
  ierr = MatZeroEntries(ctx->Id); CHKERRQ(ierr);
  ierr = MatShift(ctx->Id,1.0); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->Id, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Id, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n, NULL, &(ctx->Fp)); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(ctx->Fp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Fp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatDuplicate(ctx->Fp,MAT_DO_NOT_COPY_VALUES,&(ctx->Fpinv)); CHKERRQ(ierr);
  ierr = MatDuplicate(ctx->F,MAT_DO_NOT_COPY_VALUES,&(ctx->P)); CHKERRQ(ierr);
  ierr = MatDuplicate(ctx->F,MAT_DO_NOT_COPY_VALUES,&(ctx->temp)); CHKERRQ(ierr);

  /* Stencil matrix is symmetric. Setting symmetric flag for ICC/CHolesky preconditioner */
  if (!(ctx->matops)){
    ierr = MatSetOption(ctx->F,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);
  }

  ierr = MatTransposeMatMult(ctx->F,ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->W)); CHKERRQ(ierr);
  ierr = MatMatTransposeMult(ctx->F,ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->W1)); CHKERRQ(ierr);

  ierr = MatDuplicate(ctx->W,MAT_DO_NOT_COPY_VALUES,&(ctx->Hm)); CHKERRQ(ierr);
  ierr = MatDuplicate(ctx->W,MAT_DO_NOT_COPY_VALUES,&(ctx->Hr)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupWorkspace(UserCtx ctx)
{
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateVecs(ctx->F, &ctx->workLeft[0], &ctx->workRight[0]);CHKERRQ(ierr);
  for (i = 1; i < NWORKLEFT; i++) {
    ierr = VecDuplicate(ctx->workLeft[0], &(ctx->workLeft[i]));CHKERRQ(ierr);
  }
  for (i = 1; i < NWORKRIGHT; i++) {
    ierr = VecDuplicate(ctx->workRight[0], &(ctx->workRight[i]));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ConfigureContext(UserCtx ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ctx->m = 16;
  ctx->n = 16;
  ctx->alpha = 1.;
  ctx->relax = 1.;
  ctx->eps = 1.e-3;
  ctx->matops = 0;
  ctx->iter = 10;
  ctx->p = NORM_2;
  ctx->hStart = 1.;
  ctx->hMin = 1.e-3;
  ctx->hFactor = 0.5;
  ctx->mu = 1.0;
  ctx->taylor = PETSC_TRUE;
  ctx->use_admm = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "ex4.c");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m", "The row dimension of matrix F", "ex4.c", ctx->m, &(ctx->m), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The column dimension of matrix F", "ex4.c", ctx->n, &(ctx->n), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matrix_format","Decide format of F matrix. 0 for stencil, 1 for dense random", "ex4.c", ctx->matops, &(ctx->matops), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-iter","Iteration number for ADMM Basic Pursuit", "ex4.c", ctx->iter, &(ctx->iter), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha", "The regularization multiplier. 1 default", "ex4.c", ctx->alpha, &(ctx->alpha), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-relax", "Overrelaxation parameter.", "ex4.c", ctx->relax, &(ctx->relax), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-epsilon", "The small constant added to |x_i| in the denominator to approximate the gradient of ||x||_1", "ex4.c", ctx->eps, &(ctx->eps), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu", "The augmented lagrangian multiplier in ADMM", "ex4.c", ctx->mu, &(ctx->mu), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hStart", "Taylor test starting point. 1 default.", "ex4.c", ctx->hStart, &(ctx->hStart), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hFactor", "Taylor test multiplier factor. 0.5 default", "ex4.c", ctx->hFactor, &(ctx->hFactor), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hMin", "Taylor test ending condition. 1.e-3 default", "ex4.c", ctx->hMin, &(ctx->hMin), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-taylor","Flag for Taylor test. Default is true.", "ex4.c", ctx->taylor, &(ctx->taylor), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_admm","Use the ADMM solver in this example.", "ex4.c", ctx->use_admm, &(ctx->use_admm), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-p","Norm type.", "ex4.c", NormTypes,  ctx->p, (PetscEnum *) &(ctx->p), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* Creating random ctx */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&(ctx->rctx));CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(ctx->rctx);CHKERRQ(ierr);
  ierr = CreateMatrix(ctx);CHKERRQ(ierr);
  ierr = CreateRHS(ctx);CHKERRQ(ierr);
  ierr = SetupWorkspace(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyContext(UserCtx *ctx)
{
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&((*ctx)->F)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->W)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->W1)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->Id)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->Fp)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->Fpinv)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->temp)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->P)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->Hm)); CHKERRQ(ierr);
  ierr = MatDestroy(&((*ctx)->Hr)); CHKERRQ(ierr);
  ierr = VecDestroy(&((*ctx)->d)); CHKERRQ(ierr);
  for (i = 0; i < NWORKLEFT; i++) {
    ierr = VecDestroy(&((*ctx)->workLeft[i])); CHKERRQ(ierr);
  }
  for (i = 0; i < NWORKRIGHT; i++) {
    ierr = VecDestroy(&((*ctx)->workRight[i])); CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&((*ctx)->rctx)); CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  *ctx = NULL;
  PetscFunctionReturn(0);
}

/* compute (1/2) * ||F x - d||^2 */
PetscErrorCode ObjectiveMisfit(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  Vec y = ctx->workLeft[0];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(ctx->F, x, y);CHKERRQ(ierr);
  ierr = VecAXPY(y, -1., ctx->d);CHKERRQ(ierr);
  ierr = VecDot(y, y, J);CHKERRQ(ierr);
  *J *= 0.5;
  PetscFunctionReturn(0);
}

PetscErrorCode GradientMisfit(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscErrorCode ierr;
  Vec FTFx, FTd;

  /* work1 is A^T Ax, work2 is Ab, W is A^T A*/

  PetscFunctionBegin;
  FTFx = ctx->workRight[0];
  FTd = ctx->workRight[1];

  ierr = MatMult(ctx->W,x,FTFx); CHKERRQ(ierr);
  ierr = MatMultTranspose(ctx->F, ctx->d, FTd);CHKERRQ(ierr);
  ierr = VecWAXPY(V, -1., FTd, FTFx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode HessianMisfit(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (H != ctx->W) {ierr = MatCopy(ctx->W, H, SAME_NONZERO_PATTERN); CHKERRQ(ierr);}
  if (Hpre != ctx->W) {ierr = MatCopy(ctx->W, Hpre, SAME_NONZERO_PATTERN); CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode ObjectiveMisfitADMM(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscReal mu, workNorm, misfit;
  Vec z, u;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mu = ctx->mu;
  z = ctx->workRight[12];
  u = ctx->workRight[13];
  ierr = ObjectiveMisfit(tao, x, &misfit, _ctx);CHKERRQ(ierr);
  ierr = VecCopy(x,ctx->workRight[14]); CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(ctx->workRight[14],-1.,-1.,1.,z,u);CHKERRQ(ierr);
  ierr = VecDot(ctx->workRight[14], ctx->workRight[14], &workNorm);CHKERRQ(ierr);
  *J = misfit + 0.5 * mu * workNorm;
  PetscFunctionReturn(0);
}

PetscErrorCode GradientMisfitADMM(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscReal      mu;
  Vec            z, u;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mu = ctx->mu;
  z = ctx->workRight[12];
  u = ctx->workRight[13];
  ierr = GradientMisfit(tao, x, V, _ctx);CHKERRQ(ierr);
  ierr = VecCopy(x, ctx->workRight[14]);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(ctx->workRight[14],-1.,-1.,1.,z,u);CHKERRQ(ierr);
  ierr = VecAXPY(V, mu, ctx->workRight[14]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode HessianMisfitADMM(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCopy(ctx->W, H, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = MatShift(H, ctx->mu); CHKERRQ(ierr);
  
  if (Hpre != H) {
    ierr = MatCopy(H, Hpre, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ObjectiveRegularization(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscReal norm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNorm (x, ctx->p, &norm);CHKERRQ(ierr);
  if (ctx->p == NORM_2) {
    norm = 0.5 * norm * norm;
  }
  *J = ctx->alpha * norm;
  PetscFunctionReturn(0);
}

PetscErrorCode GradientRegularization(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    ierr = VecCopy(x, V);CHKERRQ(ierr);
  }
  else if (ctx->p == NORM_1) {
    PetscReal eps = ctx->eps;

    ierr = VecCopy(x, ctx->workRight[1]);CHKERRQ(ierr);
    ierr = VecAbs(ctx->workRight[1]); CHKERRQ(ierr);
    ierr = VecShift(ctx->workRight[1], eps);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(V, x, ctx->workRight[1]); CHKERRQ(ierr);
  }
  else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode HessianRegularization(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    /* Identity matrix scaled by mu */
    ierr = MatZeroEntries(H); CHKERRQ(ierr);
    ierr = MatShift(H,ctx->mu); CHKERRQ(ierr);	
    if (Hpre != H) {
      ierr = MatZeroEntries(Hpre); CHKERRQ(ierr);
      ierr = MatShift(Hpre,ctx->mu); CHKERRQ(ierr);
    }
  }
  else if (ctx->p == NORM_1) {
    /* 1/sqrt(x_i^2 + eps) * ( 1 - x_i^2/ABS(x_i^2+eps) ) */

    PetscReal eps = ctx->eps;
    Vec copy1,copy2,copy3;
    copy1 = ctx->workRight[1];
    copy2 = ctx->workRight[2];
    copy3 = ctx->workRight[3];

    /* copy1 : 1/sqrt(x_i^2 + eps) */
    ierr = VecCopy(x, copy1);CHKERRQ(ierr);
    ierr = VecPow(copy1,2); CHKERRQ(ierr);
    ierr = VecShift(copy1, eps);CHKERRQ(ierr);
    ierr = VecSqrtAbs(copy1);CHKERRQ(ierr);
    ierr = VecReciprocal(copy1); CHKERRQ(ierr);

    /* copy2:  x_i^2.*/
    ierr = VecCopy(x,copy2); CHKERRQ(ierr);
    ierr = VecPow(copy2,2); CHKERRQ(ierr);

    /* copy3: abs(x_i^2 + eps) */
    ierr = VecCopy(x,copy3); CHKERRQ(ierr);
    ierr = VecPow(copy3,2); CHKERRQ(ierr);
    ierr = VecShift(copy3, eps);CHKERRQ(ierr);
    ierr = VecAbs(copy3); CHKERRQ(ierr);

    /* copy2: 1 - x_i^2/abs(x_i^2 + eps) */
    ierr = VecPointwiseDivide(copy2, copy2,copy3); CHKERRQ(ierr);
    ierr = VecScale(copy2, -1.); CHKERRQ(ierr);
    ierr = VecShift(copy2, 1.); CHKERRQ(ierr);

    ierr = VecAXPY(copy1,1.,copy2); CHKERRQ(ierr);
    ierr = VecScale(copy1, ctx->mu); CHKERRQ(ierr);

    ierr = MatZeroEntries(H); CHKERRQ(ierr);
    ierr = MatDiagonalSet(H, copy1,INSERT_VALUES); CHKERRQ(ierr);
    if (Hpre != H) {
      ierr = MatZeroEntries(Hpre); CHKERRQ(ierr);
      ierr = MatDiagonalSet(Hpre, copy1,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ObjectiveRegularizationADMM(Tao tao, Vec z, PetscReal *J, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscReal mu, workNorm, reg;
  Vec x, u, temp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mu = ctx->mu;
  x = ctx->workRight[11];
  u = ctx->workRight[13];
  temp = ctx->workRight[14];
  ierr = ObjectiveRegularization(tao, z, &reg, _ctx);CHKERRQ(ierr);
  ierr = VecCopy(z,temp); CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(temp,1.,-1.,-1.,x,u);CHKERRQ(ierr);
  ierr = VecDot(temp, temp, &workNorm);CHKERRQ(ierr);
  *J = reg + 0.5 * mu * workNorm;
  PetscFunctionReturn(0);
}

PetscErrorCode GradientRegularizationADMM(Tao tao, Vec z, Vec V, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscReal      mu;
  Vec x, u, temp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mu = ctx->mu;
  x = ctx->workRight[11];
  u = ctx->workRight[13];
  temp = ctx->workRight[14];
  ierr = GradientRegularization(tao, z, V, _ctx);CHKERRQ(ierr);
  ierr = VecCopy(z, temp);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(temp,1.,-1.,-1.,x,u);CHKERRQ(ierr);
  ierr = VecAXPY(V, -mu, temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode HessianRegularizationADMM(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx ctx = (UserCtx) _ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    /* Identity matrix scaled by mu */
    ierr = MatZeroEntries(H); CHKERRQ(ierr);
    ierr = MatShift(H,ctx->mu); CHKERRQ(ierr);	
    if (Hpre != H) {
      ierr = MatZeroEntries(Hpre); CHKERRQ(ierr);
      ierr = MatShift(Hpre,ctx->mu); CHKERRQ(ierr);
    }
  }
  else if (ctx->p == NORM_1) {
	ierr = HessianMisfit(tao, x, H, Hpre, (void *) ctx); CHKERRQ(ierr);
	ierr = MatShift(H, ctx->mu); CHKERRQ(ierr);
    if (Hpre != H) { ierr = MatShift(Hpre, ctx->mu); CHKERRQ(ierr);}
  }
  else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ObjectiveComplete(Tao tao, Vec x, PetscReal *J, void *ctx)
{
  PetscReal Jm, Jr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ObjectiveMisfit(tao, x, &Jm, ctx);CHKERRQ(ierr);
  ierr = ObjectiveRegularization(tao, x, &Jr, ctx);CHKERRQ(ierr);
  *J = Jm + Jr;
  PetscFunctionReturn(0);
}

PetscErrorCode GradientComplete(Tao tao, Vec x, Vec V, void *ctx)
{
  UserCtx cntx = (UserCtx) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GradientMisfit(tao, x, cntx->workRight[2], ctx);CHKERRQ(ierr);
  ierr = GradientRegularization(tao, x, cntx->workRight[3], ctx);CHKERRQ(ierr);
  ierr = VecWAXPY(V,1,cntx->workRight[2],cntx->workRight[3]); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode HessianComplete(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  Mat tempH;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate(H, MAT_SHARE_NONZERO_PATTERN, &tempH);CHKERRQ(ierr);
  ierr = HessianMisfit(tao, x, H, H, ctx);CHKERRQ(ierr);
  ierr = HessianRegularization(tao, x, tempH, tempH, ctx);CHKERRQ(ierr);
  ierr = MatAXPY(H, 1., tempH, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (Hpre != H) {
    ierr = MatCopy(H, Hpre, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&tempH);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscReal SoftThreshold(PetscReal z, PetscReal mu)
{
  return PetscMax(0,z- mu) - PetscMax(0, -z-mu);
}

PetscErrorCode TaoSolveADMM(UserCtx ctx,  Vec x)
{
  PetscErrorCode ierr;
  PetscInt i;
  Tao tao1,tao2;
  Vec xk,z,u,diff;

  PetscFunctionBegin;

  xk = ctx->workRight[11];
  z = ctx->workRight[12];
  u = ctx->workRight[13];
  diff = ctx->workRight[15];
  ierr = VecSet(u, 0.);CHKERRQ(ierr);

  ierr = TaoCreate(PETSC_COMM_WORLD, &tao1);CHKERRQ(ierr);
  ierr = TaoSetType(tao1,TAONLS); CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(tao1, ObjectiveMisfitADMM, (void *) ctx);CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine(tao1, GradientMisfitADMM, (void *) ctx);CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao1, ctx->Hm, ctx->Hm, HessianMisfitADMM, (void *) ctx);CHKERRQ(ierr);
  //ierr = MatCreateVecs(ctx->F, NULL, &xk);CHKERRQ(ierr);
  ierr = VecSet(xk, 0.);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao1, xk);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(tao1, "misfit_");CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao1);CHKERRQ(ierr);

  ierr = TaoCreate(PETSC_COMM_WORLD, &tao2);CHKERRQ(ierr);
  ierr = TaoSetType(tao2,TAONLS); CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(tao2, ObjectiveRegularizationADMM, (void *) ctx);CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine(tao2, GradientRegularizationADMM, (void *) ctx);CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao2, ctx->Hr, ctx->Hr, HessianRegularizationADMM, (void *) ctx);CHKERRQ(ierr);
  //ierr = MatCreateVecs(ctx->F, NULL, &z);CHKERRQ(ierr);
  ierr = VecSet(z, 0.);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao2, z);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(tao2, "reg_");CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao2);CHKERRQ(ierr);

  for (i=0; i<ctx->iter; i++){
    PetscReal t1,t2,norm;

    TaoSolve(tao1);CHKERRQ(ierr); /* Updates xk */
    TaoSolve(tao2);CHKERRQ(ierr); /* Update zk */
    ierr = VecAXPBYPCZ(u,-1.,+1.,1.,xk,z); CHKERRQ(ierr);
    /*TODO iter stop check */
    /* Convergence Check */
    ierr = ObjectiveMisfit(tao1, xk, &t1, (void *) ctx);CHKERRQ(ierr);
    ierr = ObjectiveRegularization(tao2, z, &t2, (void *) ctx);CHKERRQ(ierr);
    ierr = VecWAXPY(diff,-1.,xk,z);CHKERRQ(ierr);
    ierr = VecNorm(diff,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)tao1),"ADMM %D: ||x - z|| = %g\n", i, (double) norm);CHKERRQ(ierr);
  }
 
  ierr = VecCopy(xk, x); CHKERRQ(ierr);
  ierr = TaoDestroy(&tao1);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode ADMMBasicPursuit(UserCtx ctx, Tao tao, Vec x, PetscReal *C)
{
  PetscErrorCode ierr;
  PetscInt i, nlocal;
  PetscReal *z_array;
  IS perm, iscol;
  MatFactorInfo factinfo;
  Vec z_k, u_k, x_k, max_k;

  PetscFunctionBegin;
  z_k = ctx->workRight[3];
  u_k = ctx->workRight[4];
  x_k = ctx->workRight[11];
  max_k = ctx->workRight[9];
  ierr = VecSet(z_k,0); CHKERRQ(ierr); /* z_k */
  ierr = VecSet(u_k,0); CHKERRQ(ierr); /* u_k */
  ierr = VecSet(x_k,0); CHKERRQ(ierr); /* x_k */
  ierr = VecSet(max_k,0); CHKERRQ(ierr); // compare zero vector for VecPointWiseMax

  ierr  = MatGetOrdering(ctx->W1,MATORDERINGNATURAL,&perm,&iscol);CHKERRQ(ierr);
  ierr  = ISDestroy(&iscol);CHKERRQ(ierr);

  ierr = PetscMemzero(&factinfo,sizeof(MatFactorInfo));CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&factinfo); CHKERRQ(ierr); 
  ierr = MatGetFactor(ctx->W1,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&(ctx->temp));CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic(ctx->temp,ctx->W1,perm,&factinfo);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(ctx->temp,ctx->W1,&factinfo);CHKERRQ(ierr);

  ierr = MatMatSolve(ctx->temp,ctx->Id, ctx->Fp); CHKERRQ(ierr); // Solve LLT FFTinv = I for FFTinv
  ierr = MatTransposeMatMult(ctx->F, ctx->Fp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->Fpinv)); CHKERRQ(ierr); 
  ierr = MatMatMult(ctx->Fpinv, ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->P)); CHKERRQ(ierr);
  ierr = MatScale(ctx->P, -1.0); CHKERRQ(ierr);
  ierr = MatShift(ctx->P, 1.0); CHKERRQ(ierr); /* P = I - FT*(FFT^-1)*F */

  ierr = MatMult(ctx->Fpinv, ctx->d, ctx->workRight[5]); /* q = FT*((FFT)^-1 * b) */

  for (i=0; i<ctx->iter; i++){
    // x update 
    ierr = VecWAXPY(ctx->workRight[6], -1.0, u_k, z_k); CHKERRQ(ierr); // work[6] = z-u
    ierr = MatMultAdd(ctx->P, ctx->workRight[6], ctx->workRight[5], x_k); CHKERRQ(ierr); // x = P(z-u) + q
    ierr = VecAXPBYPCZ(ctx->workRight[7], ctx->alpha, 1.0 - ctx->alpha, 0.0, x_k, z_k); CHKERRQ(ierr); // x_hat = ax + (1-a)z

    /* soft thresholding for z */
    ierr = VecGetArray(z_k, &z_array); CHKERRQ(ierr);
    ierr = VecGetLocalSize(z_k, &nlocal); CHKERRQ(ierr);
    for (i=0; i < nlocal; i++){
      z_array[i] = SoftThreshold(z_array[i], 1./ctx->mu);
    }
    ierr = VecRestoreArray(z_k, &z_array);
    
    /* SoftThreshold
    ierr = VecWAXPY(z_k, 1., ctx->workRight[7], u_k); CHKERRQ(ierr); // xhat + u for shrinkage. 
    ierr = VecCopy(z_k, ctx->workRight[8]);CHKERRQ(ierr);
    ierr = VecScale(ctx->workRight[8], -1.); CHKERRQ(ierr);
    ierr = VecShift(z_k, - 1./(ctx->mu)); CHKERRQ(ierr);
    ierr = VecShift(ctx->workRight[8], - 1./(ctx->mu)); CHKERRQ(ierr);
    ierr = VecPointwiseMax(z_k, max_k, z_k); CHKERRQ(ierr);
    ierr = VecPointwiseMax(ctx->workRight[8], max_k, ctx->workRight[8]); CHKERRQ(ierr);
    ierr = VecAXPY(z_k, -1., ctx->workRight[8]); CHKERRQ(ierr);
    */

    // u update 
    ierr = VecWAXPY(ctx->workRight[10], -1., z_k, ctx->workRight[7]); CHKERRQ(ierr); // work[10] = x_hat - z
    ierr = VecAXPY(u_k, 1., ctx->workRight[10]); CHKERRQ(ierr); // u = u + x_hat - z
  }

//  ierr = VecView(x_k, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = VecNorm(x_k,NORM_1,C); CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Second order Taylor remainder convergence test */
PetscErrorCode TaylorTest(UserCtx ctx, Tao tao, Vec x, PetscReal *C)
{
  PetscReal h,J,temp;
  PetscInt i, j;
  PetscInt numValues;
  PetscReal Jx;
  PetscReal *Js, *hs;
  PetscReal minrate = PETSC_MAX_REAL;
  PetscReal gdotdx;
  MPI_Comm       comm = PetscObjectComm((PetscObject)x);
  Vec       g, dx, xhat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(x, &g);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xhat);CHKERRQ(ierr);

  /* choose a perturbation direction */
  ierr = VecDuplicate(x, &dx);CHKERRQ(ierr);
  ierr = VecSetRandom(dx,ctx->rctx); CHKERRQ(ierr);
  /* evaluate objective at x: J(x) */
  ierr = TaoComputeObjective(tao, x, &Jx);CHKERRQ(ierr);
  /* evaluate gradient at x, save in vector g */
  ierr = TaoComputeGradient(tao, x, g);CHKERRQ(ierr);

  ierr = VecDot(g, dx, &gdotdx);CHKERRQ(ierr);

  for (numValues = 0, h = ctx->hStart; h >= ctx->hMin; h *= ctx->hFactor) numValues++;
  ierr = PetscCalloc2(numValues, &Js, numValues, &hs);CHKERRQ(ierr);

  for (i = 0, h = ctx->hStart; h >= ctx->hMin; h *= ctx->hFactor, i++) {
    PetscReal Jxhat_comp, Jxhat_pred;

    ierr = VecWAXPY(xhat, h, dx, x);CHKERRQ(ierr);
    ierr = TaoComputeObjective(tao, xhat, &Jxhat_comp);CHKERRQ(ierr);
    /* J(\hat(x)) \approx J(x) + g^T (xhat - x) = J(x) + h * g^T dx */
    Jxhat_pred = Jx + h * gdotdx;

    /* Vector to dJdm scalar? Dot?*/
    J = PetscAbsReal(Jxhat_comp - Jxhat_pred);

    ierr = PetscPrintf (comm, "J(xhat): %g, predicted: %g, diff %g\n", (double) Jxhat_comp,
                        (double) Jxhat_pred, (double) J);CHKERRQ(ierr);
    Js[i] = J;
    hs[i] = h;
  }

  for (j=1; j<numValues; j++){
    temp = PetscLogReal(Js[j] / Js[j - 1]) / PetscLogReal (hs[j] / hs[j - 1]);
    ierr = PetscPrintf (comm, "Convergence rate step %D: %g\n", j - 1, (double) temp);CHKERRQ(ierr);
    minrate = PetscMin(minrate, temp);
  }
  //ierr = VecMin(ctx->workLeft[2],NULL, &O); CHKERRQ(ierr);

  /* If O is not ~2, then the test is wrong */  

  ierr = PetscFree2(Js, hs);CHKERRQ(ierr);
  *C = minrate;
  ierr = VecDestroy(&dx);CHKERRQ(ierr);
  ierr = VecDestroy(&xhat);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main (int argc, char** argv)
{
  UserCtx        ctx;
  Tao            tao;
  Vec            x;
  Mat            H;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ierr = ConfigureContext(ctx);CHKERRQ(ierr);

  /* Define two functions that could pass as objectives to TaoSetObjectiveRoutine(): one
   * for the misfit component, and one for the regularization component */
  /* ObjectiveMisfit() and ObjectiveRegularization() */

  /* Define a single function that calls both components adds them together: the complete objective,
   * in the absence of a Tao implementation that handles separability */
  /* ObjectiveComplete() */

  /* Construct the Tao object */
  ierr = TaoCreate(PETSC_COMM_WORLD, &tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAONM); CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(tao, ObjectiveComplete, (void *) ctx);CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine(tao, GradientComplete, (void *) ctx);CHKERRQ(ierr);
  ierr = MatDuplicate(ctx->W, MAT_SHARE_NONZERO_PATTERN, &H);CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao, H, H, HessianComplete, (void *) ctx);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx->F, NULL, &x);CHKERRQ(ierr);
  ierr = VecSet(x, 0.);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao, x);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

#if 0
  ierr = ADMMBasicPursuit(ctx, tao, x, &temp); CHKERRQ(ierr);
#endif

  /* solve */
  if (ctx->use_admm) {
    ierr = TaoSolveADMM(ctx,x); CHKERRQ(ierr);
  }
  else {
    ierr = TaoSolve(tao);CHKERRQ(ierr);
  }

  /* examine solution */
  VecViewFromOptions(x, NULL, "-view_sol");CHKERRQ(ierr);

  if (ctx->taylor) {
    PetscReal rate;

    ierr = TaylorTest(ctx, tao, x, &rate);CHKERRQ(ierr);
  }

  /* cleanup */
  ierr = MatDestroy(&H);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DestroyContext(&ctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args:

  test:
    suffix: l1_1
    args: -p 1 -tao_type lmvm -alpha 1. -epsilon 1.e-7 -m 64 -n 64 -view_sol -mat_format 1

  test:
    suffix: hessian_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -tao_type nls -tao_nls_ksp_monitor

  test:
    suffix: hessian_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -tao_type nls -tao_nls_ksp_monitor

  test:
    suffix: nm_1 
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -tao_type nm

  test:
    suffix: nm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -tao_type nm 

  test:
    suffix: lmvm_1 
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -tao_type lmvm 

  test:
    suffix: lmvm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -tao_type lmvm 

  test:
    suffix: hessian_admm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -use_admm -reg_tao_type nls -misfit_tao_type nls 

  test:
    suffix: hessian_admm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -use_admm -reg_tao_type nls -misfit_tao_type nls 

  test:
    suffix: nm_admm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -use_admm -reg_tao_type nm -misfit_tao_type nm 

  test:
    suffix: nm_admm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -use_admm -reg_tao_type nm -misfit_tao_type nm

  test:
    suffix: lmvm_admm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -use_admm -reg_tao_type lmvm -misfit_tao_type lmvm 

  test:
    suffix: lmvm_admm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -use_admm -reg_tao_type lmvm -misfit_tao_type lmvm

TEST*/
