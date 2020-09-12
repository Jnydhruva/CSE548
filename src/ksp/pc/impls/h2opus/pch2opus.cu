#include <petsc/private/pcimpl.h>
#include <petsc/private/matimpl.h>

typedef struct {
  Mat         M;
  PetscScalar s0;

  /* sampler for Newton-Schultz */
  Mat      S;
  PetscInt hyperorder;
  Vec      wns[4];
  Mat      wnsmat[4];

  /* convergence testing */
  Mat T;
  Vec w;

  /* Support for PCSetCoordinates */
  PetscInt  sdim;
  PetscInt  nlocc;
  PetscReal *coords;

  /* Newton-Schultz customization */
  PetscInt  maxits;
  PetscReal rtol,atol;
  PetscBool monitor;
  PetscBool useapproximatenorms;
} PC_H2OPUS;

static PetscErrorCode PCReset_H2OPUS(PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pch2opus->sdim  = 0;
  pch2opus->nlocc = 0;
  ierr = PetscFree(pch2opus->coords);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->M);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->T);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->w);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->S);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[2]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[3]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetCoordinates_H2OPUS(PC pc, PetscInt sdim, PetscInt nlocc, PetscReal *coords)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscBool      reset = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pch2opus->sdim && sdim == pch2opus->sdim && nlocc == pch2opus->nlocc) {
    ierr  = PetscArraycmp(pch2opus->coords,coords,sdim*nlocc,&reset);CHKERRQ(ierr);
    reset = (PetscBool)!reset;
  }
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&reset,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
  if (reset) {
    ierr = PCReset_H2OPUS(pc);CHKERRQ(ierr);
    ierr = PetscMalloc1(sdim*nlocc,&pch2opus->coords);CHKERRQ(ierr);
    ierr = PetscArraycpy(pch2opus->coords,coords,sdim*nlocc);CHKERRQ(ierr);
    pch2opus->sdim  = sdim;
    pch2opus->nlocc = nlocc;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_H2OPUS(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_H2OPUS(pc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_H2OPUS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"H2OPUS options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_h2opus_maxits","Maximum number of iterations for Newton-Schultz",NULL,pch2opus->maxits,&pch2opus->maxits,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_h2opus_monitor","Monitor Newton-Schultz convergence",NULL,pch2opus->monitor,&pch2opus->monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_h2opus_atol","Absolute tolerance",NULL,pch2opus->atol,&pch2opus->atol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_h2opus_rtol","Relative tolerance",NULL,pch2opus->rtol,&pch2opus->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_h2opus_hyperorder","Hyper power order of sampling",NULL,pch2opus->hyperorder,&pch2opus->hyperorder,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyKernel_H2OPUS(PC pc, Vec x, Vec y, PetscBool t)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MatAssembled(pch2opus->M,&flg);CHKERRQ(ierr);
  if (flg) {
    if (t) {
      ierr = MatMultTranspose(pch2opus->M,x,y);CHKERRQ(ierr);
    } else {
      ierr = MatMult(pch2opus->M,x,y);CHKERRQ(ierr);
    }
  } else { /* Not assembled, initial approximation */
    Mat A = pc->useAmat ? pc->mat : pc->pmat;

    if (pch2opus->s0 < 0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong scaling");
    /* X_0 = s0 * A^T */
    if (t) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);
    }
    ierr = VecScale(y,pch2opus->s0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyMatKernel_H2OPUS(PC pc, Mat X, Mat Y, PetscBool t)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MatAssembled(pch2opus->M,&flg);CHKERRQ(ierr);
  if (flg) {
    if (t) {
      ierr = MatTransposeMatMult(pch2opus->M,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
    } else {
      ierr = MatMatMult(pch2opus->M,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
    }
  } else { /* Not assembled, initial approximation */
    Mat A = pc->useAmat ? pc->mat : pc->pmat;

    if (pch2opus->s0 < 0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong scaling");
    /* X_0 = s0 * A^T */
    if (t) {
      ierr = MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
    } else {
      ierr = MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
    }
    ierr = MatScale(Y,pch2opus->s0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyMat_H2OPUS(PC pc, Mat X, Mat Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyMatKernel_H2OPUS(pc,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTransposeMat_H2OPUS(PC pc, Mat X, Mat Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyMatKernel_H2OPUS(pc,X,Y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_H2OPUS(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyKernel_H2OPUS(pc,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_H2OPUS(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyKernel_H2OPUS(pc,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* used to test norm of (M^-1 A - I) */
static PetscErrorCode MatMultKernel_MAmI(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS       *pch2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  pch2opus = (PC_H2OPUS*)pc->data;
  if (!pch2opus->w) {
    ierr = MatCreateVecs(pch2opus->M,&pch2opus->w,NULL);CHKERRQ(ierr);
  }
  A = pc->useAmat ? pc->mat : pc->pmat;
  if (t) {
    ierr = PCApplyTranspose_H2OPUS(pc,x,pch2opus->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,pch2opus->w,y);CHKERRQ(ierr);
  } else {
    ierr = MatMult(A,x,pch2opus->w);CHKERRQ(ierr);
    ierr = PCApply_H2OPUS(pc,pch2opus->w,y);CHKERRQ(ierr);
  }
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_MAmI(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_MAmI(A,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_MAmI(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_MAmI(A,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* HyperPower kernel:
Y = R = x
for i = 1 . . . l - 1 do
  R = (I - AXk)R
  Y = Y + R
Y = XkY
*/
static PetscErrorCode MatMultKernel_Hyper(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  A = pc->useAmat ? pc->mat : pc->pmat;
  pch2opus = (PC_H2OPUS*)pc->data;
  ierr = MatCreateVecs(pch2opus->M,pch2opus->wns[0] ? NULL : &pch2opus->wns[0],pch2opus->wns[1] ? NULL : &pch2opus->wns[1]);CHKERRQ(ierr);
  ierr = MatCreateVecs(pch2opus->M,pch2opus->wns[2] ? NULL : &pch2opus->wns[2],pch2opus->wns[3] ? NULL : &pch2opus->wns[3]);CHKERRQ(ierr);
  ierr = VecCopy(x,pch2opus->wns[0]);CHKERRQ(ierr);
  ierr = VecCopy(x,pch2opus->wns[3]);CHKERRQ(ierr);
  if (t) {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      ierr = MatMultTranspose(A,pch2opus->wns[0],pch2opus->wns[1]);CHKERRQ(ierr);
      ierr = PCApplyTranspose_H2OPUS(pc,pch2opus->wns[1],pch2opus->wns[2]);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(pch2opus->wns[3],-1.,1.,1.,pch2opus->wns[2],pch2opus->wns[0]);CHKERRQ(ierr);
    }
    ierr = PCApplyTranspose_H2OPUS(pc,pch2opus->wns[3],y);CHKERRQ(ierr);
  } else {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      ierr = PCApply_H2OPUS(pc,pch2opus->wns[0],pch2opus->wns[1]);CHKERRQ(ierr);
      ierr = MatMult(A,pch2opus->wns[1],pch2opus->wns[2]);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(pch2opus->wns[3],-1.,1.,1.,pch2opus->wns[2],pch2opus->wns[0]);CHKERRQ(ierr);
    }
    ierr = PCApply_H2OPUS(pc,pch2opus->wns[3],y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Hyper(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_Hyper(M,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Hyper(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_Hyper(M,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Hyper power kernel, MatMat version */
static PetscErrorCode MatMatMultKernel_Hyper(Mat M, Mat X, Mat Y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  A = pc->useAmat ? pc->mat : pc->pmat;
  pch2opus = (PC_H2OPUS*)pc->data;
  ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[2]);CHKERRQ(ierr);
  ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[3]);CHKERRQ(ierr);
  ierr = MatCopy(X,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatCopy(X,pch2opus->wnsmat[3],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (t) {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      ierr = MatTransposeMatMult(A,pch2opus->wnsmat[0],MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[1]);CHKERRQ(ierr);
      ierr = PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[1],pch2opus->wnsmat[2]);CHKERRQ(ierr);
      ierr = MatAXPY(pch2opus->wnsmat[0],-1.,pch2opus->wnsmat[2],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(pch2opus->wnsmat[3],1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[3],Y);CHKERRQ(ierr);
  } else {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      ierr = PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[0],pch2opus->wnsmat[1]);CHKERRQ(ierr);
      ierr = MatMatMult(A,pch2opus->wnsmat[1],MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[2]);CHKERRQ(ierr);
      ierr = MatAXPY(pch2opus->wnsmat[0],-1.,pch2opus->wnsmat[2],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(pch2opus->wnsmat[3],1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[3],Y);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&pch2opus->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_Hyper(Mat M, Mat X, Mat Y,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultKernel_Hyper(M,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Basic Newton-Schultz sampler: (2 * I - M * A ) * M */
static PetscErrorCode MatMultKernel_NS(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  A = pc->useAmat ? pc->mat : pc->pmat;
  pch2opus = (PC_H2OPUS*)pc->data;
  ierr = MatCreateVecs(pch2opus->M,pch2opus->wns[0] ? NULL : &pch2opus->wns[0],pch2opus->wns[1] ? NULL : &pch2opus->wns[1]);CHKERRQ(ierr);
  if (t) {
    ierr = PCApplyTranspose_H2OPUS(pc,x,y);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,y,pch2opus->wns[1]);CHKERRQ(ierr);
    ierr = PCApplyTranspose_H2OPUS(pc,pch2opus->wns[1],pch2opus->wns[0]);CHKERRQ(ierr);
    ierr = VecAXPBY(y,-1.,2.,pch2opus->wns[0]);CHKERRQ(ierr);
  } else {
    ierr = PCApply_H2OPUS(pc,x,y);CHKERRQ(ierr);
    ierr = MatMult(A,y,pch2opus->wns[0]);CHKERRQ(ierr);
    ierr = PCApply_H2OPUS(pc,pch2opus->wns[0],pch2opus->wns[1]);CHKERRQ(ierr);
    ierr = VecAXPBY(y,-1.,2.,pch2opus->wns[1]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_NS(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_NS(M,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_NS(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_NS(M,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* (2 * I - M * A ) * M, MatMat version */
static PetscErrorCode MatMatMultKernel_NS(Mat M, Mat X, Mat Y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  A = pc->useAmat ? pc->mat : pc->pmat;
  pch2opus = (PC_H2OPUS*)pc->data;
  ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  if (t) {
    ierr = PCApplyTransposeMat_H2OPUS(pc,X,Y);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(A,Y,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[1]);CHKERRQ(ierr);
    ierr = PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[1],pch2opus->wnsmat[0]);CHKERRQ(ierr);
    ierr = MatScale(Y,2.);CHKERRQ(ierr);
    ierr = MatAXPY(Y,-1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    ierr = PCApplyMat_H2OPUS(pc,X,Y);CHKERRQ(ierr);
    ierr = MatMatMult(A,Y,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[0]);CHKERRQ(ierr);
    ierr = PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[0],pch2opus->wnsmat[1]);CHKERRQ(ierr);
    ierr = MatScale(Y,2.);CHKERRQ(ierr);
    ierr = MatAXPY(Y,-1.,pch2opus->wnsmat[1],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&pch2opus->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_NS(Mat M, Mat X, Mat Y, void *)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultKernel_NS(M,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatNorm_H2OPUS(Mat,NormType,PetscReal*);

static PetscErrorCode PCH2OpusSetUpSampler_Private(PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  Mat            A = pc->useAmat ? pc->mat : pc->pmat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pch2opus->S) {
    PetscInt M,N,m,n;

    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)A),m,n,M,N,pc,&pch2opus->S);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(pch2opus->S,A,A);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    ierr = MatShellSetVecType(pch2opus->S,VECCUDA);CHKERRQ(ierr);
#endif
  }
  if (pch2opus->hyperorder >= 2) {
    ierr = MatShellSetOperation(pch2opus->S,MATOP_MULT,(void (*)(void))MatMult_Hyper);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_Hyper);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_Hyper,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_Hyper,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
  } else {
    ierr = MatShellSetOperation(pch2opus->S,MATOP_MULT,(void (*)(void))MatMult_NS);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_NS);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_NS,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_NS,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
  }
  ierr = MatPropagateSymmetryOptions(A,pch2opus->S);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* NS */
static PetscErrorCode PCSetUp_H2OPUS(PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  Mat            A = pc->useAmat ? pc->mat : pc->pmat;
  PetscErrorCode ierr;
  NormType       norm = NORM_2;
  PetscReal      initerr,err;

  PetscFunctionBegin;
  if (!pch2opus->T) {
    PetscInt M,N,m,n;

    ierr = MatGetSize(pc->pmat,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(pc->pmat,&m,&n);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)pc->pmat),m,n,M,N,pc,&pch2opus->T);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(pch2opus->T,pc->pmat,pc->pmat);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->T,MATOP_MULT,(void (*)(void))MatMult_MAmI);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->T,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_MAmI);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->T,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    ierr = MatShellSetVecType(pch2opus->T,VECCUDA);CHKERRQ(ierr);
#endif
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)pch2opus->T);CHKERRQ(ierr);
  }
  if (!pch2opus->M) {
    Mat       Ain = pc->pmat;
    PetscBool ish2opus,flg;
    PetscReal onenormA,infnormA;
    void      (*normfunc)(void);

    ierr = PetscObjectTypeCompare((PetscObject)Ain,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
    if (!ish2opus) {
      Ain  = pc->mat;
      ierr = PetscObjectTypeCompare((PetscObject)Ain,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
    }
    if (!ish2opus) {
      ierr = MatCreateH2OpusFromMat(A,pch2opus->sdim,pch2opus->coords,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&pch2opus->M);CHKERRQ(ierr);
    } else {
      ierr = MatDuplicate(Ain,MAT_SHARE_NONZERO_PATTERN,&pch2opus->M);CHKERRQ(ierr);
    }

    ierr = MatGetOperation(A,MATOP_NORM,&normfunc);CHKERRQ(ierr);
    if (!normfunc || pch2opus->useapproximatenorms) {
      ierr = MatSetOperation(A,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS);CHKERRQ(ierr);
    }
    ierr = MatNorm(A,NORM_1,&onenormA);CHKERRQ(ierr);
    ierr = MatGetOption(A,MAT_SYMMETRIC,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = MatNorm(A,NORM_INFINITY,&infnormA);CHKERRQ(ierr);
    } else infnormA = onenormA;
    ierr = MatSetOperation(A,MATOP_NORM,normfunc);CHKERRQ(ierr);
    pch2opus->s0 = 1./(infnormA*onenormA);
  }
  ierr = MatNorm(pch2opus->T,norm,&initerr);CHKERRQ(ierr);
  if (initerr > pch2opus->atol) {
    PetscInt i;

    ierr = PCH2OpusSetUpSampler_Private(pc);CHKERRQ(ierr);
    err  = initerr;
    if (pch2opus->monitor) { ierr = PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: %g %g\n",0,(double)err,(double)(err/initerr));CHKERRQ(ierr); }
    for (i = 0; i < pch2opus->maxits; i++) {
      Mat         M;
      const char* prefix;

      ierr = MatDuplicate(pch2opus->M,MAT_SHARE_NONZERO_PATTERN,&M);CHKERRQ(ierr);
      ierr = MatGetOptionsPrefix(M,&prefix);CHKERRQ(ierr);
      if (!prefix) {
        ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
        ierr = MatSetOptionsPrefix(M,prefix);CHKERRQ(ierr);
        ierr = MatAppendOptionsPrefix(M,"pc_h2opus_inv_");CHKERRQ(ierr);
      }
#if 0
  {
     Mat Sd1,Sd2,Id;
     PetscReal err;
     ierr = MatComputeOperator(pch2opus->S,MATDENSE,&Sd1);CHKERRQ(ierr);
     ierr = MatDuplicate(Sd1,MAT_COPY_VALUES,&Id);CHKERRQ(ierr);
     ierr = MatZeroEntries(Id);CHKERRQ(ierr);
     ierr = MatShift(Id,1.);CHKERRQ(ierr);
     ierr = MatMatMult(pch2opus->S,Id,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Sd2);CHKERRQ(ierr);
     ierr = MatAXPY(Sd2,-1.,Sd1,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
     ierr = MatNorm(Sd2,NORM_FROBENIUS,&err);CHKERRQ(ierr);
     ierr = PetscPrintf(PetscObjectComm((PetscObject)Sd2),"ERR %g\n",err);CHKERRQ(ierr);
     ierr = MatViewFromOptions(Sd2,NULL,"-Sd_view");CHKERRQ(ierr);
     ierr = MatDestroy(&Sd1);CHKERRQ(ierr);
     ierr = MatDestroy(&Sd2);CHKERRQ(ierr);
     ierr = MatDestroy(&Id);CHKERRQ(ierr);
  }
#endif
      ierr = MatH2OpusSetSamplingMat(M,pch2opus->S,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
      if (pc->setfromoptionscalled) {
        ierr = MatSetFromOptions(M);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if 0
      {
         Mat Md;
         ierr = MatComputeOperator(M,MATDENSE,&Md);CHKERRQ(ierr);
         ierr = MatViewFromOptions(Md,NULL,"-Md_view");CHKERRQ(ierr);
         ierr = MatDestroy(&Md);CHKERRQ(ierr);
         ierr = MatComputeOperator(pch2opus->S,MATDENSE,&Md);CHKERRQ(ierr);
         ierr = MatViewFromOptions(Md,NULL,"-Md_view");CHKERRQ(ierr);
         ierr = MatDestroy(&Md);CHKERRQ(ierr);
      }
#endif
      ierr = MatDestroy(&pch2opus->M);CHKERRQ(ierr);
      pch2opus->M = M;
      ierr = MatNorm(pch2opus->T,norm,&err);CHKERRQ(ierr);
      if (pch2opus->monitor) { ierr = PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: %g %g\n",i+1,(double)err,(double)(err/initerr));CHKERRQ(ierr); }
      if (err < pch2opus->atol || err < pch2opus->rtol*initerr) break;
    }
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PCCreate_H2OPUS(PC pc)
{
  PetscErrorCode ierr;
  PC_H2OPUS      *pch2opus;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&pch2opus);CHKERRQ(ierr);
  pc->data = (void*)pch2opus;

  pch2opus->atol       = 1.e-2;
  pch2opus->rtol       = 1.e-6;
  pch2opus->maxits     = 50;
  pch2opus->hyperorder = 1; /* default to basic NewtonSchultz */

  pc->ops->destroy        = PCDestroy_H2OPUS;
  pc->ops->setup          = PCSetUp_H2OPUS;
  pc->ops->apply          = PCApply_H2OPUS;
  pc->ops->matapply       = PCApplyMat_H2OPUS;
  pc->ops->applytranspose = PCApplyTranspose_H2OPUS;
  pc->ops->reset          = PCReset_H2OPUS;
  pc->ops->setfromoptions = PCSetFromOptions_H2OPUS;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_H2OPUS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
