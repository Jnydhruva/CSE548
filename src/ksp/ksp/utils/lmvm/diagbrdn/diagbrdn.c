#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  "Full" memory implementation of only the diagonal terms in a symmetric Broyden approximation.
*/

typedef struct {
  Vec invDnew, invD, BFGS, DFP, U, V, W;    /* work vectors for diagonal scaling */
  PetscReal *yts, *yty, *sts;               /* scalar arrays for recycling dot products */
  PetscReal theta, rho, alpha, beta, xi;        /* convex combination factors for the scalar or diagonal scaling */
  PetscReal delta, delta_min, delta_max, sigma;
  PetscInt sigma_hist;                      /* length of update history to be used for scaling */
  PetscBool allocated;
  PetscBool forward;
} Mat_DiagBrdn;

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_DiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  ierr = VecPointwiseMult(dX, ldb->invD, F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_DiagBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);
  ierr = VecPointwiseDivide(Z, X, ldb->invD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_DiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          old_k, i;
  PetscScalar       yty = 0.0, yts = 0.0, sts = 0.0;
  PetscScalar       ytDy, stDs, ytDs, yy_sum, ss_sum, ys_sum, sigma, denom, start;
  PetscScalar       curvtol;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAYPX(lmvm->Xprev, -1.0, X);CHKERRQ(ierr);
    ierr = VecAYPX(lmvm->Fprev, -1.0, F);CHKERRQ(ierr);
    /* Accept the update */
    old_k = lmvm->k;
    ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
    /* If we hit the memory limit, shift the yty and yts arrays */
    if (old_k == lmvm->k) {
      for (i = 0; i <= lmvm->k-1; ++i) {
        ldb->yty[i] = ldb->yty[i+1];
        ldb->yts[i] = ldb->yts[i+1];
        ldb->sts[i] = ldb->sts[i+1];
      }
    }
    /* Accumulate the latest yTy and yTs dot products */
    ierr = VecDotBegin(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[lmvm->k], lmvm->S[lmvm->k], &yts);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[lmvm->k], lmvm->S[lmvm->k], &sts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[lmvm->k], lmvm->S[lmvm->k], &yts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[lmvm->k], lmvm->S[lmvm->k], &sts);CHKERRQ(ierr);
    ldb->yty[lmvm->k] = PetscRealPart(yty);
    ldb->yts[lmvm->k] = PetscRealPart(yts);
    ldb->sts[lmvm->k] = PetscRealPart(sts);

    /* Restart upon curvature check */
    if (sts < lmvm->eps){
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * sts;
    }
    if (yts > curvtol){
      if (ldb->forward) {
        /* We are doing diagonal scaling of the forward Hessian B */
        ierr = VecCopy(ldb->invD, ldb->invDnew);CHKERRQ(ierr);
        ierr = VecReciprocal(ldb->invDnew);CHKERRQ(ierr);

        /*  V = y*y */
        ierr = VecPointwiseMult(ldb->V, lmvm->Y[lmvm->k], lmvm->Y[lmvm->k]);CHKERRQ(ierr);

        /*  W = inv(D)*s */
        ierr = VecPointwiseMult(ldb->W, ldb->invDnew, lmvm->S[lmvm->k]);CHKERRQ(ierr);
        ierr = VecDot(ldb->W, lmvm->S[lmvm->k], &stDs);CHKERRQ(ierr);

        /*  Safeguard stDs */
        stDs = PetscMax(1.0e-8, stDs);

        if (1.0 != ldb->theta) {
          /*  BFGS portion of the update */
          /*  U = (inv(D)*s)*(inv(D)*s) */
          ierr = VecPointwiseMult(ldb->U, ldb->W, ldb->W);CHKERRQ(ierr);

          /*  Assemble */
          ierr = VecAXPBY(ldb->BFGS, -1.0/stDs, 0.0, ldb->U);CHKERRQ(ierr);
        }
        if (0.0 != ldb->theta) {
          /*  DFP portion of the update */
          /*  U = inv(D)*s*y */
          ierr = VecPointwiseMult(ldb->U, ldb->W, lmvm->Y[lmvm->k]);CHKERRQ(ierr);

          /*  Assemble */
          ierr = VecAXPBY(ldb->DFP, stDs/ldb->yts[lmvm->k], 0.0, ldb->V);CHKERRQ(ierr);
          ierr = VecAXPY(ldb->DFP, -2.0, ldb->U);CHKERRQ(ierr);
        }
        /* Save computation time on special cases */
        if (0.0 == ldb->theta) {
          ierr = VecAXPY(ldb->invDnew, 1.0, ldb->BFGS);CHKERRQ(ierr);
        } else if (1.0 == ldb->theta) {
          ierr = VecAXPY(ldb->invDnew, 1.0/ldb->yts[lmvm->k], ldb->DFP);CHKERRQ(ierr);
        } else {
          /*  Broyden update Dkp1 = Dk + (1-theta)*P + theta*Q + y_i^2/yts*/
          ierr = VecAXPBYPCZ(ldb->invDnew, 1.0-ldb->theta, (ldb->theta)/ldb->yts[lmvm->k], 1.0, ldb->BFGS, ldb->DFP);CHKERRQ(ierr);
        }

        ierr = VecAXPY(ldb->invDnew, 1.0/ldb->yts[lmvm->k], ldb->V); CHKERRQ(ierr);
        /*  Obtain inverse and ensure positive definite */
        ierr = VecReciprocal(ldb->invDnew);CHKERRQ(ierr);
        ierr = VecAbs(ldb->invDnew);CHKERRQ(ierr);

      } else {
        /* Inverse Hessian update instead. */
        ierr = VecCopy(ldb->invD, ldb->invDnew);CHKERRQ(ierr);

        /*  V = s*s */
        ierr = VecPointwiseMult(ldb->V, lmvm->S[lmvm->k], lmvm->S[lmvm->k]);CHKERRQ(ierr);

        /*  W = D*y */
        ierr = VecPointwiseMult(ldb->W, ldb->invDnew, lmvm->Y[lmvm->k]);CHKERRQ(ierr);
        ierr = VecDot(ldb->W, lmvm->Y[lmvm->k], &ytDy);CHKERRQ(ierr);

        /*  Safeguard ytDy */
        ytDy = PetscMax(1.0e-8, ytDy); 
        /* Broyden diagonal is Dkp1 = Dk + ss^T/yts + (1-theta)*BFGS + theta*DFP */
        /* Let's build the components of this. Already have Dkp1 = Dk */
        /* A couple components are scaled by xi, a parameter inspired by the KD2015 method. Default 1. */
        if (1.0 != ldb->theta) {
          /*  BFGS portion of the update */
          /*  U = s*Dy */
          ierr = VecPointwiseMult(ldb->U, ldb->W, lmvm->S[lmvm->k]);CHKERRQ(ierr);

          /*  Assemble */
          ierr = VecAXPBY(ldb->BFGS, ytDy/ldb->yts[lmvm->k], 0.0, ldb->V);CHKERRQ(ierr);
          ierr = VecAXPY(ldb->BFGS, -(1+ldb->xi), ldb->U);CHKERRQ(ierr);
        }
        /* Save computation time on special cases */
        if (0.0 != ldb->theta) {
          /*  DFP portion of the update */

          /*  U = (inv(D)*y)*(inv(D)*y) */
          ierr = VecPointwiseMult(ldb->U, ldb->W, ldb->W);CHKERRQ(ierr);

          /*  Assemble */
          ierr = VecAXPBY(ldb->DFP, -ldb->xi*ldb->xi/ytDy, 0.0, ldb->U);CHKERRQ(ierr);
        }

        if (0.0 == ldb->theta) {
          ierr = VecAXPY(ldb->invDnew, 1.0, ldb->BFGS);CHKERRQ(ierr);
        } else if (1.0 == ldb->theta) {
          ierr = VecAXPY(ldb->invDnew, 1.0, ldb->DFP);CHKERRQ(ierr);
        } else {
          /*  Broyden update Dkp1 += (1-theta)*BFGS + theta*DFP */
          ierr = VecAXPBYPCZ(ldb->invDnew, (1.0-ldb->theta)/ldb->yts[lmvm->k], ldb->theta, 1.0, ldb->BFGS, ldb->DFP);CHKERRQ(ierr);
        }
        /* Dkp1 += ss^T/yts to the full update */
        ierr = VecAXPY(ldb->invDnew, 1.0/ldb->yts[lmvm->k], ldb->V); CHKERRQ(ierr);
        /*  Ensure positive definite */
        ierr = VecAbs(ldb->invDnew);CHKERRQ(ierr);
      }

      /*  Start with re-scaling on the newly computed diagonal */
      if (0.5 == ldb->beta) {
        if (1 == PetscMin(lmvm->nupdates, ldb->sigma_hist)) {
          ierr = VecPointwiseMult(ldb->V, lmvm->Y[0], ldb->invDnew);CHKERRQ(ierr);
          ierr = VecDot(ldb->V, lmvm->Y[0], &ytDy);CHKERRQ(ierr);
          yy_sum = PetscRealPart(ytDy);

          ierr = VecPointwiseDivide(ldb->W, lmvm->S[0], ldb->invDnew);CHKERRQ(ierr);
          ierr = VecDot(ldb->W, lmvm->S[0], &stDs);CHKERRQ(ierr);
          ss_sum = PetscRealPart(stDs);

          ys_sum = ldb->yts[0];
        } else {
          ierr = VecCopy(ldb->invDnew, ldb->U);CHKERRQ(ierr);
          /* TODO: why is there a reciprocal? */
          /* ierr = VecReciprocal(ldb->U);CHKERRQ(ierr); */

          /*  Compute summations for scalar scaling */
          yy_sum = 0;       /*  No safeguard required */
          ys_sum = 0;       /*  No safeguard required */
          ss_sum = 0;       /*  No safeguard required */
          start = PetscMax(0, lmvm->k-ldb->sigma_hist+1);
          for (i = start; i < PetscMin(lmvm->nupdates, ldb->sigma_hist); ++i) {
            ierr = VecPointwiseMult(ldb->V, lmvm->Y[i], ldb->U);CHKERRQ(ierr);
            ierr = VecDot(ldb->V, lmvm->Y[i], &ytDy);CHKERRQ(ierr);
            yy_sum += PetscRealPart(ytDy);

            ierr = VecPointwiseMult(ldb->W, lmvm->S[i], ldb->U);CHKERRQ(ierr);
            ierr = VecDot(ldb->W, lmvm->S[i], &stDs);CHKERRQ(ierr);
            ss_sum += PetscRealPart(stDs);
            ys_sum += ldb->yts[i];
          }
        }
      } else if (0.0 == ldb->beta) {
        if (1 == PetscMin(lmvm->nupdates, ldb->sigma_hist)) {
          /*  Compute summations for scalar scaling */
          ierr = VecPointwiseDivide(ldb->W, lmvm->S[0], ldb->invDnew);CHKERRQ(ierr);

          ierr = VecDot(ldb->W, lmvm->Y[0], &ytDs);CHKERRQ(ierr);
          ys_sum = PetscRealPart(ytDs);
          ierr = VecDot(ldb->W, ldb->W, &stDs);CHKERRQ(ierr);
          ss_sum = PetscRealPart(stDs);
          yy_sum = ldb->yty[0];
        } else {
          ierr = VecCopy(ldb->invDnew, ldb->U);CHKERRQ(ierr);
          /* TODO: why is there a reciprocal? */
          ierr = VecReciprocal(ldb->U);CHKERRQ(ierr);

          /*  Compute summations for scalar scaling */
          yy_sum = 0;       /*  No safeguard required */
          ys_sum = 0;       /*  No safeguard required */
          ss_sum = 0;       /*  No safeguard required */
          start = PetscMax(0, lmvm->k-ldb->sigma_hist+1);
          for (i = start; i < PetscMin(lmvm->nupdates, ldb->sigma_hist); ++i) {
            ierr = VecPointwiseMult(ldb->W, lmvm->S[i], ldb->U);CHKERRQ(ierr);
            ierr = VecDot(ldb->W, lmvm->Y[i], &ytDs);CHKERRQ(ierr);
            ys_sum += PetscRealPart(ytDs);

            ierr = VecDot(ldb->W, ldb->W, &stDs);CHKERRQ(ierr);
            ss_sum += PetscRealPart(stDs);

            yy_sum += ldb->yty[i];
          }
        }
      } else if (1.0 == ldb->beta) {
        /*  Compute summations for scalar scaling */
        yy_sum = 0; /*  No safeguard required */
        ys_sum = 0; /*  No safeguard required */
        ss_sum = 0; /*  No safeguard required */
        start = PetscMax(0, lmvm->k-ldb->sigma_hist+1);
        for (i = start; i < PetscMin(lmvm->nupdates, ldb->sigma_hist); ++i) {
          ierr = VecPointwiseMult(ldb->V, lmvm->Y[i], ldb->invDnew);CHKERRQ(ierr);
          ierr = VecDot(ldb->V, lmvm->S[i], &ytDs);CHKERRQ(ierr);
          ys_sum += PetscRealPart(ytDs);

          ierr = VecDot(ldb->V, ldb->V, &ytDy);CHKERRQ(ierr);
          yy_sum += PetscRealPart(ytDy);

          ss_sum += ldb->sts[i];
        }
      } else {
        ierr = VecCopy(ldb->invDnew, ldb->U);CHKERRQ(ierr);
        ierr = VecPow(ldb->U, ldb->beta-1);CHKERRQ(ierr);

        /*  Compute summations for scalar scaling */
        yy_sum = 0; /*  No safeguard required */
        ys_sum = 0; /*  No safeguard required */
        ss_sum = 0; /*  No safeguard required */
        start = PetscMax(0, lmvm->k-ldb->sigma_hist+1);
        for (i = start; i < PetscMin(lmvm->nupdates, ldb->sigma_hist); ++i) {
          ierr = VecPointwiseMult(ldb->V, ldb->invDnew, lmvm->Y[i]);CHKERRQ(ierr);
          ierr = VecPointwiseMult(ldb->W, ldb->U, lmvm->S[i]);CHKERRQ(ierr);

          ierr = VecDot(ldb->V, ldb->V, &ytDy);CHKERRQ(ierr);
          ierr = VecDot(ldb->V, ldb->W, &ytDs);CHKERRQ(ierr);
          ierr = VecDot(ldb->W, ldb->W, &stDs);CHKERRQ(ierr);

          yy_sum += PetscRealPart(ytDy);
          ys_sum += PetscRealPart(ytDs);
          ss_sum += PetscRealPart(stDs);
        }
      }

      if (0.0 == ldb->alpha) {
        /*  Safeguard ys_sum  */
        ys_sum = PetscMax(1.0e-8, ys_sum);
        sigma = ss_sum / ys_sum;
      } else if (-1.0 == ldb->alpha){
        sigma = 1.0;
      } else if (1.0 == ldb->alpha) {
        /*  Safeguard yy_sum  */
        ys_sum = PetscMax(1.0e-8, ys_sum);
        sigma = ys_sum / yy_sum;
      } else {
        denom = 2*ldb->alpha*yy_sum;

        /*  Safeguard denom */
        denom = PetscMax(1.0e-8, denom);

        sigma = ((2*ldb->alpha-1)*ys_sum + PetscSqrtReal((2*ldb->alpha-1)*(2*ldb->alpha-1)*ys_sum*ys_sum - 4*ldb->alpha*(ldb->alpha-1)*yy_sum*ss_sum)) / denom;
      }

      /*  If Q has small values, then Q^(r_beta - 1)
       can have very large values.  Hence, ys_sum
       and ss_sum can be infinity.  In this case,
       sigma can either be not-a-number or infinity. */

      if (PetscIsInfOrNanReal(sigma)) {
        /*  sigma is not-a-number; skip rescaling */
      } else if (!sigma) {
        /*  sigma is zero; this is a bad case; skip rescaling */
      } else {
        /*  sigma is positive */
        ierr = VecScale(ldb->invDnew, sigma);CHKERRQ(ierr);
      }

      /* Combine the old diagonal and the new diagonal using a convex limiter */
      if (1.0 == ldb->rho) {
        ierr = VecCopy(ldb->invDnew, ldb->invD);CHKERRQ(ierr);
      } else if (ldb->rho) {
        ierr = VecAXPBY(ldb->invD, 1.0-ldb->rho, ldb->rho, ldb->invDnew);CHKERRQ(ierr);
      }
    } else {
      ierr = MatLMVMReset(B, PETSC_FALSE); CHKERRQ(ierr);
    }
    /* End DiagBrdn update */

  }
  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_DiagBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *bctx = (Mat_DiagBrdn*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_DiagBrdn      *mctx = (Mat_DiagBrdn*)mdata->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  mctx->theta = bctx->theta;
  mctx->alpha = bctx->alpha;
  mctx->beta = bctx->beta;
  mctx->rho = bctx->rho;
  mctx->delta = bctx->delta;
  mctx->delta_min = bctx->delta_min;
  mctx->delta_max = bctx->delta_max;
  mctx->sigma = bctx->sigma;
  mctx->sigma_hist = bctx->sigma_hist;
  ierr = VecCopy(bctx->invD, mctx->invD);CHKERRQ(ierr);
  for (i=0; i<=bdata->k; ++i) {
    mctx->yty[i] = bctx->yty[i];
    mctx->yts[i] = bctx->yts[i];
    mctx->sts[i] = mctx->sts[i];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_LMVMDiagBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn       *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Restricted Broyden method for approximating SPD Jacobian actions (MATLMVMDIAGBRDN)");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_theta","(developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling","",ldb->theta,&ldb->theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_rho","(developer) update limiter in the J0 scaling","",ldb->rho,&ldb->rho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_alpha","(developer) convex ratio in the J0 scaling","",ldb->alpha,&ldb->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_xi","(developer) parameter in the inverse Broyden scaling intended for the Kou-Dai BNCG method","",ldb->xi,&ldb->xi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_lmvm_forward","Forward -> Update diagonal scaling for B. Else -> diagonal scaling for H.","",ldb->forward,&ldb->forward,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_beta","(developer) exponential factor in the diagonal J0 scaling","",ldb->beta,&ldb->beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_lmvm_sigma_hist","(developer) number of past updates to use in the default J0 scalar","",ldb->sigma_hist,&ldb->sigma_hist,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ldb->theta < 0.0) || (ldb->theta > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the diagonal J0 scale cannot be outside the range of [0, 1]");
  if (ldb->alpha != -1.0){
    if ((ldb->alpha < 0.0) || (ldb->alpha > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio in the J0 scaling cannot be outside the range of [0, 1], or -1 for no scaling");
  }
  if ((ldb->rho < 0.0) || (ldb->rho > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex update limiter in the J0 scaling cannot be outside the range of [0, 1]");
  if (ldb->sigma_hist < 0) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "J0 scaling history length cannot be negative");
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/

static PetscErrorCode MatReset_DiagBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecSet(ldb->invD, ldb->delta); CHKERRQ(ierr);
  if (destructive && ldb->allocated) {
    ierr = PetscFree3(ldb->yty, ldb->yts, ldb->sts);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->invDnew);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->invD);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->BFGS);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->DFP);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->U);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->V);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->W);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_DiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = PetscMalloc3(lmvm->m, &ldb->yty, lmvm->m, &ldb->yts, lmvm->m, &ldb->sts);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->invDnew);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->invD);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->BFGS);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->DFP);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->U);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->V);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->W);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_DiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ldb->allocated) {
    ierr = PetscFree3(ldb->yty, ldb->yts, ldb->sts);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->invDnew);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->invD);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->BFGS);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->DFP);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->U);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->V);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->W);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_DiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = PetscMalloc3(lmvm->m, &ldb->yty, lmvm->m, &ldb->yts, lmvm->m, &ldb->sts);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->invDnew);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->invD);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->BFGS);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->DFP);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->U);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->V);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->W);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_DiagBrdn      *ldb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMDIAGBRDN);CHKERRQ(ierr);
  B->ops->setup = MatSetUp_DiagBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDiagBrdn;
  B->ops->destroy = MatDestroy_DiagBrdn;
  B->ops->solve = MatSolve_DiagBrdn;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->m = 1;
  lmvm->ops->allocate = MatAllocate_DiagBrdn;
  lmvm->ops->reset = MatReset_DiagBrdn;
  lmvm->ops->mult = MatMult_DiagBrdn;
  lmvm->ops->update = MatUpdate_DiagBrdn;
  lmvm->ops->copy = MatCopy_DiagBrdn;

  ierr = PetscNewLog(B, &ldb);CHKERRQ(ierr);
  lmvm->ctx = (void*)ldb;
  ldb->theta = 0.0;
  ldb->xi = 1.0;
  ldb->alpha = 1.0;
  ldb->rho = 1.0;
  ldb->forward = PETSC_TRUE;
  ldb->beta = 0.5;
  ldb->sigma = 1.0;
  ldb->delta = 1.0;
  ldb->delta_min = 1e-7;
  ldb->delta_max = 100.0;
  ldb->sigma_hist = 1;
  ldb->allocated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDiagBrdn - DiagBrdn creates a symmetric Broyden-type diagonal matrix used 
   for approximating Hessians. It consists of a convex combination of DFP and BFGS 
   diagonal approximation schemes, such that DiagBrdn = (1-theta)*BFGS + theta*DFP. 
   To preserve symmetric positive-definiteness, we restrict theta to be in [0, 1]. 
   We also ensure positive definiteness by taking the VecAbs() of the final vector.

   There are two ways of approximating the diagonal: using the forward (B) update 
   schemes for BFGS and DFP and then taking the inverse, or directly working with 
   the inverse (H) update schemes for the BFGS and DFP updates, derived using the 
   Sherman-Morrison-Woodbury formula. We have implemented both, controlled by a 
   parameter below.

   In order to use the DiagBrdn matrix with other vector types, i.e. doing MatMults 
   and MatSolves, the matrix must first be created using MatCreate() and MatSetType(), 
   followed by MatLMVMAllocate(). Then it will be available for updating 
   (via MatLMVMUpdate) in one's favored solver implementation. 
   This allows for MPI compatibility.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

   Options Database Keys:
.   -mat_lmvm_theta - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
.   -mat_lmvm_rho - (developer) update limiter for the J0 scaling
.   -mat_lmvm_alpha - (developer) coefficient factor for the quadratic subproblem in J0 scaling
.   -mat_lmvm_beta - (developer) exponential factor for the diagonal J0 scaling
.   -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling
.   -mat_lmvm_forward - (developer) whether or not to use the forward or backward Broyden update to the diagonal

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMDIAGBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(),
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMDiagBrdn(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMDIAGBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
