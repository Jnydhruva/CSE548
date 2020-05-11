#include <petscmat.h>

static char help[] = "Tests MatMatMult with MAT_REUSE_MATRIX and already allocated dense result.\n\n";

static PetscErrorCode CheckLocal(Mat A, Mat B, PetscScalar *a, PetscScalar *b)
{
  PetscErrorCode ierr;
  PetscBool      wA = PETSC_FALSE, wB = PETSC_FALSE;

  PetscFunctionBegin;
  if (a) {
    const PetscScalar *Aa;
    ierr = MatDenseGetArrayRead(A,&Aa);CHKERRQ(ierr);
    wA   = (PetscBool)(a != Aa);
    ierr = MatDenseRestoreArrayRead(A,&Aa);CHKERRQ(ierr);
  }
  if (b) {
    const PetscScalar *Bb;
    ierr = MatDenseGetArrayRead(B,&Bb);CHKERRQ(ierr);
    wB   = (PetscBool)(b != Bb);
    ierr = MatDenseRestoreArrayRead(B,&Bb);CHKERRQ(ierr);
  }
  if (wA || wB) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong array in first Mat? %d, Wrong array in second Mat? %d",wA,wB);
  PetscFunctionReturn(0);
}

PetscErrorCode proj_mult(Mat S, Vec X, Vec Y)
{
  Mat            A,R,P;
  Vec            Ax,Ay;
  Vec            Px,Py;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)S,"_testme_A",(PetscObject*)&A);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)S,"_testme_P",(PetscObject*)&P);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)S,"_testme_R",(PetscObject*)&R);CHKERRQ(ierr);
  if (!A) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing matrix");
  if (!R && !P) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Missing projectors");
  if (R && P) SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_PLIB,"Both projectors");
  ierr = MatCreateVecs(A,&Ax,&Ay);CHKERRQ(ierr);
  if (R) {
    ierr = MatCreateVecs(R,&Py,&Px);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(P,&Px,&Py);CHKERRQ(ierr);
  }
  ierr = VecCopy(X,Px);CHKERRQ(ierr);
  if (P) {
    ierr = MatMult(P,Px,Py);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(R,Px,Py);CHKERRQ(ierr);
  }
  ierr = VecCopy(Py,Ax);CHKERRQ(ierr);
  ierr = MatMult(A,Ax,Ay);CHKERRQ(ierr);
  ierr = VecCopy(Ay,Py);CHKERRQ(ierr);
  if (P) {
    ierr = MatMultTranspose(P,Py,Px);CHKERRQ(ierr);
  } else {
    ierr = MatMult(R,Py,Px);CHKERRQ(ierr);
  }
  ierr = VecCopy(Px,Y);CHKERRQ(ierr);
  ierr = VecDestroy(&Px);CHKERRQ(ierr);
  ierr = VecDestroy(&Py);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&Ay);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyPtShellPMultNumeric(Mat S, Mat P, Mat PtAP)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)PtAP,"_testme_A",(PetscObject)A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)PtAP,"_testme_P",(PetscObject)P);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)PtAP,"_testme_R",NULL);CHKERRQ(ierr);
  ierr = MatShellSetOperation(PtAP,MATOP_MULT,(void (*)(void))proj_mult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyRShellRtMultNumeric(Mat S, Mat R, Mat RARt)
{
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)RARt,"_testme_A",(PetscObject)A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)RARt,"_testme_R",(PetscObject)R);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)RARt,"_testme_P",NULL);CHKERRQ(ierr);
  ierr = MatShellSetOperation(RARt,MATOP_MULT,(void (*)(void))proj_mult);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* cannot use directly C in this test, since MatMatMult numeric is this function itself
   However, users will know how to access B and C (e.g. MatDenseGetArray) and perform the proper computation */
PetscErrorCode MyMatShellMatMultNumeric(Mat S, Mat B, Mat C)
{
  PetscErrorCode ierr;
  Mat            A,CT;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CT);CHKERRQ(ierr);
  ierr = MatCopy(CT,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&CT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* cannot use directly C in this test, since MatTransposeMatMult numeric is this function itself
   However, users will know how to access B and C (e.g. MatDenseGetArray) and perform the proper computation */
PetscErrorCode MyMatTransposeShellMatMultNumeric(Mat S, Mat B, Mat C)
{
  PetscErrorCode ierr;
  Mat            A,CT;

  PetscFunctionBegin;
  ierr = MatShellGetContext(S,(void**)&A);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CT);CHKERRQ(ierr);
  ierr = MatCopy(CT,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&CT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            X,B,A,T,T2,PtAP = NULL,RARt = NULL, R = NULL;
  PetscInt       m,n,k,M = 10,N = 10,K = 5, ldx = 3, ldb = 5, ldr = 4;
  const char     *deft = MATAIJ;
  char           mattype[256];
  PetscBool      flg,symm = PETSC_FALSE,testtt = PETSC_TRUE, testnest = PETSC_TRUE, testtranspose = PETSC_TRUE, testcircular = PETSC_FALSE, local = PETSC_TRUE;
  PetscBool      xgpu = PETSC_FALSE, bgpu = PETSC_FALSE, testshellops = PETSC_FALSE, testproj = PETSC_TRUE, testrart = PETSC_TRUE;
  PetscScalar    *dataX = NULL,*dataB = NULL, *dataR = NULL;
  PetscScalar    *aX,*aB;
  PetscReal      err;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-K",&K,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-symm",&symm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-local",&local,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ldx",&ldx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ldb",&ldb,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testtranspose",&testtranspose,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testnest",&testnest,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testtt",&testtt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testcircular",&testcircular,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testshellops",&testshellops,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testproj",&testproj,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-testrart",&testrart,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-xgpu",&xgpu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-bgpu",&bgpu,NULL);CHKERRQ(ierr);
  if (M != N) testproj = PETSC_FALSE;
#if defined(PETSC_USE_COMPLEX)
  testtranspose = PETSC_FALSE;
  testtt = PETSC_FALSE;
#endif
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetRandom(A,NULL);CHKERRQ(ierr);
  if (M==N && symm) {
    Mat AT;

    ierr = MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&AT);CHKERRQ(ierr);
    ierr = MatAXPY(A,1.0,AT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&AT);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatViewFromOptions(A,NULL,"-A_init_view");CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","","");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-A_mat_type","Matrix type","MatSetType",MatList,deft,mattype,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    Mat A2;

    /* MATSEQAIJCUSPARSE does not support MAT_INITIAL_MATRIX */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
    ierr = MatConvert(A,mattype,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatMultEqual(A,A2,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      Mat AE,A2E;

      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with convert\n");CHKERRQ(ierr);
      ierr = MatComputeOperator(A,MATDENSE,&AE);CHKERRQ(ierr);
      ierr = MatComputeOperator(A2,MATDENSE,&A2E);CHKERRQ(ierr);
      ierr = MatView(AE,NULL);CHKERRQ(ierr);
      ierr = MatView(A2E,NULL);CHKERRQ(ierr);
      ierr = MatAXPY(A2E,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(A2E,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&A2E);CHKERRQ(ierr);
      ierr = MatDestroy(&AE);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A2);CHKERRQ(ierr);
  }
  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);

  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (local) {
    ierr = PetscMalloc1(PetscMax((m+ldx)*K,1),&dataX);CHKERRQ(ierr);
    ierr = PetscMalloc1(PetscMax((n+ldb)*K,1),&dataB);CHKERRQ(ierr);
  }
  ierr = MatCreateDense(PETSC_COMM_WORLD,n,PETSC_DECIDE,N,K,dataB,&B);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,K,dataX,&X);CHKERRQ(ierr);
  if (local) {
    ierr = MatDenseSetLDA(X,m+ldx);CHKERRQ(ierr);
    ierr = MatDenseSetLDA(B,n+ldb);CHKERRQ(ierr);
  }

  /* store pointer to dense data for testing */
  ierr = MatDenseGetArrayRead(B,(const PetscScalar**)&dataB);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(X,(const PetscScalar**)&dataX);CHKERRQ(ierr);
  aX   = dataX;
  aB   = dataB;
  ierr = MatDenseRestoreArrayRead(B,(const PetscScalar**)&dataB);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(X,(const PetscScalar**)&dataX);CHKERRQ(ierr);
  if (local) {
    dataX = aX;
    dataB = aB;
  }

  ierr = MatSetRandom(B,NULL);CHKERRQ(ierr);
  /* convert to CUDA if needed */
  if (bgpu) {
    ierr = MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  }
  if (xgpu) {
    ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatConvert(X,MATDENSECUDA,MAT_INPLACE_MATRIX,&X);CHKERRQ(ierr);
  }
  ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);

  /* Test reusing a previously allocated dense buffer */
  ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
  ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
  ierr = MatMatMultEqual(A,B,X,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage\n");CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
    ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(T,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
  }

  /* Test projection operations (PtAP and RARt) */
  if (testproj) {
    ierr = MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PtAP);CHKERRQ(ierr);
    ierr = MatPtAPMultEqual(A,B,PtAP,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP\n");CHKERRQ(ierr);
      ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(B,T,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T2);CHKERRQ(ierr);
      ierr = MatAXPY(T2,-1.0,PtAP,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T2,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T2);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = MatGetLocalSize(B,NULL,&k);CHKERRQ(ierr);
    ierr = PetscMalloc1(PetscMax((k+ldr)*M,1),&dataR);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,m,K,M,dataR,&R);CHKERRQ(ierr);
    ierr = MatSetRandom(R,NULL);CHKERRQ(ierr);
    if (testrart) { /* fails for AIJCUSPARSE */
      ierr = MatRARt(A,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RARt);CHKERRQ(ierr);
      ierr = MatRARtMultEqual(A,R,RARt,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with RARt\n");CHKERRQ(ierr);
        ierr = MatMatTransposeMult(A,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
        ierr = MatMatMult(R,T,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T2);CHKERRQ(ierr);
        ierr = MatAXPY(T2,-1.0,RARt,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatView(T2,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&T2);CHKERRQ(ierr);
        ierr = MatDestroy(&T);CHKERRQ(ierr);
      }
    }
  }

  /* Test MatDenseGetColumnVec and friends */
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
  ierr = MatDuplicate(T,MAT_DO_NOT_COPY_VALUES,&T2);CHKERRQ(ierr);
  for (k=0;k<K;k++) {
    Vec Xv,Tv,T2v;

    ierr = MatDenseGetColumnVecRead(X,k,&Xv);CHKERRQ(ierr);
    ierr = MatDenseGetColumnVec(T,k,&Tv);CHKERRQ(ierr);
    ierr = MatDenseGetColumnVecWrite(T2,k,&T2v);CHKERRQ(ierr);
    ierr = VecCopy(Xv,T2v);CHKERRQ(ierr);
    ierr = VecAXPY(Tv,-1.,Xv);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecRead(X,k,&Xv);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVec(T,k,&Tv);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumnVecWrite(T2,k,&T2v);CHKERRQ(ierr);
  }
  ierr = MatNorm(T,NORM_FROBENIUS,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetColumnVec\n");CHKERRQ(ierr);
    ierr = MatView(T,NULL);CHKERRQ(ierr);
  }
  ierr = MatAXPY(T2,-1.,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(T2,NORM_FROBENIUS,&err);CHKERRQ(ierr);
  if (err > PETSC_SMALL) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with MatDenseGetColumnVecWrite\n");CHKERRQ(ierr);
    ierr = MatView(T2,NULL);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  ierr = MatDestroy(&T2);CHKERRQ(ierr);

  /* Test with MatShell */
  ierr = MatConvert(A,MATSHELL,MAT_INITIAL_MATRIX,&T2);CHKERRQ(ierr);
  if (testshellops) { /* test callbacks for user defined MatProducts */
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_AB,NULL,MyMatShellMatMultNumeric,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_AB,NULL,MyMatShellMatMultNumeric,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_AtB,NULL,MyMatTransposeShellMatMultNumeric,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_AtB,NULL,MyMatTransposeShellMatMultNumeric,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
    if (testproj) {
      ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_PtAP,NULL,MyPtShellPMultNumeric,MATDENSE,MATSHELL);CHKERRQ(ierr);
      ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_PtAP,NULL,MyPtShellPMultNumeric,MATDENSECUDA,MATSHELL);CHKERRQ(ierr);
      ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_RARt,NULL,MyRShellRtMultNumeric,MATDENSE,MATSHELL);CHKERRQ(ierr);
      ierr = MatShellSetMatProductOperation(T2,MATPRODUCT_RARt,NULL,MyRShellRtMultNumeric,MATDENSECUDA,MATSHELL);CHKERRQ(ierr);
    }
  }
  ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
  ierr = MatMatMult(T2,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
  ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
  ierr = MatMatMultEqual(T2,B,X,10,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MATSHELL)\n");CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
    ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(T,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
  }
  if (testproj) {
    ierr = MatPtAPMultEqual(T2,B,PtAP,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP\n");CHKERRQ(ierr);
    }
    if (testshellops) {
      ierr = MatPtAP(T2,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatPtAPMultEqual(T2,B,T,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        Mat TE;

        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with PtAP (user defined)\n");CHKERRQ(ierr);
        ierr = MatComputeOperator(T,MATDENSE,&TE);CHKERRQ(ierr);
        ierr = MatView(TE,NULL);CHKERRQ(ierr);
        ierr = MatView(PtAP,NULL);CHKERRQ(ierr);
        ierr = MatAXPY(TE,-1.0,PtAP,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatView(TE,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&TE);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    if (RARt) {
      ierr = MatRARtMultEqual(T2,R,RARt,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with RARt\n");CHKERRQ(ierr);
      }
    }
    if (testshellops) {
      ierr = MatRARt(T2,R,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatRARtMultEqual(T2,R,T,10,&flg);CHKERRQ(ierr);
      if (!flg) {
        Mat TE;

        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with RARt (user defined)\n");CHKERRQ(ierr);
        ierr = MatComputeOperator(T,MATDENSE,&TE);CHKERRQ(ierr);
        ierr = MatView(TE,NULL);CHKERRQ(ierr);
        if (RARt) {
          ierr = MatView(RARt,NULL);CHKERRQ(ierr);
          ierr = MatAXPY(TE,-1.0,RARt,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
          ierr = MatView(TE,NULL);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&TE);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
  }

  if (testtranspose) {
    ierr = MatTransposeMatMult(T2,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatTransposeMatMultEqual(T2,X,B,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with reusage (MatTranspose, MATSHELL)\n");CHKERRQ(ierr);
      ierr = MatTransposeMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&T2);CHKERRQ(ierr);

  if (testnest) { /* test with MatNest */
    Mat        NA;
    const char *vtype;

    ierr = MatCreateNest(PETSC_COMM_WORLD,1,NULL,1,NULL,&A,&NA);CHKERRQ(ierr);
    /* needed to test against CUSPARSE matrices */
    ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
    ierr = MatSetVecType(NA,vtype);CHKERRQ(ierr);
    ierr = MatViewFromOptions(NA,NULL,"-NA_view");CHKERRQ(ierr);
    ierr = MatMatMult(NA,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatMatMultEqual(NA,B,X,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with Nest\n");CHKERRQ(ierr);
      ierr = MatMatMult(NA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&NA);CHKERRQ(ierr);
  }

  if (testtranspose) { /* test with Transpose */
    Mat TA;

    ierr = MatCreateHermitianTranspose(A,&TA);CHKERRQ(ierr);
    ierr = MatMatMult(TA,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatMatMultEqual(TA,X,B,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose\n");CHKERRQ(ierr);
      ierr = MatMatMult(TA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&TA);CHKERRQ(ierr);
  }

  if (testtt) { /* test with Transpose(Transpose) */
    Mat TA, TTA;

    ierr = MatCreateHermitianTranspose(A,&TA);CHKERRQ(ierr);
    ierr = MatCreateHermitianTranspose(TA,&TTA);CHKERRQ(ierr);
    ierr = MatMatMult(TTA,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatMatMultEqual(TTA,B,X,10,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error with Transpose(Transpose)\n");CHKERRQ(ierr);
      ierr = MatMatMult(TTA,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatAXPY(T,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatView(T,NULL);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&TA);CHKERRQ(ierr);
    ierr = MatDestroy(&TTA);CHKERRQ(ierr);
  }

  if (testcircular) { /* test circular */
    Mat AB;

    ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AB);CHKERRQ(ierr);
    ierr = MatMatMult(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&X);CHKERRQ(ierr);
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    if (M == N && N == K) {
      ierr = MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    } else {
      ierr = MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
    }
    ierr = CheckLocal(B,X,aB,aX);CHKERRQ(ierr);
    ierr = MatDestroy(&AB);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&PtAP);CHKERRQ(ierr);
  ierr = MatDestroy(&RARt);CHKERRQ(ierr);
  ierr = PetscFree(dataX);CHKERRQ(ierr);
  ierr = PetscFree(dataB);CHKERRQ(ierr);
  ierr = PetscFree(dataR);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 1
    args: -local {{0 1}} -testshellops

  test:
    output_file: output/ex70_1.out
    requires: cuda
    suffix: 1_cuda
    args: -local {{0 1}} -xgpu {{0 1}} -bgpu {{0 1}} -A_mat_type {{seqaijcusparse seqaij}} -testnest 0 -testshellops {{0 1}} -testrart 0

  test:
    TODO: VecGetSubVector seems broken with CUDA
    output_file: output/ex70_1.out
    requires: cuda
    suffix: 1_cuda_broken
    args: -local {{0 1}} -xgpu {{0 1}} -bgpu {{0 1}} -A_mat_type seqaijcusparse -testnest

  test:
    output_file: output/ex70_1.out
    nsize: 2
    suffix: 1_par
    args: -testtranspose -local {{0 1}}

  test:
    output_file: output/ex70_1.out
    requires: cuda
    nsize: 2
    suffix: 1_par_cuda
    args: -testtranspose 0 -local {{0 1}} -xgpu {{0 1}} -bgpu {{0 1}} -A_mat_type {{mpiaijcusparse mpiaij}} -testnest 0 -testrart 0

  test:
    output_file: output/ex70_1.out
    suffix: 2
    nsize: 1
    args: -M {{7 11}} -N {{12 9}} -K {{1 3}} -local {{0 1}}

  test:
    requires: cuda
    output_file: output/ex70_1.out
    suffix: 2_cuda
    nsize: 1
    args: -M 7 -N 9 -K 2 -local {{0 1}} -testnest 0 -A_mat_type {{seqdensecuda seqdense}} -xgpu {{0 1}} -bgpu {{0 1}}

  test:
    output_file: output/ex70_1.out
    suffix: 2_par
    nsize: 2
    args: -M {{7 11}} -N {{12 9}} -K {{1 3}} -local {{0 1}} -testcircular

  test:
    requires: cuda
    output_file: output/ex70_1.out
    suffix: 2_par_cuda
    nsize: 2
    args: -M 11 -N 9 -K 1 -local {{0 1}} -testcircular 0 -A_mat_type mpiaijcusparse -xgpu -bgpu -testnest 0

  test:
    output_file: output/ex70_1.out
    suffix: 3
    nsize: {{1 3}}
    args: -M 13 -N 13 -K {{1 3}} -local {{0 1}} -A_mat_type sbaij -symm -testtranspose 0 -testproj 0

  test:
    output_file: output/ex70_1.out
    suffix: 4
    nsize: 1
    args: -M 3 -N 3 -K 3 -local {{0 1}} -testcircular

  test:
    output_file: output/ex70_1.out
    suffix: 5
    nsize: {{2 4}}
    args: -M 3 -N 3 -K 3 -local 1 -testcircular -testtranspose 0

  test:
    TODO: MatCreateDense broken with some processors not having local rows
    output_file: output/ex70_1.out
    suffix: 5_broken
    nsize: {{2 4}}
    args: -M 3 -N 3 -K 3 -local 0 -testcircular -testtranspose 0

  test:
    output_file: output/ex70_1.out
    suffix: 6
    nsize: 1
    args: -M {{1 3}} -N {{2 5}} -K {{1 2}} -local {{0 1}} -testcircular -testtranspose

  test:
    output_file: output/ex70_1.out
    suffix: 7
    nsize: 1
    args: -M 13 -N 13 -K {{1 3}} -local {{0 1}} -A_mat_type dense -testnest -testcircular

TEST*/
