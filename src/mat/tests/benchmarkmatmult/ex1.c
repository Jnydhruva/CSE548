static char help[] = "Reads a matrix from the SuiteSparse Matrix Collection and vector from a file and tests the matrix-vector multiplication.\n\
Input arguments are:\n\
  -A <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscmat.h>
#include <petscksp.h>

int main(int argc, char **args)
{
  PetscInt    m, n, i;
  PetscReal   norm, norm2;
  Vec         b, u, u2;
  Mat         A;
  char        file[PETSC_MAX_PATH_LEN];
  PetscViewer fd;
  PetscBool   flg, test_sell = PETSC_FALSE, verify_sell = PETSC_FALSE;
  PetscInt    size, maxslicewidth, niter                = 10;
  PetscReal   ratio, avgslicewidth, varslicesize;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  /* Read matrix and RHS */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-A", file, PETSC_MAX_PATH_LEN, &flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD, 1, "Must indicate binary file with the -A option");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_sell", &test_sell, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-verify_sell", &verify_sell, NULL));
  if (verify_sell) test_sell = PETSC_TRUE; /* overwrite test_sell */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-niter", &niter, NULL));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatGetSize(A, &m, &n));

  /* Let the vec object trigger the first CUDA call, which takes a relatively long time to init CUDA */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-b", file, PETSC_MAX_PATH_LEN, &flg));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
  PetscCall(VecSetFromOptions(b));
  if (flg) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
    PetscCall(VecLoad(b, fd));
    PetscCall(PetscViewerDestroy(&fd));
  } else {
    PetscCall(VecSetSizes(b, PETSC_DECIDE, m));
    PetscCall(VecSet(b, 1.0));
  }
  PetscCall(VecDuplicate(b, &u));

  if (test_sell) {
    if (verify_sell) {
#if defined(PETSC_HAVE_CUDA)
      Mat B;
      PetscCall(MatConvert(A, MATAIJCUSPARSE, MAT_INITIAL_MATRIX, &B));
      PetscCall(VecDuplicate(b, &u2));
      PetscCall(MatMult(B, b, u2));
      PetscCall(MatDestroy(&B));
#else
      PetscCall(VecDuplicate(b, &u2));
      PetscCall(MatMult(A, b, u2));
#endif
    }
    /* two-step convert is much faster than the basic convert */
    PetscCall(MatConvert(A, MATSELL, MAT_INPLACE_MATRIX, &A));
    PetscCall(MatConvert(A, MATSELLCUDA, MAT_INPLACE_MATRIX, &A));
    if (size == 1) {
      PetscCall(MatSeqSELLGetFillRatio(A, &ratio));
      PetscCall(MatSeqSELLGetMaxSliceWidth(A, &maxslicewidth));
      PetscCall(MatSeqSELLGetAvgSliceWidth(A, &avgslicewidth));
      PetscCall(MatSeqSELLGetVarSliceSize(A, &varslicesize));
    }
#if defined(PETSC_HAVE_CUDA)
    PetscCall(MatConvert(A, MATSELLCUDA, MAT_INPLACE_MATRIX, &A));
  } else {
    PetscCall(MatConvert(A, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, &A));
#endif
  }
  PetscCall(MatSetFromOptions(A));
  /* Timing MatMult */
  for (i = 0; i < niter; i++) { PetscCall(MatMult(A, b, u)); }

  /* Show result */
  PetscCall(VecNorm(u, NORM_2, &norm));
  if (verify_sell) {
    PetscCall(VecAXPY(u2, -1, u));
    PetscCall(VecNorm(u2, NORM_2, &norm2));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative error: %.4e\n", (double)norm2 / norm));
  }
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));

  if (!verify_sell) {
    PetscLogEvent      event;
    PetscEventPerfInfo eventInfo;
    PetscReal          maxt;
#if defined(PETSC_HAVE_CUDA)
    PetscReal gtotf, gmaxt;
#else
    PetscReal totf;
#endif

#if defined(PETSC_HAVE_CUDA)
    if (test_sell) {
      PetscCall(PetscLogEventGetId("MatCUDACopyTo", &event));
    } else {
      PetscCall(PetscLogEventGetId("MatCUSPARSCopyTo", &event));
    }
    PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &eventInfo));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%.4e ", (double)eventInfo.time / eventInfo.count));
#endif

    PetscCall(PetscLogEventGetId("MatMult", &event));
    PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &eventInfo));

#if defined(PETSC_HAVE_CUDA)
    PetscCall(MPI_Allreduce(&eventInfo.GpuFlops, &gtotf, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));
    PetscCall(MPI_Allreduce(&eventInfo.GpuTime, &gmaxt, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, PETSC_COMM_WORLD));
#else
    PetscCall(MPI_Allreduce(&eventInfo.flops, &totf, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, PETSC_COMM_WORLD));
#endif
    PetscCall(MPI_Allreduce(&eventInfo.time, &maxt, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, PETSC_COMM_WORLD));

#if defined(PETSC_HAVE_CUDA)
    gtotf /= (double)eventInfo.count;
    gmaxt /= (double)eventInfo.count;
    maxt /= (double)eventInfo.count;
    /* The first three numbers correspond to GPU GFLOPs/sec, GPU time, total time*/
    if (test_sell && size == 1) {
      PetscReal bw;
      bw = 1e-9 * (avgslicewidth * m * (sizeof(PetscReal) + sizeof(PetscInt)) + n * (sizeof(PetscReal) + sizeof(PetscInt))) / gmaxt;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%.2lf %.4e %.4e %.6lf %d %.2lf %.2lf %.2lf\n", (double)gtotf / gmaxt / 1.e6, gmaxt, maxt, ratio, maxslicewidth, avgslicewidth, bw, varslicesize));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%.2lf %.4e %.4e\n", (double)gtotf / gmaxt / 1.e6, gmaxt, maxt));
    }
#else
    if (test_sell) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%.2lf %.4e\n", (double)totf / maxt / 1.e6, maxt));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%.2lf %.4e\n", (double)totf / maxt / 1.e6, maxt));
    }
#endif
  }
  PetscCall(PetscFinalize());
  return 0;
}
