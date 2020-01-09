#include <petscconf.h>
#include <../src/mat/impls/sell/mpi/mpisell.h> /*I "petscmat.h" I*/

PetscErrorCode MatMPISELLSetPreallocation_MPISELLCUDA(Mat B, PetscInt d_rlenmax, const PetscInt d_rlen[], PetscInt o_rlenmax, const PetscInt o_rlen[])
{
  Mat_MPISELL *b = (Mat_MPISELL *)B->data;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQSELLCUDA matrices. */
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->A));
    PetscCall(MatBindToCPU(b->A, B->boundtocpu));
    PetscCall(MatSetSizes(b->A, B->rmap->n, B->cmap->n, B->rmap->n, B->cmap->n));
    PetscCall(MatSetType(b->A, MATSEQSELLCUDA));
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->B));
    PetscCall(MatBindToCPU(b->B, B->boundtocpu));
    PetscCall(MatSetSizes(b->B, B->rmap->n, B->cmap->N, B->rmap->n, B->cmap->N));
    PetscCall(MatSetType(b->B, MATSEQSELLCUDA));
  }
  PetscCall(MatSeqSELLSetPreallocation(b->A, d_rlenmax, d_rlen));
  PetscCall(MatSeqSELLSetPreallocation(b->B, o_rlenmax, o_rlen));
  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPISELLCUDA(Mat A, Vec xx, Vec yy)
{
  Mat_MPISELL *a = (Mat_MPISELL *)A->data;
  PetscInt     nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  if (nt != A->cmap->n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of A (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", A->cmap->n, nt);
  PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->A->ops->mult)(a->A, xx, yy));
  PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B, a->lvec, yy, yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPISELLCUDA(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPISELL *a = (Mat_MPISELL *)A->data;
  PetscInt     nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  if (nt != A->cmap->n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of A (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", A->cmap->n, nt);
  PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->A->ops->multadd)(a->A, xx, yy, zz));
  PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B, a->lvec, zz, zz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPISELLCUDA(Mat A, Vec xx, Vec yy)
{
  Mat_MPISELL *a = (Mat_MPISELL *)A->data;
  PetscInt     nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  if (nt != A->rmap->n) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of A (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", A->rmap->n, nt);
  PetscCall((*a->B->ops->multtranspose)(a->B, xx, a->lvec));
  PetscCall((*a->A->ops->multtranspose)(a->A, xx, yy));
  PetscCall(VecScatterBegin(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_MPISELLCUDA(PetscOptionItems *PetscOptionsObject, Mat A)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPISELLCUDA options");
  if (A->factortype == MAT_FACTOR_NONE) { }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPISELLCUDA(Mat A, MatAssemblyType mode)
{
  Mat_MPISELL *mpisell;

  PetscFunctionBegin;
  mpisell = (Mat_MPISELL *)A->data;
  PetscCall(MatAssemblyEnd_MPISELL(A, mode));
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) { PetscCall(VecSetType(mpisell->lvec, VECSEQCUDA)); }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPISELLCUDA(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy_MPISELL(A));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpisellcuda_mpiaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPISELLSetPreallocation_C", NULL));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPISELLCUDA(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_MPISELL(A));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPISELLSetPreallocation_C", MatMPISELLSetPreallocation_MPISELLCUDA));
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA, &A->defaultvectype));

  A->ops->assemblyend   = MatAssemblyEnd_MPISELLCUDA;
  A->ops->mult          = MatMult_MPISELLCUDA;
  A->ops->multadd       = MatMultAdd_MPISELLCUDA;
  A->ops->multtranspose = MatMultTranspose_MPISELLCUDA;
  A->ops->destroy       = MatDestroy_MPISELLCUDA;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATMPISELLCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpisellcuda_mpiaij_C", MatConvert_MPISELL_MPIAIJ));
  PetscFunctionReturn(0);
}

/*@
   MatCreateSELLCUDA - Creates a sparse matrix in SELL (compressed row) format.
   This matrix will ultimately pushed down to NVIDIA GPUs. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqSELLSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   Level: intermediate

.seealso: MatCreate(), MatCreateSELL(), MatSetValues(), MATMPISELLCUDA, MATSELLCUDA
@*/
PetscErrorCode MatCreateSELLCUDA(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCall(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatSetType(*A, MATMPISELLCUDA));
    PetscCall(MatMPISELLSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz));
  } else {
    PetscCall(MatSetType(*A, MATSEQSELLCUDA));
    PetscCall(MatSeqSELLSetPreallocation(*A, d_nz, d_nnz));
  }
  PetscFunctionReturn(0);
}

/*MC
   MATSELLCUDA - MATMPISELLCUDA = "sellcuda" = "mpisellcuda" - A matrix type to be used for sparse matrices.

   Sliced ELLPACK matrix type whose data resides on NVIDIA GPUs.

   This matrix type is identical to MATSEQSELLCUDA when constructed with a single process communicator,
   and MATMPISELLCUDA otherwise.  As a result, for single process communicators,
   MatSeqSELLSetPreallocation is supported, and similarly MatMPISELLSetPreallocation is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpisellcuda - sets the matrix type to "mpisellcuda" during a call to MatSetFromOptions()

  Level: beginner

 .seealso: MatCreateSELLCUDA(), MATSEQSELLCUDA, MatCreateSeqSELLCUDA(), MatCUDAFormatOperation
M
M*/
