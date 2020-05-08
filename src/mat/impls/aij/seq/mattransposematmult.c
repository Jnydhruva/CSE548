
/*
  Defines matrix-matrix product routines
          C = A^T * B
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/dense/seq/dense.h>

PetscErrorCode MatDestroy_SeqDense_MatTransMatMult(void *data)
{
  PetscErrorCode      ierr;
  Mat_MatTransMatMult *atb = (Mat_MatTransMatMult *)data;

  PetscFunctionBegin;
  ierr = MatDestroy(&atb->mA);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->bt);CHKERRQ(ierr);
  ierr = VecDestroy(&atb->ct);CHKERRQ(ierr);
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  Mat_MatTransMatMult *atb;
  PetscBool           cisdense;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  if (C->product->data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");

  /* create output dense matrix C = A^T*B */
  ierr = MatSetSizes(C,A->cmap->n,B->cmap->N,A->cmap->n,B->cmap->N);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,"");CHKERRQ(ierr);
  if (!cisdense) {
    ierr = MatSetType(C,((PetscObject)B)->type_name);CHKERRQ(ierr);
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* create additional data structure for the product */
  ierr = PetscNew(&atb);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(A,B->cmap->N,&atb->mA);CHKERRQ(ierr);
  ierr = MatCreateVecs(atb->mA,&atb->ct,&atb->bt);CHKERRQ(ierr);
  C->product->data    = atb;
  C->product->destroy = MatDestroy_SeqDense_MatTransMatMult;

  C->ops->transposematmultnumeric = MatTransposeMatMultNumeric_SeqAIJ_SeqDense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode      ierr;
  PetscInt            i,j,k,m=A->rmap->n,n=A->cmap->n,BN=B->cmap->N;
  const PetscScalar   *Barray,*ctarray;
  PetscScalar         *Carray,*btarray;
  Mat_MatTransMatMult *atb;
  Vec                 bt,ct;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  atb=(Mat_MatTransMatMult *)C->product->data;
  if (!atb) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  bt = atb->bt;
  ct = atb->ct;

  /* transpose local array of B, then copy it to vector bt */
  ierr = MatDenseGetArrayRead(B,&Barray);CHKERRQ(ierr);
  ierr = VecGetArray(bt,&btarray);CHKERRQ(ierr);

  k=0;
  for (j=0; j<BN; j++) {
    for (i=0; i<m; i++) btarray[i*BN + j] = Barray[k++];
  }
  ierr = VecRestoreArray(bt,&btarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&Barray);CHKERRQ(ierr);

  /* compute ct = mA^T * cb */
  ierr = MatMultTranspose(atb->mA,bt,ct);CHKERRQ(ierr);

  /* transpose local array of ct to matrix C */
  ierr = MatDenseGetArray(C,&Carray);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ct,&ctarray);CHKERRQ(ierr);
  k = 0;
  for (j=0; j<BN; j++) {
    for (i=0; i<n; i++) Carray[k++] = ctarray[i*BN + j];
  }
  ierr = VecRestoreArrayRead(ct,&ctarray);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(C,&Carray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
