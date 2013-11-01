
/*
  Defines the basic matrix operations for the TAIJ  matrix storage format.
  This format is used to evaluate matrices of the form:

    [I \otimes S + A \otimes T]

  where
    S is a dense (p \times q) matrix
    T is a dense (p \times q) matrix
    A is an AIJ  (n \times n) matrix
    I is the identity matrix

  The resulting matrix is (np \times nq)

  We provide:
     MatMult()
     MatMultAdd()
     MatInvertBlockDiagonal()
  and
     MatCreateTAIJ(Mat,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],Mat*)

  This single directory handles both the sequential and parallel codes
*/

#include <../src/mat/impls/taij/taij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petsc-private/vecimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatTAIJGetAIJ"
/*@C
   MatTAIJGetAIJ - Get the AIJ matrix describing the blockwise action of the TAIJ matrix

   Not Collective, but if the TAIJ matrix is parallel, the AIJ matrix is also parallel

   Input Parameter:
.  A - the TAIJ matrix

   Output Parameter:
.  B - the AIJ matrix

   Level: advanced

   Notes: The reference count on the AIJ matrix is not increased so you should not destroy it.

.seealso: MatCreateTAIJ()
@*/
PetscErrorCode  MatTAIJGetAIJ(Mat A,Mat *B)
{
  PetscErrorCode ierr;
  PetscBool      ismpitaij,isseqtaij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPITAIJ,&ismpitaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQTAIJ,&isseqtaij);CHKERRQ(ierr);
  if (ismpitaij) {
    Mat_MPITAIJ *b = (Mat_MPITAIJ*)A->data;

    *B = b->A;
  } else if (isseqtaij) {
    Mat_SeqTAIJ *b = (Mat_SeqTAIJ*)A->data;

    *B = b->AIJ;
  } else {
    *B = A;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTAIJGetS"
/*@C
   MatTAIJGetS - Get the S matrix describing the shift action of the TAIJ matrix

   Not Collective, the entire S is stored and returned independently on all processes

   Input Parameter:
.  A - the TAIJ matrix

   Output Parameter:
.  S - the S matrix, in form of a scalar array in column-major format

   Level: advanced

   Notes: The reference count on the S matrix is not increased so you should not destroy it.

.seealso: MatCreateTAIJ()
@*/
PetscErrorCode  MatTAIJGetS(Mat A,const PetscScalar **S)
{
  Mat_SeqTAIJ *b = (Mat_SeqTAIJ*)A->data;
  PetscFunctionBegin;
  *S = b->S;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTAIJGetT"
/*@C
   MatTAIJGetT - Get the T matrix describing the shift action of the TAIJ matrix

   Not Collective, the entire T is stored and returned independently on all processes

   Input Parameter:
.  A - the TAIJ matrix

   Output Parameter:
.  T - the T matrix, in form of a scalar array in column-major format

   Level: advanced

   Notes: The reference count on the T matrix is not increased so you should not destroy it.

.seealso: MatCreateTAIJ()
@*/
PetscErrorCode  MatTAIJGetT(Mat A,const PetscScalar **T)
{
  Mat_SeqTAIJ *b = (Mat_SeqTAIJ*)A->data;
  PetscFunctionBegin;
  *T = b->T;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqTAIJ"
PetscErrorCode MatDestroy_SeqTAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqTAIJ    *b = (Mat_SeqTAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = PetscFree3(b->S,b->T,b->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_TAIJ"
PetscErrorCode MatSetUp_TAIJ(Mat A)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Must use MatCreateTAIJ() to create TAIJ matrices");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqTAIJ"
PetscErrorCode MatView_SeqTAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_MPITAIJ"
PetscErrorCode MatView_MPITAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPITAIJ"
PetscErrorCode MatDestroy_MPITAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPITAIJ    *b = (Mat_MPITAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = MatDestroy(&b->OAIJ);CHKERRQ(ierr);
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&b->w);CHKERRQ(ierr);
  ierr = PetscFree3(b->S,b->T,b->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*MC
  MATTAIJ - MATTAIJ = "taij" - A matrix type to be used to evaluate matrices of the following form:

    [I \otimes S + A \otimes T]

  where
    S is a dense (p \times q) matrix
    T is a dense (p \times q) matrix
    A is an AIJ  (n \times n) matrix
    I is the identity matrix
  The resulting matrix is (np \times nq)
  
  The matrix type is based on MATSEQAIJ for a sequential matrix A, and MATMPIAIJ for a distributed matrix A. 
  S and T are always stored independently on all processes as a PetscScalar array in column-major format.

  Operations provided:
. MatMult
. MatMultAdd
. MatInvertBlockDiagonal

  Level: advanced

.seealso: MatTAIJGetAIJ(), MatTAIJGetS(), MatTAIJGetT(), MatCreateTAIJ()
M*/

#undef __FUNCT__
#define __FUNCT__ "MatCreate_TAIJ"
PETSC_EXTERN PetscErrorCode MatCreate_TAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPITAIJ    *b;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr     = PetscNewLog(A,Mat_MPITAIJ,&b);CHKERRQ(ierr);
  A->data  = (void*)b;

  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->setup = MatSetUp_TAIJ;

  b->w    = 0;
  ierr    = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQTAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPITAIJ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TAIJMultAdd_Seq"
/* zz = yy + Axx */
PetscErrorCode TAIJMultAdd_Seq(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqTAIJ       *b = (Mat_SeqTAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *s = b->S, *t = b->T;
  const PetscScalar *x,*v,*bx;
  PetscScalar       *y,*sums;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j,l,p=b->p,q=b->q,k;

  PetscFunctionBegin;

  if (!yy) {
    ierr = VecSet(zz,0.0);CHKERRQ(ierr); 
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  if ((!s) && (!t) && (!b->isTI)) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  if (b->isTI) {
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sums = y + p*i;
      for (j=0; j<n; j++) {
        for (k=0; k<p; k++) {
          sums[k] += v[jrow+j]*x[q*idx[jrow+j]+k];
        }
      }
    }
  } else if (t) {
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sums = y + p*i;
      bx   = x + q*i;
      for (j=0; j<n; j++) {
        for (k=0; k<p; k++) {
          for (l=0; l<q; l++) {
            sums[k] += v[jrow+j]*t[k+l*p]*x[q*idx[jrow+j]+l];
          }
        }
      }
    }
  }
  if (s) {
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sums = y + p*i;
      bx   = x + q*i;
      if (i < b->AIJ->cmap->n) {
        for (j=0; j<q; j++) {
          for (k=0; k<p; k++) {
            sums[k] += s[k+j*p]*bx[j];
          }
        }
      }
    }
  }

  ierr = PetscLogFlops((2.0*p*q-p)*m+2*p*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqTAIJ_N"
PetscErrorCode MatMult_SeqTAIJ_N(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TAIJMultAdd_Seq(A,xx,PETSC_NULL,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqTAIJ_N"
PetscErrorCode MatMultAdd_SeqTAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TAIJMultAdd_Seq(A,xx,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc-private/kernels/blockinvert.h>

#undef __FUNCT__
#define __FUNCT__ "MatInvertBlockDiagonal_SeqTAIJ_N"
PetscErrorCode MatInvertBlockDiagonal_SeqTAIJ_N(Mat A,const PetscScalar **values)
{
  Mat_SeqTAIJ       *b  = (Mat_SeqTAIJ*)A->data;
  Mat_SeqAIJ        *a  = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *S  = b->S;
  const PetscScalar *T  = b->T;
  const PetscScalar *v  = a->a;
  const PetscInt     p  = b->p, q = b->q, m = b->AIJ->rmap->n, *idx = a->j, *ii = a->i;
  PetscErrorCode    ierr;
  PetscInt          i,j,*v_pivots,dof,dof2;
  PetscScalar       *diag,aval,*v_work;

  PetscFunctionBegin;
  if (p != q) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATTAIJ: Block size must be square to calculate inverse.");
  if ((!S) && (!T) && (!b->isTI)) {
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATTAIJ: Cannot invert a zero matrix.");
  }

  dof  = p;
  dof2 = dof*dof;

  if (b->ibdiagvalid) {
    if (values) *values = b->ibdiag;
    PetscFunctionReturn(0);
  }
  if (!b->ibdiag) {
    ierr = PetscMalloc(dof2*m*sizeof(PetscScalar),&b->ibdiag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,dof2*m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (values) *values = b->ibdiag;
  diag = b->ibdiag;

  ierr = PetscMalloc2(dof,PetscScalar,&v_work,dof,PetscInt,&v_pivots);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    if (S) {
      ierr = PetscMemcpy(diag,S,dof2*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(diag,dof2*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (b->isTI) {
      aval = 0;
      for (j=ii[i]; j<ii[i+1]; j++) if (idx[j] == i) aval = v[j];
      for (j=0; j<dof; j++) diag[j+dof*j] += aval;
    } else if (T) {
      aval = 0;
      for (j=ii[i]; j<ii[i+1]; j++) if (idx[j] == i) aval = v[j];
      for (j=0; j<dof2; j++) diag[j] += aval*T[j];
    }
    ierr = PetscKernel_A_gets_inverse_A(dof,diag,v_pivots,v_work);CHKERRQ(ierr);
    diag += dof2;
  }
  ierr = PetscFree2(v_work,v_pivots);CHKERRQ(ierr);

  b->ibdiagvalid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*===================================================================================*/

#undef __FUNCT__
#define __FUNCT__ "TAIJMultAdd_MPI"
PetscErrorCode TAIJMultAdd_MPI(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPITAIJ    *b = (Mat_MPITAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!yy) {
    ierr = VecSet(zz,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  /* start the scatter */
  ierr = VecScatterBegin(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multadd)(b->AIJ,xx,zz,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->OAIJ->ops->multadd)(b->OAIJ,b->w,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_MPITAIJ_dof"
PetscErrorCode MatMult_MPITAIJ_dof(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TAIJMultAdd_MPI(A,xx,PETSC_NULL,yy);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_MPITAIJ_dof"
PetscErrorCode MatMultAdd_MPITAIJ_dof(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TAIJMultAdd_MPI(A,xx,yy,zz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvertBlockDiagonal_MPITAIJ_dof"
PetscErrorCode MatInvertBlockDiagonal_MPITAIJ_dof(Mat A,const PetscScalar **values)
{
  Mat_MPITAIJ     *b = (Mat_MPITAIJ*)A->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = (*b->AIJ->ops->invertblockdiagonal)(b->AIJ,values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "MatGetRow_SeqTAIJ"
PetscErrorCode MatGetRow_SeqTAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **values)
{
  Mat_SeqTAIJ     *b    = (Mat_SeqTAIJ*) A->data;
  PetscErrorCode  ierr,diag;
  PetscInt        nzaij,nz,*colsaij,*idx,i,j,p=b->p,q=b->q,r=row/p,s=row%p,c;
  PetscScalar     *vaij,*v,*S=b->S,*T=b->T;

  PetscFunctionBegin;
  if (b->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  b->getrowactive = PETSC_TRUE;
  if (row < 0 || row >= A->rmap->n) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D out of range",row);

  if ((!S) && (!T) && (!b->isTI)) {
    if (ncols)    *ncols  = 0;
    if (cols)     *cols   = NULL;
    if (values)   *values = NULL;
    PetscFunctionReturn(0);
  }

  if (T || b->isTI) {
    ierr  = MatGetRow_SeqAIJ(b->AIJ,r,&nzaij,&colsaij,&vaij);CHKERRQ(ierr);
    diag  = PETSC_FALSE;
    c     = nzaij;
    for (i=0; i<nzaij; i++) {
      /* check if this row contains a diagonal entry */
      if (colsaij[i] == r) {
        diag = PETSC_TRUE;
        c = i;
      }
    }
  } else nzaij = c = 0;

  /* calculate size of row */
  nz = 0;
  if (S)            nz += q;
  if (T || b->isTI) nz += (diag && S ? (nzaij-1)*q : nzaij*q);

  if (cols || values) {
    ierr = PetscMalloc2(nz,PetscInt,&idx,nz,PetscScalar,&v);CHKERRQ(ierr);
    if (b->isTI) {
      for (i=0; i<nzaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = colsaij[i]*q+j;
          v[i*q+j]   = (j==s ? vaij[i] : 0);
        }
      }
    } else if (T) {
      for (i=0; i<nzaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = colsaij[i]*q+j;
          v[i*q+j]   = vaij[i]*T[s+j*p];
        }
      }
    }
    if (S) {
      for (j=0; j<q; j++) {
        idx[c*q+j] = r*q+j; 
        v[c*q+j]  += S[s+j*p];
      }
    }
  }

  if (ncols)    *ncols  = nz;
  if (cols)     *cols   = idx;
  if (values)   *values = v;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreRow_SeqTAIJ"
PetscErrorCode MatRestoreRow_SeqTAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree2(*idx,*v);CHKERRQ(ierr);
  ((Mat_SeqTAIJ*)A->data)->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetRow_MPITAIJ"
PetscErrorCode MatGetRow_MPITAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **values)
{
  Mat_MPITAIJ     *b      = (Mat_MPITAIJ*) A->data;
  Mat             MatAIJ  = ((Mat_SeqTAIJ*)b->AIJ->data)->AIJ;
  Mat             MatOAIJ = ((Mat_SeqTAIJ*)b->OAIJ->data)->AIJ;
  Mat             AIJ     = b->A;
  PetscErrorCode  ierr;
  const PetscInt  rstart=A->rmap->rstart,rend=A->rmap->rend,p=b->p,q=b->q,*garray;
  PetscInt        nz,*idx,ncolsaij,ncolsoaij,*colsaij,*colsoaij,r,s,c,i,j,lrow;
  PetscScalar     *v,*vals,*ovals,*S=b->S,*T=b->T;
  PetscBool       diag;

  PetscFunctionBegin;
  if (b->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  b->getrowactive = PETSC_TRUE;
  if (row < rstart || row >= rend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only local rows");
  lrow = row - rstart;

  if ((!S) && (!T) && (!b->isTI)) {
    if (ncols)    *ncols  = 0;
    if (cols)     *cols   = NULL;
    if (values)   *values = NULL;
    PetscFunctionReturn(0);
  }

  r = lrow/p;
  s = lrow%p;

  if (T || b->isTI) {
    ierr = MatMPIAIJGetSeqAIJ(AIJ,NULL,NULL,&garray);
    ierr = MatGetRow_SeqAIJ(MatAIJ,lrow/p,&ncolsaij,&colsaij,&vals);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(MatOAIJ,lrow/p,&ncolsoaij,&colsoaij,&ovals);CHKERRQ(ierr);

    diag  = PETSC_FALSE;
    c     = ncolsaij + ncolsoaij;
    for (i=0; i<ncolsaij; i++) {
      /* check if this row contains a diagonal entry */
      if (colsaij[i] == r) {
        diag = PETSC_TRUE;
        c = i;
      }
    }
  } else c = 0;

  /* calculate size of row */
  nz = 0;
  if (S)            nz += q;
  if (T || b->isTI) nz += (diag && S ? (ncolsaij+ncolsoaij-1)*q : (ncolsaij+ncolsoaij)*q);

  if (cols || values) {
    ierr = PetscMalloc2(nz,PetscInt,&idx,nz,PetscScalar,&v);CHKERRQ(ierr);
    if (b->isTI) {
      for (i=0; i<ncolsaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = (colsaij[i]+rstart/p)*q+j;
          v[i*q+j]   = (j==s ? vals[i] : 0.0);
        }
      }
      for (i=0; i<ncolsoaij; i++) {
        for (j=0; j<q; j++) {
          idx[(i+ncolsaij)*q+j] = garray[colsoaij[i]]*q+j;
          v[(i+ncolsaij)*q+j]   = (j==s ? ovals[i]: 0.0);
        }
      }
    } else if (T) {
      for (i=0; i<ncolsaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = (colsaij[i]+rstart/p)*q+j;
          v[i*q+j]   = vals[i]*T[s+j*p];
        }
      }
      for (i=0; i<ncolsoaij; i++) {
        for (j=0; j<q; j++) {
          idx[(i+ncolsaij)*q+j] = garray[colsoaij[i]]*q+j;
          v[(i+ncolsaij)*q+j]   = ovals[i]*T[s+j*p];
        }
      }
    }
    if (S) {
      for (j=0; j<q; j++) {
        idx[c*q+j] = (r+rstart/p)*q+j; 
        v[c*q+j]  += S[s+j*p];
      }
    }
  }

  if (ncols)  *ncols  = nz;
  if (cols)   *cols   = idx;
  if (values) *values = v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreRow_MPITAIJ"
PetscErrorCode MatRestoreRow_MPITAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree2(*idx,*v);CHKERRQ(ierr);
  ((Mat_SeqTAIJ*)A->data)->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if 0
These functions are commented out since MATTAIJ will use the basic MatConvert (which uses MatGetRow()).
They are retained to provide an idea on how to implement MatConvert for MATTAIJ.

#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqTAIJ_SeqAIJ"
PETSC_EXTERN PetscErrorCode MatConvert_SeqTAIJ_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_SeqTAIJ    *b   = (Mat_SeqTAIJ*)A->data;
  Mat            a    = b->AIJ,B;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)a->data;
  PetscErrorCode ierr;
  PetscInt       m,n,i,ncols,*ilen,nmax = 0,*icols,j,k,l,p=b->p,q=b->q;
  PetscInt       *cols,*srow,*scol;
  PetscScalar    *vals,*s,*t;

  PetscFunctionBegin;
  ierr = MatGetSize(a,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc(p*m*sizeof(PetscInt),&ilen);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nmax = PetscMax(nmax,aij->ilen[i]);
    for (j=0; j<p; j++) ilen[p*i+j] = aij->ilen[i]*q;
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*p,n*q,0,ilen,&B);CHKERRQ(ierr);
  ierr = PetscFree(ilen);CHKERRQ(ierr);
  ierr = PetscMalloc5(q,PetscInt,&icols,p*q,PetscScalar,&s,
                      p,PetscInt,&srow,q,PetscInt,&scol,
                      p*q,PetscScalar,&t);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) srow[j] = i*p+j;
    for (k=0; k<q; k++) scol[k] = i*q+k;
    for (j=0; j<p; j++) {
      for (k=0; k<q; k++) {
        s[j*q+k] = b->S[j+k*p];
      }
    }
    ierr = MatSetValues_SeqAIJ(B,p,srow,q,scol,s,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(a,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (k=0; k<ncols; k++) {
      for (j=0; j<q; j++) icols[j] = q*cols[k]+j;
      for (j=0; j<p; j++) {
        for (l=0; l<q; l++) {
          t[j*q+l] = b->T[j+l*p]*vals[k];
        }
      }
      ierr = MatSetValues_SeqAIJ(B,p,srow,q,icols,t,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow_SeqAIJ(a,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree5(icols,s,srow,scol,t);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPITAIJ_MPIAIJ"
PETSC_EXTERN PetscErrorCode MatConvert_MPITAIJ_MPIAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_MPITAIJ    *taij   = (Mat_MPITAIJ*)A->data;
  Mat            MatAIJ  = ((Mat_SeqTAIJ*)taij->AIJ->data)->AIJ,B;
  Mat            MatOAIJ = ((Mat_SeqTAIJ*)taij->OAIJ->data)->AIJ;
  Mat_SeqAIJ     *AIJ    = (Mat_SeqAIJ*) MatAIJ->data;
  Mat_SeqAIJ     *OAIJ   =(Mat_SeqAIJ*) MatOAIJ->data;
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*) taij->A->data;
  PetscInt       dof     = taij->dof,i,j,*dnz = NULL,*onz = NULL,nmax = 0,onmax = 0;
  PetscInt       *oicols = NULL,*icols = NULL,ncols,*cols = NULL,oncols,*ocols = NULL;
  PetscInt       rstart,cstart,*garray,ii,k;
  PetscErrorCode ierr;
  PetscScalar    *vals,*ovals;

  PetscFunctionBegin;
  ierr = PetscMalloc2(A->rmap->n,PetscInt,&dnz,A->rmap->n,PetscInt,&onz);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n/dof; i++) {
    nmax  = PetscMax(nmax,AIJ->ilen[i]);
    onmax = PetscMax(onmax,OAIJ->ilen[i]);
    for (j=0; j<dof; j++) {
      dnz[dof*i+j] = AIJ->ilen[i];
      onz[dof*i+j] = OAIJ->ilen[i];
    }
  }
  ierr = MatCreateAIJ(PetscObjectComm((PetscObject)A),A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,0,dnz,0,onz,&B);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);

  ierr   = PetscMalloc2(nmax,PetscInt,&icols,onmax,PetscInt,&oicols);CHKERRQ(ierr);
  rstart = dof*taij->A->rmap->rstart;
  cstart = dof*taij->A->cmap->rstart;
  garray = mpiaij->garray;

  ii = rstart;
  for (i=0; i<A->rmap->n/dof; i++) {
    ierr = MatGetRow_SeqAIJ(MatAIJ,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(MatOAIJ,i,&oncols,&ocols,&ovals);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      for (k=0; k<ncols; k++) {
        icols[k] = cstart + dof*cols[k]+j;
      }
      for (k=0; k<oncols; k++) {
        oicols[k] = dof*garray[ocols[k]]+j;
      }
      ierr = MatSetValues_MPIAIJ(B,1,&ii,ncols,icols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues_MPIAIJ(B,1,&ii,oncols,oicols,ovals,INSERT_VALUES);CHKERRQ(ierr);
      ii++;
    }
    ierr = MatRestoreRow_SeqAIJ(MatAIJ,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqAIJ(MatOAIJ,i,&oncols,&ocols,&ovals);CHKERRQ(ierr);
  }
  ierr = PetscFree2(icols,oicols);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    PetscInt refct = ((PetscObject)A)->refct; /* save ((PetscObject)A)->refct */
    ((PetscObject)A)->refct = 1;

    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);

    ((PetscObject)A)->refct = refct; /* restore ((PetscObject)A)->refct */
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrix_TAIJ"
PetscErrorCode  MatGetSubMatrix_TAIJ(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBegin;
  ierr = MatConvert(mat,MATAIJ,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isrow,iscol,cll,newmat);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatCreateTAIJ"
/*@C
  MatCreateTAIJ - Creates a matrix type to be used for matrices of the following form:

    [I \otimes S + A \otimes T]

  where
    S is a dense (p \times q) matrix
    T is a dense (p \times q) matrix
    A is an AIJ  (n \times n) matrix
    I is the identity matrix
  The resulting matrix is (np \times nq)
  
  The matrix type is based on MATSEQAIJ for a sequential matrix A, and MATMPIAIJ for a distributed matrix A. 
  S is always stored independently on all processes as a PetscScalar array in column-major format.
  
  Collective

  Input Parameters:
+ A - the AIJ matrix
. S - the S matrix, stored as a PetscScalar array (column-major)
. T - the T matrix, stored as a PetscScalar array (column-major)
. p - number of rows in S and T
- q - number of columns in S and T

  Output Parameter:
. taij - the new TAIJ matrix

  Operations provided:
+ MatMult
. MatMultAdd
. MatInvertBlockDiagonal
- MatView

  Level: advanced

.seealso: MatTAIJGetAIJ(), MATTAIJ
@*/
PetscErrorCode  MatCreateTAIJ(Mat A,PetscInt p,PetscInt q,const PetscScalar S[],const PetscScalar T[],Mat *taij)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n,i,j;
  Mat            B;
  PetscBool      isTI = PETSC_FALSE;

  PetscFunctionBegin;

  /* check if T is an identity matrix */
  if (T && (p == q)) {
    isTI = PETSC_TRUE;
    for (i=0; i<p; i++) {
      for (j=0; j<q; j++) {
        if (i == j) {
          /* diagonal term must be 1 */
          if (T[i+j*p] != 1.0) isTI = PETSC_FALSE;
        } else {
          /* off-diagonal term must be 0 */
          if (T[i+j*p] != 0.0) isTI = PETSC_FALSE;
        }
      }
    }
  }

  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,p*A->rmap->n,q*A->cmap->n,p*A->rmap->N,q*A->cmap->N);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->rmap,p);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,q);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

  B->assembled = PETSC_TRUE;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);

  if (size == 1) {
    Mat_SeqTAIJ *b;

    ierr = MatSetType(B,MATSEQTAIJ);CHKERRQ(ierr);

    B->ops->setup   = NULL;
    B->ops->destroy = MatDestroy_SeqTAIJ;
    B->ops->view    = MatView_SeqTAIJ;
    b               = (Mat_SeqTAIJ*)B->data;
    b->p            = p;
    b->q            = q;
    b->AIJ          = A;
    b->isTI         = isTI;
    if (S) {
      ierr = PetscMalloc(p*q*sizeof(PetscScalar),&b->S);CHKERRQ(ierr);
      ierr = PetscMemcpy(b->S,S,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
    } else  b->S = NULL;
    if (T && (!isTI)) {
      ierr = PetscMalloc(p*q*sizeof(PetscScalar),&b->T);CHKERRQ(ierr);
      ierr = PetscMemcpy(b->T,T,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
    } else b->T = NULL;

    B->ops->mult                = MatMult_SeqTAIJ_N;
    B->ops->multadd             = MatMultAdd_SeqTAIJ_N;
    B->ops->invertblockdiagonal = MatInvertBlockDiagonal_SeqTAIJ_N;
    B->ops->getrow              = MatGetRow_SeqTAIJ;
    B->ops->restorerow          = MatRestoreRow_SeqTAIJ;

    /*
    This is commented out since MATTAIJ will use the basic MatConvert (which uses MatGetRow()).
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqtaij_seqaij_C",MatConvert_SeqTAIJ_SeqAIJ);CHKERRQ(ierr);
    */

  } else {
    Mat_MPIAIJ  *mpiaij = (Mat_MPIAIJ*)A->data;
    Mat_MPITAIJ *b;
    IS          from,to;
    Vec         gvec;

    ierr = MatSetType(B,MATMPITAIJ);CHKERRQ(ierr);

    B->ops->setup   = NULL;
    B->ops->destroy = MatDestroy_MPITAIJ;
    B->ops->view    = MatView_MPITAIJ;

    b       = (Mat_MPITAIJ*)B->data;
    b->p    = p;
    b->q    = q;
    b->A    = A;
    b->isTI = isTI;
    if (S) {
      ierr = PetscMalloc(p*q*sizeof(PetscScalar),&b->S);CHKERRQ(ierr);
      ierr = PetscMemcpy(b->S,S,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
    } else  b->S = NULL;
    if (T &&(!isTI)) {
      ierr = PetscMalloc(p*q*sizeof(PetscScalar),&b->T);CHKERRQ(ierr);
      ierr = PetscMemcpy(b->T,T,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
    } else b->T = NULL;

    ierr = MatCreateTAIJ(mpiaij->A,p,q,S   ,T,&b->AIJ);CHKERRQ(ierr); 
    ierr = MatCreateTAIJ(mpiaij->B,p,q,NULL,T,&b->OAIJ);CHKERRQ(ierr);

    ierr = VecGetSize(mpiaij->lvec,&n);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF,&b->w);CHKERRQ(ierr);
    ierr = VecSetSizes(b->w,n*q,n*q);CHKERRQ(ierr);
    ierr = VecSetBlockSize(b->w,q);CHKERRQ(ierr);
    ierr = VecSetType(b->w,VECSEQ);CHKERRQ(ierr);

    /* create two temporary Index sets for build scatter gather */
    ierr = ISCreateBlock(PetscObjectComm((PetscObject)A),q,n,mpiaij->garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n*q,0,1,&to);CHKERRQ(ierr);

    /* create temporary global vector to generate scatter context */
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),q,q*A->cmap->n,q*A->cmap->N,NULL,&gvec);CHKERRQ(ierr);

    /* generate the scatter context */
    ierr = VecScatterCreate(gvec,from,b->w,to,&b->ctx);CHKERRQ(ierr);

    ierr = ISDestroy(&from);CHKERRQ(ierr);
    ierr = ISDestroy(&to);CHKERRQ(ierr);
    ierr = VecDestroy(&gvec);CHKERRQ(ierr);

    B->ops->mult                = MatMult_MPITAIJ_dof;
    B->ops->multadd             = MatMultAdd_MPITAIJ_dof;
    B->ops->invertblockdiagonal = MatInvertBlockDiagonal_MPITAIJ_dof;
    B->ops->getrow              = MatGetRow_MPITAIJ;
    B->ops->restorerow          = MatRestoreRow_MPITAIJ;

  }
  B->ops->getsubmatrix = MatGetSubMatrix_TAIJ;
  ierr  = MatSetUp(B);CHKERRQ(ierr);
  *taij = B;
  ierr  = MatViewFromOptions(B,NULL,"-mat_view");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
