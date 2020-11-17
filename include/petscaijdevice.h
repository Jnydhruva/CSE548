#if !defined(PETSCAIJDEVICE_H)
#define PETSCAIJDEVICE_H

#include <petscmat.h>
#include <petsc/private/matimpl.h>

#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv)              \
  {                                                                    \
  inserted = 0;                                                        \
  if (col <= lastcol1)  low1 = 0;                                      \
  else                 high1 = nrow1;                                  \
  lastcol1 = col;                                                      \
  while (high1-low1 > 5) {                                             \
    t = (low1+high1)/2;                                                \
    if (rp1[t] > col) high1 = t;                                       \
    else              low1  = t;                                       \
  }                                                                    \
  for (_i=low1; _i<high1; _i++) {                                      \
    if (rp1[_i] > col) break;                                          \
    if (rp1[_i] == col) {                                              \
      if (addv == ADD_VALUES) {                                        \
        atomicAdd(&ap1[_i],value);                                     \
      }                                                                \
      else ap1[_i] = value;                                            \
      inserted = 1;                                                    \
      break;                                                           \
    }                                                                  \
  }                                                                    \
}

#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv)              \
  {                                                                    \
  inserted = 0;                                                        \
  if (col <= lastcol2) low2 = 0;                                       \
  else high2 = nrow2;                                                  \
  lastcol2 = col;                                                      \
  while (high2-low2 > 5) {                                             \
    t = (low2+high2)/2;                                                \
    if (rp2[t] > col) high2 = t;                                       \
    else              low2  = t;                                       \
  }                                                                    \
  for (_i=low2; _i<high2; _i++) {                                      \
    if (rp2[_i] > col) break;                                          \
    if (rp2[_i] == col) {                                              \
      if (addv == ADD_VALUES) {                                        \
        atomicAdd(&ap2[_i],value);                                     \
      }                                                                \
      else ap2[_i] = value;                                            \
      inserted = 1;                                                    \
      break;                                                           \
    }                                                                  \
  }                                                                    \
}

#define SETERR {printf("[%d]ERROR, MatSetValuesDevice A: Location (%d,%d) not found\n",(int)d_mat->rank, (int)im[i],(int)in[i]);return PETSC_ERR_ARG_OUTOFRANGE;}

#if defined(PETSC_HAVE_CUDA)
static __device__
#endif
PetscErrorCode MatSetValuesDevice(PetscSplitCSRDataStructure *d_mat, PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  MatScalar       value;
  const PetscInt  *rp1,*rp2 = NULL,*ai = d_mat->diag.i, *aj = d_mat->diag.j;
  const PetscInt  *bi = d_mat->offdiag.i, *bj = d_mat->offdiag.j;
  MatScalar       *ba = d_mat->offdiag.a, *aa = d_mat->diag.a;
  PetscInt        nrow1,nrow2,_i,low1,high1,low2,high2,t,lastcol1,lastcol2,inserted;
  MatScalar       *ap1,*ap2 = NULL;
  PetscBool       roworiented = PETSC_TRUE;
  PetscInt        i,j,row,col;
  const PetscInt rstart = d_mat->rstart,rend = d_mat->rend, cstart = d_mat->rstart,cend = d_mat->rend,N = d_mat->N;

  for (i=0; i<m; i++) {
    if (im[i] >= rstart && im[i] < rend) { // ignore off processor rows
      row      = im[i] - rstart;
      lastcol1 = -1;
      rp1      = aj + ai[row];
      ap1      = aa + ai[row];
      nrow1    = ai[row+1] - ai[row];
      low1     = 0;
      high1    = nrow1;
      if (bj) {
        lastcol2 = -1;
        rp2      = bj + bi[row];
        ap2      = ba + bi[row];
        nrow2    = bi[row+1] - bi[row];
        low2     = 0;
        high2    = nrow2;
      }
      for (j=0; j<n; j++) {
        value = roworiented ? v[i*n+j] : v[i+j*m];
        if (in[j] >= cstart && in[j] < cend) {
          col   = in[j] - cstart;
          MatSetValues_SeqAIJ_A_Private(row,col,value,is);
          if (!inserted) SETERR;
        } else if (in[j] < 0) {
          continue; // need to check for > N also
        } else if (in[j] >= N) {
          printf("[%d]ERROR, MatSetValuesDevice A: Column location %d out of range\n",(int)d_mat->rank, (int)in[i]);
          return PETSC_ERR_ARG_OUTOFRANGE;
        } else {
          col = d_mat->colmap[in[j]] - 1;
          if (col < 0) SETERR;
          MatSetValues_SeqAIJ_B_Private(row,col,value,is);
          if (!inserted) SETERR;
        }
      }
    }
  }
  return 0;
}

#undef MatSetValues_SeqAIJ_A_Private
#undef MatSetValues_SeqAIJ_B_Private
#undef SETERR

#endif // PETSCAIJDEVICE_H
