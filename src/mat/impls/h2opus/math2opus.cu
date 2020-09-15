#include <h2opus.h>
#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_geometric_construction.h>
#include <h2opus/distributed/distributed_hgemv.h>
#include <h2opus/distributed/distributed_horthog.h>
#include <h2opus/distributed/distributed_hcompress.h>
#include <h2opus/util/boxentrygen.h>
#include <petsc/private/matimpl.h>
#include <petscsf.h>

#define MatH2OpusGetThrustPointer(v) thrust::raw_pointer_cast((v).data())

#if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
#define PETSC_H2OPUS_USE_GPU
#endif
#if defined(PETSC_H2OPUS_USE_GPU)
#define MatH2OpusUpdateIfNeeded(A) MatBindToCPU(A,(A)->boundtocpu)
#else
#define MatH2OpusUpdateIfNeeded(A) 0
#endif

// TODO H2OPUS:
// kernel needs (global?) id of points (issues with Chebyshev points and coupling matrix computation)
// DistributedHMatrix
//   unsymmetric ?
//   transpose for distributed_hgemv?
//   clearData()
// Unify interface for sequential and parallel?
// Reuse geometric construction (almost possible, only the unsymmetric case is explicitly handled)
// Namespace H2OPUS stuff?
// Diagnostics? FLOPS, MEMORY USAGE IN PARALLEL
//
template <class T> class PetscPointCloud : public H2OpusDataSet<T>
{
  private:
    int dimension;
    size_t num_points;
    std::vector<std::vector<T>> pts;

  public:
    PetscPointCloud(int dim, size_t num_pts, const T coords[])
    {
      this->dimension = dim;
      this->num_points = num_pts;

      pts.resize(dim);
      for (int i = 0; i < dim; i++)
        pts[i].resize(num_points);

      for (size_t n = 0; n < num_points; n++)
        for (int i = 0; i < dim; i++)
          pts[i][n] = coords[n*dim + i];
    }

    PetscPointCloud(const PetscPointCloud<T>& other)
    {
      this->dimension = other.dimension;
      this->num_points = other.num_points;
      this->pts = other.pts;
    }

    size_t getDimension() const
    {
        return dimension;
    }

    size_t getDataSetSize() const
    {
        return num_points;
    }

    T getDataPoint(size_t idx, size_t dim) const
    {
        return pts[dim][idx];
    }
};

template<class T> class PetscFunctionGenerator
{
private:
  MatH2OpusKernel k;
  int             dim;
  void            *ctx;

public:
    PetscFunctionGenerator(MatH2OpusKernel k, int dim, void* ctx) { this->k = k; this->dim = dim; this->ctx = ctx; }
    PetscFunctionGenerator(PetscFunctionGenerator& other) { this->k = other.k; this->dim = other.dim; this->ctx = other.ctx; }
    T operator()(PetscReal *pt1, PetscReal *pt2)
    {
        return (T)(this->k ? (*this->k)(this->dim,pt1,pt2,this->ctx) : 0);
    }
};

#include <../src/mat/impls/h2opus/math2opussampler.hpp>

typedef struct {
  distributedH2OpusHandle_t handle;

  /* two different classes at the moment */
  HMatrix *hmatrix;
  DistributedHMatrix *dist_hmatrix;

  /* May use permutations */
  PetscSF sf;
  PetscLayout h2opus_rmap;
  thrust::host_vector<PetscScalar> *xx,*yy;
  PetscInt xxs,yys;
  PetscBool multsetup;

  /* GPU */
#if defined(PETSC_H2OPUS_USE_GPU)
  HMatrix_GPU *hmatrix_gpu;
  DistributedHMatrix_GPU *dist_hmatrix_gpu;
  thrust::device_vector<PetscScalar> *xx_gpu,*yy_gpu;
  PetscInt xxs_gpu,yys_gpu;
#endif

  /* construction from matvecs */
  PetscMatrixSampler* sampler;

  /* Admissibility */
  PetscReal eta;
  PetscInt  leafsize;

  /* for dof reordering */
  PetscPointCloud<PetscReal> *ptcloud;

  /* kernel for generating matrix entries */
  PetscFunctionGenerator<PetscScalar> *kernel;

  /* basis orthogonalized? */
  PetscBool orthogonal;

  /* customization */
  PetscInt  basisord;
  PetscInt  max_rank;
  PetscInt  bs;
  PetscReal rtol;
  PetscInt  norm_max_samples;
  PetscBool check_construction;

  /* keeps track of MatScale values */
  PetscScalar s;
} Mat_H2OPUS;

static PetscErrorCode MatDestroy_H2OPUS(Mat A)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  h2opusDestroyDistributedHandle(a->handle);
  delete a->hmatrix;
  delete a->dist_hmatrix;
  ierr = PetscSFDestroy(&a->sf);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&a->h2opus_rmap);CHKERRQ(ierr);
  delete a->xx;
  delete a->yy;
#if defined(PETSC_H2OPUS_USE_GPU)
  delete a->hmatrix_gpu;
  delete a->dist_hmatrix_gpu;
  delete a->xx_gpu;
  delete a->yy_gpu;
#endif
  delete a->sampler;
  delete a->ptcloud;
  delete a->kernel;
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFGetVectorSF(PetscSF sf, PetscInt nv, PetscInt ldr, PetscInt ldl, PetscSF *vsf)
{
  PetscSF           rankssf;
  const PetscSFNode *iremote;
  PetscSFNode       *viremote,*rremotes;
  const PetscInt    *ilocal;
  PetscInt          *vilocal = NULL,*ldrs;
  const PetscMPIInt *ranks;
  PetscMPIInt       *sranks;
  PetscInt          nranks,nr,nl,vnr,vnl,i,v,j,maxl;
  MPI_Comm          comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (nv == 1) {
    ierr = PetscObjectReference((PetscObject)sf);CHKERRQ(ierr);
    *vsf = sf;
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nr,&nl,&ilocal,&iremote);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sf,NULL,&maxl);CHKERRQ(ierr);
  maxl += 1;
  if (ldl == PETSC_DECIDE) ldl = maxl;
  if (ldr == PETSC_DECIDE) ldr = nr;
  if (ldr < nr) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid leading dimension %D < %D",ldr,nr);
  if (ldl < maxl) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid leading dimension %D < %D",ldl,maxl);
  vnr  = nr*nv;
  vnl  = nl*nv;
  ierr = PetscMalloc1(vnl,&viremote);CHKERRQ(ierr);
  if (ilocal) {
    ierr = PetscMalloc1(vnl,&vilocal);CHKERRQ(ierr);
  }

  /* TODO: Should this special SF be available, e.g.
     PetscSFGetRanksSF or similar? */
  ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&sranks);CHKERRQ(ierr);
  ierr = PetscArraycpy(sranks,ranks,nranks);CHKERRQ(ierr);
  ierr = PetscSortMPIInt(nranks,sranks);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&rremotes);CHKERRQ(ierr);
  for (i=0;i<nranks;i++) {
    rremotes[i].rank  = sranks[i];
    rremotes[i].index = 0;
  }
  ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,&rankssf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(rankssf,1,nranks,NULL,PETSC_OWN_POINTER,rremotes,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&ldrs);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(rankssf,MPIU_INT,&ldr,ldrs);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(rankssf,MPIU_INT,&ldr,ldrs);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&rankssf);CHKERRQ(ierr);

  j = -1;
  for (i=0;i<nl;i++) {
    const PetscInt r  = iremote[i].rank;
    const PetscInt ii = iremote[i].index;

    if (j < 0 || sranks[j] != r) {
      ierr = PetscFindMPIInt(r,nranks,sranks,&j);CHKERRQ(ierr);
    }
    if (j < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to locate neighbor rank %D",r);
    for (v=0;v<nv;v++) {
      viremote[v*nl + i].rank  = r;
      viremote[v*nl + i].index = v*ldrs[j] + ii;
      if (ilocal) vilocal[v*nl + i] = v*ldl + ilocal[i];
    }
  }
  ierr = PetscFree(sranks);CHKERRQ(ierr);
  ierr = PetscFree(ldrs);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,vsf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*vsf,vnr,vnl,vilocal,PETSC_OWN_POINTER,viremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSign(Vec v, Vec s)
{
  const PetscScalar *av;
  PetscScalar       *as;
  PetscInt          i,n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,2);
  ierr = VecGetArrayRead(v,&av);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(s,&as);CHKERRQ(ierr);
  ierr = VecGetLocalSize(s,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&i);CHKERRQ(ierr);
  if (i != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Invalid local sizes %D != %D",i,n);
  for (i=0;i<n;i++) as[i] = PetscAbsScalar(av[i]) < 0 ? -1. : 1.;
  ierr = VecRestoreArrayWrite(s,&as);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* these are approximate norms */
/* NORM_2: Estimating the matrix p-norm Nicholas J. Higham
   NORM_1/NORM_INFINITY: A block algorithm for matrix 1-norm estimation, with an application to 1-norm pseudospectra Higham, Nicholas J. and Tisseur, Francoise */
static PetscErrorCode MatApproximateNorm_Private(Mat A, NormType normtype, PetscInt normsamples, PetscReal* n)
{
  Vec            x,y,w,z;
  PetscReal      normz,adot;
  PetscScalar    dot;
  PetscInt       i,j,N,jold = -1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (normtype) {
  case NORM_INFINITY:
  case NORM_1:
    if (normsamples < 0) normsamples = 10; /* pure guess */
    if (normtype == NORM_INFINITY) {
      Mat B;
      ierr = MatCreateTranspose(A,&B);CHKERRQ(ierr);
      A = B;
    } else {
      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&z,&w);CHKERRQ(ierr);
    ierr = VecGetSize(x,&N);CHKERRQ(ierr);
    ierr = VecSet(x,1./N);CHKERRQ(ierr);
    ierr = VecSetOption(x,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    *n   = 0.0;
    for (i = 0; i < normsamples; i++) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      ierr = VecSign(y,w);CHKERRQ(ierr);
      ierr = MatMultTranspose(A,w,z);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_INFINITY,&normz);CHKERRQ(ierr);
      ierr = VecDot(x,z,&dot);CHKERRQ(ierr);
      adot = PetscAbsScalar(dot);
      ierr = PetscInfo4(A,"%s norm it %D -> (%g %g)\n",NormTypes[normtype],i,(double)normz,(double)adot);CHKERRQ(ierr);
      if (normz <= adot && i > 0) {
        ierr = VecNorm(y,NORM_1,n);CHKERRQ(ierr);
        break;
      }
      ierr = VecSet(x,0.);CHKERRQ(ierr);
      ierr = VecMax(z,&j,&normz);CHKERRQ(ierr);
      if (j == jold) {
        ierr = VecNorm(y,NORM_1,n);CHKERRQ(ierr);
        ierr = PetscInfo2(A,"%s norm it %D -> breakdown (j==jold)\n",NormTypes[normtype],i);CHKERRQ(ierr);
        break;
      }
      jold = j;
      ierr = VecSetValue(x,j,1.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&w);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
    break;
  case NORM_2:
    if (normsamples < 0) normsamples = 20; /* pure guess */
    ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&z,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = VecNormalize(x,NULL);CHKERRQ(ierr);
    *n   = 0.0;
    for (i = 0; i < normsamples; i++) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      ierr = VecNormalize(y,n);CHKERRQ(ierr);
      ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_2,&normz);CHKERRQ(ierr);
      ierr = VecDot(x,z,&dot);CHKERRQ(ierr);
      adot = PetscAbsScalar(dot);
      ierr = PetscInfo5(A,"%s norm it %D -> %g (%g %g)\n",NormTypes[normtype],i,(double)*n,(double)normz,(double)adot);CHKERRQ(ierr);
      if (normz <= adot) break;
      if (i < normsamples - 1) {
        Vec t;

        ierr = VecNormalize(z,NULL);CHKERRQ(ierr);
        t = x;
        x = z;
        z = t;
      }
    }
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"%s norm not supported",NormTypes[normtype]);
  }
  ierr = PetscInfo3(A,"%s norm %g computed in %D iterations\n",NormTypes[normtype],(double)*n,i);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatNorm_H2OPUS(Mat A, NormType normtype, PetscReal* n)
{
  PetscErrorCode ierr;
  PetscBool      ish2opus;
  PetscInt       nmax = PETSC_DECIDE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (ish2opus) {
    Mat_H2OPUS *a = (Mat_H2OPUS*)A->data;

    nmax = a->norm_max_samples;
  }
  ierr = MatApproximateNorm_Private(A,normtype,nmax,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultNKernel_H2OPUS(Mat A, PetscBool transA, Mat B, Mat C)
{
  Mat_H2OPUS     *h2opus = (Mat_H2OPUS*)A->data;
  h2opusHandle_t handle = h2opus->handle->handle;
  PetscBool      boundtocpu = PETSC_TRUE;
  PetscScalar    *xx,*yy,*uxx,*uyy;
  PetscInt       blda,clda;
  PetscMPIInt    size;
  PetscSF        bsf,csf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  if (h2opus->sf) {
    PetscInt n;

    ierr = PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)B,"_math2opus_vectorsf",(PetscObject*)&bsf);CHKERRQ(ierr);
    if (!bsf) {
      ierr = PetscSFGetVectorSF(h2opus->sf,B->cmap->N,blda,PETSC_DECIDE,&bsf);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)B,"_math2opus_vectorsf",(PetscObject)bsf);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)bsf);CHKERRQ(ierr);
    }
    ierr = PetscObjectQuery((PetscObject)C,"_math2opus_vectorsf",(PetscObject*)&csf);CHKERRQ(ierr);
    if (!csf) {
      ierr = PetscSFGetVectorSF(h2opus->sf,B->cmap->N,clda,PETSC_DECIDE,&csf);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)C,"_math2opus_vectorsf",(PetscObject)csf);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)csf);CHKERRQ(ierr);
    }
    blda = n;
    clda = n;
  }
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (boundtocpu) {
    if (h2opus->sf) {
      PetscInt n;

      ierr = PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
      if (h2opus->xxs < B->cmap->n) { h2opus->xx->resize(n*B->cmap->N); h2opus->xxs = B->cmap->N; }
      if (h2opus->yys < B->cmap->n) { h2opus->yy->resize(n*B->cmap->N); h2opus->yys = B->cmap->N; }
    }
    ierr = MatDenseGetArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(C,&yy);CHKERRQ(ierr);
    if (h2opus->sf) {
      uxx  = MatH2OpusGetThrustPointer(*h2opus->xx);
      uyy  = MatH2OpusGetThrustPointer(*h2opus->yy);
      ierr = PetscSFBcastBegin(bsf,MPIU_SCALAR,xx,uxx);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bsf,MPIU_SCALAR,xx,uxx);CHKERRQ(ierr);
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (size > 1) {
      if (!h2opus->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      if (transA && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
      distributed_hgemv(/*transA ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix, uxx, blda, 0.0, uyy, clda, B->cmap->N, h2opus->handle);
    } else {
      if (!h2opus->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hgemv(transA ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix, uxx, blda, 0.0, uyy, clda, B->cmap->N, handle);
    }
    ierr = MatDenseRestoreArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (h2opus->sf) {
      ierr = PetscSFReduceBegin(csf,MPIU_SCALAR,uyy,yy,MPIU_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(csf,MPIU_SCALAR,uyy,yy,MPIU_REPLACE);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArrayWrite(C,&yy);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  } else {
    PetscBool ciscuda,biscuda;

    if (h2opus->sf) {
      PetscInt n;

      ierr = PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
      if (h2opus->xxs_gpu < B->cmap->n) { h2opus->xx_gpu->resize(n*B->cmap->N); h2opus->xxs_gpu = B->cmap->N; }
      if (h2opus->yys_gpu < B->cmap->n) { h2opus->yy_gpu->resize(n*B->cmap->N); h2opus->yys_gpu = B->cmap->N; }
    }
    /* If not of type seqdensecuda, convert on the fly (i.e. allocate GPU memory) */
    ierr = PetscObjectTypeCompareAny((PetscObject)B,&biscuda,MATSEQDENSECUDA,MATMPIDENSECUDA,"");CHKERRQ(ierr);
    if (!biscuda) {
      ierr = MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompareAny((PetscObject)C,&ciscuda,MATSEQDENSECUDA,MATMPIDENSECUDA,"");CHKERRQ(ierr);
    if (!ciscuda) {
      C->assembled = PETSC_TRUE;
      ierr = MatConvert(C,MATDENSECUDA,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
    }
    ierr = MatDenseCUDAGetArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    ierr = MatDenseCUDAGetArrayWrite(C,&yy);CHKERRQ(ierr);
    if (h2opus->sf) {
      uxx  = MatH2OpusGetThrustPointer(*h2opus->xx_gpu);
      uyy  = MatH2OpusGetThrustPointer(*h2opus->yy_gpu);
      ierr = PetscSFBcastBegin(bsf,MPIU_SCALAR,xx,uxx);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bsf,MPIU_SCALAR,xx,uxx);CHKERRQ(ierr);
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (size > 1) {
      if (!h2opus->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      if (transA && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
      distributed_hgemv(/* transA ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix_gpu, uxx, blda, 0.0, uyy, clda, B->cmap->N, h2opus->handle);
    } else {
      if (!h2opus->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      hgemv(transA ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix_gpu, uxx, blda, 0.0, uyy, clda, B->cmap->N, handle);
    }
    ierr = MatDenseCUDARestoreArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (h2opus->sf) {
      ierr = PetscSFReduceBegin(csf,MPIU_SCALAR,uyy,yy,MPIU_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(csf,MPIU_SCALAR,uyy,yy,MPIU_REPLACE);CHKERRQ(ierr);
    }
    ierr = MatDenseCUDARestoreArrayWrite(C,&yy);CHKERRQ(ierr);
    if (!biscuda) {
      ierr = MatConvert(B,MATDENSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
    }
    if (!ciscuda) {
      ierr = MatConvert(C,MATDENSE,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
    }
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_H2OPUS(Mat C)
{
  Mat_Product    *product = C->product;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatMultNKernel_H2OPUS(product->A,PETSC_FALSE,product->B,C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatMultNKernel_H2OPUS(product->A,PETSC_TRUE,product->B,C);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProduct type %s is not supported",MatProductTypes[product->type]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_H2OPUS(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscBool      cisdense;
  Mat            A,B;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  A = product->A;
  B = product->B;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatSetSizes(C,A->rmap->n,B->cmap->n,A->rmap->N,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(C,product->A,product->B);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,MATMPIDENSECUDA,"");CHKERRQ(ierr);
    if (!cisdense) { ierr = MatSetType(C,((PetscObject)product->B)->type_name);CHKERRQ(ierr); }
    ierr = MatSetUp(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatSetSizes(C,A->cmap->n,B->cmap->n,A->cmap->N,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(C,product->A,product->B);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,MATMPIDENSECUDA,"");CHKERRQ(ierr);
    if (!cisdense) { ierr = MatSetType(C,((PetscObject)product->B)->type_name);CHKERRQ(ierr); }
    ierr = MatSetUp(C);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProduct type %s is not supported",MatProductTypes[product->type]);
  }
  C->ops->productsymbolic = NULL;
  C->ops->productnumeric = MatProductNumeric_H2OPUS;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_H2OPUS(Mat C)
{
  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->type == MATPRODUCT_AB || C->product->type == MATPRODUCT_AtB) {
    C->ops->productsymbolic = MatProductSymbolic_H2OPUS;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultKernel_H2OPUS(Mat A, Vec x, PetscScalar sy, Vec y, PetscBool trans)
{
  Mat_H2OPUS     *h2opus = (Mat_H2OPUS*)A->data;
  h2opusHandle_t handle = h2opus->handle->handle;
  PetscBool      boundtocpu = PETSC_TRUE;
  PetscInt       n;
  PetscScalar    *xx,*yy,*uxx,*uyy;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  if (h2opus->sf) {
    ierr = PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
  } else n = A->rmap->n;
  if (boundtocpu) {
    ierr = VecGetArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (sy == 0.0) {
      ierr = VecGetArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    }
    if (h2opus->sf) {
      uxx = MatH2OpusGetThrustPointer(*h2opus->xx);
      uyy = MatH2OpusGetThrustPointer(*h2opus->yy);

      ierr = PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,xx,uxx);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,xx,uxx);CHKERRQ(ierr);
      if (sy != 0.0) {
        ierr = PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,yy,uyy);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,yy,uyy);CHKERRQ(ierr);
      }
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (size > 1) {
      if (!h2opus->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      if (trans && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
      distributed_hgemv(/*trans ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix, uxx, n, sy, uyy, n, 1, h2opus->handle);
    } else {
      if (!h2opus->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hgemv(trans ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix, uxx, n, sy, uyy, n, 1, handle);
    }
    ierr = VecRestoreArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (h2opus->sf) {
      ierr = PetscSFReduceBegin(h2opus->sf,MPIU_SCALAR,uyy,yy,MPIU_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(h2opus->sf,MPIU_SCALAR,uyy,yy,MPIU_REPLACE);CHKERRQ(ierr);
    }
    if (sy == 0.0) {
      ierr = VecRestoreArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
    }
#if defined(PETSC_H2OPUS_USE_GPU)
  } else {
    ierr = VecCUDAGetArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (sy == 0.0) {
      ierr = VecCUDAGetArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecCUDAGetArray(y,&yy);CHKERRQ(ierr);
    }
    if (h2opus->sf) {
      uxx = MatH2OpusGetThrustPointer(*h2opus->xx_gpu);
      uyy = MatH2OpusGetThrustPointer(*h2opus->yy_gpu);

      ierr = PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,xx,uxx);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,xx,uxx);CHKERRQ(ierr);
      if (sy != 0.0) {
        ierr = PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,yy,uyy);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,yy,uyy);CHKERRQ(ierr);
      }
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (size > 1) {
      if (!h2opus->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      if (trans && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
      distributed_hgemv(/*trans ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix_gpu, uxx, n, sy, uyy, n, 1, h2opus->handle);
    } else {
      if (!h2opus->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      hgemv(trans ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix_gpu, uxx, n, sy, uyy, n, 1, handle);
    }
    ierr = VecCUDARestoreArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (h2opus->sf) {
      ierr = PetscSFReduceBegin(h2opus->sf,MPIU_SCALAR,uyy,yy,MPIU_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(h2opus->sf,MPIU_SCALAR,uyy,yy,MPIU_REPLACE);CHKERRQ(ierr);
    }
    if (sy == 0.0) {
      ierr = VecCUDARestoreArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecCUDARestoreArray(y,&yy);CHKERRQ(ierr);
    }
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_H2OPUS(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatH2OpusUpdateIfNeeded(A);CHKERRQ(ierr);
  ierr = MatMultKernel_H2OPUS(A,x,0.0,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_H2OPUS(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatH2OpusUpdateIfNeeded(A);CHKERRQ(ierr);
  ierr = MatMultKernel_H2OPUS(A,x,0.0,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_H2OPUS(Mat A, Vec x, Vec y, Vec z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatH2OpusUpdateIfNeeded(A);CHKERRQ(ierr);
  ierr = VecCopy(y,z);CHKERRQ(ierr);
  ierr = MatMultKernel_H2OPUS(A,x,1.0,z,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_H2OPUS(Mat A, Vec x, Vec y, Vec z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatH2OpusUpdateIfNeeded(A);CHKERRQ(ierr);
  ierr = VecCopy(y,z);CHKERRQ(ierr);
  ierr = MatMultKernel_H2OPUS(A,x,1.0,z,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_H2OPUS(Mat A, PetscScalar s)
{
  Mat_H2OPUS *a = (Mat_H2OPUS*)A->data;

  PetscFunctionBegin;
  a->s *= s;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_H2OPUS(PetscOptionItems *PetscOptionsObject,Mat A)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"H2OPUS options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_leafsize","Leaf size when constructed from kernel",NULL,a->leafsize,&a->leafsize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_h2opus_eta","Admissibility condition tolerance",NULL,a->eta,&a->eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_order","Basis order for off-diagonal sampling when constructed from kernel",NULL,a->basisord,&a->basisord,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_maxrank","Maximum rank when constructed from matvecs",NULL,a->max_rank,&a->max_rank,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_samples","Number of samples to be taken concurrently when constructing from matvecs",NULL,a->bs,&a->bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_h2opus_rtol","Relative tolerance for construction from sampling",NULL,a->rtol,&a->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_h2opus_check","Check error when constructing from sampling during MatAssemblyEnd()",NULL,a->check_construction,&a->check_construction,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatH2OpusSetCoords_H2OPUS(Mat,PetscInt,const PetscReal[],MatH2OpusKernel,void*);

static PetscErrorCode MatH2OpusInferCoordinates_Private(Mat A)
{
  Mat_H2OPUS        *a = (Mat_H2OPUS*)A->data;
  Vec               c;
  PetscInt          spacedim;
  const PetscScalar *coords;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (a->ptcloud) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)A,"__math2opus_coords",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c && a->sampler) {
    Mat S = a->sampler->GetSamplingMat();

    ierr = PetscObjectQuery((PetscObject)S,"__math2opus_coords",(PetscObject*)&c);CHKERRQ(ierr);
#if 0
    if (!c) {
      PetscBool ish2opus;

      ierr = PetscObjectTypeCompare((PetscObject)S,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
      if (ish2opus) {
        Mat_H2OPUS *s = (Mat_H2OPUS*)S->data;

        a->ptcloud = new PetscPointCloud<PetscReal>(*s->ptcloud);
      }
    }
#endif
  }
  if (!c) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Missing coordinates");
  ierr = VecGetArrayRead(c,&coords);CHKERRQ(ierr);
  ierr = VecGetBlockSize(c,&spacedim);CHKERRQ(ierr);
  ierr = MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,NULL,NULL);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(c,&coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUpMultiply_H2OPUS(Mat A)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  IS             is;
  PetscInt       n,*idx;
  int            *iidx;
  PetscCopyMode  own;
  PetscBool      rid;

  PetscFunctionBegin;
  if (a->multsetup) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    iidx = MatH2OpusGetThrustPointer(a->dist_hmatrix->basis_tree.basis_branch.index_map);
    n    = a->dist_hmatrix->basis_tree.basis_branch.index_map.size();
  } else {
    iidx = MatH2OpusGetThrustPointer(a->hmatrix->u_basis_tree.index_map);
    n    = a->hmatrix->u_basis_tree.index_map.size();
  }
  if (PetscDefined(USE_64BIT_INDICES)) {
    PetscInt i;

    own  = PETSC_OWN_POINTER;
    ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
    for (i=0;i<n;i++) idx[i] = iidx[i];
  } else {
    own  = PETSC_USE_POINTER;
    idx  = iidx;
  }
  ierr = ISCreateGeneral(comm,n,idx,own,&is);CHKERRQ(ierr);
  ierr = ISIdentity(is,&rid);CHKERRQ(ierr);
  if (!rid) {
    ierr = PetscSFCreate(comm,&a->sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(a->sf,A->rmap,n,NULL,PETSC_OWN_POINTER,idx);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
    a->xx_gpu  = new thrust::device_vector<PetscScalar>(n);
    a->yy_gpu  = new thrust::device_vector<PetscScalar>(n);
    a->xxs_gpu = 1;
    a->yys_gpu = 1;
#endif
    a->xx  = new thrust::host_vector<PetscScalar>(n);
    a->yy  = new thrust::host_vector<PetscScalar>(n);
    a->xxs = 1;
    a->yys = 1;
  }
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  a->multsetup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_H2OPUS(Mat A, MatAssemblyType assemblytype)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  h2opusHandle_t handle = a->handle->handle;
  PetscBool      kernel = PETSC_FALSE;
  PetscBool      boundtocpu = PETSC_TRUE;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  /* TODO REUSABILITY of geometric construction */
  delete a->hmatrix;
  delete a->dist_hmatrix;
#if defined(PETSC_H2OPUS_USE_GPU)
  delete a->hmatrix_gpu;
  delete a->dist_hmatrix_gpu;
#endif
  /* TODO: other? */
  H2OpusBoxCenterAdmissibility adm(a->eta);

  if (size > 1) {
    a->dist_hmatrix = new DistributedHMatrix(A->rmap->n/*,A->symmetric*/);
  } else {
    a->hmatrix = new HMatrix(A->rmap->n,A->symmetric);
  }
  ierr = MatH2OpusInferCoordinates_Private(A);CHKERRQ(ierr);
  if (!a->ptcloud) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing pointcloud");
  if (a->kernel) {
    BoxEntryGen<PetscScalar, H2OPUS_HWTYPE_CPU, PetscFunctionGenerator<PetscScalar>> entry_gen(*a->kernel);
    if (size > 1) {
      buildDistributedHMatrix(*a->dist_hmatrix,a->ptcloud,adm,entry_gen,a->leafsize,a->basisord,a->handle);
    } else {
      buildHMatrix(*a->hmatrix,a->ptcloud,adm,entry_gen,a->leafsize,a->basisord);
    }
    kernel = PETSC_TRUE;
  } else {
    if (size > 1) SETERRQ(comm,PETSC_ERR_SUP,"Construction from sampling not supported in parallel");
    buildHMatrixStructure(*a->hmatrix,a->ptcloud,a->leafsize,adm);
  }

  ierr = MatSetUpMultiply_H2OPUS(A);CHKERRQ(ierr);

#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
  if (!boundtocpu) {
    if (size > 1) {
      a->dist_hmatrix_gpu = new DistributedHMatrix_GPU(*a->dist_hmatrix);
    } else {
      a->hmatrix_gpu = new HMatrix_GPU(*a->hmatrix);
    }
  }
#endif
  if (size == 1) {
    if (!kernel && a->sampler) {
      PetscReal Anorm;
      bool      verbose = false;

      ierr = MatApproximateNorm_Private(a->sampler->GetSamplingMat(),NORM_2,PETSC_DECIDE,&Anorm);CHKERRQ(ierr);
      if (boundtocpu) {
        a->sampler->SetGPUSampling(false);
        hara(a->sampler, *a->hmatrix, a->max_rank, 10 /* TODO */,a->rtol*Anorm,a->bs,handle,verbose);
#if defined(PETSC_H2OPUS_USE_GPU)
      } else {
        a->sampler->SetGPUSampling(true);
        hara(a->sampler, *a->hmatrix_gpu, a->max_rank, 10 /* TODO */,a->rtol*Anorm,a->bs,handle,verbose);
#endif
      }
    }
  }
#if defined(PETSC_H2OPUS_USE_GPU)
  if (kernel) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = boundtocpu ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
#endif

  if (!a->s) a->s = 1.0;
  A->assembled = PETSC_TRUE;

  if (a->sampler) {
    PetscBool check = a->check_construction;

    ierr = PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_check",&check,NULL);CHKERRQ(ierr);
    if (check) {
      Mat       E,Ae;
      PetscReal n1,ni,n2;
      PetscReal n1A,niA,n2A;
      void      (*normfunc)(void);

      Ae   = a->sampler->GetSamplingMat();
      ierr = MatConvert(A,MATSHELL,MAT_INITIAL_MATRIX,&E);CHKERRQ(ierr);
      ierr = MatShellSetOperation(E,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS);CHKERRQ(ierr);
      ierr = MatAXPY(E,-1.0,Ae,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_1,&n1);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_INFINITY,&ni);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_2,&n2);CHKERRQ(ierr);

      ierr = MatGetOperation(Ae,MATOP_NORM,&normfunc);CHKERRQ(ierr);
      ierr = MatSetOperation(Ae,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_1,&n1A);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_INFINITY,&niA);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_2,&n2A);CHKERRQ(ierr);
      n1A  = PetscMax(n1A,PETSC_SMALL);
      n2A  = PetscMax(n2A,PETSC_SMALL);
      niA  = PetscMax(niA,PETSC_SMALL);
      ierr = MatSetOperation(Ae,MATOP_NORM,normfunc);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)A),"MATH2OPUS construction errors: NORM_1 %g, NORM_INFINITY %g, NORM_2 %g (%g %g %g)\n",(double)n1,(double)ni,(double)n2,(double)(n1/n1A),(double)(ni/niA),(double)(n2/n2A));
      ierr = MatDestroy(&E);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_H2OPUS(Mat A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not yet supported");
  else {
    a->hmatrix->clearData();
#if defined(PETSC_H2OPUS_USE_GPU)
    if (a->hmatrix_gpu) a->hmatrix_gpu->clearData();
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_H2OPUS(Mat B, MatDuplicateOption op, Mat *nA)
{
  Mat            A;
  Mat_H2OPUS     *a, *b = (Mat_H2OPUS*)B->data;
#if defined(PETSC_H2OPUS_USE_GPU)
  PetscBool      iscpu = PETSC_FALSE;
#else
  PetscBool      iscpu = PETSC_TRUE;
#endif
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,B->rmap->n,B->cmap->n,B->rmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATH2OPUS);CHKERRQ(ierr);
  ierr = MatPropagateSymmetryOptions(B,A);CHKERRQ(ierr);

  a = (Mat_H2OPUS*)A->data;
  a->s = b->s;
  a->ptcloud = new PetscPointCloud<PetscReal>(*b->ptcloud);
  if (op == MAT_COPY_VALUES && b->kernel) a->kernel = new PetscFunctionGenerator<PetscScalar>(*b->kernel);

  if (b->dist_hmatrix) { a->dist_hmatrix = new DistributedHMatrix(*b->dist_hmatrix); }
#if defined(PETSC_H2OPUS_USE_GPU)
  if (b->dist_hmatrix_gpu) { a->dist_hmatrix_gpu = new DistributedHMatrix_GPU(*b->dist_hmatrix_gpu); }
#endif
  if (b->hmatrix) {
    a->hmatrix = new HMatrix(*b->hmatrix);
    if (op == MAT_DO_NOT_COPY_VALUES) a->hmatrix->clearData();
  }
#if defined(PETSC_H2OPUS_USE_GPU)
  if (b->hmatrix_gpu) {
    a->hmatrix_gpu = new HMatrix_GPU(*b->hmatrix_gpu);
    if (op == MAT_DO_NOT_COPY_VALUES) a->hmatrix_gpu->clearData();
  }
#endif

  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetUpMultiply_H2OPUS(A);CHKERRQ(ierr);
  if (op == MAT_COPY_VALUES) {
    A->assembled = PETSC_TRUE;
    a->orthogonal = b->orthogonal;
#if defined(PETSC_H2OPUS_USE_GPU)
    iscpu = B->boundtocpu;
    A->offloadmask = B->offloadmask;
#endif
  }
  ierr = MatBindToCPU(A,iscpu);CHKERRQ(ierr);

  *nA = A;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_H2OPUS(Mat A, PetscViewer view)
{
  Mat_H2OPUS        *h2opus = (Mat_H2OPUS*)A->data;
  PetscBool         isascii;
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(view,&format);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(view,"  H-Matrix constructed from %s\n",h2opus->sampler ? "Mat" : (h2opus->kernel ? "Kernel" : "None"));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view,"  PointCloud dim %D\n",h2opus->ptcloud ? h2opus->ptcloud->getDimension() : 0);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view,"  Admissibility parameters: leaf size %D, eta %g\n",h2opus->leafsize,(double)h2opus->eta);CHKERRQ(ierr);
    if (h2opus->sampler) {
      ierr = PetscViewerASCIIPrintf(view,"  Sampling parameters: max_rank %D, samples %D, tolerance %g\n",h2opus->max_rank,h2opus->bs,(double)h2opus->rtol);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(view,"  Offdiagonal blocks approximation order %D\n",h2opus->basisord);CHKERRQ(ierr);
    }
    if (size == 1) {
      double dense_mem_cpu = h2opus->hmatrix ? h2opus->hmatrix->getDenseMemoryUsage() : 0;
      double low_rank_cpu = h2opus->hmatrix ? h2opus->hmatrix->getLowRankMemoryUsage() : 0;
#if defined(PETSC_H2OPUS_USE_GPU)
      double dense_mem_gpu = h2opus->hmatrix_gpu ? h2opus->hmatrix_gpu->getDenseMemoryUsage() : 0;
      double low_rank_gpu = h2opus->hmatrix_gpu ? h2opus->hmatrix_gpu->getLowRankMemoryUsage() : 0;
#endif
      ierr = PetscViewerASCIIPrintf(view,"  Memory consumption (CPU): %g (dense) %g (low rank) %g GB (total)\n", dense_mem_cpu, low_rank_cpu, low_rank_cpu + dense_mem_cpu);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
      ierr = PetscViewerASCIIPrintf(view,"  Memory consumption (GPU): %g (dense) %g (low rank) %g GB (total)\n", dense_mem_gpu, low_rank_gpu, low_rank_gpu + dense_mem_gpu);CHKERRQ(ierr);
#endif
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatH2OpusSetCoords_H2OPUS(Mat A, PetscInt spacedim, const PetscReal coords[], MatH2OpusKernel kernel, void *kernelctx)
{
  Mat_H2OPUS     *h2opus = (Mat_H2OPUS*)A->data;
  PetscReal      *gcoords;
  PetscInt       N;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscBool      cong;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatHasCongruentLayouts(A,&cong);CHKERRQ(ierr);
  if (!cong) SETERRQ(comm,PETSC_ERR_SUP,"Only for square matrices with congruent layouts");
  N    = A->rmap->N;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    PetscSF      sf;
    MPI_Datatype dtype;

    ierr = MPI_Type_contiguous(spacedim,MPIU_REAL,&dtype);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&dtype);CHKERRQ(ierr);

    ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphWithPattern(sf,A->rmap,PETSCSF_PATTERN_ALLGATHER);CHKERRQ(ierr);
    ierr = PetscMalloc1(spacedim*N,&gcoords);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf,dtype,coords,gcoords);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,dtype,coords,gcoords);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = MPI_Type_free(&dtype);CHKERRQ(ierr);
  } else gcoords = (PetscReal*)coords;

  delete h2opus->ptcloud;
  delete h2opus->kernel;
  h2opus->ptcloud = new PetscPointCloud<PetscReal>(spacedim,N,gcoords);
  if (kernel) h2opus->kernel = new PetscFunctionGenerator<PetscScalar>(kernel,spacedim,kernelctx);
  if (gcoords != coords) { ierr = PetscFree(gcoords);CHKERRQ(ierr); }
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_H2OPUS_USE_GPU)
PetscErrorCode MatBindToCPU_H2OPUS(Mat A, PetscBool flg)
{
  PetscMPIInt    size;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (flg && A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (size > 1) {
      if (!a->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      if (!a->dist_hmatrix) dist_hmatrix = new DistributedHMatrix(*a->dist_hmatrix_gpu);
      else *a->dist_hmatrix = *a->dist_hmatrix_gpu;
    } else {
      if (!a->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      if (!a->hmatrix) hmatrix = new HMatrix(*a->hmatrix_gpu);
      else *a->hmatrix = *a->hmatrix_gpu;
    }
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  } else if (!flg && A->offloadmask == PETSC_OFFLOAD_CPU) {
    if (size > 1) {
      if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      if (!a->dist_hmatrix_gpu) dist_hmatrix_gpu = new DistributedHMatrix(*a->dist_hmatrix);
      else *a->dist_hmatrix_gpu = *a->dist_hmatrix;
    } else {
      if (!a->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      if (!a->hmatrix_gpu) hmatrix_gpu = new HMatrix(*a->hmatrix);
      else *a->hmatrix_gpu = *a->hmatrix;
    }
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}
#endif

PETSC_EXTERN PetscErrorCode MatCreate_H2OPUS(Mat A)
{
  Mat_H2OPUS     *a;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data = (void*)a;

  a->eta              = 0.9;
  a->leafsize         = 32;
  a->basisord         = 4;
  a->max_rank         = 64;
  a->bs               = 32;
  a->rtol             = 1.e-4;
  a->s                = 1.0;
  a->norm_max_samples = 10;
  h2opusCreateDistributedHandleComm(&a->handle,PetscObjectComm((PetscObject)A));

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATH2OPUS);CHKERRQ(ierr);
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->destroy          = MatDestroy_H2OPUS;
  A->ops->view             = MatView_H2OPUS;
  A->ops->assemblyend      = MatAssemblyEnd_H2OPUS;
  A->ops->mult             = MatMult_H2OPUS;
  A->ops->multtranspose    = MatMultTranspose_H2OPUS;
  A->ops->multadd          = MatMultAdd_H2OPUS;
  A->ops->multtransposeadd = MatMultTransposeAdd_H2OPUS;
  A->ops->scale            = MatScale_H2OPUS;
  A->ops->duplicate        = MatDuplicate_H2OPUS;
  A->ops->setfromoptions   = MatSetFromOptions_H2OPUS;
  A->ops->norm             = MatNorm_H2OPUS;
  A->ops->zeroentries      = MatZeroEntries_H2OPUS;
#if defined(PETSC_H2OPUS_USE_GPU)
  A->ops->bindtocpu        = MatBindToCPU_H2OPUS;
#endif

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdense_C",MatProductSetFromOptions_H2OPUS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdensecuda_C",MatProductSetFromOptions_H2OPUS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidense_C",MatProductSetFromOptions_H2OPUS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidensecuda_C",MatProductSetFromOptions_H2OPUS);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatH2OpusOrthogonalize(Mat A)
{
  PetscErrorCode ierr;
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscMPIInt    size;
  PetscBool      boundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (!ish2opus) PetscFunctionReturn(0);
  if (a->orthogonal) PetscFunctionReturn(0);
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size > 1) {
    if (boundtocpu) {
      if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      distributed_horthog(*a->dist_hmatrix, a->handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (!a->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      distributed_horthog(*a->dist_hmatrix_gpu, a->handle);
#endif
    }
  } else {
    h2opusHandle_t handle = a->handle->handle;
    if (boundtocpu) {
      if (!a->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      horthog(*a->hmatrix, handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (!a->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      horthog(*a->hmatrix_gpu, handle);
#endif
    }
  }
  a->orthogonal = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatH2OpusCompress(Mat A, PetscReal tol)
{
  PetscErrorCode ierr;
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscMPIInt    size;
  PetscBool      boundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (!ish2opus) PetscFunctionReturn(0);
  ierr = MatH2OpusOrthogonalize(A);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size > 1) {
    if (boundtocpu) {
      if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      distributed_hcompress(*a->dist_hmatrix, tol, a->handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (!a->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      distributed_hcompress(*a->dist_hmatrix_gpu, tol, a->handle);
#endif
    }
  } else {
    h2opusHandle_t handle = a->handle->handle;
    if (boundtocpu) {
      if (!a->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hcompress(*a->hmatrix, tol, handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (!a->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      hcompress(*a->hmatrix_gpu, tol, handle);
#endif
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatH2OpusSetSamplingMat(Mat A, Mat B, PetscInt bs, PetscReal tol)
{
  PetscBool      ish2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (ish2opus) {
    Mat_H2OPUS *a = (Mat_H2OPUS*)A->data;

    if (!a->sampler) a->sampler = new PetscMatrixSampler();
    a->sampler->SetSamplingMat(B);
    if (bs > 0) a->bs = bs;
    if (tol > 0.) a->rtol = tol;
    delete a->kernel;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateH2OpusFromKernel(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt spacedim, const PetscReal coords[], MatH2OpusKernel kernel, void *kernelctx, PetscReal eta, PetscInt leafsize, PetscInt basisord, Mat* nA)
{
  Mat            A;
  Mat_H2OPUS     *h2opus;
#if defined(PETSC_H2OPUS_USE_GPU)
  PetscBool      iscpu = PETSC_FALSE;
#else
  PetscBool      iscpu = PETSC_TRUE;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATH2OPUS);CHKERRQ(ierr);
  ierr = MatBindToCPU(A,iscpu);CHKERRQ(ierr);
  ierr = MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,kernel,kernelctx);CHKERRQ(ierr);

  h2opus = (Mat_H2OPUS*)A->data;
  if (eta > 0.) h2opus->eta = eta;
  if (leafsize > 0) h2opus->leafsize = leafsize;
  if (basisord > 0) h2opus->basisord = basisord;

  *nA = A;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateH2OpusFromMat(Mat B, PetscInt spacedim, const PetscReal coords[], PetscReal eta, PetscInt leafsize, PetscInt maxrank, PetscInt bs, PetscReal rtol, Mat *nA)
{
  Mat            A;
  Mat_H2OPUS     *h2opus;
  MPI_Comm       comm;
#if defined(PETSC_H2OPUS_USE_GPU)
  PetscBool      iscpu = PETSC_FALSE;
#else
  PetscBool      iscpu = PETSC_TRUE;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(B,spacedim,2);
  PetscValidPointer(nA,4);
  ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,B->rmap->n,B->cmap->n,B->rmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATH2OPUS);CHKERRQ(ierr);
  ierr = MatBindToCPU(A,iscpu);CHKERRQ(ierr);
  if (spacedim) {
    ierr = MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = MatPropagateSymmetryOptions(B,A);CHKERRQ(ierr);
  /* if (!A->symmetric) SETERRQ(comm,PETSC_ERR_SUP,"Unsymmetric sampling does not work"); */

  h2opus = (Mat_H2OPUS*)A->data;
  h2opus->sampler = new PetscMatrixSampler(B);
  if (eta > 0.) h2opus->eta = eta;
  if (leafsize > 0) h2opus->leafsize = leafsize;
  if (maxrank > 0) h2opus->max_rank = maxrank;
  if (bs > 0) h2opus->bs = bs;
  if (rtol > 0.) h2opus->rtol = rtol;
  *nA = A;
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}
