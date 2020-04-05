#include <petscmat.h>
#include <hara.h>

#ifndef __MATHARA_HPP
#define __MATHARA_HPP

class PetscMatrixSampler : public HMatrixSampler
{
protected:
  Mat  A;
  Vec  x,y;
  bool gpusampling;
  bool usemult;

private:
  void Init();

public:
  PetscMatrixSampler();
  PetscMatrixSampler(Mat);
  ~PetscMatrixSampler();
  void SetSamplingMat(Mat);
  void SetGPUSampling(bool);
  void SetUseMult(bool);
  virtual void sample(Hara_Real*,Hara_Real*,int);
  Mat GetSamplingMat() { return A; }
};

void PetscMatrixSampler::Init()
{
  this->A = NULL;
  this->x = NULL;
  this->y = NULL;
  this->usemult = false;
  this->gpusampling = false;
}

PetscMatrixSampler::PetscMatrixSampler()
{
  Init();
}

PetscMatrixSampler::PetscMatrixSampler(Mat A)
{
  Init();
  SetSamplingMat(A);
}

void PetscMatrixSampler::SetSamplingMat(Mat A)
{
  PetscErrorCode ierr;

  ierr = PetscObjectReference((PetscObject)A);CHKERRCONTINUE(ierr);
  ierr = MatDestroy(&this->A);CHKERRCONTINUE(ierr);
  this->A = A;
  ierr = VecDestroy(&this->x);CHKERRCONTINUE(ierr);
  ierr = VecDestroy(&this->y);CHKERRCONTINUE(ierr);
  ierr = MatCreateVecs(A,&this->x,&this->y);CHKERRCONTINUE(ierr);
#if defined(PETSC_HAVE_CUDA)
  /* always use CUDA vectors */
  ierr = VecSetType(this->x,VECCUDA);CHKERRCONTINUE(ierr);
  ierr = VecSetType(this->y,VECCUDA);CHKERRCONTINUE(ierr);
#endif
}

void PetscMatrixSampler::SetUseMult(bool usemult)
{
  this->usemult = usemult;
}

void PetscMatrixSampler::SetGPUSampling(bool gpusampling)
{
  this->gpusampling = gpusampling;
}

PetscMatrixSampler::~PetscMatrixSampler()
{
  PetscErrorCode ierr;

  ierr = MatDestroy(&A);CHKERRCONTINUE(ierr);
  ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
  ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
}

void PetscMatrixSampler::sample(Hara_Real *x, Hara_Real *y, int samples)
{
  PetscErrorCode ierr;
  if (this->usemult || samples == 1) {
    PetscInt nl;

    ierr = MatGetLocalSize(this->A,&nl,NULL);CHKERRCONTINUE(ierr);
    for (int i = 0; i < samples; i++) {
      if (this->gpusampling) {
#if defined(PETSC_HAVE_CUDA)
        ierr = VecCUDAPlaceArray(this->x,x);CHKERRCONTINUE(ierr);
        ierr = VecCUDAPlaceArray(this->y,y);CHKERRCONTINUE(ierr);
#endif
      } else {
        ierr = VecPlaceArray(this->x,x);CHKERRCONTINUE(ierr);
        ierr = VecPlaceArray(this->y,y);CHKERRCONTINUE(ierr);
      }
      ierr = MatMult(this->A,this->x,this->y);CHKERRCONTINUE(ierr);
      if (this->gpusampling) {
#if defined(PETSC_HAVE_CUDA)
        ierr = VecCUDAResetArray(this->x);CHKERRCONTINUE(ierr);
        ierr = VecCUDAResetArray(this->y);CHKERRCONTINUE(ierr);
#endif
      } else {
        ierr = VecResetArray(this->x);CHKERRCONTINUE(ierr);
        ierr = VecResetArray(this->y);CHKERRCONTINUE(ierr);
      }
      x += nl;
      y += nl;
    }
  } else {
    MPI_Comm comm = PetscObjectComm((PetscObject)this->A);
    Mat      X,Y;
    PetscInt M,N,m,n;

    ierr = MatGetLocalSize(this->A,&m,&n);CHKERRCONTINUE(ierr);
    ierr = MatGetSize(this->A,&M,&N);CHKERRCONTINUE(ierr);
    if (!this->gpusampling) {
      ierr = MatCreateDense(comm,n,samples,N,samples,x,&X);CHKERRCONTINUE(ierr);
      ierr = MatCreateDense(comm,m,samples,M,samples,y,&Y);CHKERRCONTINUE(ierr);
    } else {
#if defined(PETSC_HAVE_CUDA)
      ierr = MatCreateDenseCUDA(comm,n,samples,N,samples,x,&X);CHKERRCONTINUE(ierr);
      ierr = MatCreateDenseCUDA(comm,m,samples,M,samples,y,&Y);CHKERRCONTINUE(ierr);
#endif
    }
    ierr = MatMatMult(this->A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRCONTINUE(ierr);
    ierr = MatDestroy(&X);CHKERRCONTINUE(ierr);
    ierr = MatDestroy(&Y);CHKERRCONTINUE(ierr);
  }
}
#endif
