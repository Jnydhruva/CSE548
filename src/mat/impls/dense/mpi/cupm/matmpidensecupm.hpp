#ifndef PETSCMATMPIDENSECUPM_HPP
#define PETSCMATMPIDENSECUPM_HPP

#ifdef __cplusplus
  #include <../src/mat/impls/dense/mpi/mpidense.h>
  #include <petsc/private/matdensecupmimpl.h>
  #include <../src/mat/impls/dense/seq/cupm/matseqdensecupm.hpp>
  #include <../src/vec/vec/impls/mpi/cupm/vecmpicupm.hpp>

namespace Petsc
{

namespace mat
{

namespace cupm
{

namespace impl
{

template <device::cupm::DeviceType T>
class MatMPIDense_CUPM : MatDense_CUPM<T, MatMPIDense_CUPM<T>> {
public:
  MATDENSECUPM_HEADER(T, MatMPIDense_CUPM<T>);

private:
  using MatSeqDense_CUPM_T = MatSeqDense_CUPM<T>;

  PETSC_NODISCARD static constexpr Mat_MPIDense *MatIMPLCast_(Mat) noexcept;
  PETSC_NODISCARD static constexpr MatType       MATIMPLCUPM_() noexcept;

  static PetscErrorCode SetPreallocation_(Mat, PetscDeviceContext, PetscScalar *) noexcept;

public:
  PETSC_NODISCARD static constexpr const char *MatConvert_mpidensecupm_mpidense_C() noexcept;

  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_mpiaij_mpidensecupm_C() noexcept;
  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_mpidensecupm_mpiaij_C() noexcept;

  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_mpiaijcupmsparse_mpidensecupm_C() noexcept;
  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_mpidensecupm_mpiaijcupmsparse_C() noexcept;

  static PetscErrorCode Create(Mat) noexcept;

  static PetscErrorCode BindToCPU(Mat, PetscBool) noexcept;
  static PetscErrorCode Convert_MPIDenseCUPM_MPIDense(Mat, MatType, MatReuse, Mat *) noexcept;
  static PetscErrorCode Convert_MPIDense_MPIDenseCUPM(Mat, MatType, MatReuse, Mat *) noexcept;

  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode GetArray(Mat, PetscScalar **, PetscDeviceContext = nullptr) noexcept;
  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode RestoreArray(Mat, PetscScalar **, PetscDeviceContext = nullptr) noexcept;

private:
  template <PetscMemType mtype, PetscMemoryAccessMode mode>
  static PetscErrorCode GetArrayC_(Mat m, PetscScalar **p) noexcept
  {
    return GetArray<mtype, mode>(m, p);
  }

  template <PetscMemType mtype, PetscMemoryAccessMode mode>
  static PetscErrorCode RestoreArrayC_(Mat m, PetscScalar **p) noexcept
  {
    return RestoreArray<mtype, mode>(m, p);
  }

public:
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetColumnVec(Mat, PetscInt, Vec *) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreColumnVec(Mat, PetscInt, Vec *) noexcept;

  static PetscErrorCode PlaceArray(Mat, const PetscScalar *) noexcept;
  static PetscErrorCode ReplaceArray(Mat, const PetscScalar *) noexcept;
  static PetscErrorCode ResetArray(Mat) noexcept;

  static PetscErrorCode Shift(Mat, PetscScalar) noexcept;
};

// ==========================================================================================
// MatMPIDense_CUPM -- Private API
// ==========================================================================================

template <device::cupm::DeviceType T>
inline constexpr Mat_MPIDense *MatMPIDense_CUPM<T>::MatIMPLCast_(Mat m) noexcept
{
  return static_cast<Mat_MPIDense *>(m->data);
}

template <device::cupm::DeviceType T>
inline constexpr MatType MatMPIDense_CUPM<T>::MATIMPLCUPM_() noexcept
{
  return MATMPIDENSECUPM();
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::SetPreallocation_(Mat A, PetscDeviceContext dctx, PetscScalar *device_array) noexcept
{
  PetscFunctionBegin;
  if (auto &mimplA = MatIMPLCast(A)->A) {
    PetscCall(MatSetType(mimplA, MATSEQDENSECUPM()));
    PetscCall(MatSeqDense_CUPM_T::SetPreallocation(mimplA, dctx, device_array));
  } else {
    PetscCall(MatCreateSeqDenseCUPM<T>(PETSC_COMM_SELF, A->rmap->n, A->cmap->N, device_array, &mimplA, dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MatMPIDense_CUPM -- Public API
// ==========================================================================================

template <device::cupm::DeviceType T>
inline constexpr const char *MatMPIDense_CUPM<T>::MatConvert_mpidensecupm_mpidense_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatConvert_mpidensecuda_mpidense_C" : "MatConvert_mpidensehip_mpidense_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatMPIDense_CUPM<T>::MatProductSetFromOptions_mpiaij_mpidensecupm_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_mpiaij_mpidensecuda_C" : "MatProductSetFromOptions_mpiaij_mpidensehip_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatMPIDense_CUPM<T>::MatProductSetFromOptions_mpidensecupm_mpiaij_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatMatProductSetFromOptions_mpidensecuda_mpiaij_C" : "MatProductSetFromOptions_mpidensehip_mpiaij_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatMPIDense_CUPM<T>::MatProductSetFromOptions_mpiaijcupmsparse_mpidensecupm_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_mpiaijcusparse_mpidensecuda_C" : "MatProductSetFromOptions_mpiaijhipsparse_mpidensehip_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatMPIDense_CUPM<T>::MatProductSetFromOptions_mpidensecupm_mpiaijcupmsparse_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_mpidensecuda_mpiaijcusparse_C" : "MatProductSetFromOptions_mpidensehip_mpiaijhipsparse_C";
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::Create(Mat A) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatCreate_MPIDense(A));
  PetscCall(Convert_MPIDense_MPIDenseCUPM(A, MATMPIDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::BindToCPU(Mat A, PetscBool usehost) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto pobj  = PetscObjectCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm(pobj), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm(pobj), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (const auto mimpl_A = mimpl->A) PetscCall(MatBindToCPU(mimpl_A, usehost));
  A->boundtocpu = usehost;
  PetscCall(PetscStrFreeAllocpy(usehost ? PETSCRANDER48 : PETSCDEVICERAND(), &A->defaultrandtype));
  if (!usehost) {
    PetscBool iscupm;

    PetscCall(PetscObjectTypeCompare(PetscObjectCast(mimpl->cvec), VecMPI_CUPM::VECMPICUPM(), &iscupm));
    if (!iscupm) PetscCall(VecDestroy(&mimpl->cvec));
    PetscCall(PetscObjectTypeCompare(PetscObjectCast(mimpl->cmat), MATMPIDENSECUPM(), &iscupm));
    if (!iscupm) PetscCall(MatDestroy(&mimpl->cmat));
  }

  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVec_C", MatDenseGetColumnVec_MPIDense, GetColumnVec<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVec_C", MatDenseRestoreColumnVec_MPIDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVecRead_C", MatDenseGetColumnVecRead_MPIDense, GetColumnVec<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVecRead_C", MatDenseRestoreColumnVecRead_MPIDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVecWrite_C", MatDenseGetColumnVecWrite_MPIDense, GetColumnVec<PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVecWrite_C", MatDenseRestoreColumnVecWrite_MPIDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_WRITE>);

  MatSetOp_CUPM(usehost, A, shift, MatShift_MPIDense, Shift);

  if (const auto mimpl_cmat = mimpl->cmat) PetscCall(MatBindToCPU(mimpl_cmat, usehost));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::Convert_MPIDenseCUPM_MPIDense(Mat M, MatType, MatReuse reuse, Mat *newmat) noexcept
{
  const auto B    = *newmat;
  const auto pobj = PetscObjectCast(B);

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(M, MAT_COPY_VALUES, newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(M, B, SAME_NONZERO_PATTERN));
  }

  PetscCall(BindToCPU(B, PETSC_TRUE));
  PetscCall(PetscStrFreeAllocpy(VECSTANDARD, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName(pobj, MATMPIDENSE));
  PetscCall(PetscObjectComposeFunction(pobj, MatConvert_mpidensecupm_mpidense_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_mpiaij_mpidensecupm_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_mpidensecupm_mpiaij_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_mpiaijcupmsparse_mpidensecupm_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_mpidensecupm_mpiaijcupmsparse_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArray_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArrayRead_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArrayWrite_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArray_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArrayRead_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArrayWrite_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMPlaceArray_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMResetArray_C(), nullptr));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMReplaceArray_C(), nullptr));
  if (auto &m_A = MatIMPLCast(B)->A) PetscCall(MatConvert(m_A, MATSEQDENSE, MAT_INPLACE_MATRIX, &m_A));

  B->ops->bindtocpu = nullptr;
  B->offloadmask    = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::Convert_MPIDense_MPIDenseCUPM(Mat M, MatType, MatReuse reuse, Mat *newmat) noexcept
{
  const auto B    = *newmat;
  const auto pobj = PetscObjectCast(B);

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(M, MAT_COPY_VALUES, newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(M, B, SAME_NONZERO_PATTERN));
  }

  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
  PetscCall(PetscStrFreeAllocpy(VecMPI_CUPM::VECCUPM(), &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName(pobj, MATMPIDENSECUPM()));
  PetscCall(PetscObjectComposeFunction(pobj, MatConvert_mpidensecupm_mpidense_C(), Convert_MPIDenseCUPM_MPIDense));
  PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_mpiaij_mpidensecupm_C(), MatProductSetFromOptions_MPIAIJ_MPIDense));
  PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_mpiaijcupmsparse_mpidensecupm_C(), MatProductSetFromOptions_MPIAIJ_MPIDense));
  PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_mpidensecupm_mpiaij_C(), MatProductSetFromOptions_MPIDense_MPIAIJ));
  PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_mpidensecupm_mpiaijcupmsparse_C(), MatProductSetFromOptions_MPIDense_MPIAIJ));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArray_C(), GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArrayRead_C(), GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArrayWrite_C(), GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArray_C(), RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArrayRead_C(), RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArrayWrite_C(), RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMPlaceArray_C(), PlaceArray));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMResetArray_C(), ResetArray));
  PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMReplaceArray_C(), ReplaceArray));
  if (auto &m_A = MatIMPLCast(B)->A) {
    PetscCall(MatConvert(m_A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &m_A));
    B->offloadmask = PETSC_OFFLOAD_BOTH;
  } else {
    B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  PetscCall(BindToCPU(B, PETSC_FALSE));
  B->ops->bindtocpu = BindToCPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
template <PetscMemType, PetscMemoryAccessMode access>
inline PetscErrorCode MatMPIDense_CUPM<T>::GetArray(Mat A, PetscScalar **array, PetscDeviceContext) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArray_Private<T, access>(MatIMPLCast(A)->A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemType, PetscMemoryAccessMode access>
inline PetscErrorCode MatMPIDense_CUPM<T>::RestoreArray(Mat A, PetscScalar **array, PetscDeviceContext) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArray_Private<T, access>(MatIMPLCast(A)->A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatMPIDense_CUPM<T>::GetColumnVec(Mat A, PetscInt col, Vec *v) noexcept
{
  const auto mimpl   = MatIMPLCast(A);
  const auto mimpl_A = mimpl->A;
  const auto pobj    = PetscObjectCast(A);
  auto      &cvec    = mimpl->cvec;
  PetscInt   lda;

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm(pobj), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm(pobj), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  mimpl->vecinuse = col + 1;
  if (!cvec) PetscCall(vec::cupm::impl::VecCreateMPICUPMWithArray<T>(PetscObjectComm(pobj), A->rmap->bs, A->rmap->n, A->rmap->N, nullptr, &cvec));
  PetscCall(MatDenseGetLDA(mimpl_A, &lda));
  PetscCall(MatDenseCUPMGetArray_Private<T, access>(mimpl_A, const_cast<PetscScalar **>(&mimpl->ptrinuse)));
  PetscCall(vec::cupm::impl::VecCUPMPlaceArrayAsync<T>(cvec, mimpl->ptrinuse + static_cast<std::size_t>(col) * static_cast<std::size_t>(lda)));
  if (access == PETSC_MEMORY_ACCESS_READ) PetscCall(VecLockReadPush(cvec));
  *v = cvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatMPIDense_CUPM<T>::RestoreColumnVec(Mat A, PetscInt col, Vec *v) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto cvec  = mimpl->cvec;

  PetscFunctionBegin;
  PetscCheck(mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  mimpl->vecinuse = 0;
  PetscCall(MatDenseCUPMRestoreArray_Private<T, access>(mimpl->A, const_cast<PetscScalar **>(&mimpl->ptrinuse)));
  if (access == PETSC_MEMORY_ACCESS_READ) PetscCall(VecLockReadPop(cvec));
  PetscCall(vec::cupm::impl::VecCUPMResetArrayAsync<T>(cvec));
  if (v) *v = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::PlaceArray(Mat A, const PetscScalar *array) noexcept
{
  const auto mimpl = MatIMPLCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm((PetscObject)A), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm((PetscObject)A), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUPMPlaceArray<T>(mimpl->A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::ReplaceArray(Mat A, const PetscScalar *array) noexcept
{
  const auto mimpl = MatIMPLCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm((PetscObject)A), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm((PetscObject)A), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUPMReplaceArray<T>(mimpl->A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::ResetArray(Mat A) noexcept
{
  const auto mimpl = MatIMPLCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm((PetscObject)A), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm((PetscObject)A), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUPMResetArray<T>(mimpl->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMPIDense_CUPM<T>::Shift(Mat A, PetscScalar alpha) noexcept
{
  PetscInt           lda;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(PetscInfo(A, "Performing Shift on backend\n"));
  PetscCall(Shift_Base(dctx, DeviceArrayReadWrite(dctx, A), alpha, lda, A->rmap->rstart, A->rmap->rend, A->cmap->N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace mat

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCMATMPIDENSECUPM_HPP
