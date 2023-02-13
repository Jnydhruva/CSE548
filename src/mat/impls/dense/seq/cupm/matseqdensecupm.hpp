#ifndef PETSCMATSEQDENSECUPM_HPP
#define PETSCMATSEQDENSECUPM_HPP

#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

#if defined(__cplusplus)
  #include <petsc/private/matdensecupmimpl.h>
  #include <petsc/private/deviceimpl.h> // PetscDeviceContextGetOptionalNullContext_Internal()
  #include <petsc/private/randomimpl.h> // _p_PetscRandom
  #include <petsc/private/vecimpl.h>    // _p_Vec
  #include <petsc/private/cupmobject.hpp>
  #include <petsc/private/cupmsolverinterface.hpp>

  #include <petsc/private/cpp/type_traits.hpp> // PetscObjectCast()
  #include <petsc/private/cpp/utility.hpp>     // util::exchange()

  #include <../src/vec/vec/impls/seq/cupm/vecseqcupm.hpp> // for VecSeq_CUPM

namespace Petsc
{

namespace mat
{

namespace cupm
{

namespace impl
{

template <device::cupm::DeviceType T>
class MatSeqDense_CUPM : MatDense_CUPM<T, MatSeqDense_CUPM<T>> {
public:
  MATDENSECUPM_HEADER(T, MatSeqDense_CUPM<T>);

private:
  struct Mat_SeqDenseCUPM {
    PetscScalar *d_v; // pointer to the matrix on the GPU
    PetscBool    user_alloc;
    PetscScalar *unplacedarray; // if one called MatCUPMDensePlaceArray(), this is where it stashed the original
    PetscBool    unplaced_user_alloc;
    // factorization support
    PetscCuBLASInt *d_fact_ipiv; // device pivots
    PetscScalar    *d_fact_tau;  // device QR tau vector
    PetscScalar    *d_fact_work; // device workspace
    PetscCuBLASInt  fact_lwork;
    PetscCuBLASInt *d_fact_info; // device info
    // workspace
    Vec workvec;
  };

  static PetscErrorCode SetPreallocation_(Mat, PetscDeviceContext, PetscScalar *) noexcept;

  static PetscErrorCode HostToDevice_(Mat, PetscDeviceContext) noexcept;
  static PetscErrorCode DeviceToHost_(Mat, PetscDeviceContext) noexcept;

  static PetscErrorCode CheckCUPMSolverInfo_(const cupmBlasInt_t *, cupmStream_t) noexcept;

  template <typename Derived>
  struct SolveCommon;
  struct SolveQR;
  struct SolveCholesky;
  struct SolveLU;

  template <typename Solver, bool transpose>
  static PetscErrorCode MatSolve_Factored_Dispatch_(Mat, Vec, Vec) noexcept;
  template <typename Solver, bool transpose>
  static PetscErrorCode MatMatSolve_Factored_Dispatch_(Mat, Mat, Mat) noexcept;
  template <bool transpose>
  static PetscErrorCode MatMultAdd_Dispatch_(Mat, Vec, Vec, Vec) noexcept;

  PETSC_NODISCARD static constexpr MatType       MATIMPLCUPM_() noexcept;
  PETSC_NODISCARD static constexpr Mat_SeqDense *MatIMPLCast_(Mat) noexcept;

public:
  PETSC_NODISCARD static constexpr Mat_SeqDenseCUPM *MatCUPMCast(Mat) noexcept;

  // define these by hand since they don't fit the above mold
  PETSC_NODISCARD static constexpr const char *MatConvert_seqdensecupm_seqdense_C() noexcept;
  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_seqaij_seqdensecupm_C() noexcept;

  static PetscErrorCode Create(Mat) noexcept;
  static PetscErrorCode Destroy(Mat) noexcept;
  static PetscErrorCode SetUp(Mat) noexcept;
  static PetscErrorCode Reset(Mat) noexcept;

  static PetscErrorCode BindToCPU(Mat, PetscBool) noexcept;
  static PetscErrorCode Convert_SeqDense_SeqDenseCUPM(Mat, MatType, MatReuse, Mat *) noexcept;
  static PetscErrorCode Convert_SeqDenseCUPM_SeqDense(Mat, MatType, MatReuse, Mat *) noexcept;

  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode GetArray(Mat, PetscScalar **, PetscDeviceContext = nullptr) noexcept;
  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode RestoreArray(Mat, PetscScalar **, PetscDeviceContext = nullptr) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetArrayAndMemType(Mat, PetscScalar **, PetscMemType *, PetscDeviceContext = nullptr) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreArrayAndMemType(Mat, PetscScalar **, PetscDeviceContext = nullptr) noexcept;

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

  template <PetscMemoryAccessMode mode>
  static PetscErrorCode GetArrayAndMemTypeC_(Mat m, PetscScalar **p, PetscMemType *tp) noexcept
  {
    return GetArrayAndMemType<mode>(m, p, tp);
  }

  template <PetscMemoryAccessMode mode>
  static PetscErrorCode RestoreArrayAndMemTypeC_(Mat m, PetscScalar **p) noexcept
  {
    return RestoreArrayAndMemType<mode>(m, p);
  }

public:
  static PetscErrorCode PlaceArray(Mat, const PetscScalar *) noexcept;
  static PetscErrorCode ReplaceArray(Mat, const PetscScalar *) noexcept;
  static PetscErrorCode ResetArray(Mat) noexcept;

  template <bool transpose_A, bool transpose_B>
  static PetscErrorCode MatMatMult_Numeric_Dispatch(Mat, Mat, Mat) noexcept;
  static PetscErrorCode Copy(Mat, Mat, MatStructure) noexcept;
  static PetscErrorCode ZeroEntries(Mat) noexcept;
  static PetscErrorCode Scale(Mat, PetscScalar) noexcept;
  static PetscErrorCode Shift(Mat, PetscScalar) noexcept;
  static PetscErrorCode AXPY(Mat, PetscScalar, Mat, MatStructure) noexcept;
  static PetscErrorCode Duplicate(Mat, MatDuplicateOption, Mat *) noexcept;
  static PetscErrorCode SetRandom(Mat, PetscRandom) noexcept;

  static PetscErrorCode GetColumnVector(Mat, Vec, PetscInt) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetColumnVec(Mat, PetscInt, Vec *) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreColumnVec(Mat, PetscInt, Vec *) noexcept;

  static PetscErrorCode GetFactor(Mat, MatFactorType, Mat *) noexcept;
  static PetscErrorCode InvertFactors(Mat) noexcept;

  static PetscErrorCode GetSubMatrix(Mat, PetscInt, PetscInt, PetscInt, PetscInt, Mat *) noexcept;
  static PetscErrorCode RestoreSubMatrix(Mat, Mat *) noexcept;

  static PetscErrorCode SetLDA(Mat, PetscInt) noexcept;
};

// Declare these here so that the functions below can make use of them
namespace
{

template <device::cupm::DeviceType T>
inline PetscErrorCode MatCreateSeqDenseCUPM(MPI_Comm comm, PetscInt m, PetscInt n, PetscScalar *data, Mat *A, PetscDeviceContext dctx = nullptr) noexcept
{
  Mat mat;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCheck(size <= 1, comm, PETSC_ERR_ARG_WRONG, "Invalid communicator size %d", size);
  }
  PetscCall(MatCreate(comm, &mat));
  PetscCall(MatSetSizes(mat, m, n, m, n));
  PetscCall(MatSetType(mat, MatSeqDense_CUPM<T>::MATSEQDENSECUPM()));
  PetscCall(MatSeqDense_CUPM<T>::SetPreallocation(mat, dctx, data));
  *A = mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMatMultNumeric_SeqDenseCUPM_SeqDenseCUPM(Mat A, Mat B, Mat C, PetscBool TA, PetscBool TB) noexcept
{
  PetscFunctionBegin;
  if (TA) {
    if (TB) {
      PetscCall(MatSeqDense_CUPM<T>::template MatMatMult_Numeric_Dispatch<true, true>(A, B, C));
    } else {
      PetscCall(MatSeqDense_CUPM<T>::template MatMatMult_Numeric_Dispatch<true, false>(A, B, C));
    }
  } else {
    if (TB) {
      PetscCall(MatSeqDense_CUPM<T>::template MatMatMult_Numeric_Dispatch<false, true>(A, B, C));
    } else {
      PetscCall(MatSeqDense_CUPM<T>::template MatMatMult_Numeric_Dispatch<false, false>(A, B, C));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSolverTypeRegister_DENSECUPM() noexcept
{
  PetscFunctionBegin;
  for (auto ftype : util::make_array(MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_QR)) {
    PetscCall(MatSolverTypeRegister(MatSeqDense_CUPM<T>::MATSOLVERCUPM(), MATSEQDENSE, ftype, MatSeqDense_CUPM<T>::GetFactor));
    PetscCall(MatSolverTypeRegister(MatSeqDense_CUPM<T>::MATSOLVERCUPM(), MatSeqDense_CUPM<T>::MATSEQDENSECUPM(), ftype, MatSeqDense_CUPM<T>::GetFactor));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

// ==========================================================================================
// MatSeqDense_CUPM - Private API - Utility
// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::SetPreallocation_(Mat m, PetscDeviceContext dctx, PetscScalar *user_device_array) noexcept
{
  const auto mcu = MatCUPMCast(m);

  PetscFunctionBegin;
  if (PetscLikely(mcu->d_v)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  {
    const auto ncols = m->cmap->n;
    const auto nrows = m->rmap->n;
    const auto lda = MatIMPLCast(m)->lda = [](PetscBLASInt lda, PetscInt nrows) {
      if (lda <= 0) {
        // CPU preallocation has not yet been performed
        lda = static_cast<PetscBLASInt>(nrows);
      }
      return lda;
    }(MatIMPLCast(m)->lda, nrows);
    cupmStream_t stream;

    PetscCall(checkCupmBlasIntCast(nrows));
    PetscCall(checkCupmBlasIntCast(ncols));
    PetscCall(GetHandlesFrom_(dctx, &stream));
    if (!mcu->user_alloc) PetscCallCUPM(cupmFreeAsync(mcu->d_v, stream));
    if (user_device_array) {
      mcu->user_alloc = PETSC_TRUE;
      mcu->d_v        = user_device_array;
    } else {
      const auto size = lda * ncols;

      mcu->user_alloc = PETSC_FALSE;
      PetscCall(PetscIntMultError(lda, ncols, nullptr));
      PetscCall(PetscCUPMMallocAsync(&mcu->d_v, size, stream));
      PetscCall(PetscCUPMMemsetAsync(mcu->d_v, 0, size, stream));
    }
  }
  m->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::HostToDevice_(Mat m, PetscDeviceContext dctx) noexcept
{
  auto      &offloadmask = m->offloadmask;
  const auto nrows       = m->rmap->n;
  const auto ncols       = m->cmap->n;
  const auto copy        = offloadmask == PETSC_OFFLOAD_CPU || offloadmask == PETSC_OFFLOAD_UNALLOCATED;

  PetscFunctionBegin;
  PetscCheckTypeName(m, MATSEQDENSECUPM());
  if (m->boundtocpu) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscInfo(m, "%s matrix %" PetscInt_FMT " x %" PetscInt_FMT "\n", copy ? "Copy" : "Reusing", nrows, ncols));
  if (copy) {
    cupmStream_t stream;

    // Allocate GPU memory if not present
    PetscCall(SetPreallocation(m, dctx));
    PetscCall(GetHandlesFrom_(dctx, &stream));
    PetscCall(PetscLogEventBegin(MAT_DenseCopyToGPU, m, 0, 0, 0));
    {
      const auto mimpl = MatIMPLCast(m);
      const auto lda   = mimpl->lda;
      const auto src   = mimpl->v;
      const auto dest  = MatCUPMCast(m)->d_v;

      if (lda > nrows) {
        PetscCall(PetscCUPMMemcpy2DAsync(dest, lda, src, lda, nrows, ncols, cupmMemcpyHostToDevice, stream));
      } else {
        PetscCall(PetscCUPMMemcpyAsync(dest, src, lda * ncols, cupmMemcpyHostToDevice, stream));
      }
    }
    PetscCall(PetscLogEventEnd(MAT_DenseCopyToGPU, m, 0, 0, 0));
    // order important, ensure that offloadmask is PETSC_OFFLOAD_BOTH
    offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::DeviceToHost_(Mat m, PetscDeviceContext dctx) noexcept
{
  auto      &offloadmask = m->offloadmask;
  const auto nrows       = m->rmap->n;
  const auto ncols       = m->cmap->n;
  const auto copy        = offloadmask == PETSC_OFFLOAD_GPU;

  PetscFunctionBegin;
  PetscCheckTypeName(m, MATSEQDENSECUPM());
  PetscCall(PetscInfo(m, "%s matrix %" PetscInt_FMT " x %" PetscInt_FMT "\n", copy ? "Copy" : "Reusing", nrows, ncols));
  if (copy) {
    cupmStream_t stream;

    // MatCreateSeqDenseCUPM may not allocate CPU memory. Allocate if needed
    PetscCall(MatSeqDenseSetPreallocation(m, nullptr));
    PetscCall(GetHandlesFrom_(dctx, &stream));
    PetscCall(PetscLogEventBegin(MAT_DenseCopyFromGPU, m, 0, 0, 0));
    {
      const auto mimpl = MatIMPLCast(m);
      const auto lda   = mimpl->lda;
      const auto dest  = mimpl->v;
      const auto src   = MatCUPMCast(m)->d_v;

      if (lda > nrows) {
        PetscCall(PetscCUPMMemcpy2DAsync(dest, lda, src, lda, nrows, ncols, cupmMemcpyDeviceToHost, stream));
      } else {
        PetscCall(PetscCUPMMemcpyAsync(dest, src, lda * ncols, cupmMemcpyDeviceToHost, stream));
      }
    }
    PetscCall(PetscLogEventEnd(MAT_DenseCopyFromGPU, m, 0, 0, 0));
    // order is important, MatSeqDenseSetPreallocation() might set offloadmask
    offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::CheckCUPMSolverInfo_(const cupmBlasInt_t *fact_info, cupmStream_t stream) noexcept
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    cupmBlasInt_t info = 0;

    PetscCall(PetscCUPMMemcpyAsync(&info, fact_info, 1, cupmMemcpyDeviceToHost, stream));
    if (stream) PetscCallCUPM(cupmStreamSynchronize(stream));
    static_assert(std::is_same<decltype(info), int>::value, "");
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cupmSolver %d", -info);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MatSeqDense_CUPM - Private API - Solver Dispatch
// ==========================================================================================

// specific solvers called through the dispatch_() family of functions
template <device::cupm::DeviceType T>
template <typename Derived>
struct MatSeqDense_CUPM<T>::SolveCommon {
  using derived_type = Derived;

  template <typename F>
  static PetscErrorCode factor_prepare(Mat A, cupmStream_t stream, F &&cupmSolverComputeFactLwork) noexcept
  {
    const auto m   = static_cast<cupmBlasInt_t>(A->rmap->n);
    const auto n   = static_cast<cupmBlasInt_t>(A->cmap->n);
    const auto mcu = MatCUPMCast(A);

    PetscFunctionBegin;
    PetscCall(PetscInfo(A, "%s factor %d x %d on backend\n", derived_type::NAME(), m, n));
    A->factortype             = derived_type::MATFACTORTYPE();
    A->ops->solve             = MatSolve_Factored_Dispatch_<derived_type, false>;
    A->ops->solvetranspose    = MatSolve_Factored_Dispatch_<derived_type, true>;
    A->ops->matsolve          = MatMatSolve_Factored_Dispatch_<derived_type, false>;
    A->ops->matsolvetranspose = MatMatSolve_Factored_Dispatch_<derived_type, true>;

    PetscCall(PetscStrFreeAllocpy(MATSOLVERCUPM(), &A->solvertype));
    if (!mcu->d_fact_info) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_info, 1, stream));
    if (!mcu->fact_lwork) {
      PetscCallCUPMSOLVER(cupmSolverComputeFactLwork(&mcu->fact_lwork));
      PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_work, mcu->fact_lwork, stream));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <device::cupm::DeviceType T>
struct MatSeqDense_CUPM<T>::SolveLU : SolveCommon<SolveLU> {
  using base_type = SolveCommon<SolveLU>;

  static constexpr const char   *NAME() noexcept { return "LU"; }
  static constexpr MatFactorType MATFACTORTYPE() noexcept { return MAT_FACTOR_LU; }

  static PetscErrorCode factor(Mat A, IS, IS, const MatFactorInfo *) noexcept
  {
    const auto         m = static_cast<cupmBlasInt_t>(A->rmap->n);
    const auto         n = static_cast<cupmBlasInt_t>(A->cmap->n);
    cupmStream_t       stream;
    cupmSolverHandle_t handle;
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    if (!m || !n) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(GetHandles_(&dctx, &handle, &stream));
    {
      const auto mcu = MatCUPMCast(A);
      const auto lda = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);
      const auto da  = DeviceArrayReadWrite(dctx, A);

      // clang-format off
      PetscCall(
        base_type::factor_prepare(
          A, stream,
          [&](cupmBlasInt_t *fact_lwork)
          {
            return cupmSolverXgetrf_bufferSize(handle, m, n, da.cupmdata(), lda, fact_lwork);
          }
        )
      );
      // clang-format on
      if (!mcu->d_fact_ipiv) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_ipiv, n, stream));

      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMSOLVER(cupmSolverXgetrf(handle, m, n, da.cupmdata(), lda, mcu->d_fact_work, mcu->d_fact_ipiv, mcu->d_fact_info));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(CheckCUPMSolverInfo_(mcu->d_fact_info, stream));
    }
    PetscCall(PetscLogGpuFlops(2.0 * n * n * m / 3.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <bool transpose>
  static PetscErrorCode solve(Mat A, PetscScalar *x, cupmBlasInt_t ldx, cupmBlasInt_t m, cupmBlasInt_t nrhs, cupmBlasInt_t k, PetscDeviceContext dctx, cupmStream_t stream) noexcept
  {
    const auto         mcu       = MatCUPMCast(A);
    const auto         fact_info = mcu->d_fact_info;
    const auto         fact_ipiv = mcu->d_fact_ipiv;
    const auto         lda       = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);
    cupmSolverHandle_t handle;

    PetscFunctionBegin;
    PetscCall(GetHandlesFrom_(dctx, &handle));
    PetscCall(PetscInfo(A, "%s solve %d x %d on backend\n", NAME(), m, k));
    PetscCall(PetscLogGpuTimeBegin());
    {
      const auto da = DeviceArrayRead(dctx, A);

      PetscCallCUPMSOLVER(cupmSolverXgetrs(handle, transpose ? CUPMBLAS_OP_T : CUPMBLAS_OP_N, m, nrhs, da.cupmdata(), lda, fact_ipiv, x, ldx, fact_info));
      PetscCall(CheckCUPMSolverInfo_(fact_info, stream));
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(nrhs * (2.0 * m * m - m)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <device::cupm::DeviceType T>
struct MatSeqDense_CUPM<T>::SolveCholesky : SolveCommon<SolveCholesky> {
  using base_type = SolveCommon<SolveCholesky>;

  static constexpr const char   *NAME() noexcept { return "Cholesky"; }
  static constexpr MatFactorType MATFACTORTYPE() noexcept { return MAT_FACTOR_CHOLESKY; }

  static PetscErrorCode factor(Mat A, IS, const MatFactorInfo *) noexcept
  {
    const auto         n = static_cast<cupmBlasInt_t>(A->rmap->n);
    PetscDeviceContext dctx;
    cupmSolverHandle_t handle;
    cupmStream_t       stream;

    PetscFunctionBegin;
    if (!n || !A->cmap->n) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCheck(A->spd == PETSC_BOOL3_TRUE, PETSC_COMM_SELF, PETSC_ERR_SUP, "cupmSolversytrs unavailable. Use MAT_FACTOR_LU");
    PetscCall(GetHandles_(&dctx, &handle, &stream));
    {
      const auto mcu = MatCUPMCast(A);
      const auto lda = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);
      const auto da  = DeviceArrayReadWrite(dctx, A);

      // clang-format off
      PetscCall(
        base_type::factor_prepare(
          A, stream,
          [&](cupmBlasInt_t *fact_lwork)
          {
            return cupmSolverXpotrf_bufferSize(
              handle, CUPMBLAS_FILL_MODE_LOWER, n, da.cupmdata(), lda, fact_lwork
            );
          }
        )
      );
      // clang-format on
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMSOLVER(cupmSolverXpotrf(handle, CUPMBLAS_FILL_MODE_LOWER, n, da.cupmdata(), lda, mcu->d_fact_work, mcu->fact_lwork, mcu->d_fact_info));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(CheckCUPMSolverInfo_(mcu->d_fact_info, stream));
    }
    PetscCall(PetscLogGpuFlops(1.0 * n * n * n / 3.0));

  #if 0
    // At the time of writing this interface (cuda 10.0), cusolverDn does not implement *sytrs
    // and *hetr* routines. The code below should work, and it can be activated when *sytrs
    // routines will be available
    if (!mcu->d_fact_ipiv) PetscCall(PetscCUPMMallocAsync(mcu->d_fact_ipiv, n, stream));
    if (!mcu->fact_lwork) {
      PetscCallCUPMSOLVER(cupmSolverDnXsytrf_bufferSize(handle, n, da.cupmdata(), lda, &mcu->fact_lwork));
      PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_work, mcu->fact_lwork, stream));
    }
    if (mcu->d_fact_info) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_info, 1, stream));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMSOLVER(cupmSolverXsytrf(handle, CUPMBLAS_FILL_MODE_LOWER, n, da, lda, mcu->d_fact_ipiv, mcu->d_fact_work, mcu->fact_lwork, mcu->d_fact_info));
    PetscCall(PetscLogGpuTimeEnd());
  #endif
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <bool transpose>
  static PetscErrorCode solve(Mat A, PetscScalar *x, cupmBlasInt_t ldx, cupmBlasInt_t m, cupmBlasInt_t nrhs, cupmBlasInt_t k, PetscDeviceContext dctx, cupmStream_t stream) noexcept
  {
    const auto         mcu       = MatCUPMCast(A);
    const auto         fact_info = mcu->d_fact_info;
    const auto         lda       = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);
    cupmSolverHandle_t handle;

    PetscFunctionBegin;
    PetscAssert(mcu->d_fact_ipiv, PETSC_COMM_SELF, PETSC_ERR_LIB, "cupmSolversytrs not implemented");
    PetscCall(GetHandlesFrom_(dctx, &handle));
    PetscCall(PetscInfo(A, "%s solve %d x %d on backend\n", NAME(), m, k));
    {
      const auto da = DeviceArrayRead(dctx, A);

      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMSOLVER(cupmSolverXpotrs(handle, CUPMBLAS_FILL_MODE_LOWER, m, nrhs, da.cupmdata(), lda, x, ldx, fact_info));
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(CheckCUPMSolverInfo_(fact_info, stream));
    PetscCall(PetscLogGpuFlops(nrhs * (2.0 * m * m - m)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <device::cupm::DeviceType T>
struct MatSeqDense_CUPM<T>::SolveQR : SolveCommon<SolveQR> {
  using base_type = SolveCommon<SolveQR>;

  static constexpr const char   *NAME() noexcept { return "QR"; }
  static constexpr MatFactorType MATFACTORTYPE() noexcept { return MAT_FACTOR_QR; }

  static PetscErrorCode factor(Mat A, IS, const MatFactorInfo *) noexcept
  {
    const auto         m     = static_cast<cupmBlasInt_t>(A->rmap->n);
    const auto         n     = static_cast<cupmBlasInt_t>(A->cmap->n);
    const auto         min   = std::min(m, n);
    const auto         mimpl = MatIMPLCast(A);
    cupmStream_t       stream;
    cupmSolverHandle_t handle;
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    if (!m || !n) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(GetHandles_(&dctx, &handle, &stream));
    mimpl->rank = min;
    {
      const auto mcu = MatCUPMCast(A);
      const auto lda = static_cast<cupmBlasInt_t>(mimpl->lda);
      const auto da  = DeviceArrayReadWrite(dctx, A);

      if (!mcu->workvec) PetscCall(vec::cupm::impl::VecCreateSeqCUPMAsync<T>(PetscObjectComm(PetscObjectCast(A)), m, &mcu->workvec));
      if (!mcu->d_fact_tau) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_tau, min, stream));
      // clang-format off
      PetscCall(
        base_type::factor_prepare(
          A, stream,
          [&](cupmBlasInt_t *fact_lwork)
          {
            return cupmSolverXgeqrf_bufferSize(handle, m, n, da.cupmdata(), lda, fact_lwork);
          }
        )
      );
      // clang-format on
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMSOLVER(cupmSolverXgeqrf(handle, m, n, da.cupmdata(), lda, mcu->d_fact_tau, mcu->d_fact_work, mcu->fact_lwork, mcu->d_fact_info));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(CheckCUPMSolverInfo_(mcu->d_fact_info, stream));
    }
    PetscCall(PetscLogGpuFlops(2.0 * min * min * (std::max(m, n) - min / 3.0)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <bool transpose>
  static PetscErrorCode solve(Mat A, PetscScalar *x, cupmBlasInt_t ldx, cupmBlasInt_t m, cupmBlasInt_t nrhs, cupmBlasInt_t k, PetscDeviceContext dctx, cupmStream_t stream) noexcept
  {
    const auto         mimpl      = MatIMPLCast(A);
    const auto         rank       = static_cast<cupmBlasInt_t>(mimpl->rank);
    const auto         lda        = static_cast<cupmBlasInt_t>(mimpl->lda);
    const auto         mcu        = MatCUPMCast(A);
    const auto         fact_info  = mcu->d_fact_info;
    const auto         fact_tau   = mcu->d_fact_tau;
    const auto         fact_work  = mcu->d_fact_work;
    const auto         fact_lwork = mcu->fact_lwork;
    cupmSolverHandle_t solver_handle;
    cupmBlasHandle_t   blas_handle;

    PetscFunctionBegin;
    PetscCall(GetHandlesFrom_(dctx, &blas_handle, &solver_handle));
    PetscCall(PetscInfo(A, "%s solve %d x %d on backend\n", NAME(), m, k));
    PetscCall(PetscLogGpuTimeBegin());
    {
      const auto         da  = DeviceArrayRead(dctx, A);
      const cupmScalar_t one = 1.;

      if (transpose) {
        PetscCallCUPMBLAS(cupmBlasXtrsm(blas_handle, CUPMBLAS_SIDE_LEFT, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_T, CUPMBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da, lda, x, ldx));
        PetscCallCUPMSOLVER(cupmSolverXormqr(solver_handle, CUPMBLAS_SIDE_LEFT, CUPMBLAS_OP_N, m, nrhs, rank, da, lda, fact_tau, x, ldx, fact_work, fact_lwork, fact_info));
        PetscCall(CheckCUPMSolverInfo_(fact_info, stream));
      } else {
        constexpr auto op = PetscDefined(USE_COMPLEX) ? CUPMBLAS_OP_C : CUPMBLAS_OP_T;

        PetscCallCUPMSOLVER(cupmSolverXormqr(solver_handle, CUPMBLAS_SIDE_LEFT, op, m, nrhs, rank, da, lda, fact_tau, x, ldx, fact_work, fact_lwork, fact_info));
        PetscCall(CheckCUPMSolverInfo_(fact_info, stream));
        PetscCallCUPMBLAS(cupmBlasXtrsm(blas_handle, CUPMBLAS_SIDE_LEFT, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da, lda, x, ldx));
      }
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogFlops(nrhs * (4.0 * m * rank - (rank * rank))));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <device::cupm::DeviceType T>
template <typename Solver, bool transpose>
inline PetscErrorCode MatSeqDense_CUPM<T>::MatSolve_Factored_Dispatch_(Mat A, Vec x, Vec y) noexcept
{
  using namespace vec::cupm::impl;
  const auto         pobj_A  = PetscObjectCast(A);
  const auto         m       = static_cast<cupmBlasInt_t>(A->rmap->n);
  const auto         k       = static_cast<cupmBlasInt_t>(A->cmap->n);
  auto              &workvec = MatCUPMCast(A)->workvec;
  PetscScalar       *y_array = nullptr;
  PetscDeviceContext dctx;
  PetscBool          xiscupm, yiscupm, aiscupm;
  bool               use_y_array_directly;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCheck(A->factortype != MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix must be factored to solve");
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(x), VecSeq_CUPM::VECSEQCUPM(), &xiscupm));
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(y), VecSeq_CUPM::VECSEQCUPM(), &yiscupm));
  PetscCall(PetscObjectTypeCompare(pobj_A, MATSEQDENSECUPM(), &aiscupm));
  PetscAssert(aiscupm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Matrix A is somehow not CUPM?????????????????????????????");
  PetscCall(GetHandles_(&dctx, &stream));
  use_y_array_directly = yiscupm && (k >= m);
  {
    const PetscScalar *x_array;
    const auto         xisdevice = xiscupm && PetscOffloadDevice(x->offloadmask);
    const auto         copy_mode = xisdevice ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice;

    if (!use_y_array_directly && !workvec) PetscCall(VecCreateSeqCUPMAsync<T>(PetscObjectComm(pobj_A), m, &workvec));
    // The logic here is to try to minimize the amount of memory copying:
    //
    // If we call VecCUPMGetArrayRead(X, &x) every time xiscupm and the data is not offloaded
    // to the GPU yet, then the data is copied to the GPU. But we are only trying to get the
    // data in order to copy it into the y array. So the array x will be wherever the data
    // already is so that only one memcpy is performed
    if (xisdevice) {
      PetscCall(VecCUPMGetArrayReadAsync<T>(x, &x_array, dctx));
    } else {
      PetscCall(VecGetArrayRead(x, &x_array));
    }
    PetscCall(VecCUPMGetArrayWriteAsync<T>(use_y_array_directly ? y : workvec, &y_array, dctx));
    PetscCall(PetscCUPMMemcpyAsync(y_array, x_array, m, copy_mode, stream));
    if (xisdevice) {
      PetscCall(VecCUPMRestoreArrayReadAsync<T>(x, &x_array, dctx));
    } else {
      PetscCall(VecRestoreArrayRead(x, &x_array));
    }
  }

  if (!aiscupm) PetscCall(MatConvert(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  PetscCall(Solver{}.template solve<transpose>(A, y_array, m, m, 1, k, dctx, stream));
  if (!aiscupm) PetscCall(MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));

  if (use_y_array_directly) {
    PetscCall(VecCUPMRestoreArrayWriteAsync<T>(y, &y_array, dctx));
  } else {
    const auto   copy_mode = yiscupm ? cupmMemcpyDeviceToDevice : cupmMemcpyDeviceToHost;
    PetscScalar *yv;

    // The logic here is that the data is not yet in either y's GPU array or its CPU array.
    // There is nothing in the interface to say where the user would like it to end up. So we
    // choose the GPU, because it is the faster option
    if (yiscupm) {
      PetscCall(VecCUPMGetArrayWriteAsync<T>(y, &yv, dctx));
    } else {
      PetscCall(VecGetArray(y, &yv));
    }
    PetscCall(PetscCUPMMemcpyAsync(yv, y_array, k, copy_mode, stream));
    if (yiscupm) {
      PetscCall(VecCUPMRestoreArrayWriteAsync<T>(y, &yv, dctx));
    } else {
      PetscCall(VecRestoreArray(y, &yv));
    }
    PetscCall(VecCUPMRestoreArrayWriteAsync<T>(workvec, &y_array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <typename Solver, bool transpose>
inline PetscErrorCode MatSeqDense_CUPM<T>::MatMatSolve_Factored_Dispatch_(Mat A, Mat B, Mat X) noexcept
{
  const auto         m = static_cast<cupmBlasInt_t>(A->rmap->n);
  const auto         k = static_cast<cupmBlasInt_t>(A->cmap->n);
  cupmBlasInt_t      nrhs, ldb, ldx, ldy;
  PetscScalar       *y;
  PetscBool          biscupm, xiscupm, aiscupm;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCheck(A->factortype != MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix must be factored to solve");
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(B), MATSEQDENSECUPM(), &biscupm));
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(X), MATSEQDENSECUPM(), &xiscupm));
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(A), MATSEQDENSECUPM(), &aiscupm));
  PetscCall(GetHandles_(&dctx, &stream));
  {
    PetscInt n;

    PetscCall(MatGetSize(B, nullptr, &n));
    nrhs = cupmBlasIntCast(n);
    PetscCall(MatDenseGetLDA(B, &n));
    ldb = cupmBlasIntCast(n);
    PetscCall(MatDenseGetLDA(X, &n));
    ldx = cupmBlasIntCast(n);
  }
  {
    // The logic here is to try to minimize the amount of memory copying:
    //
    // If we call MatDenseCUPMGetArrayRead(B, &b) every time biscupm and the data is not
    // offloaded to the GPU yet, then the data is copied to the GPU. But we are only trying to
    // get the data in order to copy it into the y array. So the array b will be wherever the
    // data already is so that only one memcpy is performed
    const auto         bisdevice = biscupm && PetscOffloadDevice(B->offloadmask);
    const auto         copy_mode = bisdevice ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice;
    const PetscScalar *b;

    if (bisdevice) {
      b = DeviceArrayRead(dctx, B);
    } else if (biscupm) {
      b = HostArrayRead(dctx, B);
    } else {
      PetscCall(MatDenseGetArrayRead(B, &b));
    }

    if (ldx < m || !xiscupm) {
      // X's array cannot serve as the array (too small or not on device), B's array cannot
      // serve as the array (const), so allocate a new array
      ldy = m;
      PetscCall(PetscCUPMMallocAsync(&y, nrhs * m));
    } else {
      // X's array should serve as the array
      ldy = ldx;
      y   = DeviceArrayWrite(dctx, X);
    }
    PetscCall(PetscCUPMMemcpy2DAsync(y, ldy, b, ldb, m, nrhs, copy_mode, stream));
    if (!bisdevice && !biscupm) PetscCall(MatDenseRestoreArrayRead(B, &b));
  }

  // convert to CUPM twice??????????????????????????????????
  // but A should already be CUPM??????????????????????????????????????
  if (!aiscupm) PetscCall(MatConvert(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  PetscCall(Solver{}.template solve<transpose>(A, y, ldy, m, nrhs, k, dctx, stream));
  if (!aiscupm) PetscCall(MatConvert(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));

  if (ldx < m || !xiscupm) {
    const auto   copy_mode = xiscupm ? cupmMemcpyDeviceToDevice : cupmMemcpyDeviceToHost;
    PetscScalar *x;

    // The logic here is that the data is not yet in either X's GPU array or its CPU
    // array. There is nothing in the interface to say where the user would like it to end up.
    // So we choose the GPU, because it is the faster option
    if (xiscupm) {
      x = DeviceArrayWrite(dctx, X);
    } else {
      PetscCall(MatDenseGetArray(X, &x));
    }
    PetscCall(PetscCUPMMemcpy2DAsync(x, ldx, y, ldy, k, nrhs, copy_mode, stream));
    if (!xiscupm) PetscCall(MatDenseRestoreArray(X, &x));
    PetscCallCUPM(cupmFreeAsync(y, stream));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// MatMultAdd_SeqDenseCUDA
// MatMultTransposeAdd_SeqDenseCUDA
// MatMult_SeqDenseCUDA
// MatMultTranspose_SeqDenseCUDA
template <device::cupm::DeviceType T>
template <bool transpose>
inline PetscErrorCode MatSeqDense_CUPM<T>::MatMultAdd_Dispatch_(Mat A, Vec xx, Vec yy, Vec zz) noexcept
{
  const auto         m   = static_cast<cupmBlasInt_t>(A->rmap->n);
  const auto         n   = static_cast<cupmBlasInt_t>(A->cmap->n);
  const auto         lda = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);
  cupmBlasHandle_t   handle;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (yy && yy != zz) PetscCall(VecSeq_CUPM::copy(yy, zz)); // mult add
  if (!m || !n) {
    // mult only
    if (!yy) PetscCall(VecSeq_CUPM::set(zz, 0.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscInfo(A, "Matrix-vector product %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", m, n));
  PetscCall(GetHandles_(&dctx, &handle));
  {
    const auto         da     = DeviceArrayRead(dctx, A);
    const auto         xarray = VecSeq_CUPM::DeviceArrayRead(dctx, xx);
    const auto         zarray = VecSeq_CUPM::DeviceArrayReadWrite(dctx, zz);
    const cupmScalar_t one = 1.0, zero = 0.0;

    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXgemv(handle, transpose ? CUPMBLAS_OP_T : CUPMBLAS_OP_N, m, n, &one, da, lda, xarray, 1, (yy ? &one : &zero), zarray, 1));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(2.0 * m * n - (yy ? 0 : m)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MatSeqDense_CUPM - Public API
// ==========================================================================================

template <device::cupm::DeviceType T>
inline constexpr MatType MatSeqDense_CUPM<T>::MATIMPLCUPM_() noexcept
{
  return MATSEQDENSECUPM();
}

template <device::cupm::DeviceType T>
inline constexpr typename MatSeqDense_CUPM<T>::Mat_SeqDenseCUPM *MatSeqDense_CUPM<T>::MatCUPMCast(Mat m) noexcept
{
  return static_cast<Mat_SeqDenseCUPM *>(m->spptr);
}

template <device::cupm::DeviceType T>
inline constexpr Mat_SeqDense *MatSeqDense_CUPM<T>::MatIMPLCast_(Mat m) noexcept
{
  return static_cast<Mat_SeqDense *>(m->data);
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatSeqDense_CUPM<T>::MatConvert_seqdensecupm_seqdense_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatConvert_seqdensecuda_seqdense_C" : "MatConvert_seqdensehip_seqdense_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatSeqDense_CUPM<T>::MatProductSetFromOptions_seqaij_seqdensecupm_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_seqaij_seqdensecuda_C" : "MatProductSetFromOptions_seqaij_seqdensehip_C";
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Create(Mat A) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
  PetscCall(MatCreate_SeqDense(A));
  PetscCall(Convert_SeqDense_SeqDenseCUPM(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Destroy(Mat A) noexcept
{
  PetscFunctionBegin;
  // prevent copying back data if we own the data pointer
  if (!MatIMPLCast(A)->user_alloc) A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscCall(Convert_SeqDenseCUPM_SeqDense(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatDestroy_SeqDense(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::SetUp(Mat A) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (!A->preallocated) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(SetPreallocation(A, dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Reset(Mat A) noexcept
{
  PetscFunctionBegin;
  if (const auto mcu = MatCUPMCast(A)) {
    cupmStream_t stream;

    PetscCheck(!mcu->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDense%sResetArray() must be called first", cupmNAME());
    PetscCall(GetHandles_(&stream));
    if (!mcu->user_alloc) PetscCallCUPM(cupmFreeAsync(mcu->d_v, stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_fact_tau, stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_fact_ipiv, stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_fact_info, stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_fact_work, stream));
    PetscCall(VecDestroy(&mcu->workvec));
    PetscCall(PetscFree(A->spptr /* mcu */));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::BindToCPU(Mat A, PetscBool usehost) noexcept
{
  const auto mimpl    = MatIMPLCast(A);
  const auto pobj = PetscObjectCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  A->boundtocpu = usehost;
  PetscCall(PetscStrFreeAllocpy(usehost ? PETSCRANDER48 : PETSCDEVICERAND(), &A->defaultrandtype));
  if (usehost) {
    PetscDeviceContext dctx;

    // make sure we have an up-to-date copy on the CPU
    PetscCall(GetHandles_(&dctx));
    PetscCall(DeviceToHost_(A, dctx));
  } else {
    PetscBool iscupm;

    PetscCall(PetscObjectTypeCompare(PetscObjectCast(mimpl->cvec), VecSeq_CUPM::VECSEQCUPM(), &iscupm));
    if (!iscupm) PetscCall(VecDestroy(&mimpl->cvec));
    PetscCall(PetscObjectTypeCompare(PetscObjectCast(mimpl->cmat), MATSEQDENSECUPM(), &iscupm));
    if (!iscupm) PetscCall(MatDestroy(&mimpl->cmat));
  }

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseGetArray_C",
    MatDenseGetArray_SeqDense,
    GetArrayC_<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>
  );

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseGetArrayRead_C",
    MatDenseGetArray_SeqDense,
    GetArrayC_<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>
  );

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseGetArrayWrite_C",
    MatDenseGetArray_SeqDense,
    GetArrayC_<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>
  );

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseGetArrayAndMemType_C",
    nullptr,
    GetArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_READ_WRITE>
  );

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseRestoreArrayAndMemType_C",
    nullptr,
    RestoreArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_READ_WRITE>
  );

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseGetArrayReadAndMemType_C",
    nullptr,
    GetArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_READ>
  );

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseRestoreArrayReadAndMemType_C",
    nullptr,
    RestoreArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_READ>
  );

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseGetArrayWriteAndMemType_C",
    nullptr,
    GetArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_WRITE>
  );

  MatComposeOp_CUPM(
    usehost, pobj, "MatDenseRestoreArrayWriteAndMemType_C",
    nullptr,
    RestoreArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_WRITE>
  );

  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVec_C", MatDenseGetColumnVec_SeqDense, GetColumnVec<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVec_C", MatDenseRestoreColumnVec_SeqDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVecRead_C", MatDenseGetColumnVecRead_SeqDense, GetColumnVec<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVecRead_C", MatDenseRestoreColumnVecRead_SeqDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVecWrite_C", MatDenseGetColumnVecWrite_SeqDense, GetColumnVec<PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVecWrite_C", MatDenseRestoreColumnVecWrite_SeqDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetSubMatrix_C", MatDenseGetSubMatrix_SeqDense, GetSubMatrix);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreSubMatrix_C", MatDenseRestoreSubMatrix_SeqDense, RestoreSubMatrix);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseSetLDA_C", MatDenseSetLDA_SeqDense, SetLDA);
  MatComposeOp_CUPM(usehost, pobj, "MatQRFactor_C", MatQRFactor_SeqDense, SolveQR::factor);

  MatSetOp_CUPM(usehost, A, duplicate, MatDuplicate_SeqDense, Duplicate);
  MatSetOp_CUPM(usehost, A, mult, MatMult_SeqDense, [](Mat A, Vec xx, Vec yy) { return MatMultAdd_Dispatch_</* transpose */ false>(A, xx, nullptr, yy); });
  MatSetOp_CUPM(usehost, A, multtranspose, MatMultTranspose_SeqDense, [](Mat A, Vec xx, Vec yy) { return MatMultAdd_Dispatch_</* transpose */ true>(A, xx, nullptr, yy); });
  MatSetOp_CUPM(usehost, A, multadd, MatMultAdd_SeqDense, MatMultAdd_Dispatch_</* transpose */ false>);
  MatSetOp_CUPM(usehost, A, multtransposeadd, MatMultTransposeAdd_SeqDense, MatMultAdd_Dispatch_</* transpose */ true>);
  MatSetOp_CUPM(usehost, A, matmultnumeric, MatMatMultNumeric_SeqDense_SeqDense, MatMatMult_Numeric_Dispatch</* transpose_A */ false, /* transpose_B */ false>);
  MatSetOp_CUPM(usehost, A, mattransposemultnumeric, MatMatTransposeMultNumeric_SeqDense_SeqDense, MatMatMult_Numeric_Dispatch</* transpose_A */ false, /* transpose_B */ true>);
  MatSetOp_CUPM(usehost, A, transposematmultnumeric, MatTransposeMatMultNumeric_SeqDense_SeqDense, MatMatMult_Numeric_Dispatch</* transpose_A */ true, /* transpose_B */ false>);
  MatSetOp_CUPM(usehost, A, axpy, MatAXPY_SeqDense, AXPY);
  MatSetOp_CUPM(usehost, A, choleskyfactor, MatCholeskyFactor_SeqDense, SolveCholesky::factor);
  MatSetOp_CUPM(usehost, A, lufactor, MatLUFactor_SeqDense, SolveLU::factor);
  MatSetOp_CUPM(usehost, A, getcolumnvector, MatGetColumnVector_SeqDense, GetColumnVector);
  MatSetOp_CUPM(usehost, A, scale, MatScale_SeqDense, Scale);
  MatSetOp_CUPM(usehost, A, shift, MatShift_SeqDense, Shift);
  MatSetOp_CUPM(usehost, A, copy, MatCopy_SeqDense, Copy);
  MatSetOp_CUPM(usehost, A, zeroentries, MatZeroEntries_SeqDense, ZeroEntries);
  MatSetOp_CUPM(usehost, A, setup, MatSetUp_SeqDense, SetUp);
  MatSetOp_CUPM(usehost, A, setrandom, MatSetRandom_SeqDense, SetRandom);
  // seemingly always the same
  A->ops->productsetfromoptions = MatProductSetFromOptions_SeqDense;

  if (const auto cmat = mimpl->cmat) PetscCall(MatBindToCPU(cmat, usehost));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Convert_SeqDenseCUPM_SeqDense(Mat M, MatType type, MatReuse reuse, Mat *newmat) noexcept
{
  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    // TODO these cases should be optimized
    PetscCall(MatConvert_Basic(M, type, reuse, newmat));
  } else {
    const auto B    = *newmat;
    const auto pobj = PetscObjectCast(B);

    PetscCall(BindToCPU(B, PETSC_TRUE));
    PetscCall(Reset(B));
    PetscCall(PetscStrFreeAllocpy(VECSTANDARD, &B->defaultvectype));
    // cvec might be VECSEQCUPM. Destroy it and rebuild a VECSEQ when needed
    PetscCall(VecDestroy(&MatIMPLCast(B)->cvec));
    PetscCall(PetscObjectChangeTypeName(pobj, MATSEQDENSE));

    PetscCall(PetscObjectComposeFunction(pobj, MatConvert_seqdensecupm_seqdense_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArray_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArrayRead_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArrayWrite_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArray_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArrayRead_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArrayWrite_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMPlaceArray_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMResetArray_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMReplaceArray_C(), nullptr));
    PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_seqaij_seqdensecupm_C(), nullptr));

    B->ops->bindtocpu = nullptr;
    B->ops->destroy   = MatDestroy_SeqDense;
    B->offloadmask    = PETSC_OFFLOAD_CPU;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Convert_SeqDense_SeqDenseCUPM(Mat M, MatType type, MatReuse reuse, Mat *newmat) noexcept
{
  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    // TODO these cases should be optimized
    PetscCall(MatConvert_Basic(M, type, reuse, newmat));
  } else {
    const auto B    = *newmat;
    const auto pobj = PetscObjectCast(B);

    PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
    PetscCall(PetscStrFreeAllocpy(VecSeq_CUPM::VECCUPM(), &B->defaultvectype));
    PetscCall(PetscObjectChangeTypeName(pobj, MATSEQDENSECUPM()));
    PetscCall(PetscObjectComposeFunction(pobj, MatConvert_seqdensecupm_seqdense_C(), Convert_SeqDenseCUPM_SeqDense));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArray_C(), GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArrayRead_C(), GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMGetArrayWrite_C(), GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArray_C(), RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArrayRead_C(), RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMRestoreArrayWrite_C(), RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMPlaceArray_C(), PlaceArray));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMResetArray_C(), ResetArray));
    PetscCall(PetscObjectComposeFunction(pobj, MatDenseCUPMReplaceArray_C(), ReplaceArray));
    PetscCall(PetscObjectComposeFunction(pobj, MatProductSetFromOptions_seqaij_seqdensecupm_C(), MatProductSetFromOptions_SeqAIJ_SeqDense));
    // cvec might be VECSEQ. Destroy it and rebuild a VECSEQCUPM when needed
    PetscCall(VecDestroy(&MatIMPLCast(B)->cvec));

    {
      MatSeqDense_CUPM *mcu;

      PetscCall(PetscNew(&mcu));
      B->spptr = mcu;
    }
    B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;

    PetscCall(BindToCPU(B, PETSC_FALSE));
    B->ops->bindtocpu = BindToCPU;
    B->ops->destroy   = Destroy;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
template <PetscMemType mtype, PetscMemoryAccessMode access>
inline PetscErrorCode MatSeqDense_CUPM<T>::GetArray(Mat m, PetscScalar **array, PetscDeviceContext dctx) noexcept
{
  constexpr auto hostmem     = PetscMemTypeHost(mtype);
  constexpr auto read_access = PetscMemoryAccessRead(access);

  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (hostmem) {
    if (read_access) {
      PetscCall(DeviceToHost_(m, dctx));
    } else if (!MatIMPLCast(m)->v) {
      // MatCreateSeqDenseCUPM may not allocate CPU memory. Allocate if needed
      PetscCall(MatSeqDenseSetPreallocation(m, nullptr));
    }
    *array = MatIMPLCast(m)->v;
  } else {
    if (read_access) {
      PetscCall(HostToDevice_(m, dctx));
    } else if (!MatCUPMCast(m)->d_v) {
      // write-only
      PetscCall(SetPreallocation(m, dctx, nullptr));
    }
    *array = MatCUPMCast(m)->d_v;
  }
  if (PetscMemoryAccessWrite(access)) m->offloadmask = hostmem ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemType mtype, PetscMemoryAccessMode access>
inline PetscErrorCode MatSeqDense_CUPM<T>::RestoreArray(Mat m, PetscScalar **array, PetscDeviceContext) noexcept
{
  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  if (PetscMemoryAccessWrite(access)) {
    // WRITE or READ_WRITE
    m->offloadmask = PetscMemTypeHost(mtype) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  *array = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatSeqDense_CUPM<T>::GetArrayAndMemType(Mat m, PetscScalar **array, PetscMemType *mtype, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(GetArray<PETSC_MEMTYPE_DEVICE, access>(m, array, dctx));
  if (mtype) *mtype = PETSC_MEMTYPE_CUPM();
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatSeqDense_CUPM<T>::RestoreArrayAndMemType(Mat m, PetscScalar **array, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(RestoreArray<PETSC_MEMTYPE_DEVICE, access>(m, array, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::PlaceArray(Mat A, const PetscScalar *array) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto mcu   = MatCUPMCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!mcu->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDense%sResetArray() must be called first", cupmNAME());
  if (mimpl->v) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(HostToDevice_(A, dctx));
  }
  mcu->unplacedarray       = util::exchange(mcu->d_v, const_cast<PetscScalar *>(array));
  mcu->unplaced_user_alloc = util::exchange(mcu->user_alloc, PETSC_TRUE);
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::ReplaceArray(Mat A, const PetscScalar *array) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto mcu   = MatCUPMCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!mcu->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDense%sResetArray() must be called first", cupmNAME());
  if (!mcu->user_alloc) {
    cupmStream_t stream;

    PetscCall(GetHandles_(&stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_v, stream));
  }
  mcu->d_v        = const_cast<PetscScalar *>(array);
  mcu->user_alloc = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::ResetArray(Mat A) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto mcu   = MatCUPMCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (mimpl->v) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(HostToDevice_(A, dctx));
  }
  mcu->d_v        = util::exchange(mcu->unplacedarray, nullptr);
  mcu->user_alloc = mcu->unplaced_user_alloc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

// MatTransposeMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA
// MatMatMultNumeric_SeqDenseCUDA_SeqDenseCUDA
// MatMatTransposeMultNumeric_SeqDenseCUDA_SeqDenseCUDA
template <device::cupm::DeviceType T>
template <bool transpose_A, bool transpose_B>
inline PetscErrorCode MatSeqDense_CUPM<T>::MatMatMult_Numeric_Dispatch(Mat A, Mat B, Mat C) noexcept
{
  const auto         m = C->rmap->n;
  const auto         n = C->cmap->n;
  const auto         k = transpose_A ? A->rmap->n : A->cmap->n;
  PetscInt           alda, blda, clda;
  PetscBool          Aiscupm, Biscupm;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   handle;

  PetscFunctionBegin;
  if (!m || !n || !k) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(checkCupmBlasIntCast(m));
  PetscCall(checkCupmBlasIntCast(n));
  PetscCall(checkCupmBlasIntCast(k));
  // we may end up with SEQDENSE as one of the arguments
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(A), MATSEQDENSECUPM(), &Aiscupm));
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(B), MATSEQDENSECUPM(), &Biscupm));
  if (!Aiscupm) PetscCall(MatConvert(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  if (!Biscupm) PetscCall(MatConvert(B, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &B));
  PetscCall(PetscInfo(C, "Matrix-Matrix product %d x %d x %d on backend\n", m, k, n));
  PetscCall(MatDenseGetLDA(A, &alda));
  PetscCall(MatDenseGetLDA(B, &blda));
  PetscCall(MatDenseGetLDA(C, &clda));
  PetscCall(GetHandles_(&dctx, &handle));
  {
    const auto         da  = DeviceArrayRead(dctx, A);
    const auto         db  = DeviceArrayRead(dctx, B);
    const auto         dc  = DeviceArrayWrite(dctx, C);
    const cupmScalar_t one = 1.0, zero = 0.0;

    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXgemm(handle, transpose_A ? CUPMBLAS_OP_T : CUPMBLAS_OP_N, transpose_B ? CUPMBLAS_OP_T : CUPMBLAS_OP_N, m, n, k, &one, da, alda, db, blda, &zero, dc, clda));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(1.0 * m * n * k + 1.0 * m * n * (k - 1)));
  if (!Aiscupm) PetscCall(MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  if (!Biscupm) PetscCall(MatConvert(B, MATSEQDENSE, MAT_INPLACE_MATRIX, &B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Copy(Mat A, Mat B, MatStructure str) noexcept
{
  const auto ma = A->rmap->n;
  const auto na = A->cmap->n;

  PetscFunctionBegin;
  PetscAssert(ma == B->rmap->n && na == B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "size(B) != size(A)");
  // The two matrices must have the same copy implementation to be eligible for fast copy
  if (A->ops->copy == B->ops->copy) {
    const auto         lda_a = MatIMPLCast(A)->lda;
    const auto         lda_b = MatIMPLCast(B)->lda;
    PetscDeviceContext dctx;
    cupmStream_t       stream;

    PetscCall(GetHandles_(&dctx, &stream));
    PetscCall(PetscLogGpuTimeBegin());
    {
      const auto va = DeviceArrayRead(dctx, A);
      const auto vb = DeviceArrayWrite(dctx, B);

      if (lda_a > ma || lda_b > ma) {
        PetscCall(PetscCUPMMemcpy2DAsync(vb.data(), lda_b, va.data(), lda_a, ma, na, cupmMemcpyDeviceToDevice, stream));
      } else {
        PetscCall(PetscCUPMMemcpyAsync(vb.data(), va.data(), ma * na, cupmMemcpyDeviceToDevice, stream));
      }
    }
    PetscCall(PetscLogGpuTimeEnd());
  } else {
    PetscCall(MatCopy_Basic(A, B, str));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::ZeroEntries(Mat m) noexcept
{
  const auto         ma  = m->rmap->n;
  const auto         na  = m->cmap->n;
  const auto         lda = MatIMPLCast(m)->lda;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  {
    const auto va = DeviceArrayWrite(dctx, m);

    if (lda > ma) {
      PetscCall(PetscCUPMMemset2DAsync(va.data(), lda, 0, ma, na, stream));
    } else {
      PetscCall(PetscCUPMMemsetAsync(va.data(), 0, ma * na, stream));
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Scale(Mat A, PetscScalar alpha) noexcept
{
  const auto         m   = static_cast<cupmBlasInt_t>(A->rmap->n);
  const auto         n   = static_cast<cupmBlasInt_t>(A->cmap->n);
  const auto         N   = m * n;
  const auto         lda = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);
  cupmBlasHandle_t   handle;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscInfo(A, "Performing Scale %d x %d on backend\n", m, n));
  PetscCall(GetHandles_(&dctx, &handle));
  {
    constexpr cupmBlasInt_t one = 1;
    const auto              da  = DeviceArrayReadWrite(dctx, A);

    PetscCall(PetscLogGpuTimeBegin());
    if (lda > m) {
      for (cupmBlasInt_t j = 0; j < n; ++j) PetscCallCUPMBLAS(cupmBlasXscal(handle, m, &alpha, da + lda * j, one));
    } else {
      PetscCallCUPMBLAS(cupmBlasXscal(handle, N, &alpha, da, one));
    }
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Shift(Mat A, PetscScalar alpha) noexcept
{
  const auto         m = A->rmap->n;
  const auto         n = A->cmap->n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscInfo(A, "Performing Shift %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", m, n));
  PetscCall(GetHandles_(&dctx));
  PetscCall(Shift_Base(dctx, DeviceArrayReadWrite(dctx, A), alpha, MatIMPLCast(A)->lda, 0, m, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::AXPY(Mat Y, PetscScalar alpha, Mat X, MatStructure) noexcept
{
  const auto         m_x = X->rmap->n, m_y = Y->rmap->n;
  const auto         n_x = X->cmap->n, n_y = Y->cmap->n;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   handle;

  PetscFunctionBegin;
  if (!m_x || !n_x) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscInfo(Y, "Performing AXPY %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", m_y, n_y));
  PetscCall(GetHandles_(&dctx, &handle));
  {
    constexpr cupmBlasInt_t one  = 1;
    const auto              N    = m_x * n_x;
    const auto              ldax = static_cast<cupmBlasInt_t>(MatIMPLCast(X)->lda);
    const auto              lday = static_cast<cupmBlasInt_t>(MatIMPLCast(Y)->lda);
    const auto              dx   = DeviceArrayRead(dctx, X);
    const auto              dy   = alpha == 0.0 ? DeviceArrayWrite(dctx, Y).cupmdata() : DeviceArrayReadWrite(dctx, Y).cupmdata();

    PetscCall(PetscLogGpuTimeBegin());
    if (ldax > m_x || lday > m_x) {
      for (cupmBlasInt_t j = 0; j < n_x; j++) PetscCallCUPMBLAS(cupmBlasXaxpy(handle, m_x, &alpha, dx.cupmdata() + j * ldax, one, dy + j * lday, one));
    } else {
      PetscCallCUPMBLAS(cupmBlasXaxpy(handle, N, &alpha, dx.cupmdata(), one, dy, one));
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(PetscMax(2 * N - 1, 0)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::Duplicate(Mat A, MatDuplicateOption cpvalues, Mat *B) noexcept
{
  const auto pobj      = PetscObjectCast(A);
  const auto m         = A->rmap->n;
  const auto n         = A->cmap->n;
  const auto hcpvalues = (cpvalues == MAT_COPY_VALUES && A->offloadmask != PETSC_OFFLOAD_CPU) ? MAT_DO_NOT_COPY_VALUES : cpvalues;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm(pobj), B));
  PetscCall(MatSetSizes(*B, m, n, m, n));
  PetscCall(MatSetType(*B, pobj->type_name));
  PetscCall(MatDuplicateNoCreate_SeqDense(*B, A, hcpvalues));
  if (cpvalues == MAT_COPY_VALUES && hcpvalues != MAT_COPY_VALUES) PetscCall(Copy(A, *B, SAME_NONZERO_PATTERN));
  // allocate memory if needed
  if (cpvalues != MAT_COPY_VALUES) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(SetPreallocation(*B, dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::SetRandom(Mat A, PetscRandom rng) noexcept
{
  PetscBool iscurand;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(A), PETSCDEVICERAND(), &iscurand));
  if (iscurand) {
    const auto         m = A->rmap->n;
    const auto         n = A->cmap->n;
    PetscInt           lda;
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(MatDenseGetLDA(A, &lda));
    {
      const auto a = DeviceArrayWrite(dctx, A);

      if (lda > m) {
        for (PetscInt i = 0; i < n; i++) PetscCall(PetscRandomGetValues(rng, m, a.data() + i * lda));
      } else {
        PetscInt mn;

        PetscCall(PetscIntMultError(m, n, &mn));
        PetscCall(PetscRandomGetValues(rng, mn, a));
      }
    }
  } else {
    PetscCall(MatSetRandom_SeqDense(A, rng));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::GetColumnVector(Mat A, Vec v, PetscInt col) noexcept
{
  const auto         mimpl       = MatIMPLCast(A);
  const auto         mcu         = MatCUPMCast(A);
  const auto         offloadmask = A->offloadmask;
  const auto         n           = A->rmap->n;
  const auto         col_offset  = [&](const PetscScalar *ptr) { return ptr + col * mimpl->lda; };
  PetscBool          viscupm;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny(PetscObjectCast(v), &viscupm, VecSeq_CUPM::VECSEQCUPM(), VecSeq_CUPM::VECMPICUPM(), VecSeq_CUPM::VECCUPM(), ""));
  PetscCall(GetHandles_(&dctx, &stream));
  if (viscupm && !v->boundtocpu) {
    // update device data
    const auto x = VecSeq_CUPM::DeviceArrayWrite(dctx, v);

    if (PetscOffloadDevice(offloadmask)) {
      PetscCall(PetscCUPMMemcpyAsync(x.data(), col_offset(DeviceArrayRead(dctx, A)), n, cupmMemcpyHostToHost, stream));
    } else {
      PetscCall(PetscCUPMMemcpyAsync(x.data(), col_offset(HostArrayRead(dctx, A)), n, cupmMemcpyHostToDevice, stream));
    }
  } else {
    PetscScalar *x;

    // update host data
    PetscCall(VecGetArrayWrite(v, &x));
    if (PetscOffloadUnallocated(offloadmask) || PetscOffloadHost(offloadmask)) {
      PetscCall(PetscArraycpy(x, col_offset(HostArrayRead(dctx, A)), n));
    } else if (PetscOffloadDevice(offloadmask)) {
      PetscCall(PetscCUPMMemcpyAsync(x, col_offset(DeviceArrayRead(dctx, A)), n, cupmMemcpyDeviceToHost, stream));
    }
    PetscCall(VecRestoreArrayWrite(v, &x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatSeqDense_CUPM<T>::GetColumnVec(Mat A, PetscInt col, Vec *v) noexcept
{
  const auto         mimpl = MatIMPLCast(A);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  mimpl->vecinuse = col + 1;
  PetscCall(GetHandles_(&dctx));
  PetscCall(GetArray<PETSC_MEMTYPE_DEVICE, access>(A, const_cast<PetscScalar **>(&mimpl->ptrinuse), dctx));
  if (!mimpl->cvec) {
    // we pass the data of A, to prevent allocating needless GPU memory the first time
    // VecCUPMPlaceArray is called
    PetscCall(vec::cupm::impl::VecCreateSeqCUPMWithArraysAsync<T>(PetscObjectComm(PetscObjectCast(A)), A->rmap->bs, A->rmap->n, nullptr, mimpl->ptrinuse, &mimpl->cvec));
  }
  PetscCall(vec::cupm::impl::VecCUPMPlaceArrayAsync<T>(mimpl->cvec, mimpl->ptrinuse + static_cast<std::size_t>(col) * static_cast<std::size_t>(mimpl->lda)));
  if (access == PETSC_MEMORY_ACCESS_READ) PetscCall(VecLockReadPush(mimpl->cvec));
  *v = mimpl->cvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatSeqDense_CUPM<T>::RestoreColumnVec(Mat A, PetscInt, Vec *v) noexcept
{
  const auto         mimpl = MatIMPLCast(A);
  const auto         cvec  = mimpl->cvec;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheck(mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  mimpl->vecinuse = 0;
  if (access == PETSC_MEMORY_ACCESS_READ) PetscCall(VecLockReadPop(cvec));
  PetscCall(vec::cupm::impl::VecCUPMResetArrayAsync<T>(cvec));
  PetscCall(GetHandles_(&dctx));
  PetscCall(RestoreArray<PETSC_MEMTYPE_DEVICE, access>(A, const_cast<PetscScalar **>(&mimpl->ptrinuse), dctx));
  if (v) *v = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::GetFactor(Mat A, MatFactorType ftype, Mat *fact_out) noexcept
{
  const auto m = A->rmap->n;
  const auto n = A->cmap->n;
  Mat        fact;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm(PetscObjectCast(A)), &fact));
  PetscCall(MatSetSizes(fact, m, n, m, n));
  PetscCall(MatSetType(fact, MATSEQDENSECUPM()));
  fact->factortype = ftype;
  switch (ftype) {
  case MAT_FACTOR_LU:
  case MAT_FACTOR_ILU: // fall-through
    fact->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqDense;
    fact->ops->ilufactorsymbolic = MatLUFactorSymbolic_SeqDense;
    break;
  case MAT_FACTOR_CHOLESKY:
  case MAT_FACTOR_ICC: // fall-through
    fact->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
    break;
  case MAT_FACTOR_QR: {
    const auto pobj = PetscObjectCast(fact);

    PetscCall(PetscObjectComposeFunction(pobj, "MatQRFactor_C", MatQRFactor_SeqDense));
    PetscCall(PetscObjectComposeFunction(pobj, "MatQRFactorSymbolic_C", MatQRFactorSymbolic_SeqDense));
  } break;
  case MAT_FACTOR_NONE:
  case MAT_FACTOR_ILUDT:     // fall-through
  case MAT_FACTOR_NUM_TYPES: // fall-through
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MatFactorType %s not supported", MatFactorTypes[ftype]);
  }
  PetscCall(PetscStrFreeAllocpy(MATSOLVERCUPM(), &fact->solvertype));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, const_cast<char **>(fact->preferredordering) + MAT_FACTOR_LU));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, const_cast<char **>(fact->preferredordering) + MAT_FACTOR_ILU));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, const_cast<char **>(fact->preferredordering) + MAT_FACTOR_CHOLESKY));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, const_cast<char **>(fact->preferredordering) + MAT_FACTOR_ICC));
  *fact_out = fact;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::InvertFactors(Mat A) noexcept
{
  const auto         mimpl = MatIMPLCast(A);
  const auto         mcu   = MatCUPMCast(A);
  const auto         n     = static_cast<cupmBlasInt_t>(A->cmap->n);
  const auto         lda   = static_cast<cupmBlasInt_t>(mimpl->lda);
  cupmSolverHandle_t handle;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCheck(PETSC_PKG_CUDA_VERSION_GE(10, 1, 0), PETSC_COMM_SELF, PETSC_ERR_SUP, "Upgrade to CUDA version 10.1.0 or higher");
  if (!n || !A->rmap->n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(A->factortype == MAT_FACTOR_CHOLESKY, PETSC_COMM_SELF, PETSC_ERR_LIB, "Factor type %s not implemented", MatFactorTypes[A->factortype]);
  // spd
  PetscCheck(!mcu->d_fact_ipiv, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnsytri not implemented");
  PetscCall(GetHandles_(&dctx, &handle, &stream));
  {
    const auto    da = DeviceArrayReadWrite(dctx, A);
    cupmBlasInt_t il;

    PetscCallCUPMSOLVER(cupmSolverXpotri_bufferSize(handle, CUPMBLAS_FILL_MODE_LOWER, n, da.cupmdata(), lda, &il));
    if (il > mcu->fact_lwork) {
      PetscCallCUPM(cupmFreeAsync(mcu->d_fact_work, stream));
      PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_work, il, stream));
      mcu->fact_lwork = il;
    }
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMSOLVER(cupmSolverXpotri(handle, CUPMBLAS_FILL_MODE_LOWER, n, da.cupmdata(), lda, mcu->d_fact_work, mcu->fact_lwork, mcu->d_fact_info));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(CheckCUPMSolverInfo_(mcu->d_fact_info, stream));
  // TODO (write cuda kernel)
  PetscCall(MatSeqDenseSymmetrize_Private(A, PETSC_TRUE));
  PetscCall(PetscLogGpuFlops(1.0 * n * n * n / 3.0));

  A->ops->solve          = nullptr;
  A->ops->solvetranspose = nullptr;
  A->ops->matsolve       = nullptr;
  A->factortype          = MAT_FACTOR_NONE;

  PetscCall(PetscFree(A->solvertype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::GetSubMatrix(Mat A, PetscInt rbegin, PetscInt rend, PetscInt cbegin, PetscInt cend, Mat *mat) noexcept
{
  const auto         mimpl        = MatIMPLCast(A);
  const auto         lda          = mimpl->lda;
  const auto         array_offset = [&](PetscScalar *ptr) { return ptr + rbegin + static_cast<std::size_t>(cbegin) * lda; };
  const auto         n            = rend - rbegin;
  const auto         m            = cend - cbegin;
  auto              &cmat         = mimpl->cmat;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  mimpl->matinuse = cbegin + 1;
  PetscCall(GetHandles_(&dctx));
  PetscCall(HostToDevice_(A, dctx));
  if (cmat && ((m != cmat->cmap->N) || (n != cmat->rmap->N))) PetscCall(MatDestroy(&cmat));
  {
    const auto device_array = array_offset(MatCUPMCast(A)->d_v);

    if (cmat) {
      PetscCall(PlaceArray(cmat, device_array));
    } else {
      if (PetscDefined(USE_DEBUG)) {
        MPI_Comm    comm;
        PetscMPIInt size;

        PetscCall(PetscObjectGetComm(PetscObjectCast(A), &comm));
        PetscCallMPI(MPI_Comm_size(comm, &size));
        PetscCheck(size == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Sequential MatDense has a parallel submatrix on a communicator (of size %d) larger than itself", size);
      }
      PetscCall(MatCreateSeqDenseCUPM<T>(PETSC_COMM_SELF, n, m, device_array, &cmat, dctx));
    }
  }
  PetscCall(MatDenseSetLDA(cmat, lda));
  // place CPU array if present but do not copy any data
  if (const auto host_array = mimpl->v) {
    cmat->offloadmask = PETSC_OFFLOAD_GPU;
    PetscCall(MatDensePlaceArray(cmat, array_offset(host_array)));
  }
  cmat->offloadmask = A->offloadmask;
  *mat              = cmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::RestoreSubMatrix(Mat A, Mat *m) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto cmat  = mimpl->cmat;
  const auto reset = static_cast<bool>(mimpl->v);
  bool       copy, was_offload_host;

  PetscFunctionBegin;
  PetscCheck(mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetSubMatrix() first");
  PetscCheck(cmat, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column matrix");
  PetscCheck(*m == cmat, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not the matrix obtained from MatDenseGetSubMatrix()");
  mimpl->matinuse = 0;
  // calls to ResetArray may change it, so save it here
  was_offload_host = cmat->offloadmask == PETSC_OFFLOAD_CPU;
  if (was_offload_host && !reset) {
    copy = true;
    PetscCall(MatSeqDenseSetPreallocation(A, nullptr));
  } else {
    copy = false;
  }
  PetscCall(ResetArray(cmat));
  if (reset) PetscCall(MatDenseResetArray(cmat));
  if (copy) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(DeviceToHost_(A, dctx));
  } else {
    A->offloadmask = was_offload_host ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  cmat->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  *m                = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSeqDense_CUPM<T>::SetLDA(Mat A, PetscInt lda) noexcept
{
  const auto mimpl     = MatIMPLCast(A);
  const auto mcu       = MatCUPMCast(A);
  const auto m         = A->rmap->n;
  const auto n         = A->cmap->n;
  const auto have_data = (m > 0 && n > 0) ? mcu->d_v != nullptr : false;

  PetscFunctionBegin;
  PetscCheck(mcu->user_alloc || !have_data || mimpl->lda == lda, PETSC_COMM_SELF, PETSC_ERR_ORDER, "LDA cannot be changed after allocation of internal storage");
  PetscCheck(lda >= m, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "LDA %" PetscInt_FMT " must be at least matrix dimension %" PetscInt_FMT, lda, m);
  mimpl->lda = lda;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace mat

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCMATSEQDENSECUPM_HPP
