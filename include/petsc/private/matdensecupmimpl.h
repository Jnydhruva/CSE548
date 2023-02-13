#ifndef PETSCMATDENSECUPMIMPL_H
#define PETSCMATDENSECUPMIMPL_H

#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#ifdef __cplusplus
  #include <petsc/private/matimpl.h>
  #include <petsc/private/deviceimpl.h>
  #include <petsc/private/cupmsolverinterface.hpp>
  #include <petsc/private/cupmobject.hpp>

  #include "../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp"
  #include "../src/sys/objects/device/impls/cupm/kernels.hpp"

  #include <thrust/device_vector.h>
  #include <thrust/device_ptr.h>
  #include <thrust/iterator/counting_iterator.h>
  #include <thrust/iterator/transform_iterator.h>
  #include <thrust/iterator/permutation_iterator.h>
  #include <thrust/transform.h>

namespace Petsc
{

namespace vec
{

namespace cupm
{

namespace impl
{

template <device::cupm::DeviceType>
class VecSeq_CUPM;
template <device::cupm::DeviceType>
class VecMPI_CUPM;

} // namespace impl

} // namespace cupm

} // namespace vec

namespace mat
{

namespace cupm
{

namespace impl
{

template <device::cupm::DeviceType T>
class MatDense_CUPM_Base : protected device::cupm::impl::CUPMObject<T> {
public:
  PETSC_CUPMOBJECT_HEADER(T);

  #define MatDenseCUPMComposedOpDecl(OP_NAME) \
    PETSC_NODISCARD static constexpr const char *PetscConcat(MatDenseCUPM, OP_NAME)() noexcept \
    { \
      return T == device::cupm::DeviceType::CUDA ? PetscStringize(PetscConcat(MatDenseCUDA, OP_NAME)) : PetscStringize(PetscConcat(MatDenseHIP, OP_NAME)); \
    }

  // clang-format off
  MatDenseCUPMComposedOpDecl(GetArray_C)
  MatDenseCUPMComposedOpDecl(GetArrayRead_C)
  MatDenseCUPMComposedOpDecl(GetArrayWrite_C)
  MatDenseCUPMComposedOpDecl(RestoreArray_C)
  MatDenseCUPMComposedOpDecl(RestoreArrayRead_C)
  MatDenseCUPMComposedOpDecl(RestoreArrayWrite_C)
  MatDenseCUPMComposedOpDecl(PlaceArray_C)
  MatDenseCUPMComposedOpDecl(ReplaceArray_C)
  MatDenseCUPMComposedOpDecl(ResetArray_C)
    // clang-format on

  #undef MatDenseCUPMComposedOpDecl

    PETSC_NODISCARD static constexpr MatType MATSEQDENSECUPM() noexcept;
  PETSC_NODISCARD static constexpr MatType       MATMPIDENSECUPM() noexcept;
  PETSC_NODISCARD static constexpr MatType       MATDENSECUPM() noexcept;
  PETSC_NODISCARD static constexpr MatSolverType MATSOLVERCUPM() noexcept;
};

template <device::cupm::DeviceType T>
inline constexpr MatType MatDense_CUPM_Base<T>::MATSEQDENSECUPM() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? MATSEQDENSECUDA : MATSEQDENSEHIP;
}

template <device::cupm::DeviceType T>
inline constexpr MatType MatDense_CUPM_Base<T>::MATMPIDENSECUPM() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? MATMPIDENSECUDA : MATMPIDENSEHIP;
}

template <device::cupm::DeviceType T>
inline constexpr MatType MatDense_CUPM_Base<T>::MATDENSECUPM() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? MATDENSECUDA : MATDENSEHIP;
}

template <device::cupm::DeviceType T>
inline constexpr MatSolverType MatDense_CUPM_Base<T>::MATSOLVERCUPM() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? MATSOLVERCUDA : MATSOLVERHIP;
}

  #define MATDENSECUPM_BASE_HEADER(T) \
    PETSC_CUPMOBJECT_HEADER(T); \
    using VecSeq_CUPM = ::Petsc::vec::cupm::impl::VecSeq_CUPM<T>; \
    using VecMPI_CUPM = ::Petsc::vec::cupm::impl::VecMPI_CUPM<T>; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MATSEQDENSECUPM; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MATMPIDENSECUPM; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MATDENSECUPM; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MATSOLVERCUPM; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMGetArray_C; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMGetArrayRead_C; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMGetArrayWrite_C; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMRestoreArray_C; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMRestoreArrayRead_C; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMRestoreArrayWrite_C; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMPlaceArray_C; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMReplaceArray_C; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM_Base<T>::MatDenseCUPMResetArray_C

// forward declare
template <device::cupm::DeviceType>
class MatSeqDense_CUPM;
template <device::cupm::DeviceType>
class MatMPIDense_CUPM;

// The reason MatDense_CUPM and MatDense_CUPM_Base exist is to separate out the CRTP code from
// the non-crtp code so that the generic functions can be called via templates below
template <device::cupm::DeviceType T, typename Derived>
class MatDense_CUPM : protected MatDense_CUPM_Base<T> {
protected:
  MATDENSECUPM_BASE_HEADER(T);

  static PetscErrorCode PetscStrFreeAllocpy(const char target[], char **dest) noexcept
  {
    PetscFunctionBegin;
    PetscValidPointer(dest, 2);
    PetscValidCharPointer(*dest, 2);
    PetscCall(PetscFree(*dest));
    PetscCall(PetscStrallocpy(target, dest));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <PetscMemType, PetscMemoryAccessMode>
  class MatrixArray;

  // Cast the Mat to its host struct, i.e. return the result of (Mat_SeqDense *)m->data
  template <typename U = Derived>
  PETSC_NODISCARD static constexpr auto    MatIMPLCast(Mat m) noexcept PETSC_DECLTYPE_AUTO_RETURNS(U::MatIMPLCast_(m))
  PETSC_NODISCARD static constexpr MatType MATIMPLCUPM() noexcept;

  static PetscErrorCode SetPreallocation(Mat, PetscDeviceContext, PetscScalar * = nullptr) noexcept;
  static PetscErrorCode Shift_Base(PetscDeviceContext, PetscScalar *, PetscScalar, PetscInt, PetscInt, PetscInt, PetscInt) noexcept;

  PETSC_NODISCARD static auto DeviceArrayRead(PetscDeviceContext dctx, Mat m) noexcept PETSC_DECLTYPE_AUTO_RETURNS(MatrixArray<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>{dctx, m})
  PETSC_NODISCARD static auto DeviceArrayWrite(PetscDeviceContext dctx, Mat m) noexcept PETSC_DECLTYPE_AUTO_RETURNS(MatrixArray<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>{dctx, m})
  PETSC_NODISCARD static auto DeviceArrayReadWrite(PetscDeviceContext dctx, Mat m) noexcept PETSC_DECLTYPE_AUTO_RETURNS(MatrixArray<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>{dctx, m})
  PETSC_NODISCARD static auto HostArrayRead(PetscDeviceContext dctx, Mat m) noexcept PETSC_DECLTYPE_AUTO_RETURNS(MatrixArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>{dctx, m})
  PETSC_NODISCARD static auto HostArrayWrite(PetscDeviceContext dctx, Mat m) noexcept PETSC_DECLTYPE_AUTO_RETURNS(MatrixArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>{dctx, m})
  PETSC_NODISCARD static auto HostArrayReadWrite(PetscDeviceContext dctx, Mat m) noexcept PETSC_DECLTYPE_AUTO_RETURNS(MatrixArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>{dctx, m})
};

// ==========================================================================================
// MatDense_CUPM::MatrixArray
// ==========================================================================================

template <device::cupm::DeviceType T, typename D>
template <PetscMemType MT, PetscMemoryAccessMode MA>
class MatDense_CUPM<T, D>::MatrixArray : public device::cupm::impl::RestoreableArray<T, MT, MA> {
  using base_type = device::cupm::impl::RestoreableArray<T, MT, MA>;

public:
  MatrixArray(PetscDeviceContext, Mat) noexcept;
  ~MatrixArray() noexcept;

private:
  Mat m_ = nullptr;
};

// ==========================================================================================
// MatDense_CUPM::MatrixArray - Public API
// ==========================================================================================

template <device::cupm::DeviceType T, typename D>
template <PetscMemType MT, PetscMemoryAccessMode MA>
inline MatDense_CUPM<T, D>::MatrixArray<MT, MA>::MatrixArray(PetscDeviceContext dctx, Mat m) noexcept : base_type{dctx}, m_{m}
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, D::template GetArray<MT, MA>(m, &this->ptr_, dctx));
  PetscFunctionReturnVoid();
}

template <device::cupm::DeviceType T, typename D>
template <PetscMemType MT, PetscMemoryAccessMode MA>
inline MatDense_CUPM<T, D>::MatrixArray<MT, MA>::~MatrixArray() noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, D::template RestoreArray<MT, MA>(m_, &this->ptr_, this->dctx_));
  PetscFunctionReturnVoid();
}

// ==========================================================================================
// MatDense_CUPM -- Protected API
// ==========================================================================================

template <device::cupm::DeviceType T, typename D>
inline constexpr MatType MatDense_CUPM<T, D>::MATIMPLCUPM() noexcept
{
  return D::MATIMPLCUPM_();
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode MatDense_CUPM<T, D>::SetPreallocation(Mat A, PetscDeviceContext dctx, PetscScalar *device_array) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(A, MAT_CLASSID, 1, MATIMPLCUPM());
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  PetscCall(D::SetPreallocation_(A, dctx, device_array));
  A->preallocated = PETSC_TRUE;
  A->assembled    = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace detail
{

template <typename Iterator>
class strided_range {
public:
  using difference_type = typename thrust::iterator_difference<Iterator>::type;

  struct stride_functor {
    PETSC_NODISCARD PETSC_HOSTDEVICE_INLINE_DECL difference_type operator()(const difference_type &i) const noexcept { return stride * i; }

    difference_type stride;
  };

  using CountingIterator    = thrust::counting_iterator<difference_type>;
  using TransformIterator   = thrust::transform_iterator<stride_functor, CountingIterator>;
  using PermutationIterator = thrust::permutation_iterator<Iterator, TransformIterator>;
  using iterator            = PermutationIterator; // type of the strided_range iterator

  constexpr strided_range(Iterator first, Iterator last, difference_type stride) noexcept : first(first), last(last), stride(stride) { }

  PETSC_NODISCARD iterator begin() const noexcept
  {
    return PermutationIterator{
      first, TransformIterator{CountingIterator{0}, stride_functor{stride}}
    };
  }

  PETSC_NODISCARD iterator end() const noexcept { return begin() + ((last - first) + (stride - 1)) / stride; }

protected:
  Iterator        first;
  Iterator        last;
  difference_type stride;
};

} // namespace detail

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode MatDense_CUPM<T, D>::Shift_Base(PetscDeviceContext dctx, PetscScalar *da, PetscScalar alpha, PetscInt lda, PetscInt rstart, PetscInt rend, PetscInt cols) noexcept
{
  const auto rend2 = std::min(rend, cols);

  PetscFunctionBegin;
  if (rend2 > rstart) {
    using strided_range_type = detail::strided_range<thrust::device_vector<PetscScalar>::iterator>;
    const auto         dptr  = thrust::device_pointer_cast(da);
    const std::size_t  begin = rstart * lda;
    const std::size_t  end   = rend2 - rstart + rend2 * lda;
    strided_range_type diagonal{dptr + begin, dptr + end, lda + 1};
    cupmStream_t       stream;

    PetscCall(GetHandlesFrom_(dctx, &stream));
    // clang-format off
    PetscCallThrust(
      THRUST_CALL(
        thrust::transform,
        stream,
        diagonal.begin(), diagonal.end(), diagonal.begin(),
        device::cupm::functors::plus_equals<PetscScalar>{alpha}
      )
    );
    // clang-format on
    PetscCall(PetscLogGpuFlops(rend2 - rstart));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #define MatComposeOp_CUPM(use_host, pobj, op_str, op_host, ...) \
    do { \
      if (use_host) { \
        PetscCall(PetscObjectComposeFunction(pobj, op_str, op_host)); \
      } else { \
        PetscCall(PetscObjectComposeFunction(pobj, op_str, __VA_ARGS__)); \
      } \
    } while (0)

  #define MatSetOp_CUPM(use_host, mat, op_name, op_host, ...) \
    do { \
      if (use_host) { \
        (mat)->ops->op_name = op_host; \
      } else { \
        (mat)->ops->op_name = __VA_ARGS__; \
      } \
    } while (0)

  #define MATDENSECUPM_HEADER(T, ...) \
    MATDENSECUPM_BASE_HEADER(T); \
    friend class ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::PetscStrFreeAllocpy; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::MatIMPLCast; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::MATIMPLCUPM; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::SetPreallocation; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::DeviceArrayRead; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::DeviceArrayWrite; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::DeviceArrayReadWrite; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::HostArrayRead; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::HostArrayWrite; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::HostArrayReadWrite; \
    using ::Petsc::mat::cupm::impl::MatDense_CUPM<T, __VA_ARGS__>::Shift_Base

} // namespace impl

namespace
{

template <device::cupm::DeviceType T, PetscMemoryAccessMode access>
inline PetscErrorCode MatDenseCUPMGetArray_Private(Mat A, PetscScalar **array) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(array, 2);
  switch (access) {
  case PETSC_MEMORY_ACCESS_READ:
    PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMGetArrayRead_C(), (Mat, PetscScalar **), (A, array));
    break;
  case PETSC_MEMORY_ACCESS_WRITE:
    PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMGetArrayWrite_C(), (Mat, PetscScalar **), (A, array));
    break;
  case PETSC_MEMORY_ACCESS_READ_WRITE:
    PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMGetArray_C(), (Mat, PetscScalar **), (A, array));
    break;
  }
  if (PetscMemoryAccessWrite(access)) PetscCall(PetscObjectStateIncrease(PetscObjectCast(A)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, PetscMemoryAccessMode access>
inline PetscErrorCode MatDenseCUPMRestoreArray_Private(Mat A, PetscScalar **array) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(array, 2);
  switch (access) {
  case PETSC_MEMORY_ACCESS_READ:
    PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMRestoreArrayRead_C(), (Mat, PetscScalar **), (A, array));
    break;
  case PETSC_MEMORY_ACCESS_WRITE:
    PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMRestoreArrayWrite_C(), (Mat, PetscScalar **), (A, array));
    break;
  case PETSC_MEMORY_ACCESS_READ_WRITE:
    PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMRestoreArray_C(), (Mat, PetscScalar **), (A, array));
    break;
  }
  if (PetscMemoryAccessWrite(access)) {
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(A)));
    A->offloadmask = PETSC_OFFLOAD_GPU;
  }
  *array = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMGetArray(Mat A, PetscScalar **array) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArray_Private<T, PETSC_MEMORY_ACCESS_READ_WRITE>(A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMGetArrayRead(Mat A, const PetscScalar **array) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArray_Private<T, PETSC_MEMORY_ACCESS_READ>(A, const_cast<PetscScalar **>(array)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMGetArrayWrite(Mat A, PetscScalar **array) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArray_Private<T, PETSC_MEMORY_ACCESS_WRITE>(A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMRestoreArray(Mat A, PetscScalar **array) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArray_Private<T, PETSC_MEMORY_ACCESS_READ_WRITE>(A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMRestoreArrayRead(Mat A, const PetscScalar **array) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArray_Private<T, PETSC_MEMORY_ACCESS_READ>(A, const_cast<PetscScalar **>(array)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMRestoreArrayWrite(Mat A, PetscScalar **array) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArray_Private<T, PETSC_MEMORY_ACCESS_WRITE>(A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMPlaceArray(Mat A, const PetscScalar *array) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMPlaceArray_C(), (Mat, const PetscScalar *), (A, array));
  PetscCall(PetscObjectStateIncrease(PetscObjectCast(A)));
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMReplaceArray(Mat A, const PetscScalar *array) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMReplaceArray_C(), (Mat, const PetscScalar *), (A, array));
  PetscCall(PetscObjectStateIncrease(PetscObjectCast(A)));
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDenseCUPMResetArray(Mat A) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscUseMethod(A, impl::MatDense_CUPM_Base<T>::MatDenseCUPMResetArray_C(), (Mat), (A));
  PetscCall(PetscObjectStateIncrease(PetscObjectCast(A)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatCreateDenseCUPM(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscScalar *data, Mat *A) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidPointer(A, 7);
  PetscCall(MatCreate(comm, A));
  PetscValidLogicalCollectiveBool(*A, !!data, 6);
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCall(MatSetType(*A, impl::MatDense_CUPM_Base<T>::MATDENSECUPM()));
  PetscCall(impl::MatSeqDense_CUPM<T>::GetHandles_(&dctx));
  PetscCall(impl::MatSeqDense_CUPM<T>::SetPreallocation(*A, dctx, data));
  PetscCall(impl::MatMPIDense_CUPM<T>::SetPreallocation(*A, dctx, data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

} // namespace cupm

} // namespace mat

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCMATDENSECUPMIMPL_H
