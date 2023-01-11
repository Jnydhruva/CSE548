#ifndef PETSCCUPMSOLVERINTERFACE_HPP
#define PETSCCUPMSOLVERINTERFACE_HPP

#if defined(__cplusplus)
  #include <petsc/private/cupmblasinterface.hpp>
  #include <petsc/private/petscadvancedmacros.h>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

  #define PetscCallCUPMSOLVER(...) \
    do { \
      const cupmSolverError_t cupmsolver_stat_p_ = __VA_ARGS__; \
      if (PetscUnlikely(cupmsolver_stat_p_ != CUPMSOLVER_STATUS_SUCCESS)) { \
        if (((cupmsolver_stat_p_ == CUPMSOLVER_STATUS_NOT_INITIALIZED) || (cupmsolver_stat_p_ == CUPMSOLVER_STATUS_ALLOC_FAILED) || (cupmsolver_stat_p_ == CUPMSOLVER_STATUS_INTERNAL_ERROR)) && PetscDeviceInitialized(PETSC_DEVICE_CUPM())) { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                  "%s error %d (%s). " \
                  "This indicates the GPU may have run out resources", \
                  cupmSolverName(), static_cast<PetscErrorCode>(cupmsolver_stat_p_), cupmSolverGetErrorName(cupmsolver_stat_p_)); \
        } \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "%s error %d (%s)", cupmSolverName(), static_cast<PetscErrorCode>(cupmsolver_stat_p_), cupmSolverGetErrorName(cupmsolver_stat_p_)); \
      } \
    } while (0)

  #ifndef PetscConcat3
    #define PetscConcat3(a, b, c) PetscConcat(PetscConcat(a, b), c)
  #endif

  #if PetscDefined(USE_COMPLEX)
    #define PETSC_CUPMSOLVER_FP_TYPE_SPECIAL un
  #else
    #define PETSC_CUPMSOLVER_FP_TYPE_SPECIAL or
  #endif // USE_COMPLEX

  #define PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupm_name, their_prefix, fp_type, suffix) PETSC_CUPM_ALIAS_FUNCTION(cupm_name, PetscConcat3(their_prefix, fp_type, suffix))

template <DeviceType>
struct SolverInterfaceImpl;

  #if PetscDefined(HAVE_CUDA)
template <>
struct SolverInterfaceImpl<DeviceType::CUDA> : BlasInterface<DeviceType::CUDA> {
  // typedefs
  using cupmSolverHandle_t = cusolverDnHandle_t;
  using cupmSolverError_t  = cusolverStatus_t;

  // values
  static const auto CUPMSOLVER_STATUS_SUCCESS         = CUSOLVER_STATUS_SUCCESS;
  static const auto CUPMSOLVER_STATUS_NOT_INITIALIZED = CUSOLVER_STATUS_NOT_INITIALIZED;
  static const auto CUPMSOLVER_STATUS_ALLOC_FAILED    = CUSOLVER_STATUS_ALLOC_FAILED;
  static const auto CUPMSOLVER_STATUS_INTERNAL_ERROR  = CUSOLVER_STATUS_INTERNAL_ERROR;

  // utility functions
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverCreate, cusolverDnCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverDestroy, cusolverDnDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverSetStream, cusolverDnSetStream)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverGetStream, cusolverDnGetStream)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrf, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potrf)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrf_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrs, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potrs)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotri, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potri)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotri_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potri_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXsytrf, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, sytrf)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXsytrf_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, sytrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrf, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, getrf)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrf_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, getrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrs, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, getrs)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgeqrf, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, geqrf)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgeqrf_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, geqrf_bufferSize)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXormqr, cusolverDn, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_U, PETSC_CUPMSOLVER_FP_TYPE_SPECIAL), mqr)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXormqr_bufferSize, cusolverDn, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_U, PETSC_CUPMSOLVER_FP_TYPE_SPECIAL), mqr_bufferSize)

  static const char *cupmSolverGetErrorName(cupmSolverError_t status) noexcept { return PetscCUSolverGetErrorName(status); }
};
  #endif

  #if PetscDefined(HAVE_HIP)
template <>
struct SolverInterfaceImpl<DeviceType::HIP> : BlasInterface<DeviceType::HIP> {
  // typedefs
  using cupmSolverHandle_t = hipsolverHandle_t;
  using cupmSolverError_t  = hipsolverStatus_t;

  static const auto CUPMSOLVER_STATUS_SUCCESS         = HIPSOLVER_STATUS_SUCCESS;
  static const auto CUPMSOLVER_STATUS_NOT_INITIALIZED = HIPSOLVER_STATUS_NOT_INITIALIZED;
  static const auto CUPMSOLVER_STATUS_ALLOC_FAILED    = HIPSOLVER_STATUS_ALLOC_FAILED;
  static const auto CUPMSOLVER_STATUS_INTERNAL_ERROR  = HIPSOLVER_STATUS_INTERNAL_ERROR;

  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverCreate, hipsolverCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverDestroy, hipsolverDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverSetStream, hipsolverSetStream)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverGetStream, hipsolverGetStream)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrf, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potrf)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrf_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrs, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potrs)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotri, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potri)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotri_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potri_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXsytrf, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, sytrf)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXsytrf_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, sytrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrf, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, getrf)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrf_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, getrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrs, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, getrs)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgeqrf, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, geqrf)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgeqrf_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, geqrf_bufferSize)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXormqr, hipsolver, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_U, PETSC_CUPMSOLVER_FP_TYPE_SPECIAL), mqr)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXormqr_bufferSize, hipsolver, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_U, PETSC_CUPMSOLVER_FP_TYPE_SPECIAL), mqr_bufferSize)

  static const char *cupmSolverGetErrorName(cupmSolverError_t status) noexcept { return PetscHIPSolverGetErrorName(status); }
};
  #endif

  #define PETSC_CUPMSOLVER_IMPL_CLASS_HEADER(T) \
    PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(T); \
    /* introspection */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverGetErrorName; \
    /* types */ \
    using cupmSolverHandle_t = typename ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverHandle_t; \
    using cupmSolverError_t  = typename ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverError_t; \
    /* values */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_STATUS_SUCCESS; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_STATUS_NOT_INITIALIZED; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_STATUS_ALLOC_FAILED; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_STATUS_INTERNAL_ERROR; \
    /* utility functions */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverCreate; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverDestroy; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverGetStream; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverSetStream; \
    /* blas functions */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotrf; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotrf_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotrs; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotri; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotri_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXsytrf; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXsytrf_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgetrf; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgetrf_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgetrs; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgeqrf; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgeqrf_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXormqr; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXormqr_bufferSize

template <DeviceType T>
struct SolverInterface : SolverInterfaceImpl<T> {
  PETSC_NODISCARD static constexpr const char *cupmSolverName() noexcept { return T == DeviceType::CUDA ? "cuSOLVER" : "hipSOLVER"; }
};

  #define PETSC_CUPMSOLVER_INHERIT_INTERFACE_TYPEDEFS_USING(T) \
    PETSC_CUPMSOLVER_IMPL_CLASS_HEADER(T); \
    using ::Petsc::device::cupm::impl::SolverInterface<T>::cupmSolverName

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCCUPMSOLVERINTERFACE_HPP
