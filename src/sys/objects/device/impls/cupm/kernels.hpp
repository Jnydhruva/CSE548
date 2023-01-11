#ifndef PETSC_DEVICE_CUPM_KERNELS_HPP
#define PETSC_DEVICE_CUPM_KERNELS_HPP

#include <petscdevice_cupm.h>

#if defined(__cplusplus)

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace kernels
{

namespace util
{

template <typename SizeType, typename T>
PETSC_DEVICE_INLINE_DECL static void grid_stride_1D(const SizeType size, T &&func) noexcept
{
  for (SizeType i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) func(i);
  return;
}

} // namespace util

} // namespace kernels

namespace functors
{

template <typename T>
class plus_equals {
public:
  using value_type = T;

  PETSC_HOSTDEVICE_DECL constexpr explicit plus_equals(value_type v = value_type{}) noexcept : v_(std::move(v)) { }

  PETSC_NODISCARD PETSC_HOSTDEVICE_INLINE_DECL constexpr T operator()(const T &val) const noexcept { return val + v_; }

private:
  value_type v_;
};

} // namespace functors

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_DEVICE_CUPM_KERNELS_HPP
