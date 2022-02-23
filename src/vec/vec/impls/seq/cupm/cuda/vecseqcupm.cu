#include  "../vecseqcupm.hpp" /*I <petscvec.h> I*/

namespace Petsc { namespace Vector { namespace CUPM { namespace Impl {

template struct VecSeq_CUPM<Petsc::Device::CUPM::DeviceType::CUDA>;

}}}} // namespace Petsc::Vector::CUPM::Impl

using namespace Petsc::Vector::CUPM::Impl;

static const auto VecSeq_CUDA = VecSeq_CUPM<Petsc::Device::CUPM::DeviceType::CUDA>{};

PetscErrorCode VecCreate_SeqCUDA(Vec v)
{
  PetscFunctionBegin;
  PetscCall(VecSeq_CUDA.create_async(v));
  PetscFunctionReturn(0);
}

/*@
  VecCreateSeqCUDA - Creates a standard, sequential, array-style vector.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm - the communicator, must be PETSC_COMM_SELF
- n    - the vector length

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  This function may initialize PetscDevice, which may incur a device synchronization.

  Level: intermediate

.seealso: PetscDeviceInitialize(), VecCreate(), VecCreateSeq(), VecCreateSeqCUDAWithArray(),
VecCreateMPI(), VecCreateMPICUDA(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
@*/
PetscErrorCode VecCreateSeqCUDA(MPI_Comm comm, PetscInt n, Vec *v)
{
  PetscFunctionBegin;
  PetscValidPointer(v,3);
  PetscCall(VecSeq_CUDA.createseqcupm_async(comm,0,n,v,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
  VecCreateSeqCUDAWithArrays - Creates a sequential, array-style vector using CUDA, where the
  user provides the complete array space to store the vector values.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm     - the communicator, must be PETSC_COMM_SELF
. bs       - the block size
. n        - the local vector length
. cpuarray - CPU memory where the vector elements are to be stored (or NULL)
- gpuarray - GPU memory where the vector elements are to be stored (or NULL)

  Output Parameter:
. v - the vector

  Notes:
  If the user-provided array is NULL, then VecCUDAPlaceArray() can be used at a later stage to
  SET the array for storing the vector values. Otherwise, the array must be allocated on the
  device.

  If both cpuarray and gpuarray are provided, the provided arrays must have identical
  values.

  The arrays are NOT freed when the vector is destroyed via VecDestroy(). The user must free
  them themselves, but not until the vector is destroyed.

  This function may initialize PetscDevice, which may incur a device synchronization.

  Level: intermediate

.seealso: PetscDeviceInitialize(), VecCreate(), VecCreateSeqWithArray(), VecCreateSeqCUDA(),
VecCreateSeqCUDAWithArray(), VecCreateMPICUDA(), VecCreateMPICUDAWithArray(),
VecCreateMPICUDAWithArrays(), VecCUDAPlaceArray()
C@*/
PetscErrorCode VecCreateSeqCUDAWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v)
{
  PetscFunctionBegin;
  if (n && cpuarray) PetscValidScalarPointer(cpuarray,4);
  PetscValidPointer(v,6);
  PetscCall(VecSeq_CUDA.createseqcupmwithbotharrays_async(comm,bs,n,cpuarray,gpuarray,v));
  PetscFunctionReturn(0);
}

/*@C
  VecCreateSeqCUDAWithArray - Creates a sequential, array-style vector using CUDA, where the
  user provides the device array space to store the vector values.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm     - the communicator, must be PETSC_COMM_SELF
. bs       - the block size
. n        - the vector length
- gpuarray - GPU memory where the vector elements are to be stored (or NULL)

  Output Parameter:
. v - the vector

  Notes:
  If the user-provided array is NULL, then VecCUDAPlaceArray() can be used at a later stage to
  SET the array for storing the vector values. Otherwise, the array must be allocated on the
  device.

  The array is NOT freed when the vector is destroyed via VecDestroy(). The user must free the
  array themselves, but not until the vector is destroyed.

  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  This function may initialize PetscDevice, which may incur a device synchronization.

  Level: intermediate

.seealso: PetscDeviceInitialize(), VecCreate(), VecCreateSeq(), VecCreateSeqWithArray(),
VecCreateMPIWithArray(), VecCreateSeqCUDA(), VecCreateMPICUDAWithArray(), VecCUDAPlaceArray(),
VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
@*/
PetscErrorCode VecCreateSeqCUDAWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar gpuarray[], Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecCreateSeqCUDAWithArrays(comm,bs,n,nullptr,gpuarray,v));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAGetArray - Provides access to the device buffer inside a vector

  Asynchronous

  Input Parameter:
. v - the vector

  Output Parameter:
. a - the device buffer

  Notes:
  This routine has semantics similar to VecGetArray(); the returned buffer points to a
  consistent view of the vector data. This may involve copying data from the host to the device
  if the data on the device is out of date. It is also assumed that the returned buffer is
  immediately modified, marking the host data out of date. This is similar to intent(inout) in
  fortran.

  If the user does require strong memory guarantees, they are encocuraged to use
  VecCUDAGetArrayRead() and/or VecCUDAGetArrayWrite() instead.

  The user must call VecCUDARestoreArray() when they are finished using the array.

  Fortran note:
  This function is not currently available from Fortran.

  Developer Notes:
  If the device memory hasn't been allocated previously it will be allocated as part of this
  routine.

  Level: intermediate

.seealso: VecCUDARestoreArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(),
VecGetArrayRead(), VecGetArrayWrite()
@*/
PetscErrorCode VecCUDAGetArray(Vec v, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCall(VecSeq_CUDA.getarray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ_WRITE>(v,a));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDARestoreArray - Restore a device buffer previously acquired with VecCUDAGetArray().

  Asynchronous

  Input Parameters:
+ v - the vector
- a - the device buffer

  Notes:
  The restored pointer is invalid after this function returns. This function also marks the
  host data as out of date. Subsequent access to the vector data on the host side via
  VecGetArray() will incur a (synchronous) data transfer.

  Fortran note:
  This function is not currently available from Fortran.

  Level: intermediate

.seealso: VecCUDAGetArray(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(),
VecRestoreArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDARestoreArray(Vec v, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCall(VecSeq_CUDA.restorearray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ_WRITE>(v,a));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAGetArrayRead - Provides read access to the CUDA buffer inside a vector.

  Asynchronous

  Input Parameter:
. v - the vector

  Output Parameter:
. a - the CUDA pointer.

  Notes:
  See VecCUDAGetArray() for data movement semantics of this function.

  This function assumes that the user will not modify the vector data. This is analgogous to
  intent(in) in Fortran.

  The device pointer must be restored by calling VecCUDARestoreArrayRead(). If the data on the
  host side was previously up to date it will remain so, i.e. data on both the device and the
  host is up to date. Accessing data on the host side does not incur a device to host data
  transfer.

  Fortran note:
  This function is not currently available from Fortran.

  Level: intermediate

.seealso: VecCUDARestoreArrayRead(), VecCUDAGetArray(), VecCUDAGetArrayWrite(), VecGetArray(),
VecGetArrayRead()
@*/
PetscErrorCode VecCUDAGetArrayRead(Vec v, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCall(VecSeq_CUDA.getarray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ>(v,const_cast<PetscScalar**>(a)));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDARestoreArrayRead - Restore a CUDA device pointer previously acquired with
  VecCUDAGetArrayRead().

  Input Parameters:
+ v - the vector
- a - the CUDA device pointer

  Notes:
  This routine does not modify the corresponding array on the host in any way. The pointer is
  invalid after this function returns.

  Fortran note:
  This function is not currently available from Fortran.

  Level: intermediate

.seealso: VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecCUDAGetArray(), VecGetArray(),
VecRestoreArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDARestoreArrayRead(Vec v, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCall(VecSeq_CUDA.restorearray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ>(v,const_cast<PetscScalar**>(a)));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAGetArrayWrite - Provides write access to the CUDA buffer inside a vector.

  Input Parameter:
. v - the vector

  Output Parameter:
. a - the CUDA pointer

  Notes:
  The data pointed to by the device pointer is uninitialized. The user may not read from this
  data. Furthermore, the entire array needs to be filled by the user to obtain well-defined
  behaviour. The device memory will be allocated by this function if it hasn't been allocated
  previously. This is analogous to intent(out) in Fortran.

  The device pointer needs to be released with VecCUDARestoreArrayWrite(). When the pointer is
  released the host data of the vector is marked as out of data. Subsequent access of the host
  data with e.g. VecGetArray() incurs a device to host data transfer.

  Fortran Note:
  This function is not currently available from Fortran.

  Level: advanced

.seealso: VecCUDARestoreArrayWrite(), VecCUDAGetArray(), VecCUDAGetArrayRead(),
VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDAGetArrayWrite(Vec v, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCall(VecSeq_CUDA.getarray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::WRITE>(v,a));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDARestoreArrayWrite - Restore a CUDA device pointer previously acquired with
  VecCUDAGetArrayWrite().

  Input Parameters:
+ v - the vector
- a - the CUDA device pointer.  This pointer is invalid after VecCUDARestoreArrayWrite() returns.

  Notes:
  Data on the host will be marked as out of date. Subsequent access of the data on the host
  side e.g. with VecGetArray() will incur a device to host data transfer.

  Fortran Note:
  This function is not currently available from Fortran.

  Level: intermediate

.seealso: VecCUDAGetArrayWrite(), VecCUDAGetArray(), VecCUDAGetArrayRead(),
VecCUDAGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PetscErrorCode VecCUDARestoreArrayWrite(Vec v, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCall(VecSeq_CUDA.restorearray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::WRITE>(v,a));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAPlaceArray - Allows one to replace the GPU array in a vector with a GPU array provided
  by the user.

  Not Collective

  Input Parameters:
+ vec - the vector
- array - the GPU array

  Notes:
  This routine is useful to avoid copying an array into a vector, though you can return to the
  original GPU array with a call to VecCUDAResetArray().

  It is not possible to use VecCUDAPlaceArray() and VecPlaceArray() at the same time on the
  same vector.

  `vec` does not take ownership of `array` in any way. The user must free `array` themselves
  but be careful not to do so before the vector has either been destroyed, had its original
  array restored with `VecCUDAResetArray()` or permanently replaced with
  `VecCUDAReplaceArray()`.

  Level: advanced

.seealso: VecPlaceArray(), VecGetArray(), VecRestoreArray(), VecReplaceArray(),
VecResetArray(), VecCUDAResetArray(), VecCUDAReplaceArray()
@*/
PetscErrorCode VecCUDAPlaceArray(Vec vin, const PetscScalar a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  PetscCall(VecSeq_CUDA.placearray_async<PETSC_MEMTYPE_DEVICE>(vin,a));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAReplaceArray - Permanently replace the GPU array in a vector with a GPU array provided
  by the user.

  Not Collective

  Input Parameters:
+ vec   - the vector
- array - the GPU array

  Notes:
  This is useful to avoid copying a GPU array into a vector.

  This frees the memory associated with the old GPU array. The vector takes ownership of the
  passed array so it CANNOT be freed by the user. It will be freed when the vector is
  destroyed.

  Fortran Notes:
  Not supported from Fortran

  Level: advanced

.seealso: VecGetArray(), VecRestoreArray(), VecPlaceArray(), VecResetArray(),
VecCUDAResetArray(), VecCUDAPlaceArray(), VecReplaceArray()
@*/
PetscErrorCode VecCUDAReplaceArray(Vec vin, const PetscScalar a[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  PetscCall(VecSeq_CUDA.replacearray_async<PETSC_MEMTYPE_DEVICE>(vin,a));
  PetscFunctionReturn(0);
}

/*@C
  VecCUDAResetArray - Resets a vector to use its default memory.

  Not Collective

  Input Parameters:
. vec - the vector

  Notes:
  Call this after the use of VecCUDAPlaceArray().

  Level: advanced

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray(),
VecResetArray(), VecCUDAPlaceArray(), VecCUDAReplaceArray()
@*/
PetscErrorCode VecCUDAResetArray(Vec vin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  PetscCall(VecSeq_CUDA.resetarray_async<PETSC_MEMTYPE_DEVICE>(vin));
  PetscFunctionReturn(0);
}
