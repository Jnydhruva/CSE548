#include <petscmanagedtype.hpp>

#include <iostream>

using namespace Petsc;

void old() {
#if 0
  PetscScalar norm[2];

  ManagedScalar scal(dctx, wrap_host_memory(norm), 2);
  scal.set_name("scal");

  PetscScalar   dot[2] = {2, 2};
  ManagedScalar scal2(dctx, wrap_host_memory(dot), 2);
  scal2.set_name("scal2");

  auto ret = scal[0];

  std::cout << "expr begin\n";
  ManagedScalar sum = (scal2.with(dctx2) + scal2.with(dctx3) + scal).with(dctx) * scal2;
  std::cout << "expr end\n";
  sum.set_name("sum");
  std::cout << "sum[0] " << sum[0] << std::endl;
  std::cout << "scal[0] " << scal[0] << std::endl;
  scal[0] = 2;
  std::cout << "scal[0] " << scal[0] << std::endl;
  scal[0] += norm[0];
  std::cout << "scal[0] " << scal[0] << std::endl;
  //PetscScalar x = scal[{1,dctx}];
#endif
}

template <typename T>
PetscErrorCode view(PetscDeviceContext dctx, ManagedType<T> &scal, const std::string &note) {
  T *array;

  PetscFunctionBegin;
  std::cout << "<========== " << note << " ==========>\n";
  PetscCall(scal.get_array(dctx, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &array));
  for (std::size_t i = 0; i < scal.size(); ++i) std::cout << array[i] << std::endl;
  std::cout << "<========== done ==========>\n";
  PetscFunctionReturn(0);
}

PetscErrorCode foo(PetscDeviceContext dctx) {
  ManagedReal x(dctx, 2);
  ManagedReal y, z;

  PetscFunctionBegin;
  x.at(dctx, 0) = 2;
  auto xit      = x.begin();
  auto xitp1    = xit + 1;
  (void)xitp1;
  std::cout << "loop begin ============================\n";
  for (auto v : x) { std::cout << v << std::endl; }
  std::cout << "loop end ============================\n";
  std::cout << "front\n";
  x.front(dctx) = 2;
  x.back(dctx)  = 4;
  auto w        = x.front(dctx) * 8;
  std::cout << w << std::endl;
  std::cout << "============================\n";
  x.front(dctx) = 5;
  PetscFunctionReturn(0);
}

template <typename T>
struct PetscManagedType {
  T                *host;
  T                *device;
  PetscDeviceType   dtype;
  PetscOffloadMask  mask;
  PetscCopyMode     d_cmode;
  PetscCopyMode     h_cmode;
  PetscInt          n;
  PetscObjectId     id;
  PetscManagedType *parent;
  PetscInt          lock; // >1  = locked, 0 = unlocked
  PetscBool         pure; // is offload to be believed
};

PetscErrorCode bar() {
  PetscDeviceContext dctx, dctx2, dctx3, dctx4;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextDuplicate(dctx, &dctx2));
  PetscCall(PetscObjectSetName((PetscObject)dctx2, "dctx2"));
  PetscCall(PetscDeviceContextDuplicate(dctx, &dctx3));
  PetscCall(PetscObjectSetName((PetscObject)dctx3, "dctx3"));
  PetscCall(PetscDeviceContextDuplicate(dctx, &dctx4));
  PetscCall(PetscObjectSetName((PetscObject)dctx4, "dctx4"));

  ManagedType<PetscScalar> scal(dctx, nullptr, nullptr, 5, PETSC_OWN_POINTER, PETSC_OWN_POINTER, PETSC_OFFLOAD_UNALLOCATED);
  PetscCheck(scal.size() == 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "size %zu != 5", scal.size());

  PetscScalar *array;

  PetscCheck(scal.offload_mask() == PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offloadmask %s is not PETSC_OFFLOAD_UNALLOCATED", PetscOffloadMaskToString(scal.offload_mask()));
  PetscCall(scal.get_array(dctx, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE, PETSC_TRUE, &array));
  PetscCheck(scal.size() == 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "size %zu != 5", scal.size());
  PetscCheck(scal.offload_mask() == PETSC_OFFLOAD_CPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offloadmask %s is not PETSC_OFFLOAD_CPU", PetscOffloadMaskToString(scal.offload_mask()));
  for (std::size_t i = 0; i < scal.size(); ++i) array[i] = i;
  PetscCall(view(dctx, scal, "scal"));

  std::cout << "save expr\n";
  auto expr = scal * scal;
  std::cout << expr.size() << std::endl;

  std::cout << "eval expr\n";
  auto evaluated = eval(dctx, expr);
  std::cout << expr.size() << std::endl;
  std::cout << evaluated.size() << std::endl;

  std::cout << "<--- build result --->\n";
  ManagedType<PetscScalar> result = eval(dctx, expr);
  std::cout << "<--- finished result --->\n";
  PetscCall(result.get_array(dctx, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE, PETSC_TRUE, &array));
  PetscCheck(result.size() == 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "size %zu != 5", result.size());
  PetscCheck(result.offload_mask() == PETSC_OFFLOAD_CPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offloadmask %s is not PETSC_OFFLOAD_CPU", PetscOffloadMaskToString(result.offload_mask()));
  for (std::size_t i = 0; i < scal.size(); ++i) PetscCheck(array[i] == i * i, PETSC_COMM_SELF, PETSC_ERR_PLIB, "array[%zu] %g != %g", i, array[i], (double)(i * i));
  PetscCall(view(dctx, result, "result"));

  std::for_each(result.begin(), result.end(), [](auto &v) { v += 2; });
  PetscCall(result.get_array(dctx, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE, PETSC_TRUE, &array));
  PetscCheck(result.size() == 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "size %zu != 5", result.size());
  PetscCheck(result.offload_mask() == PETSC_OFFLOAD_CPU, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offloadmask %s is not PETSC_OFFLOAD_CPU", PetscOffloadMaskToString(result.offload_mask()));
  for (std::size_t i = 0; i < scal.size(); ++i) PetscCheck(array[i] == i * i + 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "array[%zu] %g != %g", i, array[i], (double)(i * i + 2));

  ManagedType<PetscScalar> scal2(std::move(scal));
  PetscCall(view(dctx, scal2, "scal2"));

  std::cout << "=== assign\n";
  scal2.at(dctx, 0) = 2;
  PetscCall(view(dctx, scal2, "scal2"));
  PetscCall(scal.clear());

  PetscCall(foo(dctx));

  for (auto ctx : {dctx2, dctx3, dctx4}) PetscCall(PetscDeviceContextDestroy(&ctx));
  std::cout << "sizeof ManagedScalar: " << sizeof(ManagedType<PetscScalar>) << std::endl;
  std::cout << "sizeof Old PetscManagedType: " << sizeof(PetscManagedType<double>) << std::endl;
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  PetscCall(bar());
  static_assert(sizeof(ManagedScalar) <= 64, "");
  std::cout << "sizeof ManagedScalar: " << sizeof(ManagedScalar) << std::endl;
  std::cout << "sizeof Old PetscManagedType: " << sizeof(PetscManagedType<PetscScalar>) << std::endl;
  std::cout << "alignof ManagedScalar: " << alignof(ManagedScalar) << std::endl;
  std::cout << "alignof Old petscmanagedtype: " << alignof(PetscManagedType<PetscScalar>) << std::endl;
  PetscCall(PetscFinalize());
  return 0;
}
