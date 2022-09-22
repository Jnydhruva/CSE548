#ifndef PETSC_HASHMAPIJ_H
#define PETSC_HASHMAPIJ_H

#include <petsc/private/hashmap.h>

#if !defined(PETSC_HASHIJKEY)
  #define PETSC_HASHIJKEY
typedef struct _PetscHashIJKey {
  PetscInt i, j;
} PetscHashIJKey;
  #define PetscHashIJKeyHash(key)     PetscHashCombine(PetscHashInt((key).i), PetscHashInt((key).j))
  #define PetscHashIJKeyEqual(k1, k2) (((k1).i == (k2).i) ? ((k1).j == (k2).j) : 0)
#endif

PETSC_HASH_MAP(HMapIJ, PetscHashIJKey, PetscInt, PetscHashIJKeyHash, PetscHashIJKeyEqual, -1)

#endif /* PETSC_HASHMAPIJ_H */
