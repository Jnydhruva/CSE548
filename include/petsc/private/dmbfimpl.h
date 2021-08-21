#if !defined(_DMBFIMPL_H)
#define _DMBFIMPL_H

#include <petscdmbf.h> /*I "petscdmbf.h" I*/
#include <petscdm.h>   /*I "petscdm.h" I*/

PETSC_EXTERN PetscErrorCode DMBFGetP4est(DM,void*);
PETSC_EXTERN PetscErrorCode DMBFGetGhost(DM,void*);

/***************************************
 * SHAPE
 **************************************/

PETSC_EXTERN PetscErrorCode DMBFShapeSetUp(DM_BF_Shape*,size_t,size_t);
PETSC_EXTERN PetscErrorCode DMBFShapeClear(DM_BF_Shape*);
PETSC_EXTERN PetscErrorCode DMBFShapeIsSetUp(const DM_BF_Shape*,PetscBool*);
PETSC_EXTERN PetscErrorCode DMBFShapeIsValid(const DM_BF_Shape*,PetscBool*);
PETSC_EXTERN PetscErrorCode DMBFShapeCheckSetUp(const DM_BF_Shape*);
PETSC_EXTERN PetscErrorCode DMBFShapeCheckValid(const DM_BF_Shape*);

PETSC_EXTERN PetscErrorCode DMBFShapeCompare(const DM_BF_Shape*,const DM_BF_Shape*,PetscBool*);
PETSC_EXTERN PetscErrorCode DMBFShapeCopy(DM_BF_Shape*,const DM_BF_Shape*);

PETSC_EXTERN PetscErrorCode DMBFShapeSet(DM_BF_Shape*,const size_t*,const size_t*);
PETSC_EXTERN PetscErrorCode DMBFShapeGet(const DM_BF_Shape*,size_t**,size_t**,size_t*,size_t*);
PETSC_EXTERN PetscErrorCode DMBFShapeSetFromInt(DM_BF_Shape*,const PetscInt*);
PETSC_EXTERN PetscErrorCode DMBFShapeGetToInt(const DM_BF_Shape*,PetscInt**,PetscInt*,PetscInt*);

PETSC_STATIC_INLINE size_t _p_DMBFShapeOffset(const DM_BF_Shape *shape, PetscInt nmax)
{
  const size_t  n = (0<=nmax && nmax<shape->n ? nmax : shape->n);
  size_t        i, j, s, size=0;

  if (!shape->list) {
    return size;
  }

  for (i=0; i<n; i++) {
    s = shape->list[i][0];
    for (j=1; j<shape->dim; j++) {
      if (0 < shape->list[i][j]) {
        s *= shape->list[i][j];
      }
    }
    size += s + shape->pad[i];
  }
  return size;
}

/***************************************
 * CELL MEMORY
 **************************************/

typedef enum _p_DM_BF_CellMemoryIndex {
  DMBF_CELLMEMIDX_INFO          = 0,
  DMBF_CELLMEMIDX_POINTERS      = 1,
  DMBF_CELLMEMIDX_DATAREAD      = 2,
  DMBF_CELLMEMIDX_DATAREADWRITE = 3,
  DMBF_CELLMEMIDX_DATAV         = 4,
  DMBF_CELLMEMIDX_DATA          = 5,
  DMBF_CELLMEMIDX_END           = -1
} DM_BF_CellMemoryIndex;

PETSC_STATIC_INLINE size_t _p_cellMemoryOffset(const DM_BF_Shape *memory, size_t memoryIndex)
{
  switch (memoryIndex) {
    case DMBF_CELLMEMIDX_END:
      return memory->size;
    default:
      if (memoryIndex < memory->n) {
        return _p_DMBFShapeOffset(memory,memoryIndex);
      } else {
        SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unreachable code");
        return 0;
      }
  }
}

PETSC_STATIC_INLINE PetscScalar **_p_cellGetPointers(const DM_BF_Cell *cell, const DM_BF_Shape *memory)
{
  return (PetscScalar**)(((char*)cell) + _p_cellMemoryOffset(memory,(size_t)DMBF_CELLMEMIDX_POINTERS));
}

PETSC_STATIC_INLINE PetscScalar *_p_cellGetDataRead(const DM_BF_Cell *cell, const DM_BF_Shape *memory)
{
  return (PetscScalar*)(((char*)cell) + _p_cellMemoryOffset(memory,(size_t)DMBF_CELLMEMIDX_DATAREAD));
}

PETSC_STATIC_INLINE PetscScalar *_p_cellGetDataReadWrite(const DM_BF_Cell *cell, const DM_BF_Shape *memory)
{
  return (PetscScalar*)(((char*)cell) + _p_cellMemoryOffset(memory,(size_t)DMBF_CELLMEMIDX_DATAREADWRITE));
}

PETSC_STATIC_INLINE PetscScalar *_p_cellGetDataV(const DM_BF_Cell *cell, const DM_BF_Shape *memory)
{
  return (PetscScalar*)(((char*)cell) + _p_cellMemoryOffset(memory,(size_t)DMBF_CELLMEMIDX_DATAV));
}

PETSC_STATIC_INLINE PetscScalar *_p_cellGetData(const DM_BF_Cell *cell, const DM_BF_Shape *memory, size_t dataIndex)
{
  return (PetscScalar*)(((char*)cell) + _p_cellMemoryOffset(memory,(size_t)DMBF_CELLMEMIDX_DATA+dataIndex));
}

PETSC_STATIC_INLINE PetscErrorCode DMBFCellInitialize(DM_BF_Cell *cell, const DM_BF_Shape *memory)
{
  size_t         k;

  PetscFunctionBegin;
  cell->adaptFlag        = DM_ADAPT_DETERMINE;
  cell->memory           = memory;
  cell->vecViewRead      = (const PetscScalar**)PETSC_NULL;
  cell->vecViewReadWrite = (PetscScalar**)PETSC_NULL;
  cell->dataRead         = _p_cellGetDataRead(cell,memory);      //TODO deprecated
  cell->dataReadWrite    = _p_cellGetDataReadWrite(cell,memory); //TODO deprecated
  cell->dataV            = _p_cellGetDataV(cell,memory);
  cell->data             = _p_cellGetPointers(cell,memory);
  for (k=0; k<(memory->n - DMBF_CELLMEMIDX_DATA); k++) {
    cell->data[k] = _p_cellGetData(cell,memory,k);
  }
  PetscFunctionReturn(0);
}

#endif /* defined(_DMBFIMPL_H) */
