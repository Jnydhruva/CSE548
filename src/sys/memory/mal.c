/*
    Code that allows a user to dictate what malloc() PETSc uses.
*/
#include <petscsys.h>             /*I   "petscsys.h"   I*/
#include <petscvalgrind.h>
#include <stdarg.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#if defined(PETSC_HAVE_MEMKIND)
#include <memkind.h>
#endif
#if defined(PETSC_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

const char *const PetscMallocTypes[] = {"MALLOC_STANDARD","MALLOC_CUDA_UNIFIED","MALLOC_MEMKIND_DEFAULT","MALLOC_MEMKIND_HBW_PREFERRED","PetscMallocType","PETSC_",0};

#define PETSCMALLOCTYPEPUSHESMAX 64

/* this is to match the old default to always prefer memkind if available */
#if defined(PETSC_HAVE_MEMKIND)
static PetscMallocType mtypes[PETSCMALLOCTYPEPUSHESMAX] = {PETSC_MALLOC_MEMKIND_HBW_PREFERRED};
#elif defined(PETSC_USE_CUDA_UNIFIED_MEMORY)
static PetscMallocType mtypes[PETSCMALLOCTYPEPUSHESMAX] = {PETSC_MALLOC_CUDA_UNIFIED};
#else
static PetscMallocType mtypes[PETSCMALLOCTYPEPUSHESMAX] = {PETSC_MALLOC_STANDARD};
#endif
static int tipmtype = 0;

PetscErrorCode PetscPushMallocType(PetscMallocType mtype)
{
  PetscFunctionBeginHot;
  if (tipmtype > PETSCMALLOCTYPEPUSHESMAX-1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscPushMallocType(), perhaps you forgot to call PetscPopMallocType()?");
  PetscInfo2(NULL,"Push MallocType %s, old %s\n",PetscMallocTypes[mtype],PetscMallocTypes[mtypes[tipmtype]]);
  mtypes[++tipmtype] = mtype;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscPopMallocType()
{
  PetscFunctionBeginHot;
  if (!tipmtype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscPopMallocType()");
  PetscInfo2(NULL,"Pop MallocType %s, new %s\n",PetscMallocTypes[mtypes[tipmtype]],PetscMallocTypes[mtypes[tipmtype-1]]);
  tipmtype--;
  PetscFunctionReturn(0);
}

PetscBool PetscHasMallocType(PetscMallocType mtype)
{
  PetscBool has = PETSC_FALSE;
  switch (mtype) {
#if defined(PETSC_HAVE_CUDA)
  case PETSC_MALLOC_CUDA_UNIFIED:
#endif
#if defined(PETSC_HAVE_MEMKIND)
  case PETSC_MALLOC_MEMKIND_DEFAULT:
  case PETSC_MALLOC_MEMKIND_HBW_PREFERRED:
#endif
  case PETSC_MALLOC_STANDARD:
    has = PETSC_TRUE;
    break;
  default:
    has = PETSC_FALSE;
    break;
  }
  return has;
}

/*
        We want to make sure that all mallocs of double or complex numbers are complex aligned.
    1) on systems with memalign() we call that routine to get an aligned memory location
    2) on systems without memalign() we
       - allocate one sizeof(PetscScalar) extra space
       - we shift the pointer up slightly if needed to get PetscScalar aligned
       - if shifted we store at ptr[-1] the amount of shift (plus a classid)
*/
#define SHIFT_CLASSID 456123

PETSC_EXTERN PetscErrorCode PetscMallocAlign(size_t mem,PetscBool clear,int line,const char func[],const char file[],void **result)
{
  void                  *ptr;
  int                   shift;
  const size_t          sm = PETSC_MEMALIGN;
  const size_t          sp = sizeof(void*);
  const size_t          ss = sizeof(size_t);
  const size_t          se = sizeof(PetscMallocType);
  const size_t          len = mem + sm + sp + ss + se;
  const PetscMallocType currentmtype = mtypes[tipmtype];
#if defined(PETSC_HAVE_CUDA)
  cudaError_t           cerr;
#endif

  if (!mem) {*result = NULL; return 0;}
  switch (currentmtype) {
#if defined(PETSC_HAVE_CUDA)
  case PETSC_MALLOC_CUDA_UNIFIED:
    cerr = cudaMallocManaged(&ptr,len,cudaMemAttachHost);
    if (cerr != cudaSuccess) PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Likely memory corruption in heap. Cuda error:",cudaGetErrorString(cerr));
    break;
#endif
  case PETSC_MALLOC_STANDARD:
    if (clear) { /* use calloc to respect first-touch policy */
      ptr = calloc(1,len);
      clear = PETSC_FALSE;
    } else ptr = malloc(len);
    break;
#if defined(PETSC_HAVE_MEMKIND)
  case PETSC_MALLOC_MEMKIND_DEFAULT:
    ptr = memkind_malloc(MEMKIND_DEFAULT,len);
    break;
  case PETSC_MALLOC_MEMKIND_HBW_PREFERRED:
    ptr = memkind_malloc(MEMKIND_HBW_PREFERRED,len);
    /* This is odd: according to the specs, memkind should fall back to MEMKIND_DEFAULT if no HBW is available
       On my Fedora 30 workstation without KNLs, memkind 1.7.0 returns NULL */
    if (!ptr) ptr = memkind_malloc(MEMKIND_DEFAULT,len);
    break;
#endif
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unhandled memory type %s",PetscMallocTypes[currentmtype]);
  }
  if (!ptr) return PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_MEM,PETSC_ERROR_INITIAL,"Failure to request memory %.0f of type %s",(PetscLogDouble)mem,PetscMallocTypes[currentmtype]);
  shift = PETSC_MEMALIGN - (int)((PETSC_UINTPTR_T)((char*)ptr + ss + sp + se) % PETSC_MEMALIGN);

  /* store "allocated" memory, "allocated" memory size, allocation type and original pointer, so that we can realloc and free */
  *result = (void*)((char*)ptr + ss + sp + se + shift);
  *((PetscMallocType*)((char*)*result - se)) = currentmtype;
  *((size_t*)((char*)*result - ss - se))     = mem;
  *((void**)((char*)*result - sp - ss - se)) = ptr;
  if (clear) memset((char*)*result,0,mem);
#if defined(PETSC_USE_DEBUG)
  if (((size_t) (*result)) % PETSC_MEMALIGN) PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Unaligned memory generated! Expected %d, shift %d",PETSC_MEMALIGN,((size_t) (*result)) % PETSC_MEMALIGN);
#endif
#if defined(PETSC_USE_DEBUG) && defined(PETSC_HAVE_VALGRIND)
  if (PETSC_RUNNING_ON_VALGRIND) { /* make memory header not accessible */
    VALGRIND_MAKE_MEM_NOACCESS((char*)ptr + shift,ss + sp + se);
  }
#endif
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscFreeAlign(void *ptr,int line,const char func[],const char file[])
{
  PetscMallocType mtype;
  const size_t    se = sizeof(PetscMallocType);
  const size_t    sp = sizeof(void*);
  const size_t    ss = sizeof(size_t);
#if defined(PETSC_HAVE_FREE_RETURN_INT)
  int             err;
#endif
#if defined(PETSC_HAVE_CUDA)
  cudaError_t     cerr;
#endif

  if (!ptr) return 0;
#if defined(PETSC_USE_DEBUG) && defined(PETSC_HAVE_VALGRIND)
  if (PETSC_RUNNING_ON_VALGRIND) { /* make memory header accessible */
    VALGRIND_MAKE_MEM_DEFINED((char*)ptr - ss - se - sp,ss + se + sp);
  }
#endif
  mtype = *((PetscMallocType*)((char*)ptr - se));
  ptr   = *((void**)((char*)ptr - ss - se - sp));
  switch (mtype) {
  case PETSC_MALLOC_STANDARD:
#if defined(PETSC_HAVE_FREE_RETURN_INT)
    err = free(ptr);
    if (err) return PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"System free returned error %d",err);
#else
    free(ptr);
#endif
    break;
#if defined(PETSC_HAVE_CUDA)
  case PETSC_MALLOC_CUDA_UNIFIED:
    cerr = cudaFree(ptr);
    if (cerr != cudaSuccess) PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Likely memory corruption in heap. Cuda error:",cudaGetErrorString(cerr));
    break;
#endif
#if defined(PETSC_HAVE_MEMKIND)
  case PETSC_MALLOC_MEMKIND_DEFAULT:
  case PETSC_MALLOC_MEMKIND_HBW_PREFERRED:
    memkind_free(0,ptr); /* specify the kind to 0 so that memkind will look up for the right type */
    break;
#endif
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unhandled memory type %s",PetscMallocTypes[mtype]);
  }
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscReallocAlign(size_t mem, int line, const char func[], const char file[], void **result)
{
  PetscErrorCode ierr;
  if (!mem) {
    ierr = PetscFreeAlign(*result, line, func, file);
    if (ierr) return ierr;
    *result = NULL;
  } else if (!*result) {
    ierr = PetscMallocAlign(mem,PETSC_FALSE,line,func,file,result);
    if (ierr) return ierr;
  } else {
    void                  *newResult   = NULL;
    const size_t          sm           = PETSC_MEMALIGN;
    const size_t          sp           = sizeof(void*);
    const size_t          ss           = sizeof(size_t);
    const size_t          se           = sizeof(PetscMallocType);
    const size_t          len          = mem + sm + sp + ss + se;
    const PetscMallocType currentmtype = mtypes[tipmtype];
    PetscMallocType       omtype;
    size_t                omem;
    void                  *optr;
    int                   oshift;

#if defined(PETSC_USE_DEBUG) && defined(PETSC_HAVE_VALGRIND)
    if (PETSC_RUNNING_ON_VALGRIND) { /* make memory header accessible */
      VALGRIND_MAKE_MEM_DEFINED((char*)*result - ss - se - sp,ss + se + sp);
    }
#endif
    omtype = *((PetscMallocType*)((char*)*result - se));
    omem   = *((size_t*)((char*)*result - ss - se));
    optr   = *((void**)((char*)*result - ss - se - sp));
    oshift = PETSC_MEMALIGN - (int)((PETSC_UINTPTR_T)((char*)optr + ss + sp + se) % PETSC_MEMALIGN);

    switch (currentmtype) {
    case PETSC_MALLOC_STANDARD:
      if (omtype == PETSC_MALLOC_STANDARD) newResult = realloc(optr,len);
      break;
#if defined(PETSC_HAVE_MEMKIND)
    case PETSC_MALLOC_MEMKIND_DEFAULT:
    case PETSC_MALLOC_MEMKIND_HBW_PREFERRED:
      if (omtype == PETSC_MALLOC_MEMKIND_DEFAULT || omtype == PETSC_MALLOC_MEMKIND_HBW_PREFERRED) {
        newResult = memkind_realloc(currentmtype == PETSC_MALLOC_MEMKIND_DEFAULT ? MEMKIND_DEFAULT : MEMKIND_HBW_PREFERRED,optr,len);
      }
      break;
#endif
    default:
      break;
    }
    if (newResult) { /* realloc with specifics reallocation routines */
      int shift = PETSC_MEMALIGN - (int)((PETSC_UINTPTR_T)((char*)newResult + ss + sp + se) % PETSC_MEMALIGN);
      optr = newResult;
      newResult = (void*)((char*)newResult + ss + sp + se + shift);
      /* if the previous shift is not equal to the current one, adjust the contribution of realloc */
#if defined(PETSC_HAVE_MEMMOVE) && !defined(PETSC_USE_DEBUG)
      if (oshift != shift) memmove(newResult,(char*)newResult + oshift - shift,mem);
#else
      if (oshift != shift) (void)PetscMemmove(newResult,(char*)newResult + oshift - shift,mem);
#endif
      *((PetscMallocType*)((char*)newResult - se)) = currentmtype;
      *((size_t*)((char*)newResult - ss - se))     = mem;
      *((void**)((char*)newResult - sp - ss - se)) = optr;
#if defined(PETSC_USE_DEBUG) && defined(PETSC_HAVE_VALGRIND)
      if (PETSC_RUNNING_ON_VALGRIND) { /* make memory header not accessible */
        VALGRIND_MAKE_MEM_NOACCESS((char*)optr + shift,ss + sp + se);
      }
#endif
    } else { /* fallback to malloc + memcpy + free */
      ierr = PetscMallocAlign(mem,PETSC_FALSE,line,func,file,&newResult);if (ierr) PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT,"Likely memory corruption in heap");
      ierr = PetscMemcpy(newResult,*result,PetscMin(omem,mem));if (ierr) PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT,"Likely memory corruption in heap");
      ierr = PetscFreeAlign(*result,line,func,file); if (ierr) PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT,"Likely memory corruption in heap");
    }
    if (!newResult) return PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_MEM,PETSC_ERROR_INITIAL,"Memory requested %.0f",(PetscLogDouble)mem);
    *result = newResult;
  }
#if defined(PETSC_USE_DEBUG)
  if (((size_t) (*result)) % PETSC_MEMALIGN) PetscError(PETSC_COMM_SELF,line,func,file,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Unaligned memory generated! Expected %d, shift %d",PETSC_MEMALIGN,((size_t) (*result)) % PETSC_MEMALIGN);
#endif
  return 0;
}

PetscErrorCode (*PetscTrMalloc)(size_t,PetscBool,int,const char[],const char[],void**) = PetscMallocAlign;
PetscErrorCode (*PetscTrFree)(void*,int,const char[],const char[])                     = PetscFreeAlign;
PetscErrorCode (*PetscTrRealloc)(size_t,int,const char[],const char[],void**)          = PetscReallocAlign;

PETSC_INTERN PetscBool petscsetmallocvisited;
PetscBool petscsetmallocvisited = PETSC_FALSE;

/*@C
   PetscMallocSet - Sets the routines used to do mallocs and frees.
   This routine MUST be called before PetscInitialize() and may be
   called only once.

   Not Collective

   Input Parameters:
+ imalloc - the routine that provides the malloc (also provides calloc(), which is used depends on the second argument)
. ifree - the routine that provides the free
- iralloc - the routine that provides the realloc

   Level: developer

@*/
PetscErrorCode PetscMallocSet(PetscErrorCode (*imalloc)(size_t,PetscBool,int,const char[],const char[],void**),
                              PetscErrorCode (*ifree)(void*,int,const char[],const char[]),
                              PetscErrorCode (*iralloc)(size_t, int, const char[], const char[], void **))
{
  PetscFunctionBegin;
  if (petscsetmallocvisited && (imalloc != PetscTrMalloc || ifree != PetscTrFree)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"cannot call multiple times");
  PetscTrMalloc         = imalloc;
  PetscTrFree           = ifree;
  PetscTrRealloc        = iralloc;
  petscsetmallocvisited = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocClear - Resets the routines used to do mallocs and frees to the defaults.

   Not Collective

   Level: developer

   Notes:
    In general one should never run a PETSc program with different malloc() and
    free() settings for different parts; this is because one NEVER wants to
    free() an address that was malloced by a different memory management system

    Called in PetscFinalize() so that if PetscInitialize() is called again it starts with a fresh state of allocation information

@*/
PetscErrorCode PetscMallocClear(void)
{
  PetscFunctionBegin;
  PetscTrMalloc         = PetscMallocAlign;
  PetscTrFree           = PetscFreeAlign;
  PetscTrRealloc        = PetscReallocAlign;
  petscsetmallocvisited = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMemoryTrace(const char label[])
{
  PetscErrorCode        ierr;
  PetscLogDouble        mem,mal;
  static PetscLogDouble oldmem = 0,oldmal = 0;

  PetscFunctionBegin;
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);
  ierr = PetscMallocGetCurrentUsage(&mal);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s High water  %8.3f MB increase %8.3f MB Current %8.3f MB increase %8.3f MB\n",label,mem*1e-6,(mem - oldmem)*1e-6,mal*1e-6,(mal - oldmal)*1e-6);CHKERRQ(ierr);
  oldmem = mem;
  oldmal = mal;
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocSetDRAM - Set PetscMalloc to use DRAM.
     If memkind is available, change the memkind type. Otherwise, switch the
     current malloc and free routines to the PetscMallocAlign and
     PetscFreeAlign (PETSc default).

   Not Collective

   Level: developer

   Notes:
     This provides a way to do the allocation on DRAM temporarily. One
     can switch back to the previous choice by calling PetscMallocReset().

   Deprecated, use PetscPushMallocType()

.seealso: PetscMallocReset()
@*/
PetscErrorCode PetscMallocSetDRAM(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPushMallocType(PETSC_MALLOC_STANDARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocResetDRAM - Reset the changes made by PetscMallocSetDRAM

   Not Collective

   Level: developer

   Deprecated, use PetscPopMallocType()

.seealso: PetscMallocSetDRAM()
@*/
PetscErrorCode PetscMallocResetDRAM(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPopMallocType();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscBool petscmalloccoalesce =
#if defined(PETSC_USE_MALLOC_COALESCED)
  PETSC_TRUE;
#else
  PETSC_FALSE;
#endif

/*@C
   PetscMallocSetCoalesce - Use coalesced malloc when allocating groups of objects

   Not Collective

   Input Parameters:
.  coalesce - PETSC_TRUE to use coalesced malloc for multi-object allocation.

   Options Database Keys:
.  -malloc_coalesce - turn coalesced malloc on or off

   Note:
   PETSc uses coalesced malloc by default for optimized builds and not for debugging builds.  This default can be changed via the command-line option -malloc_coalesce or by calling this function.
   This function can only be called immediately after PetscInitialize()

   Level: developer

.seealso: PetscMallocA()
@*/
PetscErrorCode PetscMallocSetCoalesce(PetscBool coalesce)
{
  PetscFunctionBegin;
  petscmalloccoalesce = coalesce;
  PetscFunctionReturn(0);
}

/*@C
   PetscMallocA - Allocate and optionally clear one or more objects, possibly using coalesced malloc

   Not Collective

   Input Parameters:
+  n - number of objects to allocate (at least 1)
.  clear - use calloc() to allocate space initialized to zero
.  lineno - line number to attribute allocation (typically __LINE__)
.  function - function to attribute allocation (typically PETSC_FUNCTION_NAME)
.  filename - file name to attribute allocation (typically __FILE__)
-  bytes0 - first of n object sizes

   Output Parameters:
.  ptr0 - first of n pointers to allocate

   Notes:
   This function is not normally called directly by users, but rather via the macros PetscMalloc1(), PetscMalloc2(), or PetscCalloc1(), etc.

   Level: developer

.seealso: PetscMallocAlign(), PetscMallocSet(), PetscMalloc1(), PetscMalloc2(), PetscMalloc3(), PetscMalloc4(), PetscMalloc5(), PetscMalloc6(), PetscMalloc7(), PetscCalloc1(), PetscCalloc2(), PetscCalloc3(), PetscCalloc4(), PetscCalloc5(), PetscCalloc6(), PetscCalloc7(), PetscFreeA()
@*/
PetscErrorCode PetscMallocA(int n,PetscBool clear,int lineno,const char *function,const char *filename,size_t bytes0,void *ptr0,...)
{
  PetscErrorCode ierr;
  va_list        Argp;
  size_t         bytes[8],sumbytes;
  void           **ptr[8];
  int            i;

  PetscFunctionBegin;
  if (n > 8) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Attempt to allocate %d objects but only 8 supported",n);
  bytes[0] = bytes0;
  ptr[0] = (void**)ptr0;
  sumbytes = (bytes0 + PETSC_MEMALIGN-1) & ~(PETSC_MEMALIGN-1);
  va_start(Argp,ptr0);
  for (i=1; i<n; i++) {
    bytes[i] = va_arg(Argp,size_t);
    ptr[i] = va_arg(Argp,void**);
    sumbytes += (bytes[i] + PETSC_MEMALIGN-1) & ~(PETSC_MEMALIGN-1);
  }
  va_end(Argp);
  if (petscmalloccoalesce) {
    char *p;
    ierr = (*PetscTrMalloc)(sumbytes,clear,lineno,function,filename,(void**)&p);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      *ptr[i] = bytes[i] ? p : NULL;
      p = (char*)PetscAddrAlign(p + bytes[i]);
    }
  } else {
    for (i=0; i<n; i++) {
      ierr = (*PetscTrMalloc)(bytes[i],clear,lineno,function,filename,(void**)ptr[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscFreeA - Free one or more objects, possibly allocated using coalesced malloc

   Not Collective

   Input Parameters:
+  n - number of objects to free (at least 1)
.  lineno - line number to attribute deallocation (typically __LINE__)
.  function - function to attribute deallocation (typically PETSC_FUNCTION_NAME)
.  filename - file name to attribute deallocation (typically __FILE__)
-  ptr0 ... - first of n pointers to free

   Note:
   This function is not normally called directly by users, but rather via the macros PetscFree(), PetscFree2(), etc.

   The pointers are zeroed to prevent users from accidently reusing space that has been freed.

   Level: developer

.seealso: PetscMallocAlign(), PetscMallocSet(), PetscMallocA(), PetscFree1(), PetscFree2(), PetscFree3(), PetscFree4(), PetscFree5(), PetscFree6(), PetscFree7()
@*/
PetscErrorCode PetscFreeA(int n,int lineno,const char *function,const char *filename,void *ptr0,...)
{
  PetscErrorCode ierr;
  va_list        Argp;
  void           **ptr[8];
  int            i;

  PetscFunctionBegin;
  if (n > 8) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Attempt to allocate %d objects but only up to 8 supported",n);
  ptr[0] = (void**)ptr0;
  va_start(Argp,ptr0);
  for (i=1; i<n; i++) {
    ptr[i] = va_arg(Argp,void**);
  }
  va_end(Argp);
  if (petscmalloccoalesce) {
    for (i=0; i<n; i++) {       /* Find first nonempty allocation */
      if (*ptr[i]) break;
    }
    while (--n > i) {
      *ptr[n] = NULL;
    }
    ierr = (*PetscTrFree)(*ptr[n],lineno,function,filename);CHKERRQ(ierr);
    *ptr[n] = NULL;
  } else {
    while (--n >= 0) {
      ierr = (*PetscTrFree)(*ptr[n],lineno,function,filename);CHKERRQ(ierr);
      *ptr[n] = NULL;
    }
  }
  PetscFunctionReturn(0);
}
