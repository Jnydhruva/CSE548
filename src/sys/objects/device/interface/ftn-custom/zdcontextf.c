#include <petsc/private/fortranimpl.h>
#include <petscdevice.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdevicecontextsetfromoptions_ PETSCDEVICECONTEXTSETFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdevicecontextsetfromoptions_ petscdevicecontextsetfromoptions
#endif

PETSC_EXTERN void petscdevicecontextsetfromoptions_(int *comm, char *prefix, PetscDeviceContext *dctx, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *p;

  FIXCHAR(prefix,len,p);
  *ierr = PetscDeviceContextSetFromOptions(MPI_Comm_f2c(*comm),p,*dctx); if (*ierr) return;
  FREECHAR(prefix,p);
}
