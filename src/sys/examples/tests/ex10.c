
static char help[] = "Tests PetscArraymove()/PetscMemmove()\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i,*a,*b,*b1;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  if (PetscHasMallocType(PETSC_MALLOC_CUDA_UNIFIED)) {
    ierr = PetscPushMallocType(PETSC_MALLOC_CUDA_UNIFIED);CHKERRQ(ierr);
  } else if (PetscHasMallocType(PETSC_MALLOC_MEMKIND_HBW_PREFERRED)) {
    ierr = PetscPushMallocType(PETSC_MALLOC_MEMKIND_HBW_PREFERRED);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(10,&a);CHKERRQ(ierr);
  ierr = PetscMalloc1(20,&b);CHKERRQ(ierr);

  /*
      Nonoverlapping regions
  */
  for (i=0; i<20; i++) b[i] = i;
  ierr = PetscArraymove(a,b,10);CHKERRQ(ierr);
  ierr = PetscIntView(10,a,NULL);CHKERRQ(ierr);

  ierr = PetscFree(a);CHKERRQ(ierr);

  /*
     |        |                |       |
     b        b1              b +15    b +20
                              b1+10    b1+15
  */
  b1   = b + 5;
  ierr = PetscArraymove(b1,b,15);CHKERRQ(ierr);
  ierr = PetscIntView(15,b1,NULL);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);

  /*
     |       |                    |       |
     a       b                   a+20   a+25
                                        b+20
  */
  /* test realloc with NULL pointer  */
  ierr = PetscRealloc(25*sizeof(*a),&a);CHKERRQ(ierr);
  b    = a + 5;
  for (i=0; i<20; i++) b[i] = i;
  ierr = PetscArraymove(a,b,20);CHKERRQ(ierr);
  ierr = PetscIntView(20,a,NULL);CHKERRQ(ierr);

  /* test realloc */
  ierr = PetscRealloc(10*sizeof(*a),&a);CHKERRQ(ierr);
  ierr = PetscIntView(10,a,NULL);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:

TEST*/
