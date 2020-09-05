static char help[] = "Tests repeated VecDuplicate() with VECCUDA.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i, n = 1000, trials = 1000;
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-trials",&trials,NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(x,VECCUDA);CHKERRQ(ierr);
  ierr = VecSet(x,-1.3);CHKERRQ(ierr);
  for (i = 0; i < trials; i++) {
    Vec y;

    ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
    ierr = VecCopy(x,y);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       requires: cuda
       suffix: 1

TEST*/
