#ifndef lint
static char vcid[] = "$Id: ex55.c,v 1.1 1997/01/24 23:39:22 bsmith Exp bsmith $";
#endif

static char help[] = "Tests converting a matrix to another format with MatConvert()\n\n";

#include "mat.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat     C, A, B; 
  int     ierr, i, j, flg, ntypes = 9,size;
  MatType type[9] = {MATMPIAIJ,  MATMPIROWBS,  MATMPIBDIAG, MATMPIDENSE,
                     MATMPIBAIJ, MATSEQDENSE, MATSEQAIJ,   MATSEQBDIAG, MATSEQBAIJ};
  char    file[128];
  Vec     v;
  Viewer  fd;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg); CHKERRA(ierr);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if (size > 1) ntypes = 5;

  /* 
     Open binary file.  Note that we use BINARY_RDONLY to indicate
     reading from this file.
  */
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATMPIAIJ,&C); CHKERRA(ierr);
  ierr = VecLoad(fd,&v); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  
  for ( i=0; i<ntypes; i++ ) {
    ierr = MatConvert(C,type[i],&A); CHKERRA(ierr);
    for ( j=0; j<ntypes; j++ ) {
      ierr = MatConvert(A,type[i],&B); CHKERRA(ierr);
      ierr = MatDestroy(B);  CHKERRA(ierr);
    }
    ierr = MatDestroy(A);  CHKERRA(ierr);
  }

  if (size == 1) {
    ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,"testmat",BINARY_CREATE,&fd);CHKERRA(ierr);
    ierr = MatView(C,fd); CHKERRA(ierr);
    ierr = VecView(v,fd); CHKERRA(ierr);
    ierr = ViewerDestroy(fd); CHKERRA(ierr);
  }

  ierr = MatDestroy(C); CHKERRA(ierr);
  ierr = VecDestroy(v); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}











