/*$Id: ex16.c,v 1.4 2000/08/01 20:58:08 bsmith Exp $*/

static char help[] = "Tests VecPack routines.\n\n";

#include "petscda.h"
#include "petscpf.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int     ierr,nredundant1 = 5,nredundant2 = 2,rank,i,*ridx1,*ridx2,*lidx1,*lidx2,nlocal;
  Scalar  *redundant1,*redundant2;
  VecPack packer;
  Vec     global,local1,local2;
  PF      pf;
  DA      da1,da2;
  Viewer  sviewer;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = VecPackCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);

  redundant1 = (Scalar*)PetscMalloc(nredundant1*sizeof(Scalar));CHKPTRQ(redundant1);
  ierr = VecPackAddArray(packer,nredundant1);CHKERRQ(ierr);

  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,8,1,1,PETSC_NULL,&da1);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da1,&local1);CHKERRQ(ierr);
  ierr = VecPackAddDA(packer,da1);CHKERRQ(ierr);

  redundant2 = (Scalar*)PetscMalloc(nredundant2*sizeof(Scalar));CHKPTRQ(redundant2);
  ierr = VecPackAddArray(packer,nredundant2);CHKERRQ(ierr);

  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,6,1,1,PETSC_NULL,&da2);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da2,&local2);CHKERRQ(ierr);
  ierr = VecPackAddDA(packer,da2);CHKERRQ(ierr);

  ierr = VecPackCreateGlobalVector(packer,&global);CHKERRQ(ierr);
  ierr = PFCreate(PETSC_COMM_WORLD,1,1,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFIDENTITY,PETSC_NULL);CHKERRQ(ierr);
  ierr = PFApplyVec(pf,PETSC_NULL,global);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);
  ierr = VecView(global,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecPackScatter(packer,global,redundant1,local1,redundant2,local2);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] My part of redundant1 array\n",rank);CHKERRQ(ierr);
  ierr = PetscScalarView(nredundant1,redundant1,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] My part of da1 vector\n",rank);CHKERRQ(ierr);
  ierr = ViewerGetSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
  ierr = VecView(local1,sviewer);CHKERRQ(ierr);
  ierr = ViewerRestoreSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] My part of redundant2 array\n",rank);CHKERRQ(ierr);
  ierr = PetscScalarView(nredundant2,redundant2,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] My part of da2 vector\n",rank);CHKERRQ(ierr);
  ierr = ViewerGetSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);
  ierr = VecView(local2,sviewer);CHKERRQ(ierr);
  ierr = ViewerRestoreSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRQ(ierr);

  for (i=0; i<nredundant1; i++) redundant1[i] = (rank+2)*i;
  for (i=0; i<nredundant2; i++) redundant2[i] = (rank+10)*i;

  ierr = VecPackGather(packer,global,redundant1,local1,redundant2,local2);CHKERRQ(ierr);
  ierr = VecView(global,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* get the global numbering for each subvector/array element */
  ierr = VecPackGetGlobalIndices(packer,&ridx1,&lidx1,&ridx2,&lidx2);CHKERRQ(ierr);
  
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] Global numbering of redundant1 array\n",rank);CHKERRQ(ierr);
  ierr = PetscIntView(nredundant1,ridx1,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] Global numbering of local1 vector\n",rank);CHKERRQ(ierr);
  ierr = VecGetSize(local1,&nlocal);CHKERRQ(ierr);
  ierr = PetscIntView(nlocal,lidx1,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] Global numbering of redundant2 array\n",rank);CHKERRQ(ierr);
  ierr = PetscIntView(nredundant2,ridx2,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ViewerASCIISynchronizedPrintf(VIEWER_STDOUT_WORLD,"[%d] Global numbering of local2 vector\n",rank);CHKERRQ(ierr);
  ierr = VecGetSize(local2,&nlocal);CHKERRQ(ierr);
  ierr = PetscIntView(nlocal,lidx2,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  ierr = PetscFree(ridx1);CHKERRQ(ierr);
  ierr = PetscFree(lidx1);CHKERRQ(ierr);
  ierr = PetscFree(ridx2);CHKERRQ(ierr);
  ierr = PetscFree(lidx2);CHKERRQ(ierr);

  ierr = DADestroy(da1);CHKERRQ(ierr);
  ierr = DADestroy(da2);CHKERRQ(ierr);
  ierr = VecDestroy(local1);CHKERRQ(ierr);
  ierr = VecDestroy(local2);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = VecPackDestroy(packer);CHKERRQ(ierr);
  ierr = PetscFree(redundant1);CHKERRQ(ierr);
  ierr = PetscFree(redundant2);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
 
