static char help[] = "Test DMGetStratumSize on non-existing label\n\n";

#include <petsc.h>

int main(int argc,char **argv) {
  DM              dm;
  PetscInt        n;
  IS              is;
  const PetscInt *nindices;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD,2,PETSC_FALSE,NULL,NULL,NULL,NULL,PETSC_TRUE,&dm));
  PetscCall(DMView(dm,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMGetStratumSize(dm,"zorglub",1,&n));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Size of stratum zorglub: %" PetscInt_FMT "\n",n));
  PetscCall(DMGetStratumIS(dm,"zorglub",1,&is));
  if (!is) {
    PetscPrintf(PETSC_COMM_WORLD,"IS is null\n");
  }
  PetscCall(ISGetIndices(is,&nindices));
  PetscCall(ISView(is,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(ISRestoreIndices(is,&nindices));
  PetscCall(ISDestroy(&is));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}
/*TEST
  test:
    suffix: 0
TEST*/