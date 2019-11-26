static char help[33] = "Test Unstructured Mesh Handling\n";

# include <petscdmplex.h>
# include <petscviewer.h>
# include <petscsnes.h>
# include <petscds.h>
# include <petscksp.h>
# include <petscdmlabel.h>

# define PETSCVIEWERVTK          "vtk"
# define PETSCVIEWERASCII        "ascii"
# define VECSTANDARD             "standard"

typedef enum {NEUMANN, DIRICHLET, NONE} BCType;

typedef struct {
  PetscLogStage  stageREAD, stageCREATE, stageREFINE, stageINSERT, stageADD, stageGVD, stagePETSCFE, stageCREATEDS;
  PetscLogEvent  eventREAD, eventCREATE, eventREFINE, eventINSERT, eventADD, eventGVD, eventPETSCFE, eventCREATEDS;
  PetscBool      simplex, perfTest, fileflg, distribute, interpolate, dmRefine, VTKdisp, vtkSoln;
  /* Domain and mesh definition */
  PetscInt       dim, numFields, overlap, qorder, level, commax;
  PetscInt	 meshSize[3];
  PetscScalar    refinementLimit;
  char           filename[2048];    /* The optional mesh file */
  char           bar[19];
  VecType	 ctype;
  /* Problem definition */
  BCType         bcType;
} AppCtx;

/* ADDITIONAL FUNCTIONS */
PetscErrorCode GeneralInfo(MPI_Comm comm, AppCtx user, PetscViewer genViewer)
{
  PetscErrorCode ierr;
  const char    *string;

  PetscFunctionBeginUser;
  ierr = PetscViewerStringSPrintf(genViewer, "Dimension of mesh:%s>%d\n", user.bar + 3, user.dim);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Number of Fields:%s>%d", user.bar + 2, user.numFields);CHKERRQ(ierr);
  if (user.numFields == 100) {
    ierr = PetscViewerStringSPrintf(genViewer, "(default)");CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Ghost point overlap:%s>%d\n", user.bar + 5, user.overlap);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "\nFile read mode:%s>%s\n", user.bar, user.fileflg ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  if (user.fileflg) {
    ierr = PetscViewerStringSPrintf(genViewer, "┗ File read name:%s>%s\n", user.bar + 2, user.filename);CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "Mesh refinement:%s>%s\n", user.bar + 1, user.dmRefine ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  if (user.dmRefine) {
    ierr = PetscViewerStringSPrintf(genViewer, "┗ Refinement level:%s>%d\n", user.bar + 4, user.level);CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "Distributed dm:%s>%s\n", user.bar, user.distribute ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Interpolated dm:%s>%s\n", user.bar + 1, user.interpolate ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Performance test mode:%s>%s\n", user.bar + 7, user.perfTest ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "PETScFE enabled mode:%s>%s\n", user.bar + 6, user.usePetscFE ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
  if (user.usePetscFE) {
    ierr = PetscViewerStringSPrintf(genViewer, "┗ Quadrature order:%s>%d\n", user.bar + 4 , user.qorder);CHKERRQ(ierr);
  }
  ierr = PetscViewerStringSPrintf(genViewer, "DM Vec Type Used:%s>%s\n", user.bar + 2, user.ctype);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "VTKoutput mode:%s>%s\n", user.bar, user.VTKdisp ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "%s General Info %s\n", user.bar + 2, user.bar + 2);CHKERRQ(ierr);
  ierr = PetscViewerStringGetStringRead(genViewer, &string, NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, string);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "%s End General Info %s\n", user.bar + 2, user.bar + 5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* GENERAL PREPROCESSING */
static PetscErrorCode ProcessOpts(MPI_Comm comm, AppCtx *options)
{
  const char            *bcTypes[3]  = {"neumann", "dirichlet", "none"};
  PetscErrorCode        ierr;
  PetscInt              bc, nmax;

  PetscFunctionBeginUser;
  options->simplex              = PETSC_FALSE;
  options->perfTest             = PETSC_FALSE;
  options->fileflg              = PETSC_FALSE;
  options->distribute           = PETSC_FALSE;
  options->interpolate          = PETSC_TRUE;
  options->dmRefine             = PETSC_FALSE;
  options->VTKdisp              = PETSC_FALSE;
  options->usePetscFE           = PETSC_FALSE;
  options->useKSP               = PETSC_FALSE;
  options->vtkSoln              = PETSC_FALSE;
  options->filename[0]          = '\0';
  options->bcType               = DIRICHLET;
  options->dim                  = 2;
  options->meshSize[0]		= 2;
  options->meshSize[1]		= 2;
  options->meshSize[2]		= 2;
  options->numFields            = 1;
  options->overlap              = 0;
  options->qorder               = -1;
  options->level                = 0;
  options->refinementLimit      = 0.0;
  options->commax               = 100;
  ierr = PetscStrncpy(options->bar, "-----------------\0", 19);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, NULL, "Speedtest Options", "");CHKERRQ(ierr); {
    ierr = PetscOptionsBool("-speed", "Streamline program to only perform necessary operations for performance testing", "", options->perfTest, &options->perfTest, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-vtkout", "enable mesh distribution visualization", "", options->VTKdisp, &options->VTKdisp, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-vtk_soln","Get solution vector in VTK output", "", options->vtkSoln, &options->vtkSoln, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL, "-f", options->filename, PETSC_MAX_PATH_LEN, &options->fileflg); CHKERRQ(ierr);

    bc   = options->bcType;
    ierr = PetscOptionsEList("-bc_type", "Type of boundary condition", "ex12.c", bcTypes, 3, bcTypes[options->bcType], &bc, NULL);CHKERRQ(ierr);
    options->bcType = (BCType) bc;

    ierr = PetscOptionsIntArray("-n", "Num faces per edge", "", options->meshSize, &nmax, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-dim", &options->dim, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-num_field", &options->numFields, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-overlap", &options->overlap, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-petscfe", "Enable only making a petscFE", "", options->usePetscFE, &options->usePetscFE, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-qorder", &options->qorder, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-refine_dm_level", &options->level, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-refine_limit", &options->refinementLimit, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-max_com", &options->commax, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (options->usePetscFE) {
    options->numFields = 1;
  }
  if (nmax > options->dim) {
    SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "nmax %d greater than dim %d", nmax, options->dim);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode        ierr;
  DM                    dmDist;
  const char            *filename = user->filename;
  PetscInt              dim = user->dim, overlap = user->overlap, i, faces[dim];
  PetscBool             hasLabel = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (user->fileflg) {
    char        *dup, filenameAlt[PETSC_MAX_PATH_LEN];
    sprintf(filenameAlt, "%s%s", "./meshes/", (dup = strdup(filename)));
    free(dup);
    ierr = PetscLogStageRegister("READ Mesh Stage", &user->stageREAD);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("READ Mesh", 0, &user->eventREAD);CHKERRQ(ierr);
    ierr = PetscLogStagePush(user->stageREAD);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(user->eventREAD, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = DMPlexCreateFromFile(comm, filenameAlt, user->interpolate, dm);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(user->eventREAD, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, user->filename);CHKERRQ(ierr);
  } else {
    for (i = 0; i < dim; i++){
      /* Make the default box mesh creation with CLI options    */
      faces[i] = user->meshSize[i];
    }
    ierr = PetscLogStageRegister("CREATE Box Mesh Stage", &user->stageCREATE);CHKERRQ(ierr);
    ierr = PetscLogEventRegister("CREATE Box Mesh", 0, &user->eventCREATE);CHKERRQ(ierr);
    ierr = PetscLogStagePush(user->stageCREATE);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(user->eventCREATE, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, faces, NULL, NULL, NULL, user->interpolate, dm);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(user->eventCREATE, 0, 0, 0, 0);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "Generated_Box_Mesh");CHKERRQ(ierr);
  }

  ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  dim = user->dim;
  if (!user->fileflg) {
    DM          dmf;
    PetscInt    level = user->level;
    PetscScalar refinementLimit = user->refinementLimit;
    if (level || refinementLimit) {
      PetscPartitioner  part;

      ierr = PetscLogStageRegister("REFINE Mesh Stage", &user->stageREFINE);CHKERRQ(ierr);
      ierr = PetscLogEventRegister("REFINE Mesh", 0, &user->eventREFINE);CHKERRQ(ierr);
      ierr = PetscLogStagePush(user->stageREFINE);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(user->eventREFINE, 0, 0, 0, 0);CHKERRQ(ierr);
      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
      if (level) {
        for (i = 0; i < level; i++) {
          ierr = DMRefine(*dm, comm, &dmf);CHKERRQ(ierr);
          if (dmf) {
            const char  *name;
            ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) dmf, name);CHKERRQ(ierr);
            ierr = DMDestroy(dm);CHKERRQ(ierr);
            *dm = dmf;
          }
          ierr = DMPlexDistribute(*dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
          if (dmDist) {
            const char  *name;
            ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) dmDist, name);CHKERRQ(ierr);
            ierr = DMDestroy(dm);CHKERRQ(ierr);
            *dm = dmDist;
            user->distribute = PETSC_TRUE;
          }
        }
      } else {
        ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
        ierr = DMRefine(*dm, comm, &dmf);CHKERRQ(ierr);
        if (dmf) {
          const char *name;

          ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) dmf, name);CHKERRQ(ierr);
          ierr = DMDestroy(dm);CHKERRQ(ierr);
          *dm  = dmf;
        }
        /* Distribute mesh over processes */
        ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
        if (dmDist) {
          const char    *name;
          ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject) dmDist, name);CHKERRQ(ierr);
          ierr = DMDestroy(dm);CHKERRQ(ierr);
          *dm  = dmDist;
        }
      }
      ierr = PetscLogEventEnd(user->eventREFINE, 0, 0, 0, 0);CHKERRQ(ierr);
      ierr = PetscLogStagePop();CHKERRQ(ierr);
      user->dmRefine = PETSC_TRUE;
    }
  }
  ierr = DMPlexDistribute(*dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    const char  *name;
    ierr = PetscObjectGetName((PetscObject) *dm, &name);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dmDist, name);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm = dmDist;
    user->distribute = PETSC_TRUE;
  }
  if (user->interpolate) {
    DM  dmInterp;
    ierr = DMPlexInterpolate(*dm, &dmInterp);CHKERRQ(ierr);
    if (dmInterp) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = dmInterp;
      user->interpolate = PETSC_TRUE;
    }
    if (user->bcType == NEUMANN) {
      DMLabel   label;
      ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
      ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    } else if (user->bcType == DIRICHLET) {
      ierr = DMHasLabel(*dm, "marker", &hasLabel);CHKERRQ(ierr);
      if (!hasLabel) {
        DMLabel label;
        ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);
        ierr = DMGetLabel(*dm, "marker", &label);CHKERRQ(ierr);
        ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
        ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);
# if defined(PETSC_HAVE_CUDA)
  ierr = DMSetVecType(*dm, VECCUDA);CHKERRQ(ierr);
# endif
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  PetscErrorCode        ierr;
  MPI_Comm              comm;
  PetscFE               fe;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("CommStagePETSCFE", &user->stagePETSCFE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommPETSCFE", 0, &user->eventPETSCFE);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stagePETSCFE);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user->eventPETSCFE, 0, 0, 0, 0);CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, user->dim, user->numFields, user->simplex, NULL, user->qorder, &fe);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->eventPETSCFE, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "LaplaceFE");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("CommStageCREATEDS", &user->stageCREATEDS);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommCREATEDS", 0, &user->eventCREATEDS);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stageCREATEDS);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user->eventCREATEDS, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->eventCREATEDS, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*      Main    */
int main(int argc, char **argv)
{
  MPI_Comm              comm;
  AppCtx                user;
  PetscErrorCode        ierr;
  PetscViewer           genViewer;
  PetscPartitioner      partitioner;
  PetscPartitionerType  partitionername;
  DM                    dm;
  IS                    globalCellNumIS, globalVertNumIS;
  Vec                   solVecLocal, solVecGlobal, VDot, dummyVecGlobal, dummyVecLocal;
  PetscInt              globalVertSize, globalCellSize, commiter;
  PetscScalar           VDotResult;
  char                  genInfo[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = PetscViewerStringOpen(comm, genInfo, sizeof(genInfo), &genViewer);CHKERRQ(ierr);

  ierr = ProcessOpts(comm, &user);CHKERRQ(ierr);
  ierr = ProcessMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);

  /* Display Mesh Partition and write mesh to vtk output file */
  if (user.VTKdisp) {
    PetscViewer vtkviewerpart, vtkviewermesh;
    Vec         partition;
    char        meshName[PETSC_MAX_PATH_LEN];

    ierr = DMPlexCreateRankField(dm, &partition);CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &vtkviewerpart);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewerpart,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vtkviewerpart,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewerpart, "partition-map.vtk");CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(vtkviewerpart,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = VecView(partition, vtkviewerpart);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewerpart);CHKERRQ(ierr);
    ierr = VecDestroy(&partition);CHKERRQ(ierr);

    if (user.fileflg) {
      char      *fileEnding, *fixedFile = 0;
      size_t    lenTotal, lenEnding;

      ierr = PetscStrlen(user.filename, &lenTotal);CHKERRQ(ierr);
      ierr = PetscStrrchr(user.filename, '.', &fileEnding);CHKERRQ(ierr);
      ierr = PetscStrlen(fileEnding, &lenEnding);CHKERRQ(ierr);
      if (lenTotal > lenEnding) {
        ierr = PetscMalloc1(lenTotal, &fixedFile);CHKERRQ(ierr);
        ierr = PetscStrncpy(fixedFile, user.filename, lenTotal-lenEnding);CHKERRQ(ierr);
      } else {
        ierr = PetscStrallocpy(user.filename, &fixedFile);CHKERRQ(ierr);
      }
      ierr = PetscStrcat(meshName, fixedFile);CHKERRQ(ierr);
      ierr = PetscFree(fixedFile);CHKERRQ(ierr);
    } else {
      char      dateStr[PETSC_MAX_PATH_LEN] = {"generated-"};
      size_t    stringlen;

      ierr = PetscStrlen(dateStr, &stringlen);CHKERRQ(ierr);
      ierr = PetscGetDate(dateStr+stringlen, 20);CHKERRQ(ierr);
      ierr = PetscStrcat(meshName, dateStr);CHKERRQ(ierr);
    }
    ierr = PetscStrcat(meshName, "-mesh.vtu");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &vtkviewermesh);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewermesh,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewermesh, meshName);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vtkviewermesh,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(vtkviewermesh,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerSetUp(vtkviewermesh);CHKERRQ(ierr);
    ierr = DMPlexVTKWriteAll((PetscObject) dm, vtkviewermesh);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewermesh);CHKERRQ(ierr);
  }

  /*    Perform setup before timing     */
  ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = VecDuplicate(solVecGlobal, &dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDuplicate(solVecLocal, &dummyVecLocal);CHKERRQ(ierr);
  ierr = VecSet(dummyVecGlobal, 0.0);CHKERRQ(ierr);
  ierr = VecSet(dummyVecLocal, 0.0);CHKERRQ(ierr);
  ierr = VecAXPY(solVecGlobal, 0.0, dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecAXPY(solVecLocal, 0.0, dummyVecLocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecGlobal);CHKERRQ(ierr);
  ierr = VecDestroy(&dummyVecLocal);CHKERRQ(ierr);

  /*    Init INSERT_VALUES timing only log      */
  ierr = PetscLogStageRegister("CommStageINSERT", &user.stageINSERT);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommINSERT", 0, &user.eventINSERT);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageINSERT);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventINSERT, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
  }
  /*    Push LocalToGlobal time to log  */
  ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user.eventINSERT, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /*    Perform setup before timing     */
  ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);

  /*    Init ADD_VALUES Log     */
  ierr = PetscLogStageRegister("CommStageADDVAL", &user.stageADD);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommADDVAL", 0, &user.eventADD);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageADD);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventADD, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = DMLocalToGlobalBegin(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
    /*  Global to Local aren't implemented      */
  }
  /*    Push time to log        */
  ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user.eventADD, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /*    Perform setup before timing     */
  ierr = DMCreateGlobalVector(dm, &VDot);CHKERRQ(ierr);
  ierr = VecSet(VDot, 1);CHKERRQ(ierr);
  ierr = VecDotBegin(VDot, VDot, &VDotResult);CHKERRQ(ierr);
  ierr = VecDotEnd(VDot, VDot, &VDotResult);CHKERRQ(ierr);

  /*    Init VecDot Log */
  ierr = PetscLogStageRegister("CommStageGlblVecDot", &user.stageGVD);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CommGlblVecDot", 0, &user.eventGVD);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user.stageGVD);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(user.eventGVD, 0, 0, 0, 0);CHKERRQ(ierr);
  for (commiter = 0; commiter < user.commax; commiter++) {
    ierr = VecDotBegin(VDot, VDot, &VDotResult);CHKERRQ(ierr);
    ierr = VecDotEnd(VDot, VDot, &VDotResult);CHKERRQ(ierr);
  }
  /*    Push time to log        */
  ierr = VecDestroy(&VDot);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user.eventGVD, 0, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /*    Output vtk of global solution vector    */
  if (user.vtkSoln) {
    PetscViewer vtkviewersoln;

    ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &vtkviewersoln);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewersoln,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vtkviewersoln,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtk");CHKERRQ(ierr);
    ierr = VecView(solVecGlobal, vtkviewersoln);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewersoln);CHKERRQ(ierr);
  }

  /*    Get Some additional data about the mesh mainly for printing */
  ierr = DMPlexGetVertexNumbering(dm, &globalVertNumIS);CHKERRQ(ierr);
  ierr = ISGetSize(globalVertNumIS, &globalVertSize);CHKERRQ(ierr);
  ierr = DMPlexGetCellNumbering(dm, &globalCellNumIS);CHKERRQ(ierr);
  ierr = ISGetSize(globalCellNumIS, &globalCellSize);CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm, &partitioner);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = PetscPartitionerGetType(partitioner, &partitionername);CHKERRQ(ierr);

  /*    Aggregate all of the information for printing   */
  ierr = PetscViewerStringSPrintf(genViewer, "Partitioner Used:%s>%s\n", user.bar + 2, partitionername);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Global Node Num:%s>%d\n", user.bar + 1, globalVertSize);CHKERRQ(ierr);
  ierr = PetscViewerStringSPrintf(genViewer, "Global Cell Num:%s>%d\n", user.bar + 1, globalCellSize);CHKERRQ(ierr);
  ierr = DMGetVecType(dm, &user.ctype);CHKERRQ(ierr);

  ierr = GeneralInfo(comm, user, genViewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&genViewer);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
