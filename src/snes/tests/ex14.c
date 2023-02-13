static char help[] = "Finite element discretization on mesh patches.\n\n\n";

/*
  KNOWN BUGS:

  1) Coordinates created at the beginning in the ephemeral mesh. We really want to create only coordinate patches when we are asked to do so for FEGeom.

  IMPLEMENTATION:

  The corrector $C$ expresses maps the fine space $V$ into the kernel of restriction, or detail space $W$, and the complementary projector $I - C$ is a bijection between the coarse space $V_H$ and the optimized space $V_{vms}$. The example code makes a matrix $G$ whose rows are the optimized basis encoded in terms of fine space basis functions, so that it is $n \times N$. It is composed of the difference between the embedding of the original coarse basis $P_H$ and the corrector $C$, both of which have the same dimensions $n \times N$.

  We should be able to decompose this operation into projection at the element matrix level. I think we can bracket the element matrix with the transformation element matrices.

  I have implemented a special check that finds a k-cell of given height in the support of any point. This allows us to impose boundary conditions on ephemeral meshes.

*/

#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petscconvest.h>

#include <petsc/private/dmpleximpl.h>

typedef enum {PATCH_SYS_IDENTITY, PATCH_SYS_LOD} PatchSystemType;
const char *PatchSystemTypes[] = {"identity", "lod", "PatchSystemType", "PATCH_SYS_", NULL};

typedef struct {
  PatchSystemType patchSysType; // Type of patch system
  PetscBool       snesCheck;    // Check KSP solves with equivalent SNES solves
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0 * PETSC_PI * x[d]);
  return 0;
}

static PetscErrorCode const_mu(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 1.0;
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0 * PetscSqr(PETSC_PI) * PetscSinReal(2.0 * PETSC_PI * x[d]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static void f0_mu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[1] - 1.0;
}

static void g0_mumu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static void f0_id_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0];
}

static void g0_id_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->patchSysType = PATCH_SYS_IDENTITY;
  options->snesCheck    = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Mesh Patch Integration Options", "DMPLEX");
  PetscCall(PetscOptionsEnum("-patch_sys_type", "The type of patch system, e.g. LOD", NULL, PatchSystemTypes, (PetscEnum) options->patchSysType, (PetscEnum *) &options->patchSysType, NULL));
  PetscCall(PetscOptionsBool("-snes_check", "Check KSP solves with equivalent SNES", NULL, options->snesCheck, &options->snesCheck, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "orig_"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RefineMesh(DM dm, DM *rdm) {
  DMPlexTransform tr;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Transform"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, "refine_"));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));

  PetscCall(DMPlexCreateEphemeral(tr, rdm));
  PetscCall(PetscObjectSetName((PetscObject)*rdm, "Ephemeral Refined Mesh"));
  PetscCall(DMViewFromOptions(*rdm, NULL, "-dm_view"));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMPlexSetRegularRefinement(*rdm, PETSC_TRUE));
  PetscCall(DMSetCoarseDM(*rdm, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TODO: Patch should be on PETSC_COMM_SELF
static PetscErrorCode CreatePatch(DM dm, PetscInt cell, DM *patch) {
  DMPlexTransform tr;
  DMLabel         active;
  MPI_Comm        comm;
  PetscInt       *adj     = NULL;
  PetscInt        adjSize = PETSC_DETERMINE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMLabelCreate(comm, "active", &active));
  PetscCall(DMPlexGetAdjacency(dm, cell, &adjSize, &adj));
  for (PetscInt a = 0; a < adjSize; ++a) PetscCall(DMLabelSetValue(active, adj[a], DM_ADAPT_REFINE));
  PetscCall(PetscObjectViewFromOptions((PetscObject)active, NULL, "-active_view"));
  PetscCall(PetscFree(adj));

  PetscCall(DMPlexTransformCreate(comm, &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Transform"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, "select_"));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetActive(tr, active));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));
  PetscCall(DMLabelDestroy(&active));

  PetscCall(DMPlexCreateEphemeral(tr, patch));
  PetscCall(PetscObjectSetName((PetscObject)*patch, "Ephemeral Patch"));
  PetscCall(DMViewFromOptions(*patch, NULL, "-patch_view"));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMPlexSetLocationAlg(*patch, DM_PLEX_LOCATE_GRID_HASH));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RefinePatch(DM patch, DM *rpatch) {
  DMPlexTransform tr;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)patch), &tr));
  PetscCall(PetscObjectSetName((PetscObject)tr, "Transform"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tr, "refine_"));
  PetscCall(DMPlexTransformSetDM(tr, patch));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));

  PetscCall(DMPlexCreateEphemeral(tr, rpatch));
  PetscCall(PetscObjectSetName((PetscObject)*rpatch, "Ephemeral Refined Patch"));
  PetscCall(DMViewFromOptions(*rpatch, NULL, "-patch_view"));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;
  PetscInt       Nf;

  PetscFunctionBeginUser;
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_trig_u, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, trig_u, user));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))trig_u, NULL, user, NULL));
  if (Nf > 1) {
    PetscCall(PetscDSSetResidual(ds, 1, f0_mu, NULL));
    PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_mumu, NULL, NULL, NULL));
    PetscCall(PetscDSSetExactSolution(ds, 1, const_mu, user));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, PetscInt Nf, const char *names[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  for (PetscInt f = 0; f < Nf; ++f) {
    PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", names[f]));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, prefix, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, names[f]));
    PetscCall(DMSetField(dm, f, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPatchProblem_Identity(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_id_phi, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_id_phi, NULL, NULL, NULL));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))zero, NULL, user, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPatchProblem_LOD(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_trig_u, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, trig_u, user));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))zero, NULL, user, NULL));
  PetscCall(PetscDSSetResidual(ds, 1, f0_mu, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_mumu, NULL, NULL, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 1, const_mu, user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPatchProblem(DM patch, AppCtx *user)
{
  const char *names[] = {"phi", "mu"};

  PetscFunctionBegin;
  switch (user->patchSysType) {
  case PATCH_SYS_IDENTITY:
    PetscCall(SetupDiscretization(patch, 1, names, SetupPatchProblem_Identity, user));
    break;
  case PATCH_SYS_LOD:
    PetscCall(SetupDiscretization(patch, 2, names, SetupPatchProblem_LOD, user));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode PrintMatSetValues(PetscViewer viewer, Mat A, PetscInt point, PetscInt numRIndices, const PetscInt rindices[], PetscInt numCIndices, const PetscInt cindices[], const PetscScalar values[])
{
  PetscMPIInt rank;
  PetscInt    i, j;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  PetscCall(PetscViewerASCIIPrintf(viewer, "[%d]mat for point %" PetscInt_FMT "\n", rank, point));
  for (i = 0; i < numRIndices; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "[%d]mat row indices[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, i, rindices[i]));
  for (i = 0; i < numCIndices; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "[%d]mat col indices[%" PetscInt_FMT "] = %" PetscInt_FMT "\n", rank, i, cindices[i]));
  numCIndices = numCIndices ? numCIndices : numRIndices;
  if (!values) PetscFunctionReturn(PETSC_SUCCESS);
  for (i = 0; i < numRIndices; i++) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "[%d]", rank));
    for (j = 0; j < numCIndices; j++) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(viewer, " (%g,%g)", (double)PetscRealPart(values[i * numCIndices + j]), (double)PetscImaginaryPart(values[i * numCIndices + j])));
#else
      PetscCall(PetscViewerASCIIPrintf(viewer, " %g", (double)values[i * numCIndices + j]));
#endif
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PatchSolve - Solve the saddle-point system on the refined patch and inject the results into the corrector

  Input Parameters:
+ patch  - The patch from the coarse grid
. c      - The central cell for this patch
. rpatch - The patch from the fine grid
- user   - A user context

  Note:
  Coarse indices are those for the closure of the original seed cell. Fine indices are those for the closure of the entire refined patch, so we just indicate the whole section

  Level; advanced

.seealso: `CreatePatch()`
*/
static PetscErrorCode PatchSolve(DM dm, DM patch, PetscInt c, DM rdm, DM rpatch, Mat dP, AppCtx *user)
{
  const PetscInt  debug = 0;
  SNES            snes;
  Mat             P, M;
  Vec             u, b, psi, cpsi;
  IS              subpIS;
  PetscScalar    *elemP;
  PetscSection    s, sRef, gs, gsRef, dgs, dgsRef;
  const PetscInt *points;
  PetscInt       *closure = NULL, *rows, *cols;
  PetscInt        cell, pStart, pEnd, Ncl, Nfine = 0, Ncoarse = 0, j = 0;

  PetscFunctionBegin;
  PetscCall(SetupPatchProblem(patch, user));
  PetscCall(SetupPatchProblem(rpatch, user));
  {
    PetscInt n;

    PetscCall(DMPlexGetSubpointIS(patch, &subpIS));
    PetscCheck(subpIS, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Input patch mesh has no connection to the domain mesh (subpoint IS)");
    PetscCall(PetscObjectViewFromOptions((PetscObject)subpIS, NULL, "-subpoint_is_view"));
    PetscCall(ISGetLocalSize(subpIS, &n));
    PetscCall(ISGetIndices(subpIS, &points));
    for (cell = 0; cell < n; ++cell) if (points[cell] == c) break;
    PetscCheck(cell < n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " not found in patch mesh", c);
    PetscCall(ISRestoreIndices(subpIS, &points));
  }
  PetscCall(DMGetLocalSection(patch, &s));
  PetscCall(DMGetGlobalSection(patch, &gs));
  PetscCall(DMGetLocalSection(rpatch, &sRef));
  PetscCall(DMGetGlobalSection(rpatch, &gsRef));

  PetscCall(PetscSectionGetChart(gsRef, &pStart, &pEnd));
  for (PetscInt p = pStart, dof, cdof; p < pEnd; ++p) {
    PetscCall(PetscSectionGetFieldConstraintDof(sRef, p, 0, &cdof));
    PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
    Nfine += !cdof && dof > 0 ? dof : 0;
  }
  PetscCall(DMPlexGetTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  for (PetscInt cl = 0, dof, cdof; cl < Ncl*2; cl += 2) {
    PetscCall(PetscSectionGetConstraintDof(s, closure[cl], &cdof));
    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    Ncoarse += !cdof && dof > 0 ? dof : 0;
  }
  PetscCall(PetscCalloc3(Nfine, &rows, Ncoarse, &cols, Nfine * Ncoarse, &elemP));
  PetscCall(DMGetGlobalSection(dm, &dgs));
  PetscCall(DMGetGlobalSection(rdm, &dgsRef));
  PetscCall(ISGetIndices(subpIS, &points));
  PetscCall(PetscSectionGetChart(gsRef, &pStart, &pEnd));
  for (PetscInt p = pStart, i = 0; p < pEnd; ++p) {
    DMPolytopeType cct;  // Celltype for parrent point in coarse patch
    DMPolytopeType ct;   // Celltype for point p in refined patch
    PetscInt       cp;   // Parent point in coarse patch
    PetscInt       r;    // Replica number for point p in refined patch
    PetscInt       dcp;  // Parent point in coarse domain
    PetscInt       dp;   // Point p in refined domain
    PetscInt       doff; // Offset in global vector for refined domain
    PetscInt       dof, cdof;

    PetscCall(PetscSectionGetFieldConstraintDof(sRef, p, 0, &cdof));
    PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
    // Get parent point in coarse patch for this point in refined patch
    PetscCall(DMPlexTransformGetSourcePoint(((DM_Plex*)rpatch->data)->tr, p, &cct, &ct, &cp, &r));
    // Convert parent point in coarse patch to point in coarse domain
    dcp = points[cp];
    // Get points produced by domain point
    PetscCall(DMPlexTransformGetTargetPoint(((DM_Plex*)rdm->data)->tr, cct, ct, dcp, r, &dp));
    PetscCall(PetscSectionGetFieldOffset(dgsRef, dp, 0, &doff));
    for (PetscInt d = 0; d < (cdof ? 0 : dof); ++d) rows[i++] = doff + d;
  }
  for (PetscInt cl = 0, i = 0; cl < Ncl*2; cl += 2) {
    PetscInt dof, cdof, doff;

    PetscCall(PetscSectionGetFieldConstraintDof(s, closure[cl], 0, &cdof));
    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    PetscCall(PetscSectionGetFieldOffset(dgs, points[closure[cl]], 0, &doff));
    for (PetscInt d = 0; d < (cdof ? 0 : dof); ++d) cols[i++] = doff + d;
  }
  PetscCall(ISRestoreIndices(subpIS, &points));
  PetscCall(DMPlexRestoreTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  PetscCall(DMCreateInterpolation(patch, rpatch, &P, NULL));

  PetscCall(SNESCreate(PetscObjectComm((PetscObject)rpatch), &snes));
  PetscCall(SNESSetDM(snes, rpatch));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMPlexSetSNESLocalFEM(rpatch, user, user, user));

  PetscCall(DMCreateMassMatrix(rpatch, rpatch, &M));
  PetscCall(DMGetGlobalVector(rpatch, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "potential"));
  PetscCall(DMGetGlobalVector(rpatch, &b));
  PetscCall(PetscObjectSetName((PetscObject)b, "rhs"));
  PetscCall(DMGetGlobalVector(rpatch, &psi));
  PetscCall(PetscObjectSetName((PetscObject)psi, "fine basis"));
  PetscCall(DMGetGlobalVector(patch, &cpsi));
  PetscCall(PetscObjectSetName((PetscObject)cpsi, "coarse basis"));
  PetscCall(VecZeroEntries(cpsi));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(DMPlexGetTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  for (PetscInt cl = 0; cl < Ncl*2; cl += 2) {
    const PetscScalar *a;
    PetscInt           dof, cdof, off;

    PetscCall(PetscSectionGetConstraintDof(s, closure[cl], &cdof));
    PetscCall(PetscSectionGetFieldDof(gs, closure[cl], 0, &dof));
    PetscCall(PetscSectionGetFieldOffset(gs, closure[cl], 0, &off));
    for (PetscInt d = 0; d < (cdof ? 0 : dof); ++d, ++j) {
      PetscCall(VecSetValue(cpsi, off+d, 1., INSERT_VALUES));
      PetscCall(MatMult(P, cpsi, psi));
      // We need the weak rhs, not the coefficients
      PetscCall(MatMult(M, psi, b));
      PetscCall(SNESSolve(snes, b, u));
      PetscCall(VecSetValue(cpsi, off+d, 0., INSERT_VALUES));
      // Insert values into element matrix
      PetscCall(VecGetArrayRead(u, &a));
      // Copy out first field
      for (PetscInt p = pStart; p < pEnd; ++p) {
        PetscInt dof, cdof, off;

        PetscCall(PetscSectionGetFieldConstraintDof(sRef, p, 0, &cdof));
        PetscCall(PetscSectionGetFieldDof(gsRef, p, 0, &dof));
        PetscCall(PetscSectionGetFieldOffset(gsRef, p, 0, &off));
        for (PetscInt d = 0; d < (cdof > 0 ? 0 : dof); ++d) {
          if (PetscAbsScalar(a[off + d]) > PETSC_SMALL) elemP[(off + d) * Ncoarse + j] = a[off + d];
        }
      }
      PetscCall(VecRestoreArrayRead(u, &a));
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(patch, cell, PETSC_TRUE, &Ncl, &closure));
  PetscCheck(j == Ncoarse, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of columns %" PetscInt_FMT " != %" PetscInt_FMT " coarse space size", j, Ncoarse);
  if (debug) PetscCall(PrintMatSetValues(PETSC_VIEWER_STDOUT_SELF, dP, c, Nfine, rows, Ncoarse, cols, elemP));
  PetscCall(MatSetValues(dP, Nfine, rows, Ncoarse, cols, elemP, INSERT_VALUES));
  PetscCall(DMRestoreGlobalVector(rpatch, &u));
  PetscCall(DMRestoreGlobalVector(rpatch, &b));
  PetscCall(DMRestoreGlobalVector(rpatch, &psi));
  PetscCall(DMRestoreGlobalVector(patch, &cpsi));
  PetscCall(MatDestroy(&M));
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&P));
  PetscCall(PetscFree3(rows, cols, elemP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// u is the KSP solution
static PetscErrorCode SolveSystem_SNES(DM dm, Vec u)
{
  SNES      snes;
  Vec       uTmp;
  PetscReal nrm;
  AppCtx    *user;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, (void **)&user));
  if (!user->snesCheck) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetGlobalVector(dm, &uTmp));
  PetscCall(PetscObjectSetName((PetscObject)uTmp, "Coarse SNES Solution"));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dm), &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "check_"));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SNESSetErrorIfNotConverged(snes, PETSC_TRUE));
  PetscCall(SNESSetTolerances(snes, 10*PETSC_MACHINE_EPSILON, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMPlexSetSNESLocalFEM(dm, user, user, user));
  PetscCall(DMSNESCheckFromOptions(snes, uTmp));
  PetscCall(SNESSolve(snes, NULL, uTmp));
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecAXPY(uTmp, -1., u));
  PetscCall(VecNorm(uTmp, NORM_INFINITY, &nrm));
  PetscCall(DMRestoreGlobalVector(dm, &uTmp));
  PetscCheck(nrm < PETSC_SMALL, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "SNES Solution does not match KSP solution");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveSystems(DM dm, DM rdm, Mat P)
{
  PetscSimplePointFunc exacts[1] = {trig_u};
  KSP                  ksp, kspRef, kspRed;
  Mat                  A, Aref, Ared;
  Vec                  u, b, uref, bref, ured, bred;
  PetscReal            err;

  PetscFunctionBegin;
  // Create original coarse system
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "Coarse Solution"));
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(PetscObjectSetName((PetscObject)b, "Coarse Rhs"));
  {
    Vec ul, bl;

    PetscCall(VecSet(b, 0.));
    PetscCall(DMGetLocalVector(dm, &ul));
    PetscCall(DMGetLocalVector(dm, &bl));
    PetscCall(VecSet(ul, 0.));
    PetscCall(DMPlexSNESComputeBoundaryFEM(dm, ul, NULL));
    PetscCall(DMPlexSNESComputeResidualFEM(dm, ul, bl, NULL));
    PetscCall(VecScale(bl, -1.));
    PetscCall(DMLocalToGlobal(dm, bl, ADD_VALUES, b));
    PetscCall(DMRestoreLocalVector(dm, &ul));
    PetscCall(DMRestoreLocalVector(dm, &bl));
  }
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(DMPlexSNESComputeJacobianFEM(dm, u, A, A, NULL));
  PetscCall(MatViewFromOptions(A, NULL, "-coarse_mat_view"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, u));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(DMComputeL2Diff(dm, 0.0, exacts, NULL, u, &err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coarse L_2 Error: %g\n", (double)err));
  PetscCall(SolveSystem_SNES(dm, u));
  PetscCall(VecDestroy(&u));
  // Create refined system
  PetscCall(DMCreateGlobalVector(rdm, &uref));
  PetscCall(DMCreateGlobalVector(rdm, &bref));
  PetscCall(VecSet(uref, 0.));
  {
    Vec ul, bl;

    PetscCall(VecSet(bref, 0.));
    PetscCall(DMGetLocalVector(rdm, &ul));
    PetscCall(DMGetLocalVector(rdm, &bl));
    PetscCall(VecSet(ul, 0.));
    PetscCall(DMPlexSNESComputeBoundaryFEM(rdm, ul, NULL));
    PetscCall(DMPlexSNESComputeResidualFEM(rdm, ul, bl, NULL));
    PetscCall(VecScale(bl, -1.));
    PetscCall(DMLocalToGlobal(rdm, bl, ADD_VALUES, bref));
    PetscCall(DMRestoreLocalVector(rdm, &ul));
    PetscCall(DMRestoreLocalVector(rdm, &bl));
  }
  PetscCall(DMCreateMatrix(rdm, &Aref));
  PetscCall(DMPlexSNESComputeJacobianFEM(rdm, uref, Aref, Aref, NULL));
  PetscCall(MatViewFromOptions(Aref, NULL, "-fine_mat_view"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &kspRef));
  PetscCall(KSPSetOperators(kspRef, Aref, Aref));
  PetscCall(KSPSetFromOptions(kspRef));
  PetscCall(KSPSolve(kspRef, bref, uref));
  PetscCall(KSPDestroy(&kspRef));
  PetscCall(DMComputeL2Diff(rdm, 0.0, exacts, NULL, uref, &err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Refined L_2 Error: %g\n", (double)err));
  PetscCall(VecDestroy(&uref));
  // Create reduced system
  if (P) {
    PetscCall(DMCreateGlobalVector(dm, &ured));
    PetscCall(PetscObjectSetName((PetscObject)ured, "Reduced Solution"));
    PetscCall(DMCreateGlobalVector(dm, &bred));
    PetscCall(PetscObjectSetName((PetscObject)bred, "Reduced Rhs"));
    PetscCall(MatMultTranspose(P, bref, bred));
    PetscCall(MatPtAP(Aref, P, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Ared));
    PetscCall(MatViewFromOptions(Ared, NULL, "-reduced_mat_view"));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &kspRed));
    PetscCall(KSPSetOperators(kspRed, Ared, Ared));
    PetscCall(KSPSetFromOptions(kspRed));
    PetscCall(KSPSolve(kspRed, bred, ured));
    PetscCall(KSPDestroy(&kspRed));
    PetscCall(MatDestroy(&Ared));
    PetscCall(DMComputeL2Diff(dm, 0.0, exacts, NULL, ured, &err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reduced L_2 Error: %g\n", (double)err));
    PetscCall(VecDestroy(&ured));
    PetscCall(VecDestroy(&bred));
  }
  PetscCall(MatDestroy(&Aref));
  PetscCall(VecDestroy(&bref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm, rdm;
  Mat         P = NULL;
  PetscInt    cStart, cEnd;
  PetscBool   useCone, useClosure;
  AppCtx      user;
  const char *names[1] = {"phi"};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(RefineMesh(dm, &rdm));
  PetscCall(SetupDiscretization(dm, 1, names, SetupPrimalProblem, &user));
  PetscCall(SetupDiscretization(rdm, 1, names, SetupPrimalProblem, &user));
  //PetscCall(SolveSystems(dm, rdm, P));
  //goto end;
  // Create global prolongator
  {
    PetscSection gs, rgs;
    PetscInt     m, n;

    PetscCall(DMGetGlobalSection(dm, &gs));
    PetscCall(DMGetGlobalSection(rdm, &rgs));
    PetscCall(PetscSectionGetConstrainedStorageSize(rgs, &m));
    PetscCall(PetscSectionGetConstrainedStorageSize(gs, &n));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &P));
    PetscCall(MatSetSizes(P, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetUp(P));
    PetscCall(MatSetOption(P, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
    PetscCall(MatSetOption(P, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  }
  // Make a patch for each cell
  //   TODO: Just need a patch covering each dof
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_TRUE));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    DM patch, rpatch;

    PetscCall(CreatePatch(dm, c, &patch));
    PetscCall(RefinePatch(patch, &rpatch));
    PetscCall(PatchSolve(dm, patch, c, rdm, rpatch, P, &user));
    PetscCall(DMDestroy(&rpatch));
    PetscCall(DMDestroy(&patch));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscCall(DMSetBasicAdjacency(dm, useCone, useClosure));
  PetscCall(SolveSystems(dm, rdm, P));
  if (user.patchSysType == PATCH_SYS_IDENTITY) {
    Mat       gP;
    PetscReal nrm, gnrm;

    PetscCall(DMCreateInterpolation(dm, rdm, &gP, NULL));
    PetscCall(MatAXPY(P, -1.0, gP, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(P, NORM_FROBENIUS, &nrm));
    PetscCall(MatNorm(gP, NORM_FROBENIUS, &gnrm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ||Ph - P||_F/||P||_F = %g, ||Ph - P||_F = %g\n", (double)(nrm / gnrm), (double)nrm));
    PetscCheck(nrm < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Identity prolongator does not match canonical computation");
    PetscCall(MatDestroy(&gP));
  }
  end:
  PetscCall(MatDestroy(&P));
  PetscCall(DMDestroy(&rdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -select_dm_plex_transform_type transform_filter \
          -patch_sys_type lod -phi_petscspace_degree 1 -mu_petscspace_degree 1 -pc_type lu \
          -snes_converged_reason -snes_monitor

  test:
    suffix: check_id_0
    args: -select_dm_plex_transform_type transform_filter \
          -patch_sys_type identity -phi_petscspace_degree 1 -pc_type lu -snes_check \
          -snes_error_if_not_converged -ksp_error_if_not_converged -snes_converged_reason -snes_monitor \
            -ksp_rtol 1e-10

  # With -orig_dm_refine 1, b_red is not exactly b_coarse
  test:
    suffix: check_id_1
    args: -orig_dm_refine 1 -select_dm_plex_transform_type transform_filter \
          -patch_sys_type identity -phi_petscspace_degree 1 -pc_type lu -snes_check \
          -snes_error_if_not_converged -ksp_error_if_not_converged -snes_converged_reason -snes_monitor \
            -ksp_rtol 1e-10

  test:
    suffix: check_lod
    args: -select_dm_plex_transform_type transform_filter \
          -patch_sys_type lod -phi_petscspace_degree 1 -pc_type lu \
          -snes_error_if_not_converged -ksp_error_if_not_converged -snes_converged_reason -snes_monitor

TEST*/
