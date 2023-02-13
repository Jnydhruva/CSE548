static char help[] = "Landau collision operator with amnisotropic thermalization verification test as per Hager et al. 'A fully non-linear multi-species Fokker-Planck-Landau collision operator for simulation of fusion plasma'\n\n";

#include <petscts.h>
#include <petsclandau.h>
#include <petscdmcomposite.h>
#include <petscds.h>

/*
 call back method for DMPlexLandauAccess:

Input Parameters:
 .   dm - a DM for this field
 -   local_field - the local index in the grid for this field
 .   grid - the grid index
 +   b_id - the batch index
 -   vctx - a user context

 Input/Output Parameters:
 +   x - Vector to data to

 */
PetscErrorCode landau_field_print_access_callback(DM dm, Vec x, PetscInt local_field, PetscInt grid, PetscInt b_id, void *vctx)
{
  LandauCtx  *ctx;
  PetscScalar val;
  PetscInt    species;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &ctx));
  species = ctx->species_offset[grid] + local_field;
  val     = (PetscScalar)(LAND_PACK_IDX(b_id, grid) + (species + 1) * 10);
  PetscCall(VecSet(x, val));
  PetscCall(PetscInfo(dm, "DMPlexLandauAccess user 'add' method to grid %" PetscInt_FMT ", batch %" PetscInt_FMT " and local field %" PetscInt_FMT " with %" PetscInt_FMT " grids\n", grid, b_id, local_field, ctx->num_grids));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static const PetscReal alphai   = 1 / 1.3;
static const PetscReal kev_joul = 6.241506479963235e+15; /* 1/1000e */

// constants: [index of (anisotropic) direction of source, z x[1] shift
/* < v, n_s v_|| > */
static void f0_vz(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  f0[0] = u[0] * 2. * PETSC_PI * x[0] * x[1]; /* n r v_|| */
}
/* < v, n (v-shift)^2 > */
static void f0_v2_1d_shift(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscReal xi = PetscRealPart(constants[0]), vz = PetscRealPart(constants[1]);
  PetscInt  dir = (xi == 0) ? 0 : 1;

  if (dir == 0) vz = 0;                                              // no perp shift
  *f0 = u[0] * 2. * PETSC_PI * x[0] * (x[dir] - vz) * (x[dir] - vz); /* n r v^2_par|perp */
}
/* < v, n_e > */
static void f0_n(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  if (dim == 2) f0[0] = 2. * PETSC_PI * x[0] * u[0];
  else f0[0] = u[0];
}
static void f0_v2_shift(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscReal vz = PetscRealPart(constants[1]);
  f0[0]        = 2. * PETSC_PI * x[0] * (x[0] * x[0] + (x[1] - vz) * (x[1] - vz)) * u[0];
}
static PetscReal sign(PetscScalar x)
{
  if (PetscRealPart(x) > 0) return 1.0;
  if (PetscRealPart(x) < 0) return -1.0;
  return 0.0;
}
/* Define a Maxwellian function for testing out the operator. */
typedef struct {
  PetscReal v_0;
  PetscReal kT_m;
  PetscReal n;
  PetscReal shift;
  PetscInt  species;
} MaxwellianCtx;

static PetscErrorCode maxwellian(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  MaxwellianCtx *mctx  = (MaxwellianCtx *)actx;
  PetscReal      theta = 2 * mctx->kT_m / (mctx->v_0 * mctx->v_0); /* theta = 2kT/mc^2 */
  PetscFunctionBegin;
  /* evaluate the shifted Maxwellian */
  u[0] += alphai * mctx->n * PetscPowReal(PETSC_PI * theta, -1.5) * PetscExpReal(-(alphai * x[0] * x[0] + (x[1] - mctx->shift) * (x[1] - mctx->shift)) / theta);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetMaxwellians(DM dm, Vec X, PetscReal time, PetscReal temps[], PetscReal ns[], PetscInt grid, PetscReal shifts[], LandauCtx *ctx)
{
  PetscErrorCode (*initu[LANDAU_MAX_SPECIES])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  PetscInt       dim;
  MaxwellianCtx *mctxs[LANDAU_MAX_SPECIES], data[LANDAU_MAX_SPECIES];
  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  if (!ctx) PetscCall(DMGetApplicationContext(dm, &ctx));
  for (PetscInt ii = ctx->species_offset[grid], i0 = 0; ii < ctx->species_offset[grid + 1]; ii++, i0++) {
    mctxs[i0]        = &data[i0];
    data[i0].v_0     = ctx->v_0;                             // v_0 same for all grids
    data[i0].kT_m    = ctx->k * temps[ii] / ctx->masses[ii]; /* kT/m = v_th ^ 2*/
    data[i0].n       = ns[ii];
    initu[i0]        = maxwellian;
    data[i0].shift   = 0;
    data[i0].species = ii;
  }
  if (1) {
    data[0].shift = -sign(ctx->charges[ctx->species_offset[grid]]) * ctx->electronShift * ctx->m_0 / ctx->masses[ctx->species_offset[grid]];
  } else {
    shifts[0]     = 0.5 * PetscSqrtReal(ctx->masses[0] / ctx->masses[1]);
    shifts[1]     = 50 * (ctx->masses[0] / ctx->masses[1]);
    data[0].shift = ctx->electronShift * shifts[grid] * PetscSqrtReal(data[0].kT_m) / ctx->v_0; // shifts to not matter!!!!
  }
  PetscCall(DMProjectFunction(dm, time, initu, (void **)mctxs, INSERT_ALL_VALUES, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Monitor(TS ts, PetscInt stepi, PetscReal time, Vec X, void *actx)
{
  TSConvergedReason reason;
  LandauCtx        *ctx = (LandauCtx *)actx; /* user-defined application context */
  PetscInt          id;
  PetscReal         t;

  PetscFunctionBeginUser;
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(DMGetOutputSequenceNumber(ctx->plex[0], &id, NULL));
  if (ctx->verbose > 0) { // hacks to generate sparse data (eg, use '-dm_landau_verbose 1' and '-dm_landau_verbose -1' to get all steps printed)
    PetscInt b = PetscFloorReal(PetscLog10Real(t = (time + 1e-8) * (ctx->t_0 / 1e-4))) + 3;
    if (b >= 2) ctx->verbose = (PetscInt)PetscPowReal(10, b - 1);
    else if (b == 1) ctx->verbose = 2;
  }
  if ((ctx->verbose && stepi % ctx->verbose == 0) || reason || stepi == 1 || ctx->verbose < 0) {
    PetscInt nDMs;
    DM       pack;
    Vec     *XsubArray = NULL;
    PetscCall(TSGetDM(ts, &pack));
    PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
    PetscCall(DMSetOutputSequenceNumber(ctx->plex[0], id + 1, time));
    PetscCall(DMSetOutputSequenceNumber(ctx->plex[1], id + 1, time));
    PetscCall(PetscInfo(pack, "ex1 plot step %" PetscInt_FMT ", time = %g\n", id, (double)time));
    PetscCall(PetscMalloc(sizeof(*XsubArray) * nDMs, &XsubArray));
    PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
    PetscCall(VecViewFromOptions(XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, 0)], NULL, "-ex1_vec_view_e"));
    PetscCall(VecViewFromOptions(XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, 1)], NULL, "-ex1_vec_view_i"));
    // temps
    for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
      PetscDS     prob;
      DM          dm      = ctx->plex[grid];
      PetscScalar user[2] = {0, 0}, tt[1];
      PetscReal   vz_0 = 0, n, energy, e_perp, e_par, m_s = ctx->masses[ctx->species_offset[grid]];
      Vec         Xloc = XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, grid)];
      PetscCall(DMGetDS(dm, &prob));
      /* get n */
      PetscCall(PetscDSSetObjective(prob, 0, &f0_n));
      PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, NULL));
      n = PetscRealPart(tt[0]);
      /* get vz */
      PetscCall(PetscDSSetObjective(prob, 0, &f0_vz));
      PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, NULL));
      user[1] = vz_0 = PetscRealPart(tt[0]) / n; /* non-dimensional */
      /* energy temp */
      PetscCall(PetscDSSetConstants(prob, 2, user));
      PetscCall(PetscDSSetObjective(prob, 0, &f0_v2_shift));
      PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, ctx));
      energy = PetscRealPart(tt[0]) * ctx->v_0 * ctx->v_0 * m_s / n / 3; // scale?
      /* energy temp - perp */
      user[0] = 0; // perp
      PetscCall(PetscDSSetConstants(prob, 2, user));
      PetscCall(PetscDSSetObjective(prob, 0, &f0_v2_1d_shift));
      PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, ctx));
      e_perp = PetscRealPart(tt[0]) * ctx->v_0 * ctx->v_0 * m_s / n / 2; // scale?
      /* energy temp - par */
      user[0] = 1; // par
      PetscCall(PetscDSSetConstants(prob, 2, user));
      PetscCall(PetscDSSetObjective(prob, 0, &f0_v2_1d_shift));
      PetscCall(DMPlexComputeIntegralFEM(dm, Xloc, tt, ctx));
      e_par = PetscRealPart(tt[0]) * ctx->v_0 * ctx->v_0 * m_s / n; // scale?
      if (grid == 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "step %4d) time= %e temperature (ev): ", (int)stepi, (double)time));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s T= %9.4g T_par= %9.4g T_perp= %9.4g ", (grid == 0) ? "electron:" : ";ion:", (double)(energy * kev_joul * 1000), (double)(e_par * kev_joul * 1000), (double)(e_perp * kev_joul * 1000)));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray));
    PetscCall(PetscFree(XsubArray));

    PetscCall(DMPlexLandauPrintNorms(X, id + 1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          pack;
  Vec         X;
  PetscInt    dim = 2, nDMs;
  TS          ts;
  Mat         J;
  Vec        *XsubArray = NULL;
  LandauCtx  *ctx;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank) { /* turn off output stuff for duplicate runs */
    PetscCall(PetscOptionsClearValue(NULL, "-ex1_dm_view_e"));
    PetscCall(PetscOptionsClearValue(NULL, "-ex1_dm_view_i"));
    PetscCall(PetscOptionsClearValue(NULL, "-ex1_vec_view_e"));
    PetscCall(PetscOptionsClearValue(NULL, "-ex1_vec_view_i"));
    PetscCall(PetscOptionsClearValue(NULL, "-info"));
    PetscCall(PetscOptionsClearValue(NULL, "-snes_converged_reason"));
    PetscCall(PetscOptionsClearValue(NULL, "-pc_bjkokkos_ksp_converged_reason"));
    PetscCall(PetscOptionsClearValue(NULL, "-ksp_converged_reason"));
    PetscCall(PetscOptionsClearValue(NULL, "-ts_adapt_monitor"));
    PetscCall(PetscOptionsClearValue(NULL, "-ts_monitor"));
    PetscCall(PetscOptionsClearValue(NULL, "-snes_monitor"));
    //PetscCall(PetscOptionsClearValue(NULL, "-"));
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  /* Create a mesh */
  PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &pack));
  PetscCall(DMSetUp(pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(ctx->num_grids == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Must have two grids: use '-dm_landau_num_species_grid 1,1'");
  PetscCheck(ctx->num_species == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Must have two species: use '-dm_landau_num_species_grid 1,1'");
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
  //PetscCall(DMPlexLandauPrintNorms(X, 0));
  /* output plot names */
  PetscCall(PetscMalloc(sizeof(*XsubArray) * nDMs, &XsubArray));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
  PetscCall(PetscObjectSetName((PetscObject)XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, 0)], 0 == 0 ? "ue" : "ui"));
  PetscCall(PetscObjectSetName((PetscObject)XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, 1)], 1 == 0 ? "ue" : "ui"));
  /* add bimaxwellian anisotropic test */
  for (PetscInt b_id = 0; b_id < ctx->batch_sz; b_id++) {
    for (PetscInt grid = 0; grid < ctx->num_grids; grid++) {
      PetscReal shifts[2];
      PetscCall(SetMaxwellians(ctx->plex[grid], XsubArray[LAND_PACK_IDX(b_id, grid)], 0.0, ctx->thermal_temps, ctx->n, grid, shifts, ctx));
    }
  }
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray));
  PetscCall(PetscFree(XsubArray));
  /* plot */
  PetscCall(DMSetOutputSequenceNumber(ctx->plex[0], -1, 0.0));
  PetscCall(DMSetOutputSequenceNumber(ctx->plex[1], -1, 0.0));
  PetscCall(DMViewFromOptions(ctx->plex[0], NULL, "-ex1_dm_view_e"));
  PetscCall(DMViewFromOptions(ctx->plex[1], NULL, "-ex1_dm_view_i"));
  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetDM(ts, pack));
  PetscCall(TSSetIFunction(ts, NULL, DMPlexLandauIFunction, NULL));
  PetscCall(TSSetIJacobian(ts, J, J, DMPlexLandauIJacobian, NULL));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSMonitorSet(ts, Monitor, ctx, NULL));
  /* solve */
  PetscCall(TSSolve(ts, X));
  /* test add field method & output */
  PetscCall(DMPlexLandauAccess(pack, X, landau_field_print_access_callback, NULL));
  //PetscCall(Monitor(ts, -1, 1.0, X, ctx));
  /* clean up */
  PetscCall(DMPlexLandauDestroyVelocitySpace(&pack));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    requires: p4est !complex double defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex1_0.out
    filter: grep -v "DM"
    args: -dm_landau_amr_levels_max 0,2 -dm_landau_amr_post_refine 0 -dm_landau_amr_re_levels 2 -dm_landau_domain_radius 6,6 -dm_landau_electron_shift 1.5 -dm_landau_ion_charges 1 -dm_landau_ion_masses 2 -dm_landau_n 1,1 -dm_landau_n_0 1e20 -dm_landau_num_cells 2,4 -dm_landau_num_species_grid 1,1 -dm_landau_re_radius 2 -dm_landau_thermal_temps .3,.2 -dm_landau_type p4est -dm_landau_verbose -1 -dm_preallocate_only false -ex1_dm_view_e -ksp_type preonly -pc_type lu -petscspace_degree 3 -snes_converged_reason -snes_rtol 1.e-14 -snes_stol 1.e-14 -ts_adapt_clip .5,1.5 -ts_adapt_dt_max 5 -ts_adapt_monitor -ts_adapt_scale_solve_failed 0.5 -ts_arkimex_type 1bee -ts_dt .01 -ts_max_snes_failures -1 -ts_max_steps 1 -ts_max_time 8 -ts_monitor -ts_rtol 1e-2 -ts_type arkimex
    test:
      suffix: cpu
      args: -dm_landau_device_type cpu
    test:
      suffix: kokkos
      requires: kokkos_kernels !defined(PETSC_HAVE_CUDA_CLANG)
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos
    test:
      suffix: cuda
      requires: cuda !defined(PETSC_HAVE_CUDA_CLANG)
      args: -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda -mat_cusparse_use_cpu_solve

TEST*/
