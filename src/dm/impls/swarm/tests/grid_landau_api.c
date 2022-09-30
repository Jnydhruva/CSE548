static char help[] = "Particle interface for 2V grid-based Landau advance: create a particle distribution on circle or torus, AMR refine to domain, migrating particles to grid distribution, create particle lists on each cell, deposit to Landau grid, Landau advance, pseudo-inverse remap, migrate back to original ordering\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>
#include <petscdmforest.h>

#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

/*
 Cylindrical: (psi,theta,phi)
 Cartesian: (X,Y,Z) with Y == z; X = cos(phi)*(R_maj + r), Z = sine(phi)*(R_maj + r)
 : (psi,theta,phi)
 */

/* coordinate transformation - simple radial coordinates. Not really cylindrical as r_Minor is radius from plane axis */
#define XYToPsiTheta(__x,__y,__psi,__theta) {                           \
    __psi = PetscSqrtReal((__x)*(__x) + (__y)*(__y));                   \
    if (PetscAbsReal(__psi) < PETSC_SQRT_MACHINE_EPSILON) __theta = 0.; \
    else {                                                              \
      __theta = (__y) > 0. ? PetscAsinReal((__y)/__psi) : -PetscAsinReal(-(__y)/__psi); \
      if ((__x) < 0) __theta = PETSC_PI - __theta;                      \
      else if (__theta < 0.) __theta = __theta + 2.*PETSC_PI;           \
    }                                                                   \
  }

/* q: safty factor */
#define qsafty(__psi) (3.*pow(__psi,2.0))

#define CylToRZ( __psi, __theta, __r, __z) {            \
    __r = (__psi)*PetscCosReal(__theta);		\
    __z = (__psi)*PetscSinReal(__theta);                \
  }

// store Cartesian (X,Y,Z) for plotting 3D, (X,Y) for 2D
// (psi,theta,phi) --> (X,Y,Z)
#define cylToCart( __R_0, __psi,  __theta,  __phi, __cart)       \
  { PetscReal __R = (__R_0) + (__psi)*PetscCosReal(__theta);            \
    __cart[0] = __R*PetscCosReal(__phi);                                \
    __cart[1] = __psi*PetscSinReal(__theta);				\
    __cart[2] = -__R*PetscSinReal(__phi);       \
  }

#define CartTocyl2D(__R_0, __R, __cart, __psi,  __theta) {              \
    __R = __cart[0];                                                    \
    XYToPsiTheta(__R - __R_0, __cart[1], __psi, __theta);               \
  }

#define CartTocyl3D( __R_0, __R, __cart, __psi,  __theta,  __phi) { \
    __R = PetscSqrtReal(__cart[0]*__cart[0] + __cart[2]*__cart[2]);  \
    if (__cart[2] < 0.0) __phi =               PetscAcosReal(__cart[0]/__R);\
    else                 __phi = 2.*PETSC_PI - PetscAcosReal(__cart[0]/__R); \
    XYToPsiTheta(__R - __R_0, __cart[1], __psi, __theta);                 \
  }

// create DMs with command line options and register particle fields
static PetscErrorCode InitPlex(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  /* Get base DM from command line */
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm)); // seems to create a 2x2 mesh by default
  PetscFunctionReturn(0);
}
/* Init Swarm */
static PetscErrorCode InitSwarm(MPI_Comm comm, DM dm, DM *sw)
{
  PetscInt dim;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(comm, sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "vpar", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view_orig"));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt dim;
  PetscInt particles_per_point;  // number of particels in velocity space at each 'point'
  PetscInt n_plane_points_proc; // aproximate spatial 'points' on r,z plane / proc
  /* MPI parallel data */
  PetscMPIInt   rank,npe,particlePlaneRank,ParticlePlaneIdx; // MPI sizes and ranks
  PetscInt  steps;                            /* TS iterations */
  PetscReal stepSize;                         /* Time stepper step size */
  /* Grid */
  /* particle processor grid size */
  PetscInt np_radius;
  PetscInt np_theta;
  PetscInt np_phi; /* toroidal direction */
  /* torus geometry  */
  PetscReal  R;
  PetscReal  r;
  PetscReal  r_inflate;
  PetscReal  torus_section_rad;
} PartDDCtx;

/* Simple shift to origin */
static PetscErrorCode OriginShift2D(MPI_Comm comm, DM dm, PartDDCtx *ctx)
{
  Vec             coordinates;
  PetscScalar    *coords;
  PetscInt N;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=2) {
    PetscScalar *v = &coords[ii];
    v[0] *= 2*ctx->r*ctx->r_inflate;
    v[1] *= 2*ctx->r*ctx->r_inflate;
    v[0] -= ctx->r*ctx->r_inflate;
    v[1] -= ctx->r*ctx->r_inflate;
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

/* Extrude 2D Plex to 3D Plex */
static PetscErrorCode ExtrudeTorus(MPI_Comm comm, DM *dm, PartDDCtx *ctx)
{
  DM dmtorus;
  PetscReal L;
  //DMBoundaryType periodicity[] = {DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_PERIODIC};
  Vec             coordinates;
  PetscScalar    *coords, R_0 = ctx->R;
  PetscInt N,dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(*dm, &dim)); // probably 2
  PetscCheck(dim==2, PetscObjectComm((PetscObject)*dm), PETSC_ERR_ARG_WRONG,"DM dim (%d) != 2 for extruding",(int)dim);
  PetscCall(DMGetCoordinatesLocal(*dm, &coordinates));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=2) {
    PetscScalar *v = &coords[ii];
    v[0] *= 2*ctx->r*ctx->r_inflate;
    v[1] *= 2*ctx->r*ctx->r_inflate;
    v[0] += R_0 - ctx->r*ctx->r_inflate;
    v[1] +=     - ctx->r*ctx->r_inflate;
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  //
  L = ctx->torus_section_rad*ctx->R;
  // we could create a box mesh here but Plex starts with a 2x2 so we can just dm_refine from there, for now
  PetscCall(DMPlexExtrude(*dm, ctx->np_phi, L, PETSC_FALSE, PETSC_FALSE, NULL, NULL, &dmtorus));
  PetscCall(DMDestroy(dm));
  *dm = dmtorus;
  PetscCall(DMGetDimension(*dm, &dim));
  PetscCheck(dim==3, PetscObjectComm((PetscObject)*dm), PETSC_ERR_ARG_WRONG,"DM dim (%d) != 3 after extruding",(int)dim);
  // wrap around torus axis
  PetscCall(DMGetCoordinatesLocal(*dm, &coordinates));
  PetscCall(VecGetSize(coordinates, &N));
  PetscCall(VecGetArrayWrite(coordinates, &coords));
  // shift coordinates to center on (R,0). Assume the domain is (0,1)^2
  for (int ii=0;ii<N;ii+=3) {
    PetscScalar *v = &coords[ii], theta, psi, R;
    CartTocyl2D(R_0, R, v, psi, theta);
    PetscReal X = R_0 + psi*PetscCosReal(theta), Z = v[2], phi = Z/R_0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\t\t[%d] ExtrudeTorus %d) psi=%12.4e theta=%12.4e phi=%12.4e. R=%12.4e Cart=%12.4e,%12.4e,%12.4e. X=%12.4e Z=%12.4e ",ctx->rank, ii/3,  psi, theta, phi, R, v[0], v[1], v[2], X, Z));
    cylToCart( R_0, psi, theta, phi, v);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "--> X = %12.4e,%12.4e,%12.4e \n", v[0], v[1], v[2]));
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view_orig"));
  // set periodic - TODO
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, PartDDCtx *ctx)
{
  PetscBool phiFlag,radFlag,thetaFlag;
  PetscFunctionBeginUser;
  ctx->particles_per_point = 1; // 10
  ctx->dim = 2;
  ctx->n_plane_points_proc = 16; // 4x4 grid
  ctx->steps            = 1;
  ctx->stepSize         = 1;
  ctx->torus_section_rad = 360;
  /* mesh */
  ctx->R = 5.;
  ctx->r = 1.;
  ctx->r_inflate = 1.1;
  ctx->np_phi  = 1;
  ctx->np_radius = 1;
  ctx->np_theta  = 1;
  PetscCallMPI(MPI_Comm_rank(comm, &ctx->rank));
  PetscCallMPI(MPI_Comm_size(comm, &ctx->npe));

  PetscOptionsBegin(comm, "", "grid-based Landau particle interface", "DMPLEX");
  PetscCall(PetscOptionsInt("-dim", "parameter", "ex99.c", ctx->dim, &ctx->dim, PETSC_NULL));
  if (ctx->dim==3) PetscCall(PetscOptionsReal("-torus_section_degree_todo", "360 for entire torus", "ex99.c", ctx->torus_section_rad, &ctx->torus_section_rad, PETSC_NULL));
  else ctx->torus_section_rad = 0;
  ctx->torus_section_rad *= PETSC_PI/180.; // get into radians
  PetscCheck(ctx->dim==2 || ctx->dim==3, comm,PETSC_ERR_ARG_WRONG,"dim (%d) != 2 or 3",(int)ctx->dim);
  if (ctx->dim==3) {
    ctx->np_phi = 4;
    PetscCall(PetscOptionsInt("-np_phi", "Number of planes for particle mesh", "ex99.c", ctx->np_phi, &ctx->np_phi, &phiFlag));
  }
  else { ctx->np_phi = 1; phiFlag = PETSC_TRUE;} // == 1
  PetscCheck(ctx->dim==2 || ctx->np_phi > 2, comm,PETSC_ERR_ARG_WRONG,"num particle planes 'np_phi' (%d) > 2 in 3D",(int)ctx->np_phi);
  PetscCall(PetscOptionsInt("-np_radius", "Number of radial cells for particle mesh", "ex99.c", ctx->np_radius, &ctx->np_radius, &radFlag));
  PetscCall(PetscOptionsInt("-np_theta", "Number of theta cells for particle mesh", "ex99.c", ctx->np_theta, &ctx->np_theta, &thetaFlag));
  /* particle grids: <= npe, <= num solver planes */
  PetscCheck(ctx->npe >= ctx->np_phi, comm,PETSC_ERR_ARG_WRONG,"num particle planes np_phi (%d) > npe (%d)",(int)ctx->np_phi,ctx->npe);

  if (ctx->np_phi*ctx->np_radius*ctx->np_theta != ctx->npe) { /* recover from inconsistant grid/procs */
    PetscCheck(thetaFlag || radFlag || phiFlag,comm,PETSC_ERR_USER,"over constrained number of particle processes npe (%d) != %d",(int)ctx->npe,(int)ctx->np_phi*ctx->np_radius*ctx->np_theta);
    if (!thetaFlag && radFlag && phiFlag) ctx->np_theta = ctx->npe/(ctx->np_phi*ctx->np_radius);
    else if (thetaFlag && !radFlag && phiFlag) ctx->np_radius = ctx->npe/(ctx->np_phi*ctx->np_theta);
    else if (thetaFlag && radFlag && !phiFlag && ctx->dim==2) ctx->np_phi = ctx->npe/(ctx->np_radius*ctx->np_theta);
    else if (!thetaFlag && !radFlag && !phiFlag) {
      PetscInt npe_plane = (int)pow((double)ctx->npe,0.6667);
      ctx->np_phi = ctx->npe/npe_plane;
      ctx->np_radius = (PetscInt)(PetscSqrtReal((double)npe_plane)+0.5);
      ctx->np_theta = npe_plane/ctx->np_radius;
    }
    else if (!thetaFlag && !radFlag) {
      PetscInt npe_plane = ctx->npe/ctx->np_phi;
      ctx->np_radius = (int)(PetscSqrtReal((double)npe_plane)+0.5);
      ctx->np_theta = npe_plane/ctx->np_radius;
    }
  }
  PetscCheck(ctx->np_phi*ctx->np_radius*ctx->np_theta==ctx->npe,comm,PETSC_ERR_USER,"failed to recover npe=%d != %d",(int)ctx->npe,(int)ctx->np_phi*ctx->np_radius*ctx->np_theta);
  PetscCall(PetscOptionsInt("-particles_per_point", "Number of particles per spatial cell", "ex99.c", ctx->particles_per_point, &ctx->particles_per_point, NULL));
  PetscCall(PetscOptionsInt("-n_plane_points_proc", "parameter", "ex99.c", ctx->n_plane_points_proc, &ctx->n_plane_points_proc, PETSC_NULL));
  PetscCall(PetscOptionsInt("-steps", "Steps to take", "ex99.c", ctx->steps, &ctx->steps, PETSC_NULL));
  PetscCall(PetscOptionsReal("-dt", "dt", "ex99.c", ctx->stepSize, &ctx->stepSize, PETSC_NULL));
  /* Domain and mesh definition */
  PetscCall(PetscOptionsReal("-radius_minor", "Minor radius of torus", "ex99.c", ctx->r, &ctx->r, NULL));
  PetscCall(PetscOptionsReal("-radius_major", "Major radius of torus", "ex99.c", ctx->R, &ctx->R, NULL));
  PetscCall(PetscOptionsReal("-radius_inflation", "inflate domain factor from minor radius", "ex99.c", ctx->r_inflate, &ctx->r_inflate, NULL));

  PetscOptionsEnd();
  /* derived */
  PetscCheck(ctx->npe%ctx->np_phi==0,comm,PETSC_ERR_USER,"ctx->npe mod ctx->np_phi!=0 npe=%d != %d",(int)ctx->npe,(int)ctx->np_phi);
  PetscCheck((ctx->npe/ctx->np_phi)/ctx->np_radius == ctx->np_theta,comm,PETSC_ERR_USER,"ctx->npe/ctx->np_phi)/ctx->np_radius != ctx->np_theta np_theta=%d np_radius=%d",(int)ctx->np_theta,(int)ctx->np_radius);
  ctx->particlePlaneRank = ctx->rank%(ctx->npe/ctx->np_phi); // rank in plane = rank % nproc_plane
  ctx->ParticlePlaneIdx = ctx->rank/(ctx->npe/ctx->np_phi);  // plane index = rank / nproc_plane
  PetscFunctionReturn(0);
}

/*
 Create particle coordinates quasi-uniform on a circle
*/
static PetscErrorCode CreateParticles(DM dm, DM sw, PartDDCtx *ctx)
{
  PetscRandom rnd;
  PetscReal  *vpar, *coords,*weights, r0 = -1, dr = -1, psi;
  PetscInt   *cellid;
  PetscInt   gid,dim;
  const PetscReal rmin = ctx->r, rmaj=ctx->R;
  const PetscReal dth  = 2.0*PETSC_PI/(PetscReal)ctx->np_theta;
  const PetscInt  iths = ctx->particlePlaneRank % ctx->np_theta;
  const PetscInt  irs =  ctx->particlePlaneRank / ctx->np_theta;
  const PetscReal th0 = (PetscReal)iths*dth + 1.e-12*dth;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)dm), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, .0, .999999999));
  PetscCall(DMGetDimension(dm, &dim));
  // get r0 and dr
  psi = 0;
  for (int ii=0;ii<ctx->np_radius;ii++) {
    PetscReal tdr = PetscSqrtReal(rmin*rmin/(PetscReal)ctx->np_radius + psi*psi) - psi;
    if (irs==ii) { dr = tdr; r0 = psi; }
    psi += tdr;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "[%d) psi=%g\n",ii,psi));
  }
  /* ~length of a particle */
  const PetscReal n_points_global =  (dim==3) ? (PetscReal)(ctx->npe)*PetscPowReal((PetscReal)(ctx->n_plane_points_proc),1.5) : ctx->n_plane_points_proc;
  if (n_points_global <= 0) {
    PetscCall(DMSwarmSetLocalSizes(sw, 0, 10));
  } else {
    const PetscReal dx = (dim==3) ? PetscPowReal((PETSC_PI*rmin*rmin/4.0) * rmaj*2.0*PETSC_PI / n_points_global, 0.333) : PetscPowReal((PETSC_PI*rmin*rmin/4.0) / n_points_global, 0.5);
    const PetscInt  npart_r = (PetscInt)(dr/dx + PETSC_SQRT_MACHINE_EPSILON) + 1, npart_theta = ctx->n_plane_points_proc / npart_r + 1, npart_phi = (dim==3) ? npart_r : 1;
    const PetscInt  npart = npart_r*npart_theta*npart_phi*ctx->particles_per_point;
    const PetscReal dphi = 2.0*PETSC_PI/(PetscReal)ctx->np_phi; /* rmin for particles < rmin */
    const PetscReal phi0 = (PetscReal)ctx->ParticlePlaneIdx*dphi;
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] CreateParticles: npart(%d): r=%d, theta=%d, phi=%d. n proc: r=%d, theta=%d, phi=%d. r0 = %g dr = %g dx = %g\n",ctx->rank,npart,npart_r,npart_theta,npart_phi,ctx->np_radius,ctx->np_theta,ctx->np_phi,r0,dr,dx));
    PetscCall(DMSwarmSetLocalSizes(sw, npart, npart/10 + 2));
    PetscCall(DMSetFromOptions(sw));
    PetscCall(DMViewFromOptions(sw, NULL, "-sw_view_orig"));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weights));
    PetscCall(DMSwarmGetField(sw, "vpar", NULL, NULL, (void **)&vpar));
    PetscCallMPI(MPI_Scan(&npart, &gid, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm))); // start with local
    gid -= npart;
    const PetscReal dr2 = dr/(PetscReal)npart_r - PETSC_SQRT_MACHINE_EPSILON*dr;
    const PetscReal dth2 = dth/(PetscReal)npart_theta - PETSC_SQRT_MACHINE_EPSILON*dth;
    psi = r0 + dr2/2;
    for (int ic, ir = 0, ip = 0; ir < npart_r; ir++, psi += dr2) {
      PetscScalar value,theta,cartx[3];
      for (ic = 0, theta = th0 + dth2/2.0; ic < npart_theta; ic++, theta += dth2) {
        for (int iphi = 0; iphi < npart_phi; iphi++) {
          PetscCall(PetscRandomGetValue(rnd, &value));
          const PetscReal phi = phi0 + ((dim==3) ? value*dphi : 0.0); // random phi in processes interval, 0 for 2D
          const PetscReal qsaf = qsafty(psi/rmin);
          PetscReal thetap = theta + qsaf*phi; /* push forward to follow field-lines */
          while (thetap >= 2.*PETSC_PI) thetap -= 2.*PETSC_PI;
          while (thetap < 0.0)          thetap += 2.*PETSC_PI;
          cylToCart(((dim==3) ? rmaj : 0), psi, thetap,  phi, cartx); // store Cartesian for plotting
          for (int iv=0;iv<ctx->particles_per_point;iv++, ip++) {
            cellid[ip] = 0; // do in migrate
            vpar[ip] = (PetscReal)(-ctx->particles_per_point/2 + iv + 1)/(PetscReal)ctx->particles_per_point; // arbitrary velocity distribution function
            coords[ip*dim + 0] = cartx[0];
            coords[ip*dim + 1] = cartx[1];
            if (dim==3) coords[ip*dim + 2] = cartx[2];
            //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t[%d] cid=%d X = %12.4e,%12.4e,%12.4e  cyl: %12.4e,%12.4e,**%12.4e**\n",ctx->rank, gid, cartx[0], cartx[1], cartx[2], psi, thetap,  phi));
            weights[ip] = ++gid;
          }
        }
      }
    }
    // DMSwarmRestoreField
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weights));
    PetscCall(DMSwarmRestoreField(sw, "vpar", NULL, NULL, (void **)&vpar));
  }
  // migration
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmGetLocalSize(sw, &gid));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] CreateParticles done: npart = %d\n",ctx->rank,gid));

  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(DMLocalizeCoordinates(sw));
  PetscFunctionReturn(0);
}

static PetscErrorCode processParticles(DM dm, DM sw, PetscReal dt, PartDDCtx *ctx)
{
  PetscInt npart,dim;
  PetscReal  *vpar, *coords,*weights, rmaj = ctx->R;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetLocalSize(sw, &npart));
  PetscCall(DMGetDimension(dm, &dim)); // swarm?
  if (dim==2) rmaj = 0;
  /* Push particles with v_par only */
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "vpar", NULL, NULL, (void **)&vpar));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weights));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "[%d] push: %d\n",ctx->rank,npart));
  for (int ip = 0; ip < npart; ip++) {
    PetscReal dphi, qsaf, theta, psi, R, phi = 0.0, cartx[3], *crd = &coords[ip*dim];
    if (dim==2) { CartTocyl2D(rmaj, R, crd, psi, theta); }
    else { CartTocyl3D(rmaj, R, crd, psi, theta, phi);}
    dphi = dt*vpar[ip]/ctx->R; // the push, use R_0 for 2D also
    qsaf = qsafty(psi/ctx->r);
    phi += dphi;
    theta += qsaf*dphi; // little twist in 2D
    while (theta >= 2.*PETSC_PI) theta -= 2.*PETSC_PI;
    while (theta < 0.0)          theta += 2.*PETSC_PI;
    if (dim==2) phi = 0.0;
    cylToCart( rmaj, psi, theta, phi, cartx); // store Cartesian for plotting
    //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] push: %3d) qsaf=%12.4e phi=%12.4e, theta=%12.4e dphi=%g\n",ctx->rank,ip,qsaf,phi,theta,dphi));
    //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] push: %3d) Cart %12.4e,%12.4e,%12.4e --> %12.4e,%12.4e,%12.4e R=%12.4e, cyl: %12.4e,%12.4e,%12.4e\n",ctx->rank,ip,crd[0],crd[1],crd[2],cartx[0],cartx[1],cartx[2],R,psi, theta, phi));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] push: %3d) Cart %12.4e,%12.4e --> %12.4e,%12.4e R=%12.4e, cyl: %12.4e,%12.4e\n",ctx->rank,(int)weights[ip],crd[0],crd[1],cartx[0],cartx[1],R,psi, theta));
    for (int i=0;i<dim;i++) crd[i] = cartx[i];
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "vpar", NULL, NULL, (void **)&vpar));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weights));

  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PartDDCtx     actx,*ctx=&actx; /* work context */
  MPI_Comm           comm;
  DM                 dm, sw;
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  /* Create dm and particles */
  PetscCall(InitPlex(comm, &dm));      // get Plex stuff from command line
  PetscCall(DMGetDimension(dm, &dim)); // probably 2
  PetscCall(ProcessOptions(comm, ctx));// ctx->dim probably 3
  PetscCheck(dim <= ctx->dim && dim > 1, comm,PETSC_ERR_ARG_WRONG,"DM dim (%d) > -dim %d",(int)dim,ctx->dim);
  if (ctx->dim > dim) PetscCall(ExtrudeTorus(comm, &dm, ctx)); // 3D extrude
  else PetscCall(OriginShift2D(comm, dm, ctx)); // shift to center
  PetscCall(InitSwarm(comm, dm, &sw));
  PetscCall(CreateParticles(dm, sw, ctx));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMViewFromOptions(sw, NULL, "-sw_view"));
  for (PetscInt step = 0; step < ctx->steps; ++step) {
    PetscInt n;
    PetscCall(processParticles(dm, sw, ctx->stepSize, ctx));
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMViewFromOptions(sw, NULL, "-sw_view"));
    PetscCall(DMSwarmGetLocalSize(sw, &n));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] step %d) npart = %d\n",ctx->rank,step,n));
  }
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: !complex p4est hdf5
   testset:
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -radius_inflation 1.1 -steps 1 -dt 8 -sw_view hdf5:f.h5::append -dm_view hdf5:f.h5
   test:
     suffix: 2D
     args: -dim 2 -n_plane_points_proc 20
   test:
     suffix: 2D_4
     nsize: 4
     args: -dim 2 -n_plane_points_proc 10
   test:
     suffix: 3D
     nsize: 4
     args: -dim 3 -n_plane_points_proc 20 -np_phi 4

TEST*/
