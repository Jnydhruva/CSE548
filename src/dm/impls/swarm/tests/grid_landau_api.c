static char help[] = "Particle interface for 2V grid-based Landau advance: create a particle distribution on circle or torus, AMR refine to domain, migrating particles to grid distribution, create particle lists on each cell, deposit to Landau grid, Landau advance, pseudo-inverse remap, migrate back to original ordering\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>
//#include <petscdmforest.h>
#include <petscdmcomposite.h>
#include <petsclandau.h>

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
static PetscErrorCode InitSwarm(MPI_Comm comm, DM dm_x, DM *sw)
{
  PetscInt dim;
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm_x, &dim));
  PetscCall(DMCreate(comm, sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm_x));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "vpar", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
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
  PetscInt n_local_cell_phi; /* number of local cells in phi direction */
  /* torus geometry  */
  PetscReal  R;
  PetscReal  r;
  PetscReal  r_inflate;
  PetscReal  torus_section_rad;
  // diagnostics
  PetscInt  grid_view_idx;
  PetscInt  field_view_idx;
} PartDDCtx;

/* Simple shift to origin */
static PetscErrorCode OriginShift2D(MPI_Comm comm, DM dm, PartDDCtx *partCtx)
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
    v[0] *= 2*partCtx->r*partCtx->r_inflate;
    v[1] *= 2*partCtx->r*partCtx->r_inflate;
    v[0] -= partCtx->r*partCtx->r_inflate;
    v[1] -= partCtx->r*partCtx->r_inflate;
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(dm, coordinates));
  PetscFunctionReturn(0);
}

/* Extrude 2D Plex to 3D Plex */
static PetscErrorCode ExtrudeTorus(MPI_Comm comm, DM *dm, PartDDCtx *partCtx)
{
  DM dmtorus;
  PetscReal L;
  //DMBoundaryType periodicity[] = {DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_PERIODIC};
  Vec             coordinates;
  PetscScalar    *coords, R_0 = partCtx->R;
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
    v[0] *= 2*partCtx->r*partCtx->r_inflate;
    v[1] *= 2*partCtx->r*partCtx->r_inflate;
    v[0] += R_0 - partCtx->r*partCtx->r_inflate;
    v[1] +=     - partCtx->r*partCtx->r_inflate;
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  //
  L = partCtx->torus_section_rad*partCtx->R;
  // we could create a box mesh here but Plex starts with a 2x2 so we can just dm_refine from there, for now
  PetscCall(DMPlexExtrude(*dm, partCtx->np_phi*partCtx->n_local_cell_phi, L, PETSC_FALSE, PETSC_FALSE, NULL, NULL, &dmtorus));
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
    PetscReal Z = v[2], phi = Z/R_0;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)*dm), "\t\t[%d] ExtrudeTorus %d) psi=%12.4e theta=%12.4e phi=%12.4e. Cart=%12.4e,%12.4e,%12.4e",partCtx->rank, ii/3,  psi, theta, phi, v[0], v[1], v[2]));
    cylToCart( R_0, psi, theta, phi, v);
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)*dm), "--> X = %12.4e,%12.4e,%12.4e \n", v[0], v[1], v[2]));
  }
  PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view_orig"));
  // set periodic - TODO
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, PartDDCtx *partCtx)
{
  PetscBool phiFlag,radFlag,thetaFlag;
  PetscFunctionBeginUser;
  partCtx->particles_per_point = 1; // 10
  partCtx->dim = 2;
  partCtx->n_plane_points_proc = 16; // 4x4 grid
  partCtx->steps            = 1;
  partCtx->stepSize         = 1;
  partCtx->torus_section_rad = 360;
  /* mesh */
  partCtx->R = 5.;
  partCtx->r = 1.;
  partCtx->r_inflate = 1.1;
  partCtx->np_phi  = 1;
  partCtx->n_local_cell_phi  = 1;
  partCtx->np_radius = 1;
  partCtx->np_theta  = 1;
  partCtx->grid_view_idx = partCtx->field_view_idx = 0; // on
  PetscCallMPI(MPI_Comm_rank(comm, &partCtx->rank));
  PetscCallMPI(MPI_Comm_size(comm, &partCtx->npe));

  PetscOptionsBegin(comm, "", "grid-based Landau particle interface", "DMPLEX");
  PetscCall(PetscOptionsInt("-dim", "parameter", "grid_landau_api.c", partCtx->dim, &partCtx->dim, PETSC_NULL));
  if (partCtx->dim==3) PetscCall(PetscOptionsReal("-torus_section_degree_todo", "360 for entire torus", "grid_landau_api.c", partCtx->torus_section_rad, &partCtx->torus_section_rad, PETSC_NULL));
  else partCtx->torus_section_rad = 0;
  partCtx->torus_section_rad *= PETSC_PI/180.; // get into radians
  PetscCheck(partCtx->dim==2 || partCtx->dim==3, comm,PETSC_ERR_ARG_WRONG,"dim (%d) != 2 or 3",(int)partCtx->dim);
  PetscCall(PetscOptionsInt("-grid_view_idx", "Index of grid for diagnostics like plotting", ".c", partCtx->grid_view_idx, &partCtx->grid_view_idx, NULL));
  PetscCall(PetscOptionsInt("-field_view_idx", "Index of field for diagnostics like plotting", ".c", partCtx->field_view_idx, &partCtx->field_view_idx, NULL));
  if (partCtx->dim==3) {
    partCtx->np_phi = 4;
    PetscCall(PetscOptionsInt("-np_phi", "Number of planes for particle mesh", "grid_landau_api.c", partCtx->np_phi, &partCtx->np_phi, &phiFlag));
    PetscCall(PetscOptionsInt("-n_local_cell_phi", "number of local cells in phi direction", "grid_landau_api.c", partCtx->n_local_cell_phi, &partCtx->n_local_cell_phi, PETSC_NULL));
    PetscCheck(partCtx->np_phi*partCtx->n_local_cell_phi > 2, comm,PETSC_ERR_ARG_WRONG,"num particle planes 'np_phi' (%d) > 2 in 3D",(int)partCtx->np_phi);
  }
  else { partCtx->np_phi = 1; phiFlag = PETSC_TRUE;} // == 1
  PetscCall(PetscOptionsInt("-np_radius", "Number of radial cells for particle mesh", "grid_landau_api.c", partCtx->np_radius, &partCtx->np_radius, &radFlag));
  PetscCall(PetscOptionsInt("-np_theta", "Number of theta cells for particle mesh", "grid_landau_api.c", partCtx->np_theta, &partCtx->np_theta, &thetaFlag));
  /* particle grids: <= npe, <= num solver planes */
  PetscCheck(partCtx->npe >= partCtx->np_phi, comm,PETSC_ERR_ARG_WRONG,"num particle planes np_phi (%d) > npe (%d)",(int)partCtx->np_phi,partCtx->npe);

  if (partCtx->np_phi*partCtx->np_radius*partCtx->np_theta != partCtx->npe) { /* recover from inconsistant grid/procs */
    PetscCheck(thetaFlag || radFlag || phiFlag,comm,PETSC_ERR_USER,"over constrained number of particle processes npe (%d) != %d",(int)partCtx->npe,(int)partCtx->np_phi*partCtx->np_radius*partCtx->np_theta);
    if (!thetaFlag && radFlag && phiFlag) partCtx->np_theta = partCtx->npe/(partCtx->np_phi*partCtx->np_radius);
    else if (thetaFlag && !radFlag && phiFlag) partCtx->np_radius = partCtx->npe/(partCtx->np_phi*partCtx->np_theta);
    else if (thetaFlag && radFlag && !phiFlag && partCtx->dim==2) partCtx->np_phi = partCtx->npe/(partCtx->np_radius*partCtx->np_theta);
    else if (!thetaFlag && !radFlag && !phiFlag) {
      PetscInt npe_plane = (int)pow((double)partCtx->npe,0.6667);
      partCtx->np_phi = partCtx->npe/npe_plane;
      partCtx->np_radius = (PetscInt)(PetscSqrtReal((double)npe_plane)+0.5);
      partCtx->np_theta = npe_plane/partCtx->np_radius;
    }
    else if (!thetaFlag && !radFlag) {
      PetscInt npe_plane = partCtx->npe/partCtx->np_phi;
      partCtx->np_radius = (int)(PetscSqrtReal((double)npe_plane)+0.5);
      partCtx->np_theta = npe_plane/partCtx->np_radius;
    }
  }
  PetscCheck(partCtx->np_phi*partCtx->np_radius*partCtx->np_theta==partCtx->npe,comm,PETSC_ERR_USER,"failed to recover npe=%d != %d",(int)partCtx->npe,(int)partCtx->np_phi*partCtx->np_radius*partCtx->np_theta);
  PetscCall(PetscOptionsInt("-particles_per_point", "Number of particles per spatial cell", "grid_landau_api.c", partCtx->particles_per_point, &partCtx->particles_per_point, NULL));
  PetscCall(PetscOptionsInt("-n_plane_points_proc", "parameter", "grid_landau_api.c", partCtx->n_plane_points_proc, &partCtx->n_plane_points_proc, PETSC_NULL));
  PetscCall(PetscOptionsInt("-steps", "Steps to take", "grid_landau_api.c", partCtx->steps, &partCtx->steps, PETSC_NULL));
  PetscCall(PetscOptionsReal("-dt", "dt", "grid_landau_api.c", partCtx->stepSize, &partCtx->stepSize, PETSC_NULL));
  /* Domain and mesh definition */
  PetscCall(PetscOptionsReal("-radius_minor", "Minor radius of torus", "grid_landau_api.c", partCtx->r, &partCtx->r, NULL));
  PetscCall(PetscOptionsReal("-radius_major", "Major radius of torus", "grid_landau_api.c", partCtx->R, &partCtx->R, NULL));
  PetscCall(PetscOptionsReal("-radius_inflation", "inflate domain factor from minor radius", "grid_landau_api.c", partCtx->r_inflate, &partCtx->r_inflate, NULL));

  PetscOptionsEnd();
  /* derived */
  PetscCheck(partCtx->npe%partCtx->np_phi==0,comm,PETSC_ERR_USER,"partCtx->npe mod partCtx->np_phi!=0 npe=%d != %d",(int)partCtx->npe,(int)partCtx->np_phi);
  PetscCheck((partCtx->npe/partCtx->np_phi)/partCtx->np_radius == partCtx->np_theta,comm,PETSC_ERR_USER,"partCtx->npe/partCtx->np_phi)/partCtx->np_radius != partCtx->np_theta np_theta=%d np_radius=%d",(int)partCtx->np_theta,(int)partCtx->np_radius);
  partCtx->particlePlaneRank = partCtx->rank%(partCtx->npe/partCtx->np_phi); // rank in plane = rank % nproc_plane
  partCtx->ParticlePlaneIdx = partCtx->rank/(partCtx->npe/partCtx->np_phi);  // plane index = rank / nproc_plane
  PetscFunctionReturn(0);
}

/*
 Create particle coordinates quasi-uniform on a circle
*/
static PetscErrorCode CreateParticles(DM sw, PartDDCtx *partCtx)
{
  PetscRandom rnd;
  PetscReal  *vpar, *coords,*weights, r0 = -1, dr = -1, psi;
  PetscInt   *cellid;
  PetscInt   gid,dim;
  const PetscReal rmin = partCtx->r, rmaj=partCtx->R;
  const PetscReal dth  = 2.0*PETSC_PI/(PetscReal)partCtx->np_theta;
  const PetscInt  iths = partCtx->particlePlaneRank % partCtx->np_theta;
  const PetscInt  irs =  partCtx->particlePlaneRank / partCtx->np_theta;
  const PetscReal th0 = (PetscReal)iths*dth + 1.e-12*dth;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)sw), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, .0, .999999999));
  PetscCall(DMGetDimension(sw, &dim));
  // get r0 and dr
  psi = 0;
  for (int ii=0;ii<partCtx->np_radius;ii++) {
    PetscReal tdr = PetscSqrtReal(rmin*rmin/(PetscReal)partCtx->np_radius + psi*psi) - psi;
    if (irs==ii) { dr = tdr; r0 = psi; }
    psi += tdr;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "Radial psi_0 = %g for radial proc %d\n",psi,ii));
  }
  /* ~length of a particle */
  const PetscReal n_points_global =  (dim==3) ? (PetscReal)(partCtx->npe)*PetscPowReal((PetscReal)(partCtx->n_plane_points_proc),1.5) : partCtx->n_plane_points_proc;
  if (n_points_global <= 0) {
    PetscCall(DMSwarmSetLocalSizes(sw, 0, 10));
  } else {
    const PetscReal dx = (dim==3) ? PetscPowReal((PETSC_PI*rmin*rmin/4.0) * rmaj*2.0*PETSC_PI / n_points_global, 0.333) : PetscPowReal((PETSC_PI*rmin*rmin/4.0) / n_points_global, 0.5);
    const PetscInt  npart_r = (PetscInt)(dr/dx + PETSC_SQRT_MACHINE_EPSILON) + 1, npart_theta = partCtx->n_plane_points_proc / npart_r + 1, npart_phi = (dim==3) ? npart_r : 1;
    const PetscInt  npart = npart_r*npart_theta*npart_phi*partCtx->particles_per_point*partCtx->n_local_cell_phi;
    const PetscReal dphi_proc = 2.0*PETSC_PI/(PetscReal)partCtx->np_phi;
    const PetscReal dphi_local= dphi_proc/(PetscReal)partCtx->n_local_cell_phi;
    const PetscReal phi0 = (PetscReal)partCtx->ParticlePlaneIdx*dphi_proc;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "[%d] CreateParticles: npart(%d): r=%d, theta=%d, phi=%d. n proc: r=%d, theta=%d, phi=%d. r0 = %g dr = %g dx = %g\n",partCtx->rank,npart,npart_r,npart_theta,npart_phi,partCtx->np_radius,partCtx->np_theta,partCtx->np_phi,r0,dr,dx));
    PetscCall(DMSwarmSetLocalSizes(sw, npart, npart/10 + 2));
    PetscCall(DMSetFromOptions(sw));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weights));
    PetscCall(DMSwarmGetField(sw, "vpar", NULL, NULL, (void **)&vpar));
    PetscCallMPI(MPI_Scan(&npart, &gid, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)sw))); // start with local
    gid -= npart;
    const PetscReal dr2 = dr/(PetscReal)npart_r - PETSC_SQRT_MACHINE_EPSILON*dr;
    const PetscReal dth2 = dth/(PetscReal)npart_theta - PETSC_SQRT_MACHINE_EPSILON*dth;
    psi = r0 + dr2/2;
    for (int ic, ir = 0, ip = 0; ir < npart_r; ir++, psi += dr2) {
      PetscScalar value, theta, cartx[3];
      for (ic = 0, theta = th0 + dth2/2.0; ic < npart_theta; ic++, theta += dth2) {
        PetscScalar phi0_cell = phi0;
        for (int iphi_loc = 0; iphi_loc < partCtx->n_local_cell_phi ; iphi_loc++, phi0_cell += dphi_local) {
          for (int iphi = 0; iphi < npart_phi; iphi++) {
            PetscCall(PetscRandomGetValue(rnd, &value));
            const PetscReal phi = phi0_cell + ((dim==3) ? value*dphi_local : 0.0); // random phi in processes interval, 0 for 2D
            const PetscReal qsaf = qsafty(psi/rmin);
            PetscReal thetap = theta + qsaf*phi; /* push forward to follow field-lines */
            while (thetap >= 2.*PETSC_PI) thetap -= 2.*PETSC_PI;
            while (thetap < 0.0)          thetap += 2.*PETSC_PI;
            cylToCart(((dim==3) ? rmaj : 0), psi, thetap,  phi, cartx); // store Cartesian for plotting
            for (int iv=0;iv<partCtx->particles_per_point;iv++, ip++) {
              cellid[ip] = 0; // do in migrate
              vpar[ip] = (PetscReal)(-partCtx->particles_per_point/2 + iv + 1)/(PetscReal)partCtx->particles_per_point; // arbitrary velocity distribution function
              coords[ip*dim + 0] = cartx[0];
              coords[ip*dim + 1] = cartx[1];
              if (dim==3) coords[ip*dim + 2] = cartx[2];
              //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t[%d] cid=%d X = %12.4e,%12.4e,%12.4e  cyl: %12.4e,%12.4e,**%12.4e**\n",partCtx->rank, gid, cartx[0], cartx[1], cartx[2], psi, thetap,  phi));
              weights[ip] = ++gid;
            }
          }
        }
      }
    }
    // DMSwarmRestoreField
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weights));
    PetscCall(DMSwarmRestoreField(sw, "vpar", NULL, NULL, (void **)&vpar));
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)sw), "\t[%d] CreateParticles made %d particles (%d)\n",(int)partCtx->rank,(int)gid,(int)npart));
  }
  // migration
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmGetLocalSize(sw, &gid));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "\t[%d] CreateParticles done: number of particles: %d\n",(int)partCtx->rank,(int)gid));

  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(DMLocalizeCoordinates(sw));
  PetscFunctionReturn(0);
}

// ex7.c deposit & remap
typedef struct {
  Mat MpTrans;
  Mat Mp;
  Vec ff;
  Vec uu;
} MatShellCtx;

static PetscErrorCode MatMultMtM_SeqAIJ(Mat MtM, Vec xx, Vec yy)
{
  MatShellCtx *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(MtM, &matshellctx));
  PetscCheck(matshellctx, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  PetscCall(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  PetscCall(MatMult(matshellctx->MpTrans, matshellctx->ff, yy));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAddMtM_SeqAIJ(Mat MtM, Vec xx, Vec yy, Vec zz)
{
  MatShellCtx *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(MtM, &matshellctx));
  PetscCheck(matshellctx, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  PetscCall(MatMult(matshellctx->Mp, xx, matshellctx->ff));
  PetscCall(MatMultAdd(matshellctx->MpTrans, matshellctx->ff, yy, zz));
  PetscFunctionReturn(0);
}

static PetscErrorCode addMoments2D(DM sw, PetscReal *moments)
{
  PetscReal *wq, *coords;
  PetscInt Np;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&wq));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  for (int p = 0; p < Np; p++) {
    moments[0] += wq[p];
    moments[1] += wq[p] * coords[p * 2 + 0]; // x-momentum
    moments[2] += wq[p] * (PetscSqr(coords[p * 2 + 0]) + PetscSqr(coords[p * 2 + 1]));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&wq));

  PetscFunctionReturn(0);
}

static PetscErrorCode pushParticles(DM swarray[LANDAU_MAX_SPECIES], PetscInt dim, PetscReal dt, PartDDCtx *partCtx, LandauCtx *ctx)
{
  PetscInt npart;
  PetscReal  *vpar, *coords,*weights, rmaj = partCtx->R;

  PetscFunctionBeginUser;
  for (PetscInt sp = 0; sp < ctx->num_species; sp++) {
    DM sw = swarray[sp];
    PetscCall(DMSwarmGetLocalSize(sw, &npart));
    if (dim==2) rmaj = 0;
    /* Push particles with v_par only */
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(sw, "vpar", NULL, NULL, (void **)&vpar));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weights));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "[%d] push: %d\n",partCtx->rank,npart));
    for (int ip = 0; ip < npart; ip++) {
      PetscReal dphi, qsaf, theta, psi, R, phi = 0.0, cartx[3], *crd = &coords[ip*dim];
      if (dim==2) { CartTocyl2D(rmaj, R, crd, psi, theta); }
      else { CartTocyl3D(rmaj, R, crd, psi, theta, phi);}
      dphi = dt*vpar[ip]/partCtx->R; // the push, use R_0 for 2D also
      qsaf = qsafty(psi/partCtx->r);
      phi += dphi;
      theta += qsaf*dphi; // little twist in 2D
      while (theta >= 2.*PETSC_PI) theta -= 2.*PETSC_PI;
      while (theta < 0.0)          theta += 2.*PETSC_PI;
      if (dim==2) phi = 0.0;
      cylToCart( rmaj, psi, theta, phi, cartx); // store Cartesian for plotting
      //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] push: %3d) qsaf=%12.4e phi=%12.4e, theta=%12.4e dphi=%g\n",partCtx->rank,ip,qsaf,phi,theta,dphi));
      //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] push: %3d) Cart %12.4e,%12.4e,%12.4e --> %12.4e,%12.4e,%12.4e R=%12.4e, cyl: %12.4e,%12.4e,%12.4e\n",partCtx->rank,ip,crd[0],crd[1],crd[2],cartx[0],cartx[1],cartx[2],R,psi, theta, phi));
      //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] push: %3d) Cart %12.4e,%12.4e --> %12.4e,%12.4e R=%12.4e, cyl: %12.4e,%12.4e\n",partCtx->rank,(int)weights[ip],crd[0],crd[1],cartx[0],cartx[1],R,psi, theta));
      for (int i=0;i<dim;i++) crd[i] = cartx[i];
    }
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmRestoreField(sw, "vpar", NULL, NULL, (void **)&vpar));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weights));
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode gridToParticles(DM sw, PetscReal *moments, Vec rhs, Mat M_p)
{
  PetscBool     is_lsqr;
  KSP           ksp;
  Mat           PM_p = NULL, MtM, D;
  Vec           ff;
  PetscInt      timestep = 0, N, M, nzl;
  PetscReal     time = 0.0;
  MatShellCtx  *matshellctx;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "ftop_"));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPLSQR, &is_lsqr));
  if (!is_lsqr) {
    PetscCall(MatGetLocalSize(M_p, &M, &N));
    if (N > M) {
      PC pc;
      PetscCall(PetscInfo(sw, " M (%" PetscInt_FMT ") < M (%" PetscInt_FMT ") -- skip revert to lsqr\n", M, N));
      is_lsqr = PETSC_TRUE;
      PetscCall(KSPSetType(ksp, KSPLSQR));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCNONE)); // could put in better solver -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero
    } else {
      PetscCall(PetscNew(&matshellctx));
      PetscCall(MatCreateShell(PetscObjectComm((PetscObject)sw), N, N, PETSC_DECIDE, PETSC_DECIDE, matshellctx, &MtM));
      PetscCall(MatTranspose(M_p, MAT_INITIAL_MATRIX, &matshellctx->MpTrans));
      matshellctx->Mp = M_p;
      PetscCall(MatShellSetOperation(MtM, MATOP_MULT, (void (*)(void))MatMultMtM_SeqAIJ));
      PetscCall(MatShellSetOperation(MtM, MATOP_MULT_ADD, (void (*)(void))MatMultAddMtM_SeqAIJ));
      PetscCall(MatCreateVecs(M_p, &matshellctx->uu, &matshellctx->ff));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 1, NULL, &D));
      for (int i = 0; i < N; i++) {
        const PetscScalar *vals;
        const PetscInt    *cols;
        PetscScalar        dot = 0;
        PetscCall(MatGetRow(matshellctx->MpTrans, i, &nzl, &cols, &vals));
        for (int ii = 0; ii < nzl; ii++) dot += PetscSqr(vals[ii]);
        PetscCheck(dot != 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Row %d is empty", i);
        PetscCall(MatSetValue(D, i, i, dot, INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY));
      PetscCall(PetscInfo(sw, "createMtMKSP Have %" PetscInt_FMT " eqs, nzl = %" PetscInt_FMT "\n", N, nzl));
      PetscCall(KSPSetOperators(ksp, MtM, D));
      PetscCall(MatViewFromOptions(D, NULL, "-ftop2_D_mat_view"));
      PetscCall(MatViewFromOptions(M_p, NULL, "-ftop2_Mp_mat_view"));
      PetscCall(MatViewFromOptions(matshellctx->MpTrans, NULL, "-ftop2_MpTranspose_mat_view"));
    }
  }
  if (is_lsqr) {
    PC        pc;
    PetscBool is_bjac;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &is_bjac));
    if (is_bjac) {
      DM dm;
      PetscCall(DMSwarmGetCellDM(sw,&dm));
      PetscCall(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
      PetscCall(KSPSetOperators(ksp, M_p, PM_p));
    } else {
      PetscCall(KSPSetOperators(ksp, M_p, M_p));
    }
  }
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff)); // this grabs access !!!!!
  if (!is_lsqr) {
    PetscCall(KSPSolve(ksp, rhs, matshellctx->uu));
    PetscCall(MatMult(M_p, matshellctx->uu, ff));
    PetscCall(MatDestroy(&matshellctx->MpTrans));
    PetscCall(VecDestroy(&matshellctx->ff));
    PetscCall(VecDestroy(&matshellctx->uu));
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&MtM));
    PetscCall(PetscFree(matshellctx));
  } else {
    PetscCall(KSPSolveTranspose(ksp, rhs, ff));
  }
  PetscCall(KSPDestroy(&ksp));
  /* Visualize particle field */
  PetscCall(DMSetOutputSequenceNumber(sw, timestep, time));
  PetscCall(VecViewFromOptions(ff, NULL, "-weights_view"));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));

  /* compute energy */
  if (moments) PetscCall(addMoments2D(sw, moments));

  PetscCall(MatDestroy(&PM_p));
  PetscFunctionReturn(0);
}

// call Landau

typedef struct {
  Mat Mp[LANDAU_MAX_BATCH_SZ][LANDAU_MAX_SPECIES];
  DM  sw[LANDAU_MAX_BATCH_SZ][LANDAU_MAX_SPECIES];
  PetscReal moments[3];
} accessCtx;

// put particles onto local grid vector x
static PetscErrorCode landau_deposit(DM vdm, Vec x, PetscInt local_field, PetscInt grid, PetscInt bid, void *vctx)
{
  accessCtx    *user = (accessCtx*)vctx;
  LandauCtx *ctx;
  PetscInt npart,sp;
  DM       sw;
  PetscReal *moms = NULL;
  Vec           ff;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(vdm, &ctx));
  sp = ctx->species_offset[grid] + local_field; // make the arg ???
  sw = user->sw[bid][sp]; // particle list
  PetscCall(DMSwarmSetCellDM(sw, vdm)); // we tell the sub-swarm that you are working with this velocity grid
  PetscCall(DMSwarmGetLocalSize(sw, &npart));
  PetscCall(PetscInfo(vdm, "landau_deposit grid %" PetscInt_FMT ", batch %" PetscInt_FMT " and species %" PetscInt_FMT ", num particles = %" PetscInt_FMT ", sw=%p\n", grid, bid, sp, npart,sw));
  if (bid == ctx->batch_view_idx) moms = user->moments;
  user->Mp[bid][sp] = NULL;
  if (moms) {
    PetscCall(DMViewFromOptions(sw, NULL, "-landau_deposit_swarm_view"));
    PetscCall(DMViewFromOptions(vdm, NULL, "-landau_deposit_plex_view"));
  }
  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  PetscCall(DMCreateMassMatrix(sw, vdm, &user->Mp[bid][sp]));
  PetscCall(MatViewFromOptions(user->Mp[bid][sp], NULL, "-mass_view"));
  PetscCall(DMViewFromOptions(sw, NULL, "-mass_view"));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff)); // this grabs access !!!!!
  PetscCall(PetscObjectSetName((PetscObject)ff, "weights"));
  PetscCall(MatMultTranspose(user->Mp[bid][sp], ff, x)); // Add or not?
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff));
  PetscCall(PetscObjectSetName((PetscObject)x, "rho"));
  if (moms) {
    PetscCall(PetscObjectSetName((PetscObject)vdm, "single"));
    PetscCall(DMViewFromOptions(vdm, NULL, "-deposit_dm_view"));
    PetscCall(VecViewFromOptions(x, NULL, "-rho_view"));
  }
  if (moms) PetscCall(addMoments2D(sw, moms));

  PetscFunctionReturn(0);
}

static PetscErrorCode landau_grid_to_particles(DM vdm, Vec x, PetscInt local_field, PetscInt grid, PetscInt bid, void *vctx)
{
  accessCtx    *user = (accessCtx*)vctx;
  LandauCtx *ctx;
  PetscInt npart,dim,sp;
  DM       sw;
  PetscReal *moms=NULL;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(vdm, &dim)); // 2
  PetscCall(DMGetApplicationContext(vdm, &ctx));
  sp = ctx->species_offset[grid] + local_field; // make the arg -- TODO
  sw = user->sw[bid][sp]; // particle list
  PetscCall(DMSwarmGetLocalSize(sw, &npart));
  PetscCall(PetscInfo(vdm, "landau_grid_to_particles: batch %" PetscInt_FMT " and species %" PetscInt_FMT ", num particles = %" PetscInt_FMT "\n", bid, sp, npart));
  if (bid == ctx->batch_view_idx) moms = user->moments;
  if (moms) PetscCall(DMViewFromOptions(sw, NULL, "-swarm_view"));
  // do it
  PetscCall(gridToParticles(sw,moms,x,user->Mp[bid][sp]));
  PetscCall(MatDestroy(&user->Mp[bid][sp]));
  // print moments

  PetscFunctionReturn(0);
}

// advance Landau for whole batched system
static PetscErrorCode advanceCollisions(DM pack, Mat J, Vec X, PetscReal dt, PartDDCtx *partCtx, DM sw_loc_species[])
{
  PetscInt cStart,cEnd;
  DM dm_x;
  LandauCtx *ctx;
  TS ts;
  accessCtx user;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCall(DMSwarmGetCellDM(sw_loc_species[0],&dm_x)); // one species DM, everone has same spatial grid
  PetscCall(DMPlexGetHeightStratum(dm_x, 0, &cStart, &cEnd)); // my local part of global X grid
  // deposit: pack up particles
  for (PetscInt sp = 0; sp < ctx->num_species; sp++) {
    for(int c = 0 /* cStart */; c < cEnd; c++){
      PetscCall(DMCreate(PETSC_COMM_SELF, &user.sw[c][sp]));
      PetscCall(DMSwarmGetCellSwarm(sw_loc_species[sp], c, user.sw[c][sp]));
      PetscCall(PetscInfo(pack, "%d) cell sw = %p\n",c,user.sw[c][sp]));
    }
  }
  // deposit particels onto grid
  for (int i=0;i<3;i++) user.moments[i] = 0;
  PetscCall(DMPlexLandauAccess(pack, X, landau_deposit, &user));
  // print moments

  // Landau
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetDM(ts, pack));
  PetscCall(TSSetIFunction(ts, NULL, DMPlexLandauIFunction, NULL));
  PetscCall(TSSetIJacobian(ts, J, J, DMPlexLandauIJacobian, NULL));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSSolve(ts, X));
  PetscCall(DMPlexLandauPrintNorms(X, 1));

  // back to particles
  for (int i=0;i<3;i++) user.moments[i] = 0;
  PetscCall(DMPlexLandauAccess(pack, X, landau_grid_to_particles, &user));
  // print moments

  // release and copy new wights back (grid Landau)
  for (PetscInt sp = 0; sp < ctx->num_species; sp++) {
    for(int c = cStart; c < cEnd; c++){
      PetscCall(DMSwarmRestoreCellSwarm(sw_loc_species[sp], c, user.sw[c][sp]));
      PetscCall(DMDestroy(&user.sw[c][sp]));
    }
  }

  PetscCall(TSDestroy(&ts));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PartDDCtx     apartCtx,*partCtx=&apartCtx; /* work context */
  LandauCtx *ctx;
  MPI_Comm           comm;
  DM                 dm_x, pack, sw_loc_species[LANDAU_MAX_SPECIES];
  PetscInt dim, cStart, cEnd, nDMs, nCells;
  char   batch_opt[32] = "-dm_landau_batch_size ", numstr[32];
  Vec            X;
  Mat            J;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  /* Create Plex */
  PetscCall(InitPlex(comm, &dm_x));      // get Plex stuff from command line, start at dim==2 and extrude
  PetscCall(DMGetDimension(dm_x, &dim)); // probably 2
  PetscCall(ProcessOptions(comm, partCtx));// partCtx->dim probably 3 or 2 for debugging
  PetscCheck(dim <= partCtx->dim && dim > 1, comm,PETSC_ERR_ARG_WRONG,"DM dim (%d) > -dim %d",(int)dim,partCtx->dim);
  if (partCtx->dim > dim) PetscCall(ExtrudeTorus(comm, &dm_x, partCtx)); // 3D extrude
  else PetscCall(OriginShift2D(comm, dm_x, partCtx)); // shift to center
  // create Landau
  PetscCall(DMPlexGetHeightStratum(dm_x, 0, &cStart, &cEnd));
  nCells = cEnd - cStart;
  PetscCall(PetscSNPrintf(numstr, 32, "%d", (int)nCells));
  PetscCheck(nCells <= LANDAU_MAX_BATCH_SZ, comm, PETSC_ERR_ARG_WRONG, "num cells (%d) > MAX BATCH %d",(int)nCells,LANDAU_MAX_BATCH_SZ);
  PetscCall(PetscStrcat(batch_opt, numstr));
  PetscCall(PetscOptionsInsertString(NULL, batch_opt));
  PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &pack));
  PetscCall(DMSetUp(pack));
  PetscCall(DMPlexLandauPrintNorms(X, 0)); // empty
  PetscCall(DMSetOutputSequenceNumber(pack, 0, 0.0));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCheck(nCells==ctx->batch_sz, comm, PETSC_ERR_ARG_WRONG,"nCells != ctx->batch_sz");
  /* Create particles */
  if (partCtx->grid_view_idx >= 0 && partCtx->field_view_idx >= 0) {
    PetscCall(DMViewFromOptions(ctx->plex[partCtx->grid_view_idx], NULL, "-dm_v_view"));
  }
  for (PetscInt sp = 0; sp < ctx->num_species; sp++) {
    PetscCall(InitSwarm(comm, dm_x, &sw_loc_species[sp]));
    PetscCall(CreateParticles(sw_loc_species[sp], partCtx));
    if (partCtx->grid_view_idx >= 0 && partCtx->field_view_idx==sp) {
      PetscCall(DMViewFromOptions(dm_x, NULL, "-dm_x_view"));
      PetscCall(DMViewFromOptions(sw_loc_species[sp], NULL, "-sw_x_view"));
    }
  }
  // go
  for (PetscInt step = 0, n; step < partCtx->steps; ++step) {
    PetscCall(advanceCollisions(pack, J, X, partCtx->stepSize, partCtx, sw_loc_species));
    if (partCtx->grid_view_idx >= 0 && partCtx->field_view_idx >= 0 && ctx->batch_view_idx >= 0) {
      Vec *XsubArray = NULL;
      PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
      PetscCall(PetscMalloc(sizeof(*XsubArray) * nDMs, &XsubArray));
      PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
      PetscCall(PetscObjectSetName((PetscObject)XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, partCtx->grid_view_idx)], partCtx->grid_view_idx == 0 ? "ue" : "ui"));
      PetscCall(DMSetOutputSequenceNumber(ctx->plex[partCtx->grid_view_idx], 0, 0.0));
      PetscCall(VecViewFromOptions(XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, partCtx->grid_view_idx)], NULL, "-vec_v_view"));      // initial condition (monitor plots after step)
      PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray));                                                       // read only
      PetscCall(PetscFree(XsubArray));
    }
    PetscCall(pushParticles(sw_loc_species, dim, partCtx->stepSize, partCtx, ctx));
    for (PetscInt sp = 0; sp < ctx->num_species; sp++) {
      PetscCall(DMSwarmMigrate(sw_loc_species[sp], PETSC_TRUE)); // need to batch
      if (partCtx->grid_view_idx >= 0 && partCtx->field_view_idx==sp) PetscCall(DMViewFromOptions(sw_loc_species[sp], NULL, "-x_sw_view"));
      PetscCall(DMSwarmGetLocalSize(sw_loc_species[sp], &n));
      PetscCall(PetscSynchronizedPrintf(comm, "\t[%d] step %d) species %d, %d particles\n",partCtx->rank,step+1,sp,n));
    }
  }
  // cleanup
  for (PetscInt sp = 0; sp < ctx->num_species; sp++) PetscCall(DMDestroy(&sw_loc_species[sp]));
  PetscCall(DMDestroy(&dm_x));
  PetscCall(DMPlexLandauDestroyVelocitySpace(&pack));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     args: -dm_view hdf5:f.h5 -sw_view hdf5:f.h5::append -dm_plex_simplex 0 -radius_inflation 1.1 -dm_plex_dim 2 -ftop_ksp_type cg -ftop_pc_type jacobi -ftop_ksp_converged_reason -ftop_ksp_rtol 1.e-14 -ts_max_steps 1
     requires: !complex p4est hdf5

     test:
       suffix: 2D
       args: -dim 2 -n_plane_points_proc 20 -steps 0

     test:
       suffix: 2D_4
       nsize: 4
       args: -n_plane_points_proc 10 -steps 1 -dt 8

     test:
       suffix: 3D
       nsize: 1
       args: -dim 3 -n_plane_points_proc 40 -np_phi 1 -n_local_cell_phi 4 -steps 0

     test:
       suffix: 3D_4
       nsize: 4
       args: -dim 3 -n_plane_points_proc 10 -np_phi 4 -steps 0

TEST*/
