/*
  Note:
    -hratio is the ratio between mesh size of corse grids and fine grids
*/

static const char help[] = "1D Finite Volume solver in slope-limiter form with semidiscrete time stepping defined on a network.\n"
  "  shallow     - 1D Shallow water equations (Saint Venant System) \n"
  "                h_t + (q)_x = 0 \n"
  "                q_t + (\frac{q^2}{h} + g/2*h^2)_x = 0 \n"
  "                where, h(x,t) denotes the height of the water, q(x,t) the momentum.\n"
  "   \n"
  "  where hxs and hxf are the grid spacings for coarse and fine grids respectively.\n"
  "  exact       - Exact Riemann solver which usually needs to perform a Newton iteration to connect\n"
  "                the states across shocks and rarefactions\n"
  "  simulation  - use reference solution which is generated by smaller time step size to be true solution,\n"
  "                also the reference solution should be generated by user and stored in a binary file.\n"
  "  characteristic - Limit the characteristic variables, this is usually preferred (default)\n"
  "Several problem descriptions (initial data, physics specific features, boundary data) can be chosen with -initial N\n\n"
 "The problem size should be set with -da_grid_x M0\n\n";

/*
  Example:
    Euler timestepping:
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 1 -hratio 2 -limit minmod -ts_dt 0.05 -ts_max_time 7.0 -ymax 3 -ymin 0 -ts_type euler
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 2 -hratio 2 -limit minmod -ts_dt 0.05 -ts_max_time 2.5 -ymax 5.1 -ymin -5.1 -ts_type euler
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 3 -hratio 2 -limit minmod -ts_dt 0.05 -ts_max_time 4.0 -ymax 2 -ymin -2 -ts_type euler
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 4 -hratio 2 -limit minmod -ts_dt 0.05 -ts_max_time 4.0 -ymax 2 -ymin -2 -ts_type euler
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 5 -hratio 2 -limit minmod -ts_dt 0.10 -ts_max_time 5.0 -ymax 0.5 -ymin -0.5 -ts_type euler

    MRPK timestepping:
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 1 -hratio 2 -limit minmod -ts_dt 0.1 -ts_max_time 7.0 -ymax 3 -ymin 0 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1 -bufferwidth 4
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 2 -hratio 2 -limit minmod -ts_dt 0.1 -ts_max_time 2.5 -ymax 5.1 -ymin -5.1 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1 -bufferwidth 4
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 3 -hratio 2 -limit minmod -ts_dt 0.1 -ts_max_time 4.0 -ymax 2 -ymin -2 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1 -bufferwidth 4
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 4 -hratio 2 -limit minmod -ts_dt 0.1 -ts_max_time 4.0 -ymax 2 -ymin -2 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1 -bufferwidth 4
    mpiexec -n 1 ./ex9 -Mx 20 -network 0 -initial 5 -hratio 2 -limit minmod -ts_dt 0.2 -ts_max_time 5.0 -ymax 0.5 -ymin -0.5 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1 -bufferwidth 4

  Contributed by: Aidan Hamilton <aidan@udel.edu>

*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "./fvnet/fvnet.h"
#include "./fvnet/limiters.h"
#include <petsc/private/kernels/blockinvert.h>

PETSC_STATIC_INLINE PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }

/* --------------------------------- Shallow Water ----------------------------------- */
typedef struct {
  PetscReal gravity;
} ShallowCtx;

PETSC_STATIC_INLINE void ShallowFlux(ShallowCtx *phys,const PetscScalar *u,PetscScalar *f)
{
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

PETSC_STATIC_INLINE void ShallowFlux2(ShallowCtx *phys,const PetscScalar *u,PetscScalar *f)
{
  f[0] = u[1]*u[0];
  f[1] = PetscSqr(u[1])*u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

static PetscErrorCode PhysicsRiemann_Shallow_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g    = phys->gravity,ustar[2],cL,cR,c,cstar;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]},star;
  PetscInt                  i;

  PetscFunctionBeginUser;
  if (!(L.h > 0 && R.h > 0)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Reconstructed thickness is negative");
  cL = PetscSqrtScalar(g*L.h);
  cR = PetscSqrtScalar(g*R.h);
  c  = PetscMax(cL,cR);
  {
    /* Solve for star state */
    const PetscInt maxits = 50;
    PetscScalar tmp,res,res0=0,h0,h = 0.5*(L.h + R.h); /* initial guess */
    h0 = h;
    for (i=0; i<maxits; i++) {
      PetscScalar fr,fl,dfr,dfl;
      fl = (L.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - L.h*L.h)*(1/L.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*L.h);   /* rarefaction */
      fr = (R.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - R.h*R.h)*(1/R.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*R.h);   /* rarefaction */
      res = R.u - L.u + fr + fl;
      if (PetscIsInfOrNanScalar(res)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinity or Not-a-Number generated in computation");
      if (PetscAbsScalar(res) < 1e-8 || (i > 0 && PetscAbsScalar(h-h0) < 1e-8)) {
        star.h = h;
        star.u = L.u - fl;
        goto converged;
      } else if (i > 0 && PetscAbsScalar(res) >= PetscAbsScalar(res0)) {        /* Line search */
        h = 0.8*h0 + 0.2*h;
        continue;
      }
      /* Accept the last step and take another */
      res0 = res;
      h0 = h;
      dfl = (L.h < h) ? 0.5/fl*0.5*g*(-L.h*L.h/(h*h) - 1 + 2*h/L.h) : PetscSqrtScalar(g/h);
      dfr = (R.h < h) ? 0.5/fr*0.5*g*(-R.h*R.h/(h*h) - 1 + 2*h/R.h) : PetscSqrtScalar(g/h);
      tmp = h - res/(dfr+dfl);
      if (tmp <= 0) h /= 2;   /* Guard against Newton shooting off to a negative thickness */
      else h = tmp;
      if (!((h > 0) && PetscIsNormalScalar(h))) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FP,"non-normal iterate h=%g",(double)h);
    }
    SETERRQ1(PETSC_COMM_SELF,1,"Newton iteration for star.h diverged after %D iterations",i);
  }
converged:
  cstar = PetscSqrtScalar(g*star.h);
  if (L.u-cL < 0 && 0 < star.u-cstar) { /* 1-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(L.u/3 + 2./3*cL);
    ufan[1] = PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if (star.u+cstar < 0 && 0 < R.u+cR) { /* 2-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(R.u/3 - 2./3*cR);
    ufan[1] = -PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if ((L.h >= star.h && L.u-c >= 0) || (L.h<star.h && (star.h*star.u-L.h*L.u)/(star.h-L.h) > 0)) {
    /* 1-wave is right-travelling shock (supersonic) */
    ShallowFlux(phys,uL,flux);
  } else if ((star.h <= R.h && R.u+c <= 0) || (star.h>R.h && (R.h*R.u-star.h*star.h)/(R.h-star.h) < 0)) {
    /* 2-wave is left-travelling shock (supersonic) */
    ShallowFlux(phys,uR,flux);
  } else {
    ustar[0] = star.h;
    ustar[1] = star.h*star.u;
    ShallowFlux(phys,ustar,flux);
  }
  *maxspeed = MaxAbs(MaxAbs(star.u-cstar,star.u+cstar),MaxAbs(L.u-cL,R.u+cR));
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRiemann_Shallow_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx                *phys = (ShallowCtx*)vctx;
  PetscScalar               g = phys->gravity,fL[2],fR[2],s;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]};
  PetscReal                 tol = 1e-6;

  PetscFunctionBeginUser;
  /* Positivity preserving modification*/
  if (L.h < tol) L.u = 0.0;
  if (R.h < tol) R.u = 0.0;

  /*simple positivity preserving limiter*/
  if (L.h < 0) L.h = 0;
  if (R.h < 0) R.h = 0;

  ShallowFlux2(phys,(PetscScalar*)&L,fL);
  ShallowFlux2(phys,(PetscScalar*)&R,fR);

  s         = PetscMax(PetscAbs(L.u)+PetscSqrtScalar(g*L.h),PetscAbs(R.u)+PetscSqrtScalar(g*R.h));
  flux[0]   = 0.5*(fL[0] + fR[0]) + 0.5*s*(L.h - R.h);
  flux[1]   = 0.5*(fL[1] + fR[1]) + 0.5*s*(uL[1] - uR[1]);
  *maxspeed = s;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Conservative(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  PetscInt i,j;

  PetscFunctionBeginUser;
  for (i=0; i<m; i++) {
    for (j=0; j<m; j++) Xi[i*m+j] = X[i*m+j] = (PetscScalar)(i==j);
    speeds[i] = PETSC_MAX_REAL; /* Indicates invalid */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Shallow(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  ShallowCtx     *phys = (ShallowCtx*)vctx;
  PetscReal      c;
  PetscErrorCode ierr;
  PetscReal      tol = 1e-6;

  PetscFunctionBeginUser;
  c         = PetscSqrtScalar(u[0]*phys->gravity);

  if (u[0] < tol) { /*Use conservative variables*/
    X[0*2+0]  = 1;
    X[0*2+1]  = 0;
    X[1*2+0]  = 0;
    X[1*2+1]  = 1;
    speeds[0] = - c;
    speeds[1] =   c;
  } else {
    speeds[0] = u[1]/u[0] - c;
    speeds[1] = u[1]/u[0] + c;
    X[0*2+0]  = 1;
    X[0*2+1]  = speeds[0];
    X[1*2+0]  = 1;
    X[1*2+1]  = speeds[1];
  }

  ierr = PetscArraycpy(Xi,X,4);CHKERRQ(ierr);
  ierr = PetscKernel_A_gets_inverse_A_2(Xi,0,PETSC_FALSE,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_Shallow(void *vctx,PetscInt initial,PetscReal t,PetscReal x,PetscReal *u)
{
  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solutions not implemented for t > 0");
  switch (initial) {
    case 0:
      u[0] = (x < 25) ? 2 : 1; /* Standard Dam Break Problem */
      u[1] = (x < 25) ? 0 : 0;
      break;
    case 1:
      u[0] = (x < 10) ?   1 : 0.1; /*The Next 5 problems are standard Riemann problem tests */
      u[1] = (x < 10) ? 2.5 : 0;
      break;
    case 2:
      u[0] = (x < 25) ?  1 : 1;
      u[1] = (x < 25) ? -5 : 5;
      break;
    case 3:
      u[0] = (x < 20) ?  1 : 0;
      u[1] = (x < 20) ?  0 : 0;
      break;
    case 4:
      u[0] = (x < 30) ? 0: 1;
      u[1] = (x < 30) ? 0 : 0;
      break;
    case 5:
      u[0] = (x < 25) ?  0.1 : 0.1;
      u[1] = (x < 25) ? -0.3 : 0.3;
      break;
    case 6:
      u[0] = 1+0.5*PetscSinReal(2*PETSC_PI*x);
      u[1] = 1*u[0];
      break;
    case 7:
      u[0] = 1.0;
      u[1] = 1.0;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

/*2 edge vertex flux for edge 1 pointing in and edge 2 pointing out */
static PetscErrorCode PhysicsVertexFlux_Shallow_2Edge_InOut(const void* _fvnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed) 
{
  PetscErrorCode  ierr;
  const FVNetwork fvnet = (FVNetwork)_fvnet; 
  PetscInt        i,dof = fvnet->physics.dof; 

  PetscFunctionBeginUser; 
  /* First edge interpreted as uL, 2nd as uR. Use the user inputted Riemann function. */
  ierr = fvnet->physics.riemann(fvnet->physics.user,dof,uV,uV+dof,flux,maxspeed);CHKERRQ(ierr);
  /* Copy the flux */
  for (i = 0; i<dof; i++) {
    flux[i+dof] = flux[i];
  }
  PetscFunctionReturn(0);
}

/*2 edge vertex flux for edge 1 pointing out and edge 2 pointing in  */
static PetscErrorCode PhysicsVertexFlux_Shallow_2Edge_OutIn(const void* _fvnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed) 
{
  PetscErrorCode  ierr;
  const FVNetwork fvnet = (FVNetwork)_fvnet; 
  PetscInt        i,dof = fvnet->physics.dof; 

  PetscFunctionBeginUser; 
  /* First edge interpreted as uR, 2nd as uL. Use the user inputted Riemann function. */
  ierr = fvnet->physics.riemann(fvnet->physics.user,dof,uV+dof,uV,flux,maxspeed);CHKERRQ(ierr);
  /* Copy the flux */
  for (i = 0; i<dof; i++) {
    flux[i+dof] = flux[i];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsAssignVertexFlux_Shallow(const void* _fvnet, Junction junct)
{
  PetscFunctionBeginUser; 
  switch(junct->type)
  {
    case JUNCT: 
      if (junct->numedges == 2) {
        if (junct->dir[0] == EDGEIN) {
          if (junct->dir[1] == EDGEIN) {
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          } else { /* dir[1] == EDGEOUT */
            junct->couplingflux = PhysicsVertexFlux_Shallow_2Edge_InOut;
          }
        } else { /* dir[0] == EDGEOUT */
          if (junct->dir[1] == EDGEIN) {
            junct->couplingflux = PhysicsVertexFlux_Shallow_2Edge_OutIn;
          } else { /* dir[1] == EDGEOUT */
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          }
        }
      } else {
        /* Do the full riemann invariant solver (TO BE ADDED) */
      }
      break;
    default: 
      junct->couplingflux = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Shallow(FVNetwork fvnet)
{
  PetscErrorCode    ierr;
  ShallowCtx        *user;
  PetscFunctionList rlist = 0,rclist = 0;
  char              rname[256] = "rusanov",rcname[256] = "characteristic";

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  fvnet->physics.sample          = PhysicsSample_Shallow;
  fvnet->physics.destroy         = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.riemann         = PhysicsRiemann_Shallow_Rusanov;
  fvnet->physics.characteristic  = PhysicsCharacteristic_Shallow;
  fvnet->physics.vfluxassign     = PhysicsAssignVertexFlux_Shallow;
  fvnet->physics.user            = user;
  fvnet->physics.dof             = 2;

  ierr = PetscStrallocpy("height",&fvnet->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy("momentum",&fvnet->physics.fieldname[1]);CHKERRQ(ierr);

  user->gravity = 9.81;

  ierr = RiemannListAdd_Net(&rlist,"exact",  PhysicsRiemann_Shallow_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd_Net(&rlist,"rusanov",PhysicsRiemann_Shallow_Rusanov);CHKERRQ(ierr);
  ierr = ReconstructListAdd_Net(&rclist,"characteristic",PhysicsCharacteristic_Shallow);CHKERRQ(ierr);
  ierr = ReconstructListAdd_Net(&rclist,"conservative",PhysicsCharacteristic_Conservative);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(fvnet->comm,fvnet->prefix,"Options for Shallow","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-physics_shallow_gravity","Gravity","",user->gravity,&user->gravity,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_riemann","Riemann solver","",rlist,rname,rname,sizeof(rname),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_shallow_reconstruct","Reconstruction","",rclist,rcname,rcname,sizeof(rcname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = RiemannListFind_Net(rlist,rname,&fvnet->physics.riemann);CHKERRQ(ierr);
  ierr = ReconstructListFind_Net(rclist,rcname,&fvnet->physics.characteristic);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rlist);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSDMNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DMNetworkMonitor   monitor;

  PetscFunctionBegin;
  monitor = (DMNetworkMonitor)context;
  ierr = DMNetworkMonitorView(monitor,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  char              lname[256] = "minmod",physname[256] = "shallow",tname[256] = "fixed";
  PetscFunctionList limiters = 0,physics = 0,timestep = 0;
  MPI_Comm          comm;
  TS                ts;
  FVNetwork         fvnet;
  PetscInt          steps,draw = 0;
  PetscBool         viewdm = PETSC_FALSE;
  PetscReal         ptime,maxtime;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  IS                slow = NULL,fast = NULL,buffer = NULL;
  RhsCtx            slowrhs,fastrhs,bufferrhs;

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscMalloc1(1,&fvnet);CHKERRQ(ierr);
  ierr = PetscMemzero(fvnet,sizeof(*fvnet));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* Register limiters to be available on the command line */
  ierr = PetscFunctionListAdd(&limiters,"upwind"              ,Limit_Upwind_Uni);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"lax-wendroff"        ,Limit_LaxWendroff_Uni);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"beam-warming"        ,Limit_BeamWarming_Uni);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"fromm"               ,Limit_Fromm_Uni);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"minmod"              ,Limit_Minmod_Uni);CHKERRQ(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);

  /* Register time step functions to be available on the command line */
  ierr = PetscFunctionListAdd(&timestep,"fixed"         ,FVNetwork_GetTimeStep_Fixed);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&timestep,"adaptive"      ,FVNetwork_GetTimeStep_Adaptive);CHKERRQ(ierr);

  /* Set default values */
  fvnet->comm         = comm;
  fvnet->cfl          = 0.9;
  fvnet->networktype  = 1;
  fvnet->hratio       = 2;
  maxtime             = 1.0;
  fvnet->Mx           = 12;
  fvnet->bufferwidth  = 0;
  fvnet->monifv       = PETSC_FALSE;
  fvnet->initial      = 1;
  fvnet->ymin         = 0;
  fvnet->ymax         = 2.0;
  fvnet->bufferwidth  = 4;
  fvnet->viewfv       = PETSC_FALSE; 

  /* Command Line Options */
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-limit","Name of flux imiter to use","",limiters,lname,lname,sizeof(lname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-stepsize","Name of function to adapt the timestep size","",timestep,tname,tname,sizeof(tname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-draw","Draw solution vector, bitwise OR of (1=initial,2=final,4=final error)","",draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-initial","Initial Condition (depends on the physics)","",fvnet->initial,&fvnet->initial,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-network","Network topology to load, along with boundary condition information","",fvnet->networktype,&fvnet->networktype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-exact","Compare errors with exact solution","",fvnet->exact,&fvnet->exact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simulation","Compare errors with reference solution","",fvnet->simulation,&fvnet->simulation,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cfl","CFL number to time step at","",fvnet->cfl,&fvnet->cfl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-hratio","Spacing ratio","",fvnet->hratio,&fvnet->hratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_max_time","Max Time to Run TS","",maxtime,&maxtime,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ymin","Min y-value in plotting","",fvnet->ymin,&fvnet->ymin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ymax","Max y-value in plotting","",fvnet->ymax,&fvnet->ymax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Mx","Smallest number of cells for an edge","",fvnet->Mx,&fvnet->Mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-bufferwidth","width of the buffer regions","",fvnet->bufferwidth,&fvnet->bufferwidth,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewdm","View DMNetwork Info in stdout","",viewdm,&viewdm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-moni","Monitor FVNetwork Diagnostic Info","",fvnet->monifv,&fvnet->monifv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewfv","Display Solution","",fvnet->viewfv,&fvnet->viewfv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Choose the limiter from the list of registered limiters */
  ierr = PetscFunctionListFind(limiters,lname,&fvnet->limit);CHKERRQ(ierr);
  if (!fvnet->limit) SETERRQ1(PETSC_COMM_SELF,1,"Limiter '%s' not found",lname);

  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(FVNetwork);
    ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    ierr = (*r)(fvnet);CHKERRQ(ierr);
  }
  /* Choose the function for determining timestep */
  {
    ierr = PetscFunctionListFind(timestep,tname,&fvnet->gettimestep);CHKERRQ(ierr);
    if (!fvnet->gettimestep) SETERRQ1(PETSC_COMM_SELF,1,"Timestep function '%s' not found",tname);
  }

  /* Generate Network Data */
  ierr = FVNetworkCreate(fvnet,fvnet->networktype,fvnet->Mx);CHKERRQ(ierr);
  /* Create DMNetwork */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&fvnet->network);CHKERRQ(ierr);
  if (size == 1 && fvnet->viewfv) {
    ierr = DMNetworkMonitorCreate(fvnet->network,&fvnet->monitor);CHKERRQ(ierr);
  }
  /* Set Network Data into the DMNetwork (on proc[0]) */
  ierr = FVNetworkSetComponents(fvnet);CHKERRQ(ierr);
  /* Delete unneeded data in fvnet */
  ierr = FVNetworkCleanUp(fvnet);CHKERRQ(ierr);
  /* Distribute Network */
  ierr = DMSetUp(fvnet->network);CHKERRQ(ierr); 
  if (viewdm) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nOriginal networkdm, DMView:\n");CHKERRQ(ierr);
    ierr = DMView(fvnet->network,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = DMNetworkDistribute(&fvnet->network,0);CHKERRQ(ierr);
  if (viewdm) {
    PetscPrintf(PETSC_COMM_WORLD,"\nAfter DMNetworkDistribute, DMView:\n");CHKERRQ(ierr);
    ierr = DMView(fvnet->network,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Create Vectors */
  ierr = FVNetworkCreateVectors(fvnet);CHKERRQ(ierr);
  /* Set up component dynamic data structures */
  ierr = FVNetworkBuildDynamic(fvnet);CHKERRQ(ierr);
  /* Create a time-stepping object */
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,fvnet->network);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,fvnet);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FVNetRHS,fvnet);CHKERRQ(ierr);
  ierr = TSSetPreStep(ts,FVNetworkPreStep);CHKERRQ(ierr);
  /* Setup Multirate Partitions */
  ierr = FVNetworkGenerateMultiratePartition_Preset(fvnet);CHKERRQ(ierr);
  ierr = FVNetworkFinalizePartition(fvnet);CHKERRQ(ierr);
  ierr = FVNetworkBuildMultirateIS(fvnet,&slow,&fast,&buffer);CHKERRQ(ierr);
 
  ierr = TSRHSSplitSetIS(ts,"slow",slow);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"slowbuffer",buffer);CHKERRQ(ierr);
  ierr = TSRHSSplitSetIS(ts,"fast",fast);CHKERRQ(ierr);

  slowrhs.edgelist          = fvnet->slow_edges;
  slowrhs.vtxlist           = fvnet->slow_vert;
  slowrhs.fvnet             = fvnet;
  slowrhs.wheretoputstuff   = slow;
  slowrhs.scatter           = PETSC_NULL; /* Will be created in the rhs function */

  fastrhs.edgelist          = fvnet->fast_edges;
  fastrhs.vtxlist           = fvnet->fast_vert;
  fastrhs.fvnet             = fvnet;
  fastrhs.wheretoputstuff   = fast;
  fastrhs.scatter           = PETSC_NULL; /* Will be created in the rhs function */

  bufferrhs.vtxlist         = fvnet->buf_slow_vert;
  bufferrhs.fvnet           = fvnet;
  bufferrhs.wheretoputstuff = buffer;
  bufferrhs.scatter         = PETSC_NULL; /* Will be created in the rhs function */

  ierr = TSRHSSplitSetRHSFunction(ts,"slow",NULL,FVNetRHS_Multirate,&slowrhs);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"fast",NULL,FVNetRHS_Multirate,&fastrhs);CHKERRQ(ierr);
  ierr = TSRHSSplitSetRHSFunction(ts,"slowbuffer",NULL,FVNetRHS_Buffer,&bufferrhs);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSMPRK);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /* Compute initial conditions and starting time step */
  ierr = FVNetworkSetInitial(fvnet,fvnet->X);CHKERRQ(ierr);
  ierr = FVNetRHS(ts,0,fvnet->X,fvnet->Ftmp,fvnet);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);  /* Take runtime options */
  if (size == 1 && fvnet->viewfv) {
    ierr = TSMonitorSet(ts, TSDMNetworkMonitor, fvnet->monitor, NULL);CHKERRQ(ierr);
  }
  /* Evolve the PDE network in time */
  ierr = TSSolve(ts,fvnet->X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ptime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  if (viewdm) {
    if (!rank) printf("ts X:\n");
    ierr = VecView(fvnet->X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = FVNetworkDestroy(fvnet);CHKERRQ(ierr); /* Destroy all data within the network and within fvnet */
  if (size == 1 && fvnet->viewfv) {
    ierr = DMNetworkMonitorDestroy(&fvnet->monitor);CHKERRQ(ierr);
  }
  ierr = VecScatterDestroy(&slowrhs.scatter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&fastrhs.scatter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&bufferrhs.scatter);CHKERRQ(ierr);
  ierr = DMDestroy(&fvnet->network);CHKERRQ(ierr);
  ierr = ISDestroy(&slow);CHKERRQ(ierr);
  ierr = ISDestroy(&fast);CHKERRQ(ierr);
  ierr = ISDestroy(&buffer);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&limiters);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&timestep);CHKERRQ(ierr);
  ierr = PetscFree(fvnet);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    build:
      requires:
      depends: ./fvnet/fvnet.c ./fvnet/fvfunctions.c ./fvnet/fvnetmprk.c ./fvnet/fvnetts.c ./fvnet/limiters.c
    test:
      suffix: 1
      args: -Mx 20 -network 0 -initial 1 -hratio 2 -limit minmod -ts_dt 0.1 -ts_max_time 7.0 -ymax 3 -ymin 0 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1 -bufferwidth 4 -stepsize adaptive -moni 3
      output_file: output/ex9_1.out

    test:
      suffix: 2
      nsize: 4
      args: -Mx 20 -network 0 -initial 1 -hratio 2 -limit minmod -ts_dt 0.1 -ts_max_time 7.0 -ymax 3 -ymin 0 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1 -bufferwidth 4 -stepsize adaptive -moni 3
      output_file: output/ex9_1.out

TEST*/
