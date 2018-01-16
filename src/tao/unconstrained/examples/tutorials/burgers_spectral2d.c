
static char help[] ="Solves a simple data assimilation problem with one dimensional advection diffusion equation using TSAdjoint\n\n";

/*

    Not yet tested in parallel

*/
/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^heat equation
   Concepts: TS^diffusion equation
   Concepts: adjoints
   Processors: n
*/

/* ------------------------------------------------------------------------

   This program uses the one-dimensional advection-diffusion equation),
       u_t = mu*u_xx - a u_x,
   on the domain 0 <= x <= 1, with periodic boundary conditions

   to demonstrate solving a data assimilation problem of finding the initial conditions
   to produce a given solution at a fixed time.

   The operators are discretized with the spectral element method

  ------------------------------------------------------------------------- */

#include <petsctao.h>
#include <petscts.h>
#include <petscgll.h>
#include <petscdraw.h>
#include <petscdmda.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
  PetscInt    N;             /* grid points per elements*/
  PetscInt    Ex;              /* number of elements */
  PetscInt    Ey;              /* number of elements */
  PetscReal   tol_L2,tol_max; /* error norms */
  PetscInt    steps;          /* number of timesteps */
  PetscReal   Tend;           /* endtime */
  PetscReal   mu;             /* viscosity */
  PetscReal   Lx;              /* total length of domain */ 
  PetscReal   Ly;              /* total length of domain */     
  PetscReal   Lex; 
  PetscReal   Ley; 
  PetscInt    lenx;
  PetscInt    leny;
  PetscReal   Tadj;
} PetscParam;

typedef struct {
  Vec         obj;               /* desired end state */
  Vec         grid;              /* total grid */   
  Vec         grad;
  Vec         ic;
  Vec         curr_sol;
  Vec         true_solution;     /* actual initial conditions for the final solution */
} PetscData;

typedef struct {
  Vec         grid;              /* total grid */   
  Vec         mass;              /* mass matrix for total integration */
  Mat         stiff;             /* stifness matrix */
  Mat         keptstiff;
  Mat         grad;
  Mat         opadd;
  PetscGLL    gll;
} PetscSEMOperators;

typedef struct {
  DM                da;                /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
  PetscData         dat;
  TS                ts;
  PetscReal         initial_dt;
  PetscReal         *solutioncoefficients;
  PetscInt          ncoeff;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
extern PetscErrorCode RHSMatrixHeatgllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSMatrixAdvectiongllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSAdjointgllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSFunctionHeat(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode TrueSolution(Vec,AppCtx*);
extern PetscErrorCode ComputeObjective(PetscReal,Vec,AppCtx*);
extern PetscErrorCode MonitorError(Tao,void*);
extern PetscErrorCode ComputeSolutionCoefficients(AppCtx*);
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  Tao            tao;
  Vec            u,uu;                      /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt       xs, xm, ys,ym, j, ix,iy;
  PetscInt       indx,indy;
  PetscReal      x,y, *wrk_ptr1, **wrk_ptr2;
  MatNullSpace   nsp;
  DMDACoor2d     **coors,**coorslocal;
  Vec            global,local;
  DM             cda;
  PetscInt       jx,jy;
  PetscViewer    viewfile;
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*initialize parameters */
  appctx.param.N    = 6;  /* order of the spectral element */
  appctx.param.Ex    = 2;  /* number of elements */
  appctx.param.Ey    = 4;  /* number of elements */
  appctx.param.Lx    = 1.0;  /* length of the domain */
  appctx.param.Ly    = 4.0;  /* length of the domain */
  appctx.param.mu   = 0.01; /* diffusion coefficient */
  appctx.initial_dt = 5e-3;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend  = 2;
  appctx.ncoeff      = 2;

  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ex",&appctx.param.Ex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ey",&appctx.param.Ey,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ncoeff",&appctx.ncoeff,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-Tend",&appctx.param.Tend,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&appctx.param.mu,NULL);CHKERRQ(ierr);
  appctx.param.Lex = appctx.param.Lx/appctx.param.Ex;
  appctx.param.Ley = appctx.param.Ly/appctx.param.Ey;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscGLLCreate(appctx.param.N,PETSCGLL_VIA_LINEARALGEBRA,&appctx.SEMop.gll);CHKERRQ(ierr);
  
  appctx.param.lenx = appctx.param.Ex*(appctx.param.N-1);
  appctx.param.leny = appctx.param.Ey*(appctx.param.N-1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  //ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,lenglob,1,1,NULL,&appctx.da);CHKERRQ(ierr);
  //ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  //ierr = DMSetUp(appctx.da);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,appctx.param.lenx,appctx.param.leny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da,0,"u");CHKERRQ(ierr);
  
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMCreateGlobalVector(appctx.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.obj);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.curr_sol);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx.da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  //ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1);CHKERRQ(ierr);
  
  
  DMDASetUniformCoordinates(appctx.da,0.0,appctx.param.Lx,0.0,appctx.param.Ly,0.0,0.0);
  DMGetCoordinateDM(appctx.da,&cda);
  DMGetCoordinates(appctx.da,&global);
  DMDAVecGetArray(cda,global,&coors);
  DMDAGetCorners(appctx.da,&xs,&ys,NULL,&xm,&ym,NULL);
  
  ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2);CHKERRQ(ierr);
/* Compute function over the locally owned part of the grid */
    xs=xs/(appctx.param.N-1);
    xm=xm/(appctx.param.N-1);
    ys=ys/(appctx.param.N-1);
    ym=ym/(appctx.param.N-1);
  
  /*
     Build total grid and mass over entire mesh (multi-elemental) 

  */ 
   for (ix=xs; ix<xs+xm; ix++) 
     {for (jx=0; jx<appctx.param.N-1; jx++) 
      {for (iy=ys; iy<ys+ym; iy++) 
        {for (jy=0; jy<appctx.param.N-1; jy++)   
        {
        x = (appctx.param.Lex/2.0)*(appctx.SEMop.gll.nodes[jx]+1.0)+appctx.param.Lex*ix; 
        y = (appctx.param.Ley/2.0)*(appctx.SEMop.gll.nodes[jy]+1.0)+appctx.param.Ley*iy; 
        indx=ix*(appctx.param.N-1)+jx;
        indy=iy*(appctx.param.N-1)+jy;
        coors[indy][indx].x=x;
        coors[indy][indx].y=y;
        //printf("ind[%d][%d] and j=%d lenx =%d\n",indx,indy,indy+indx*leny,lenx);
        wrk_ptr2[indy][indx]=appctx.SEMop.gll.weights[jx]*appctx.SEMop.gll.weights[jy]*.25*appctx.param.Ley*appctx.param.Lex;
        //if (j==0) wrk_ptr2[ind]+=.5*appctx.param.Le*appctx.SEMop.gll.weights[j];
          } 
         }
       }
     }
    ierr = DMDAVecRestoreArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2);CHKERRQ(ierr);
    DMDAVecRestoreArray(cda,global,&coors);

//    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"grid.m",&viewfile);CHKERRQ(ierr);
//    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
//    ierr = PetscObjectSetName((PetscObject)appctx.SEMop.mass,"glob");
//    ierr = VecView(appctx.SEMop.mass,viewfile);CHKERRQ(ierr);
//    ierr = PetscViewerPopFormat(viewfile);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.stiff);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.grad);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.opadd);CHKERRQ(ierr);
    
   /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */
  
  /* Create the TS solver that solves the ODE and its adjoint; set its options */
  ierr = TSCreate(PETSC_COMM_WORLD,&appctx.ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(appctx.ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetType(appctx.ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetDM(appctx.ts,appctx.da);CHKERRQ(ierr);
  ierr = TSSetTime(appctx.ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx.ts,appctx.initial_dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(appctx.ts,appctx.param.steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(appctx.ts,appctx.param.Tend);CHKERRQ(ierr);
  //ierr = TSSetDuration(appctx.ts,appctx.param.steps,appctx.param.Tend);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTolerances(appctx.ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(appctx.ts);CHKERRQ(ierr);
  /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
  ierr = TSGetTimeStep(appctx.ts,&appctx.initial_dt);CHKERRQ(ierr);
  ///ierr = TSSetRHSFunction(appctx.ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  //ierr = TSSetRHSJacobian(appctx.ts,appctx.SEMop.stiff,appctx.SEMop.stiff,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
  ierr = InitialConditions(appctx.dat.ic,&appctx);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&uu);CHKERRQ(ierr);
  ierr = VecCopy(appctx.dat.ic,uu);CHKERRQ(ierr);

  ierr = TSSolve(appctx.ts,appctx.dat.ic);CHKERRQ(ierr);


    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"sol2d.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.ic,"sol");
    ierr = VecView(appctx.dat.ic,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)uu,"uu");
    ierr = VecView(uu,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);

  exit(1);
  //ierr = TSSetRHSJacobian(appctx.ts,appctx.SEMop.stiff,appctx.SEMop.stiff,RHSJacobian,&appctx);CHKERRQ(ierr);
  ierr = TSSetSaveTrajectory(appctx.ts);CHKERRQ(ierr);

  /* Set Objective and Initial conditions for the problem and compute Objective function (evolution of true_solution to final time */
  ierr = ComputeSolutionCoefficients(&appctx);CHKERRQ(ierr);
  ierr = InitialConditions(appctx.dat.ic,&appctx);CHKERRQ(ierr);
  ierr = TrueSolution(appctx.dat.true_solution,&appctx);CHKERRQ(ierr);
  ierr = ComputeObjective(appctx.param.Tend,appctx.dat.obj,&appctx);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method  */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetMonitor(tao,MonitorError,&appctx,NULL);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,appctx.dat.ic);CHKERRQ(ierr);
  /* Set routine for function and gradient evaluation  */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&appctx);CHKERRQ(ierr);
  /* Check for any TAO command line options  */
  ierr = TaoSetTolerances(tao,1e-8,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = PetscFree(appctx.solutioncoefficients);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.SEMop.stiff);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.SEMop.keptstiff);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.obj);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.grid);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);CHKERRQ(ierr);
  ierr = PetscGLLDestroy(&appctx.SEMop.gll);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  ierr = TSDestroy(&appctx.ts);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
    ierr = PetscFinalize();
    return ierr;
}

/*
    Computes the coefficients for the analytic solution to the PDE
*/
PetscErrorCode ComputeSolutionCoefficients(AppCtx *appctx)
{
  PetscErrorCode    ierr;
  PetscRandom       rand;
  PetscInt          i;

  ierr = PetscMalloc1(appctx->ncoeff,&appctx->solutioncoefficients);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,.9,1.0);CHKERRQ(ierr);
  for (i=0; i<appctx->ncoeff; i++) 
    {
    ierr = PetscRandomGetValue(rand,&appctx->solutioncoefficients[i]);CHKERRQ(ierr);
    }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the initial conditions for the Tao optimization solve (these are also initial conditions for the first TSSolve()

                       The routine TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar       **s;
  const PetscScalar *xg;
  PetscErrorCode    ierr;
  PetscInt          i,j;
  DM                cda;
  Vec          global;
  DMDACoor2d        **coors;

  ierr = DMDAVecGetArray(appctx->da,u,&s);CHKERRQ(ierr);
  //ierr = DMDAVecGetArrayRead(appctx->da,appctx->SEMop.grid,&xg);CHKERRQ(ierr);
  //lenglob  = appctx->param.Ex*(appctx->param.N-1);
    
  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);

  for (i=0; i<appctx->param.lenx; i++) 
    {for (j=0; j<appctx->param.leny; j++) 
      {
      s[j][i]=2.0*appctx->param.mu*PETSC_PI*PetscSinScalar(PETSC_PI*coors[j][i].x);
      //printf("s[%d][%d] =%f\n",j,i, s[j][i]);
      } 
     }
  
  ierr = DMDAVecRestoreArray(appctx->da,u,&s);CHKERRQ(ierr);
  //ierr = DMDAVecRestoreArrayRead(appctx->da,appctx->SEMop.grid,&xg);CHKERRQ(ierr);
   //  ierr = VecView(u,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  return 0;
}


/*
   TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function. 

             InitialConditions() computes the initial conditions for the begining of the Tao iterations

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode TrueSolution(Vec u,AppCtx *appctx)
{
  PetscScalar       *s;
  const PetscScalar *xg;
  PetscErrorCode    ierr;
  PetscInt          i,lenglob;

  ierr = DMDAVecGetArray(appctx->da,u,&s);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(appctx->da,appctx->SEMop.grid,&xg);CHKERRQ(ierr);
  lenglob  = appctx->param.Ex*(appctx->param.N-1);
  
  for (i=0; i<lenglob; i++) {
      s[i]=2.0*appctx->param.mu*PETSC_PI*PetscSinScalar(PETSC_PI*xg[i])/(2.0+PetscCosScalar(PETSC_PI*xg[i]));
      } 
  ierr = DMDAVecRestoreArray(appctx->da,u,&s);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da,appctx->SEMop.grid,&xg);CHKERRQ(ierr);
  /* make sure initial conditions do not contain the constant functions, since with periodic boundary conditions the constant functions introduce a null space */
   return 0;
}
/* --------------------------------------------------------------------- */
/*
   Sets the desired profile for the final end time

   Input Parameters:
   t - final time
   obj - vector storing the desired profile
   appctx - user-defined application context

*/
PetscErrorCode ComputeObjective(PetscReal t,Vec obj,AppCtx *appctx)
{
  PetscScalar       *s;
  const PetscScalar *xg;
  PetscErrorCode    ierr;
  PetscInt          i, lenglob;

  ierr = DMDAVecGetArray(appctx->da,obj,&s);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(appctx->da,appctx->SEMop.grid,&xg);CHKERRQ(ierr);
  lenglob  = appctx->param.Ex*(appctx->param.N-1);
  
  for (i=0; i<lenglob; i++) {
      s[i]=2.0*appctx->param.mu*PETSC_PI*PetscSinScalar(PETSC_PI*xg[i])*PetscExpScalar(-PETSC_PI*PETSC_PI*t*appctx->param.mu)\
              /(2.0+PetscExpScalar(-PETSC_PI*PETSC_PI*t*appctx->param.mu)*PetscCosScalar(PETSC_PI*xg[i]));
      } 
  ierr = DMDAVecRestoreArray(appctx->da,obj,&s);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(appctx->da,appctx->SEMop.grid,&xg);CHKERRQ(ierr);
  //ierr = VecView(obj,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx          *appctx = (AppCtx*)ctx;  
  PetscScalar     **out, **temp, **tempm,**u,**wrk1;
  PetscInt        i,j;
  PetscViewer     viewfile;

  PetscFunctionBegin;

  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);
  ierr = PetscGLLElementMassCreate(&appctx->SEMop.gll,&tempm);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(appctx->da,globalin,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,globalout,&out);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,globalout,&wrk1);CHKERRQ(ierr);

  ierr= Petsctensorprod(appctx->param.N, &tempm[0][0],&u[0][0],&temp[0][0],&wrk1[0][0]);
  ierr= Petsctensorprod(appctx->param.N, &temp[0][0],&u[0][0],&tempm[0][0],&out[0][0]);
 
  for (i=0; i<appctx->param.lenx; i++)
     {for (j=0; j<appctx->param.leny; j++)
     {out[j][i]=out[j][i]+wrk1[j][i];
      }}

  ierr = DMDAVecRestoreArray(appctx->da,globalin,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da,globalout,&out);CHKERRQ(ierr);
  VecScale(globalout, -1.0);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"rhs.m",&viewfile);CHKERRQ(ierr);

  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)globalin,"in");
  ierr = VecView(globalin,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)globalout,"out");
  ierr = VecView(globalout,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);


  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec globalin,Mat A, Mat B,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx*)ctx;  
  Vec            temp;
  
  PetscFunctionBegin;

  /* old jac something is wrong */
  ierr = MatCopy(appctx->SEMop.grad,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDiagonalScale(A,globalin,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(globalin,&temp);CHKERRQ(ierr);
  ierr = MatMult(appctx->SEMop.grad,globalin,temp);CHKERRQ(ierr);
  ierr = MatDiagonalSet(A,temp,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatScale(A,-1.0);CHKERRQ(ierr);
  ierr = MatAXPY(A,1.0,appctx->SEMop.keptstiff,DIFFERENT_NONZERO_PATTERN);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */

/*
   RHSMatrixHeat - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

   Input Parameters:
   ts - the TS context
   t - current time  (ignored)
   X - current solution (ignored)
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different matrix from which the preconditioner is built
   str - flag indicating matrix structure

*/
/*
PetscErrorCode RHSMatrixHeatgllDM(TS ts,PetscReal t,Vec X,Mat A,Mat BB,void *ctx)
{
  PetscReal      **temp, **out, **tempm, **u;
  PetscReal      vv;
  AppCtx         *appctx = (AppCtx*)ctx;    
  PetscErrorCode ierr;
  PetscInt       i,xs,xn,l,j;
  PetscInt       *rowsDM;
  PetscViewer    viewfile;
  Mat            tt;
  //   Creates the element stiffness matrix for the given gll
   
  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);
  ierr = PetscGLLElementMassCreate(&appctx->SEMop.gll,&tempm);CHKERRQ(ierr);
  

 initial stuff
  
//  scale by the size of the element 
  for (i=0; i<appctx->param.N; i++) {
    vv=-appctx->param.mu*2.0/appctx->param.Le;
    for (j=0; j<appctx->param.N; j++) temp[i][j]=temp[i][j]*vv;
  }

  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);

  xs   = xs/(appctx->param.N-1);
  xn   = xn/(appctx->param.N-1);

  ierr = PetscMalloc1(appctx->param.N,&rowsDM);CHKERRQ(ierr);
//      loop over local elements    
  for (j=xs; j<xs+xn; j++) {
    for (l=0; l<appctx->param.N; l++) {
      rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;
    }
    ierr = MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(rowsDM);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecReciprocal(appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = MatDiagonalScale(A,appctx->SEMop.mass,0);CHKERRQ(ierr);
  ierr = VecReciprocal(appctx->SEMop.mass);CHKERRQ(ierr);


   ierr = PetscMalloc1(appctx->param.N,&out);CHKERRQ(ierr);
   ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&out[0]);CHKERRQ(ierr);

   ierr = PetscMalloc(sizeof(PetscScalar)*appctx->param.N*appctx->param.N,&out);CHKERRQ(ierr);

   ierr= Petsctensorprod(appctx->param.N, &tempm[0][0],&u,&temp[0][0],&out);

   //exit(1);

   ierr = MatCreateSeqDense(PETSC_COMM_SELF,appctx->param.N,appctx->param.N,&tempm[0][0],&tt);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"tensor.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)tt,"tens");
    ierr = MatView(tt,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);
 exit(1);

  
 ierr = PetscGLLElementLaplacianDestroy(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);
  return 0;
}

*/

#undef __FUNCT__
#define __FUNCT__ "RHSMatrixAdvectiongllDM"

/*
   RHSMatrixAdvection - User-provided routine to compute the right-hand-side
   matrix for the Advection equation.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

*/
PetscErrorCode RHSMatrixAdvectiongllDM(TS ts,PetscReal t,Vec X,Mat A,Mat BB,void *ctx)
{
  PetscReal      **temp;
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       xs,xn,l,j;
  PetscInt       *rowsDM;
    
     /*
       Creates the advection matrix for the given gll
    */
    ierr = PetscGLLElementAdvectionCreate(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

    ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);

    xs   = xs/(appctx->param.N-1);
    xn   = xn/(appctx->param.N-1);

    ierr = PetscMalloc1(appctx->param.N,&rowsDM);CHKERRQ(ierr);
  
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<appctx->param.N; l++) 
      {rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;}
      ierr = MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES);CHKERRQ(ierr);
      }

   MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

   ierr = VecReciprocal(appctx->SEMop.mass);CHKERRQ(ierr);
   ierr = MatDiagonalScale(A,appctx->SEMop.mass,0);CHKERRQ(ierr);
   ierr = VecReciprocal(appctx->SEMop.mass);CHKERRQ(ierr);
  
   ierr = PetscGLLElementAdvectionDestroy(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);
   
     
   return 0;
}
/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   IC   - the input vector
   ctx - optional user-defined context, as set when calling TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient

   Notes:

          The forward equation is
              M u_t = F(U)
          which is converted to
                u_t = M^{-1} F(u)
          in the user code since TS has no direct way of providing a mass matrix. The Jacobian of this is
                 M^{-1} J
          where J is the Jacobian of F. Now the adjoint equation is
                M v_t = J^T v
          but TSAdjoint does not solve this since it can only solve the transposed system for the 
          Jacobian the user provided. Hence TSAdjoint solves
                 w_t = J^T M^{-1} w  (where w = M v)
          since there is no way to indicate the mass matrix as a seperate entitity to TS. Thus one
          must be careful in initializing the "adjoint equation" and using the result. This is
          why
              G = -2 M(u(T) - u_d)
          below (instead of -2(u(T) - u_d) and why the result is
              G = G/appctx->SEMop.mass (that is G = M^{-1}w)
          below (instead of just the result of the "adjoint solve").


*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  AppCtx           *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode    ierr;
  Vec               temp, bsol;
  PetscInt          its;
  PetscReal         ff, gnorm, cnorm, xdiff,errex; 
  TaoConvergedReason reason;    
  PetscViewer        viewfile;
  //static int counter=0; it was considered for storing line search error
  char filename[24] ;
  char data[80] ;
  
  ierr = TSSetTime(appctx->ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(appctx->ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx->ts,appctx->initial_dt);CHKERRQ(ierr);
  ierr = VecCopy(IC,appctx->dat.curr_sol);CHKERRQ(ierr);

  ierr = TSSolve(appctx->ts,appctx->dat.curr_sol);CHKERRQ(ierr);
  /*
  Store current solution for comparison
  */
  ierr = VecDuplicate(appctx->dat.curr_sol,&bsol);CHKERRQ(ierr);
  ierr = VecCopy(appctx->dat.curr_sol,bsol);CHKERRQ(ierr);
  
  ierr = VecWAXPY(G,-1.0,appctx->dat.curr_sol,appctx->dat.obj);CHKERRQ(ierr);

  /*
     Compute the L2-norm of the objective function, cost function is f
  */
  ierr = VecDuplicate(G,&temp);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,G,G);CHKERRQ(ierr);
  ierr = VecDot(temp,appctx->SEMop.mass,f);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);

  //local error evaluation   
  ierr = VecDuplicate(G,&temp);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx->dat.ic,&temp);CHKERRQ(ierr);
  ierr = VecWAXPY(temp,-1.0,appctx->dat.ic,appctx->dat.true_solution);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,temp,temp);CHKERRQ(ierr);
  //for error evaluation
  ierr = VecDot(temp,appctx->SEMop.mass,&errex);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  errex  = PetscSqrtReal(errex); 

/*
     Compute initial conditions for the adjoint integration. See Notes above
  */

  ierr = VecScale(G, -2.0);CHKERRQ(ierr);
  ierr = VecPointwiseMult(G,G,appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = TSSetCostGradients(appctx->ts,1,&G,NULL);CHKERRQ(ierr);
  ierr = TSAdjointSolve(appctx->ts);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(G,G,appctx->SEMop.mass);CHKERRQ(ierr);

  ierr=  TaoGetSolutionStatus(tao, &its, &ff, &gnorm, &cnorm, &xdiff, &reason);

  //counter++; // this was for storing the error accross line searches
  PetscPrintf(PETSC_COMM_WORLD,"iteration=%D\t cost function (TAO)=%g, cost function (L2 %g), ic error %g\n",its,(double)ff,*f,errex);
  PetscSNPrintf(filename,sizeof(filename),"PDEadjoint/optimize%02d.m",its);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  PetscSNPrintf(data,sizeof(data),"TAO(%D)=%g; L2(%D)= %g ; Err(%D)=%g\n",its+1,(double)ff,its+1,*f,its+1,errex);
  PetscViewerASCIIPrintf(viewfile,data);
  ierr = PetscObjectSetName((PetscObject)appctx->SEMop.grid,"grid");
  ierr = VecView(appctx->SEMop.grid,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.obj,"obj");
  ierr = VecView(appctx->dat.obj,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)G,"Init_adj");
  ierr = VecView(G,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)IC,  "Init_ts");
  ierr = VecView(IC,viewfile);CHKERRQ(ierr);
  //ierr = PetscObjectSetName((PetscObject)appctx->dat.senmask,  "senmask");
  //ierr = VecView(appctx->dat.senmask,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)bsol,"Curr_sol");
  ierr = VecView(bsol,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.true_solution, "exact");
  ierr = VecView(appctx->dat.true_solution,viewfile);CHKERRQ(ierr);
  //ierr = PetscObjectSetName((PetscObject)appctx->SEMop.grad, "A");
  //ierr = MatView(appctx->SEMop.grad,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MonitorError(Tao tao,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  Vec            temp;
  PetscReal      nrm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(appctx->dat.ic,&temp);CHKERRQ(ierr);
  ierr = VecWAXPY(temp,-1.0,appctx->dat.ic,appctx->dat.true_solution);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,temp,temp);CHKERRQ(ierr);
  ierr = VecDot(temp,appctx->SEMop.mass,&nrm);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  nrm  = PetscSqrtReal(nrm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error for initial conditions %g\n",(double)nrm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*TEST

   build:
     requires: !complex

   test:
     requires: !single
     args: -tao_monitor  -ts_adapt_dt_max 3.e-3 -E 10 -N 8 -ncoeff 5 

   test:
     suffix: cn
     requires: !single
     args: -tao_monitor -ts_type cn -ts_dt .003 -pc_type lu -E 10 -N 8 -ncoeff 5 

TEST*/
