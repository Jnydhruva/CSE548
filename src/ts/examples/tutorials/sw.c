#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

#define sbc_uv(a) ((a)<0 ? (-(a)-1) : (a))
#define nbc_uv(a) ((a)>My-1 ? (2*My-(a)-1) : (a))
#define sbc_h(a) ((a)<0 ? 0 : (a))
#define nbc_h(a) ((a)>My-1 ? (My-1) : (a))

typedef struct {
  PetscScalar u,v,h;
} Field;

typedef struct {
  PetscReal EarthRadius;
  PetscReal Gravity;
  PetscReal AngularSpeed;
  PetscReal alpha,phi;
  PetscInt  p;
  PetscInt  q;
} Model_SW;

PetscErrorCode InitializeLambda(DM da,Vec lambda,PetscReal x,PetscReal y)
{
   PetscInt i,j,Mx,My,xs,ys,xm,ym;
   PetscErrorCode ierr;
   Field **uarr;
   PetscFunctionBegin;

   ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
   /* locate the global i index for x and j index for y */
   i = (PetscInt)(x/(2*PETSC_PI)*(Mx-1)); /* longitude */
   j = (PetscInt)((2*y/PETSC_PI+0.5)*(My-1)); /* latitude */
   ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

   if (xs <= i && i < xs+xm && ys <= j && j < ys+ym) {
     /* the i,j vertex is on this process */
     ierr = DMDAVecGetArray(da,lambda,&uarr);CHKERRQ(ierr);
     uarr[j][i].h = 1.0;
     ierr = DMDAVecRestoreArray(da,lambda,&uarr);CHKERRQ(ierr);
   }
   PetscFunctionReturn(0);
}


PetscErrorCode InitialConditions(DM da,Vec U,Model_SW *sw)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      a,omega,g,phi,u0,dlat,lat,lon;
  Field          **uarr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  phi   = sw->phi;
  u0    = 20.0;

  ierr = DMDAVecGetArray(da,U,&uarr);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  dlat = PETSC_PI/(PetscReal)(My);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) { /* latitude */
    lat = -PETSC_PI/2.+j*dlat+dlat/2; /* shift half grid size to avoid North pole and South pole */
    for (i=xs; i<xs+xm; i++) { /* longitude */
      lon = i*dlat; /* dlon = dlat */
      uarr[j][i].u = -3.*u0*PetscSinReal(lat)*PetscCosReal(lat)*PetscCosReal(lat)*PetscSinReal(lon)+u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lon);
      uarr[j][i].v = u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscCosReal(lon);
      uarr[j][i].h = (phi+2.*omega*a*u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lat)*PetscCosReal(lat)*PetscSinReal(lon))/g;
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&uarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  Model_SW      *sw = (Model_SW*)ptr;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,p,q,ph,qh;
  PetscReal      a,g,alpha,omega,lat,dlat,dlon;
  PetscScalar    fc,fnq,fsq,uc,ue,uw,uep,uwp,ueph,uwph,un,us,unq,usq,uephnqh,uwphnqh,uephsqh,uwphsqh,vc,ve,vw,vep,vwp,vn,vs,vnqh,vsqh,vephnqh,vwphnqh,vephsqh,vwphsqh,hc,he,hw,hn,hs,heph,hwph,hnqh,hsqh;
  Field          **uarr,**farr;
  Vec            localU;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  alpha = sw->alpha;
  p     = sw->p;
  q     = sw->q;
  ph    = sw->p/2; /* staggered */
  qh    = sw->q/2; /* staggered */
  dlon  = 2.*PETSC_PI/(PetscReal)(Mx); /* longitude */
  dlat  = PETSC_PI/(PetscReal)(My); /* latitude */

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&farr);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Place a solid wall at north and south boundaries, velocities are reflected,e.g. v(-1) = v(+1)
     Height is assumed to be constant, e.g. h(-1) = h(0)
     The forced velocity at the boundaries is a source of instability which causes the height of the boundary to rapidly increase
     Copyting over the next-row values can prevent the instability
  */
  if (ys == 0) {
    for (i=xs-2; i<xs+xm+2; i++) {
      for (j=1; j<=q; j++) uarr[-j][i].u = -uarr[j-1][i].u;
      for (j=1; j<=qh; j++) {
        uarr[-j][i].h = uarr[0][i].h;
        uarr[-j][i].v = -uarr[j-1][i].v;
      }
    }
  }
  if (ys+ym == My) {
    for (i=xs-2; i<xs+xm+2; i++) {
      for (j=1; j<=q; j++) uarr[My+j-1][i].u = -uarr[My-j][i].u;
      for (j=1; j<=qh; j++) {
        uarr[My+j-1][i].h = uarr[My-1][i].h;
        uarr[My+j-1][i].v = -uarr[My-j][i].v;
      }
    }
  }
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) { /* latitude */
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);
    fnq = 2.*omega*PetscSinReal(lat+q*dlat);
    fsq = 2.*omega*PetscSinReal(lat-q*dlat);
    for (i=xs; i<xs+xm; i++) { /* longitude */
      uc      = uarr[j][i].u;
      uep     = uarr[j][i+p].u;
      uwp     = uarr[j][i-p].u;
      ueph    = uarr[j][i+ph].u;
      uwph    = uarr[j][i-ph].u;
      ue      = uarr[j][i+1].u;
      uw      = uarr[j][i-1].u;
      un      = uarr[j+1][i].u;
      us      = uarr[j-1][i].u;
      unq     = uarr[j+q][i].u;
      usq     = uarr[j-q][i].u;
      uephnqh = uarr[j+qh][i+ph].u;
      uwphnqh = uarr[j+qh][i-ph].u;
      uephsqh = uarr[j-qh][i+ph].u;
      uwphsqh = uarr[j-qh][i-ph].u;
      vc      = uarr[j][i].v;
      ve      = uarr[j][i+1].v;
      vw      = uarr[j][i-1].v;
      vep     = uarr[j][i+p].v;
      vwp     = uarr[j][i-p].v;
      vn      = uarr[j+1][i].v;
      vs      = uarr[j-1][i].v;
      vnqh    = uarr[j+qh][i].v;
      vsqh    = uarr[j-qh][i].v;
      vephnqh = uarr[j+qh][i+ph].v;
      vwphnqh = uarr[j+qh][i-ph].v;
      vephsqh = uarr[j-qh][i+ph].v;
      vwphsqh = uarr[j-qh][i-ph].v;
      hc      = uarr[j][i].h;
      he      = uarr[j][i+1].h;
      hw      = uarr[j][i-1].h;
      hn      = uarr[j+1][i].h;
      hs      = uarr[j-1][i].h;
      heph    = uarr[j][i+ph].h;
      hwph    = uarr[j][i-ph].h;
      hnqh    = uarr[j+qh][i].h;
      hsqh    = uarr[j-qh][i].h;

      farr[j][i].u = -1./(2.*a*dlat)*(uc/PetscCosReal(lat)*(ue-uw)+vc*(un-us)+2.*g/(p*PetscCosReal(lat))*(heph-hwph))
                    +(1.-alpha)*(fc+uc/a*PetscTanReal(lat))*vc
                    +alpha/2.*(fc+uep/a*PetscTanReal(lat))*vep
                    +alpha/2.*(fc+uwp/a*PetscTanReal(lat))*vwp;
      farr[j][i].v = -1./(2.*a*dlat)*(uc/PetscCosReal(lat)*(ve-vw)+vc*(vn-vs)+2.*g/q*(hnqh-hsqh))
                    -(1.-alpha)*(fc+uc/a*PetscTanReal(lat))*uc
                    -alpha/2.*(fnq+unq/a*PetscTanReal(lat+q*dlat))*unq
                    -alpha/2.*(fsq+usq/a*PetscTanReal(lat-q*dlat))*usq;
      farr[j][i].h = -1./(2.*a*dlat)*(
                     uc/PetscCosReal(lat)*(he-hw)
                    +vc*(hn-hs)
                    +2.*hc/PetscCosReal(lat)*((1.-alpha)*(ueph-uwph)+alpha/2.*(uephnqh-uwphnqh+uephsqh-uwphsqh))/p
                    +2.*hc/PetscCosReal(lat)*((1.-alpha)*(vnqh*PetscCosReal(lat+qh*dlat)-vsqh*PetscCosReal(lat-qh*dlat))+alpha/2.*(vephnqh*PetscCosReal(lat+qh*dlat)-vephsqh*PetscCosReal(lat-qh*dlat)+vwphnqh*PetscCosReal(lat+qh*dlat)-vwphsqh*PetscCosReal(lat-qh*dlat)))/q
                     );
    }
  }
  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&farr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ptx)
{
  Model_SW       *sw=(Model_SW*)ptx;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,p,q,ph,qh;
  PetscReal      a,g,alpha,omega,lat,dlat,dlon;
  Field          **uarr;
  Vec            localU;
  MatStencil     stencil[19],rowstencil;
  PetscScalar    entries[19];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  alpha = sw->alpha;
  p     = sw->p;
  q     = sw->q;
  ph    = sw->p/2; /* staggered */
  qh    = sw->q/2; /* staggered */
  dlon  = 2.*PETSC_PI/(PetscReal)(Mx); /* longitude */
  dlat  = PETSC_PI/(PetscReal)(My); /* latitude */

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,localU,&uarr);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Place a solid wall at north and south boundaries, velocities are reflected,e.g. v(-1) = v(+1)
     Height is assumed to be constant, e.g. h(-1) = h(0)
  */
  if (ys == 0) {
    for (i=xs-2; i<xs+xm+2; i++) {
      for (j=1; j<=q; j++) uarr[-j][i].u = -uarr[j-1][i].u;
      for (j=1; j<=qh; j++) {
        uarr[-j][i].h = uarr[0][i].h;
        uarr[-j][i].v = -uarr[j-1][i].v;
      }
    }
  }
  if (ys+ym == My) {
    for (i=xs-2; i<xs+xm+2; i++) {
      for (j=1; j<=q; j++) uarr[My+j-1][i].u = -uarr[My-j][i].u;
      for (j=1; j<=qh; j++) {
        uarr[My+j-1][i].h = uarr[My-1][i].h;
        uarr[My+j-1][i].v = -uarr[My-j][i].v;
      }
    }
  }

  for (i=0; i<19; i++) stencil[i].k = 0;
  rowstencil.k = 0;
  rowstencil.c = 0;
  for (j=ys; j<ys+ym; j++) { /* checked */
    PetscReal fc;
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);

    /* Relocate the ghost points at north and south boundaries */
    stencil[0].j  = nbc_uv(j+1);
    stencil[1].j  = j;
    stencil[2].j  = j;
    stencil[3].j  = j;
    stencil[4].j  = j;
    stencil[5].j  = j;
    stencil[6].j  = sbc_uv(j-1);
    stencil[7].j  = j;
    stencil[8].j  = j;
    stencil[9].j  = j;
    stencil[10].j = j;
    stencil[11].j = j;

    rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal vc,vep,vwp,uc,ue,uw,un,us,uep,uwp;
      uc  = uarr[j][i].u;
      ue  = uarr[j][i+1].u;
      uw  = uarr[j][i-1].u;
      un  = uarr[j+1][i].u;
      us  = uarr[j-1][i].u;
      uep = uarr[j][i+p].u;
      uwp = uarr[j][i-p].u;
      vc  = uarr[j][i].v;
      vep = uarr[j][i+p].v;
      vwp = uarr[j][i-p].v;

      stencil[0].i  = i;    stencil[0].c  = 0; entries[0]  = -1./(2.*a*dlat)*vc; /* un */
      stencil[1].i  = i-p; stencil[1].c  = 0; entries[1]  = alpha/2./a*PetscTanReal(lat)*vwp; /* uwp */
      stencil[2].i  = i-1;  stencil[2].c  = 0; entries[2]  = 1./(2.*a*dlat)*uc/PetscCosReal(lat); /* uw */
      stencil[3].i  = i;    stencil[3].c  = 0; entries[3]  = -1./(2.*a*dlat)/PetscCosReal(lat)*(ue-uw)+(1.-alpha)/a*PetscTanReal(lat)*vc; /* uc */
      stencil[4].i  = i+1;  stencil[4].c  = 0; entries[4]  = -entries[2]; /* ue */
      stencil[5].i  = i+p;  stencil[5].c  = 0; entries[5]  = alpha/2./a*PetscTanReal(lat)*vep; /* uep */
      stencil[6].i  = i;    stencil[6].c  = 0; entries[6]  = -entries[0]; /* us */
      stencil[7].i  = i-p;  stencil[7].c  = 1; entries[7]  = alpha/2.*(fc+uwp/a*PetscTanReal(lat)); /* vwp */
      stencil[8].i  = i;    stencil[8].c  = 1; entries[8]  = -1./(2.*a*dlat)*(un-us)+(1.-alpha)*(fc+uc/a*PetscTanReal(lat)); /* vc */
      stencil[9].i  = i+p; stencil[9].c  = 1; entries[9]  = alpha/2.*(fc+uep/a*PetscTanReal(lat)); /* vep */
      stencil[10].i = i-ph; stencil[10].c = 2; entries[10] = 1./(2.*a*dlat)*2.*g/(p*PetscCosReal(lat));/* hwph */
      stencil[11].i = i+ph; stencil[11].c = 2; entries[11] = -entries[10]; /* heph */

      /* flip the sign */
      if (j==0) entries[6] = -entries[6];
      if (j==My-1) entries[0] = -entries[0];

      rowstencil.i = i;
      /* for (int k=0;k<19;k++) entries[k] += 30000+k+10*j+1000*i; for debugging */
      ierr = MatSetValuesStencil(A,1,&rowstencil,12,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  rowstencil.c = 1;
  for (j=ys; j<ys+ym; j++) {
    PetscReal fc,fnq,fsq;
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);
    fnq = 2.*omega*PetscSinReal(lat+q*dlat);
    fsq = 2.*omega*PetscSinReal(lat-q*dlat);

    /* Relocate the ghost points at north and south boundaries */
    stencil[0].j = nbc_uv(j+q);
    stencil[1].j = j;
    stencil[2].j = sbc_uv(j-q);
    stencil[3].j = nbc_uv(j+1);
    stencil[4].j = j;
    stencil[5].j = j;
    stencil[6].j = j;
    stencil[7].j = sbc_uv(j-1);
    stencil[8].j = nbc_h(j+qh);
    stencil[9].j = sbc_h(j-qh);

    rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal uc,unq,usq,ve,vs,vw,vn,vc;
      uc  = uarr[j][i].u;
      unq = uarr[j+q][i].u;
      usq = uarr[j-q][i].u;
      ve  = uarr[j][i+1].v;
      vs  = uarr[j-1][i].v;
      vw  = uarr[j][i-1].v;
      vn  = uarr[j+1][i].v;
      vc  = uarr[j][i].v;

      stencil[0].i = i;   stencil[0].c = 0; entries[0] = -alpha/2.*(fnq+2.*unq/a*PetscTanReal(lat+q*dlat)); /* unq */
      stencil[1].i = i;   stencil[1].c = 0; entries[1] = -1./(2.*a*dlat*PetscCosReal(lat))*(ve-vw)-(1.-alpha)*(fc+2.*uc/a*PetscTanReal(lat)); /* uc */
      stencil[2].i = i;   stencil[2].c = 0; entries[2] = -alpha/2.*(fsq+2.*usq/a*PetscTanReal(lat-q*dlat)); /* usq */
      stencil[3].i = i;   stencil[3].c = 1; entries[3] = -1./(2.*a*dlat)*vc; /* vn */
      stencil[4].i = i-1; stencil[4].c = 1; entries[4] = 1./(2.*a*dlat*PetscCosReal(lat))*uc; /* vw */
      stencil[5].i = i;   stencil[5].c = 1; entries[5] = -1./(2.*a*dlat)*(vn-vs); /* vc */
      stencil[6].i = i+1; stencil[6].c = 1; entries[6] = -entries[4]; /* ve */
      stencil[7].i = i;   stencil[7].c = 1; entries[7] = -entries[3]; /* vs */
      stencil[8].i = i;   stencil[8].c = 2; entries[8] = -g/(a*dlat*q); /* hnqh */
      stencil[9].i = i;   stencil[9].c = 2; entries[9] = -entries[8]; /* hsqh */

      /* flip the sign */
      if (j < q) entries[2] = -entries[2];
      if (j > My-q-1) entries[0] = -entries[0];
      if (j == 0) entries[7] = -entries[7];
      if (j == My-1) entries[3] = -entries[3];
      rowstencil.i = i;
      /* for (int k=0;k<19;k++) entries[k] += 50000+k+10*j+1000*i; for debugging */
      ierr = MatSetValuesStencil(A,1,&rowstencil,10,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  rowstencil.c = 2;
  for (j=ys; j<ys+ym; j++) {
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */

    /* Relocate the ghost points at north and south boundaries */
    stencil[0].j  = nbc_uv(j+qh);
    stencil[1].j  = nbc_uv(j+qh);
    stencil[2].j  = j;
    stencil[3].j  = j;
    stencil[4].j  = j;
    stencil[5].j  = sbc_uv(j-qh);
    stencil[6].j  = sbc_uv(j-qh);
    stencil[7].j  = nbc_uv(j+qh);
    stencil[8].j  = nbc_uv(j+qh);
    stencil[9].j  = nbc_uv(j+qh);
    stencil[10].j = j;
    stencil[11].j = sbc_uv(j-qh);
    stencil[12].j = sbc_uv(j-qh);
    stencil[13].j = sbc_uv(j-qh);
    stencil[14].j = nbc_h(j+1);
    stencil[15].j = j;
    stencil[16].j = j;
    stencil[17].j = j;
    stencil[18].j = sbc_h(j-1);

    rowstencil.j  = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal uc,ueph,uwph,uephnqh,uwphnqh,uephsqh,uwphsqh,vc,vnqh,vsqh,vephnqh,vwphnqh,vephsqh,vwphsqh,hc,he,hw,hs,hn;
      uc      = uarr[j][i].u;
      ueph    = uarr[j][i+ph].u;
      uwph    = uarr[j][i-ph].u;
      uephnqh = uarr[j+qh][i+ph].u;
      uwphnqh = uarr[j+qh][i-ph].u;
      uephsqh = uarr[j-qh][i+ph].u;
      uwphsqh = uarr[j-qh][i-ph].u;
      vc      = uarr[j][i].v;
      vnqh    = uarr[j+qh][i].v;
      vsqh    = uarr[j-qh][i].v;
      vephnqh = uarr[j+qh][i+ph].v;
      vwphnqh = uarr[j+qh][i-ph].v;
      vephsqh = uarr[j-qh][i+ph].v;
      vwphsqh = uarr[j-qh][i-ph].v;
      hc      = uarr[j][i].h;
      he      = uarr[j][i+1].h;
      hw      = uarr[j][i-1].h;
      hs      = uarr[j-1][i].h;
      hn      = uarr[j+1][i].h;

      stencil[0].i  = i-ph; stencil[0].c  = 0; entries[0]  = 1./(2.*a*dlat*PetscCosReal(lat)*2.*p)*2.*hc*alpha; /* uwphnqh */
      stencil[1].i  = i+ph; stencil[1].c  = 0; entries[1]  = -entries[0]; /* uephnqh */
      stencil[2].i  = i-ph; stencil[2].c  = 0; entries[2]  = 1./(2.*a*dlat*PetscCosReal(lat)*p)*2.*hc*(1.-alpha); /* uwph */
      stencil[3].i  = i;    stencil[3].c  = 0; entries[3]  = -1./(2.*a*dlat*PetscCosReal(lat))*(he-hw); /* uc */
      stencil[4].i  = i+ph; stencil[4].c  = 0; entries[4]  = -entries[2]; /* ueph */
      stencil[5].i  = i-ph; stencil[5].c  = 0; entries[5]  = entries[0]; /* uwphsqh */
      stencil[6].i  = i+ph; stencil[6].c  = 0; entries[6]  = -entries[0]; /* uephsqh */
      stencil[7].i  = i-ph; stencil[7].c  = 1; entries[7]  = -1./(2.*a*dlat*PetscCosReal(lat)*2.*q)*2.*hc*alpha*PetscCosReal(lat+qh*dlat); /* vwphnqh */
      stencil[8].i  = i;    stencil[8].c  = 1; entries[8]  = -1./(2.*a*dlat*PetscCosReal(lat)*q)*2.*hc*(1.-alpha)*PetscCosReal(lat+qh*dlat); /* vnqh */
      stencil[9].i  = i+ph; stencil[9].c  = 1; entries[9]  = entries[7]; /* vephnqh */
      stencil[10].i = i;    stencil[10].c = 1; entries[10] = -1./(2.*a*dlat)*(hn-hs); /* vc */
      stencil[11].i = i-ph; stencil[11].c = 1; entries[11] = 1./(2.*a*dlat*PetscCosReal(lat)*2.*q)*(2.*hc*alpha*PetscCosReal(lat-qh*dlat)); /* vwphsqh */
      stencil[12].i = i;    stencil[12].c = 1; entries[12] = 1./(2.*a*dlat*PetscCosReal(lat)*q)*2.*hc*(1.-alpha)*PetscCosReal(lat-qh*dlat); /* vsqh */
      stencil[13].i = i+ph; stencil[13].c = 1; entries[13] = entries[11]; /* vephsqh */
      stencil[14].i = i;    stencil[14].c = 2; entries[14] = -1./(2.*a*dlat)*vc; /* hn */
      stencil[15].i = i-1;  stencil[15].c = 2; entries[15] = 1./(2.*a*dlat*PetscCosReal(lat))*uc; /* hw */
      stencil[16].i = i;    stencil[16].c = 2; entries[16] = -1./(2.*a*dlat)*(2./PetscCosReal(lat)*((1.-alpha)*(ueph-uwph)+alpha/2.*(uephnqh-uwphnqh+uephsqh-uwphsqh))/p + 2./PetscCosReal(lat)*((1.-alpha)*(vnqh*PetscCosReal(lat+qh*dlat)-vsqh*PetscCosReal(lat-qh*dlat))+alpha/2.*(vephnqh*PetscCosReal(lat+qh*dlat)-vephsqh*PetscCosReal(lat-qh*dlat)+vwphnqh*PetscCosReal(lat+qh*dlat)-vwphsqh*PetscCosReal(lat-qh*dlat)))/q); /* hc */
      stencil[17].i = i+1;  stencil[17].c = 2; entries[17] = -entries[15]; /* he */
      stencil[18].i = i;    stencil[18].c = 2; entries[18] = -entries[14]; /* hs */

      /* flip the sign */
      if (j < qh) {
        entries[5]  = -entries[5];
        entries[6]  = -entries[6];
        entries[11] = -entries[11];
        entries[12] = -entries[12];
        entries[13] = -entries[13];
      }
      if (j > My-qh-1) {
        entries[0] = -entries[0];
        entries[1] = -entries[1];
        entries[7] = -entries[7];
        entries[8] = -entries[8];
        entries[9] = -entries[9];
      }
      rowstencil.i = i;
      /* for (int k=0;k<19;k++) entries[k] += 70000+k+10*j+1000*i; for debugging */
      ierr = MatSetValuesStencil(A,1,&rowstencil,19,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscBool      forwardonly;
  TS             ts;
  Vec            U;
  DM             da;
  Model_SW       sw;
  Vec            lambda[1];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  sw.Gravity      = 9.8;
  sw.EarthRadius  = 6.37e6;
  sw.alpha        = 1./3.;
  sw.phi          = 5.768e4;
  sw.AngularSpeed = 7.292e-5;
  sw.p            = 4;
  sw.q            = 2;

  ierr = PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,150,75,PETSC_DECIDE,PETSC_DECIDE,3,4,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"h");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&U);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&sw);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,&sw);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);

  ierr = InitialConditions(da,U,&sw);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  /*
    Have the TS save its trajectory so that TSAdjointSolve() may be used
  */
  if (!forwardonly) { ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr); }

  ierr = TSSetMaxSteps(ts,3600);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,32.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  if (!forwardonly) {
    ierr = VecDuplicate(U,&lambda[0]);CHKERRQ(ierr);
    ierr = InitializeLambda(da,lambda[0],PETSC_PI,0);CHKERRQ(ierr);
    ierr = TSSetCostGradients(ts,1,lambda,NULL);CHKERRQ(ierr);
    ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
    ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
