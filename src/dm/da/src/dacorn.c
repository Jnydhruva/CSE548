#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dacorn.c,v 1.15 1998/04/27 15:58:33 curfman Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DASetCoordinates"
/*@
   DASetCoordinates - Sets into the DA a vector that indicates the 
      coordinates of the local nodes (including ghost nodes).

   Not Collective

   Input Parameter:
+  da - the distributed array
-  c - coordinate vector

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DAGetCoordinates()
@*/
int DASetCoordinates(DA da,Vec c)
{
  PetscFunctionBegin;
 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(c,VEC_COOKIE);
  da->coordinates = c;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGetCoordinates"
/*@
   DAGetCoordinates - Gets the node coordinates associated with a DA.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  c - coordinate vector

.keywords: distributed array, get, corners, nodes, local indices, coordinates

.seealso: DAGetGhostCorners(), DASetCoordinates()
@*/
int DAGetCoordinates(DA da,Vec *c)
{
  PetscFunctionBegin;
 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *c = da->coordinates;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DASetFieldName"
/*@
   DASetFieldName - Sets the names of individual field components in multicomponent
      vectors associated with a DA.

   Not Collective

   Input Parameters:
+  da - the distributed array
.  no - field number 0, 1, ... dof-1 for the DA
-  names - the name of the field (component)

.keywords: distributed array, get, component name

.seealso: DAGetFieldName()
@*/
int DASetFieldName(DA da,int no,const char name[])
{
  PetscFunctionBegin;
 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (no < 0 || no >= da->w) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Invalid field number");
  if (da->fieldname[no]) PetscFree(da->fieldname[no]);
  
  da->fieldname[no] = (char *) PetscMalloc((1+PetscStrlen(name))*sizeof(char));CHKPTRQ(da->fieldname[no]);
  PetscStrcpy(da->fieldname[no],name);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGetFieldName"
/*@
   DAGetFieldName - Gets the names of individual field components in multicomponent
      vectors associated with a DA.

   Not Collective

   Input Parameter:
+  da - the distributed array
-  no - field number 0, 1, ... dof-1 for the DA

   Output Parameter:
.  names - the name of the field (component)

.keywords: distributed array, get, component name

.seealso: DASetFieldName()
@*/
int DAGetFieldName(DA da,int no,char **name)
{
  PetscFunctionBegin;
 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (no < 0 || no >= da->w) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Invalid field number");
  *name = da->fieldname[no];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGetCorners"
/*@
   DAGetCorners - Returns the global (x,y,z) indices of the lower left
   corner of the local region, excluding ghost points.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  x,y,z - the corner indices (where y and z are optional; these are used
           for 2D and 3D problems)
-  m,n,p - widths in the corresponding directions (where n and p are optional;
           these are used for 2D and 3D problems)

   Note:
   The corner information is independent of the number of degrees of 
   freedom per node set with the DACreateXX() routine. Thus the x, y, z, and
   m, n, p can be thought of as coordinates on a logical grid, where each
   grid point has (potentially) several degrees of freedom.
   Any of y, z, n, and p can be passed in as PETSC_NULL if not needed.

.keywords: distributed array, get, corners, nodes, local indices

.seealso: DAGetGhostCorners()
@*/
int DAGetCorners(DA da,int *x,int *y,int *z,int *m, int *n, int *p)
{
  int w;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  /* since the xs, xe ... have all been multiplied by the number of degrees 
     of freedom per cell, w = da->w, we divide that out before returning.*/
  w = da->w;  
  if (x) *x = da->xs/w; if(m) *m = (da->xe - da->xs)/w;
  /* the y and z have NOT been multiplied by w */
  if (y) *y = da->ys;   if (n) *n = (da->ye - da->ys);
  if (z) *z = da->zs;   if (p) *p = (da->ze - da->zs); 
  PetscFunctionReturn(0);
} 

