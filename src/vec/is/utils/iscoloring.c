
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: iscoloring.c,v 1.34 1998/12/03 03:56:32 bsmith Exp bsmith $";
#endif

#include "sys.h"   /*I "sys.h" I*/
#include "is.h"    /*I "is.h"  I*/

#undef __FUNC__  
#define __FUNC__ "ISColoringDestroy"
/*@C
   ISColoringDestroy - Destroys a coloring context.

   Collective on ISColoring

   Input Parameter:
.  iscoloring - the coloring context

.seealso: ISColoringView(), MatGetColoring()
@*/
int ISColoringDestroy(ISColoring iscoloring)
{
  int i,ierr;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring);

  for ( i=0; i<iscoloring->n; i++ ) {
    ierr = ISDestroy(iscoloring->is[i]); CHKERRQ(ierr);
  }
  PetscCommDestroy_Private(&iscoloring->comm);
  PetscFree(iscoloring->is);
  PetscFree(iscoloring);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISColoringView"
/*@C
   ISColoringView - Views a coloring context.

   Collective on ISColoring unless Viewer is VIEWER_STDOUT_SELF

   Input Parameters:
+  iscoloring - the coloring context
-  viewer- the viewer with which to view

.seealso: ISColoringDestroy(), MatGetColoring()
@*/
int ISColoringView(ISColoring iscoloring,Viewer viewer)
{
  int        i,ierr;
  ViewerType vtype;
  FILE       *fd;

  PetscFunctionBegin;
  PetscValidPointer(iscoloring);

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    MPI_Comm comm;
    int      rank;
    ierr = PetscObjectGetComm((PetscObject)viewer,&comm); CHKERRQ(ierr);
    MPI_Comm_rank(comm,&rank);
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscSynchronizedFPrintf(comm,fd,"[%d] Number of colors %d\n",rank,iscoloring->n);
    PetscSynchronizedFlush(comm);
  }

  for ( i=0; i<iscoloring->n; i++ ) {
    ierr = ISView(iscoloring->is[i],viewer); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISColoringGetIS"
/*@C
   ISColoringGetIS - Extracts index sets from the coloring context

   Collective on ISColoring 

   Input Parameter:
.  iscoloring - the coloring context

   Output Parameters:
+  n - number of index sets in the coloring context
-  is - array of index sets

.seealso: ISColoringView()
@*/
int ISColoringGetIS(ISColoring iscoloring,int *n,IS *is[])
{
  PetscFunctionBegin;
  PetscValidPointer(iscoloring);

  *n  = iscoloring->n;
  *is = iscoloring->is;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISColoringCreate"
/*@C
    ISColoringCreate - Generates an ISColoring context from lists (provided 
    by each processor) of colors for each node.

    Collective on MPI_Comm

    Input Parameters:
+   comm - communicator for the processors creating the coloring
.   n - number of nodes on this processor
-   colors - array containing the colors for this processor, color
             numbers begin at 0.

    Output Parameter:
.   iscoloring - the resulting coloring data structure

    Options Database Key:
.   -is_coloring_view - Activates ISColoringView()

.seealso: MatColoringCreate(), ISColoringView(),ISColoringDestroy()
@*/
int ISColoringCreate(MPI_Comm comm,int n,const int colors[],ISColoring *iscoloring)
{
  int        ierr,size,rank,base,top,tag,nc,ncwork,*mcolors,**ii,i,flg;
  MPI_Status status;
  IS         *is;

  PetscFunctionBegin;
  *iscoloring = (ISColoring) PetscMalloc(sizeof(struct _p_ISColoring));CHKPTRQ(*iscoloring);
  ierr = PetscCommDuplicate_Private(comm,&(*iscoloring)->comm,&tag);CHKERRQ(ierr);
  comm = (*iscoloring)->comm;

  /* compute the number of the first node on my processor */
  MPI_Comm_size(comm,&size);

  /* should use MPI_Scan() */
  MPI_Comm_rank(comm,&rank);
  if (rank == 0) {
    base = 0;
    top  = n;
  } else {
    ierr = MPI_Recv(&base,1,MPI_INT,rank-1,tag,comm,&status);CHKERRQ(ierr);
    top = base+n;
  }
  if (rank < size-1) {
    ierr = MPI_Send(&top,1,MPI_INT,rank+1,tag,comm);CHKERRQ(ierr);
  }

  /* compute the total number of colors */
  ncwork = 0;
  for ( i=0; i<n; i++ ) {
    if (ncwork < colors[i]) ncwork = colors[i];
  }
  ncwork++;
  ierr = MPI_Allreduce(&ncwork,&nc,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);

  /* generate the lists of nodes for each color */
  mcolors = (int *) PetscMalloc( (nc+1)*sizeof(int) ); CHKPTRQ(colors);
  PetscMemzero(mcolors,nc*sizeof(int));
  for ( i=0; i<n; i++ ) {
    mcolors[colors[i]]++;
  }

  ii    = (int **) PetscMalloc( (nc+1)*sizeof(int*) ); CHKPTRQ(ii);
  ii[0] = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(ii[0]);
  for ( i=1; i<nc; i++ ) {
    ii[i] = ii[i-1] + mcolors[i-1];
  }
  PetscMemzero(mcolors,nc*sizeof(int));
  for ( i=0; i<n; i++ ) {
    ii[colors[i]][mcolors[colors[i]]++] = i + base;
  }
  is  = (IS *) PetscMalloc( (nc+1)*sizeof(IS) ); CHKPTRQ(is);
  for ( i=0; i<nc; i++ ) {
    ierr = ISCreateGeneral(comm,mcolors[i],ii[i],is+i); CHKERRQ(ierr);
  }

  (*iscoloring)->n    = nc;
  (*iscoloring)->is   = is;

  PetscFree(ii[0]);
  PetscFree(ii);
  PetscFree(mcolors);


  ierr = OptionsHasName(0,"-is_coloring_view",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ISColoringView(*iscoloring,VIEWER_STDOUT_((*iscoloring)->comm));CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISPartitioningToNumbering"
/*@C
    ISPartitioningToNumbering - Takes an ISPartitioning and on each processor
    generates an IS that contains a new global node number for each index based
    on the partitioing.

    Collective on IS

    Input Parameters
.   partitioning - a partitioning as generated by PartitioningApply()

    Output Parameter:
.   is - on each processor the index set that defines the global numbers 
         (in the new numbering) for all the nodes currently (before the partitioning) 
         on that processor

.seealso: PartitioningCreate(), AOCreateBasic()

@*/
int ISPartitioningToNumbering(IS part,IS *is)
{
  MPI_Comm comm;
  int      i,ierr,rank,size, *indices,np,n,*starts,*sums,*lsizes,*newi;

  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject) part,&comm);
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  /* count the number of partitions, make sure <= size */
  ierr = ISGetSize(part,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(part,&indices); CHKERRQ(ierr);
  np = 0;
  for ( i=0; i<n; i++ ) {
    np = PetscMax(np,indices[i]);
  }  
  if (np >= size) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Number of partitions larger than number of processors");
  }

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  lsizes = (int *) PetscMalloc( 3*size*sizeof(int) );CHKPTRQ(lsizes);
  starts = lsizes + size;
  sums   = starts + size;
  PetscMemzero(lsizes,size*sizeof(int));
  for ( i=0; i<n; i++ ) {
    lsizes[indices[i]]++;
  }  
  ierr = MPI_Allreduce(lsizes,sums,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  ierr = MPI_Scan(lsizes,starts,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  for ( i=0; i<size; i++ ) {
    starts[i] -= lsizes[i];
  }
  for ( i=1; i<size; i++ ) {
    sums[i]   += sums[i-1];
    starts[i]  += sums[i-1];
  }

  /* 
      For each local index give it the new global number
  */
  newi = (int *) PetscMalloc( (n+1)*sizeof(int) );CHKPTRQ(newi);
  for ( i=0; i<n; i++ ) {
    newi[i] = starts[indices[i]]++;
  }
  PetscFree(lsizes);

  ierr = ISRestoreIndices(part,&indices); CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n,newi,is);CHKERRQ(ierr);
  PetscFree(newi);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISPartitioningCount"
/*@C
    ISPartitioningCount - Takes a ISPartitioning and determines the number of 
    resulting elements on each processor

    Collective on IS

    Input Parameters:
.   partitioning - a partitioning as generated by PartitioningApply()

    Output Parameter:
.   count - array of length size of communicator associated with IS, contains 
           the number of elements assigned to each processor

.seealso: PartitioningCreate(), AOCreateBasic(), ISPartitioningToNumbering()

@*/
int ISPartitioningCount(IS part,int count[])
{
  MPI_Comm comm;
  int      i,ierr,size, *indices,np,n,*lsizes;

  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject) part,&comm);
  MPI_Comm_size(comm,&size);

  /* count the number of partitions, make sure <= size */
  ierr = ISGetSize(part,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(part,&indices); CHKERRQ(ierr);
  np = 0;
  for ( i=0; i<n; i++ ) {
    np = PetscMax(np,indices[i]);
  }  
  if (np >= size) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Number of partitions larger than number of processors");
  }

  /*
        lsizes - number of elements of each partition on this particular processor
        sums - total number of "previous" nodes for any particular partition
        starts - global number of first element in each partition on this processor
  */
  lsizes = (int *) PetscMalloc( size*sizeof(int) );CHKPTRQ(lsizes);
  PetscMemzero(lsizes,size*sizeof(int));
  for ( i=0; i<n; i++ ) {
    lsizes[indices[i]]++;
  }  
  ierr = ISRestoreIndices(part,&indices); CHKERRQ(ierr);
  ierr = MPI_Allreduce(lsizes,count,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  PetscFree(lsizes);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISAllGather"
/*@C
    ISAllGather - Given an index set (IS) on each processor, generates a large 
    index set (same on each processor) by concatenating together each
    processor's index set.

    Collective on IS

    Input Parameter:
.   is - the distributed index set

    Output Parameter:
.   isout - the concatenated index set (same on all processors)

    Notes: 
    ISAllGather() is clearly not scalable for large index sets.
@*/
int ISAllGather(IS is,IS *isout)
{
  int      *indices,*sizes,size,*offsets,n,*lindices,i,N,ierr;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);

  PetscObjectGetComm((PetscObject)is,&comm);
  MPI_Comm_size(comm,&size);
  sizes   = (int *) PetscMalloc(2*size*sizeof(int));CHKPTRQ(sizes);
  offsets = sizes + size;
  
  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  ierr = MPI_Allgather(&n,1,MPI_INT,sizes,1,MPI_INT,comm); CHKERRQ(ierr);
  offsets[0] = 0;
  for ( i=1;i<size; i++) offsets[i] = offsets[i-1] + sizes[i-1];
  N = offsets[size-1] + sizes[size-1];

  indices = (int *) PetscMalloc((N+1)*sizeof(int));CHKERRQ(ierr);
  ierr = ISGetIndices(is,&lindices);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(lindices,n,MPI_INT,indices,sizes,offsets,MPI_INT,comm);CHKERRQ(ierr); 
  ierr = ISRestoreIndices(is,&lindices);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF,N,indices,isout);CHKERRQ(ierr);
  PetscFree(indices);

  PetscFree(sizes);
  PetscFunctionReturn(0);
}




