#if !defined(__SFBASIC_H)
#define __SFBASIC_H

#include <../src/vec/is/sf/impls/basic/sfpack.h>

#define SFBASICHEADER \
  PetscMPIInt      niranks;         /* Number of incoming ranks (ranks accessing my roots) */                                      \
  PetscMPIInt      ndiranks;        /* Number of incoming ranks (ranks accessing my roots) in distinguished set */                 \
  PetscMPIInt      *iranks;         /* Array of ranks that reference my roots */                                                   \
  PetscInt         itotal;          /* Total number of graph edges referencing my roots */                                         \
  PetscInt         *ioffset;        /* Array of length niranks+1 holding offset in irootloc[] for each rank */                     \
  PetscInt         *irootloc;       /* Incoming roots referenced by ranks starting at ioffset[rank] */                             \
  PetscInt         *irootloc_d;     /* irootloc in device memory when needed */                                                    \
  PetscSFPackOpt   rootpackopt;     /* Optimization plans to (un)pack roots based on patterns in irootloc[]. NULL for no plans */  \
  PetscSFPackOpt   selfrootpackopt; /* Optimization plans to (un)pack roots connected to local leaves */                           \
  PetscSFPack      avail;           /* One or more entries per MPI Datatype, lazily constructed */                                 \
  PetscSFPack      inuse;           /* Buffers being used for transactions that have not yet completed */                          \
  PetscBool        selfrootdups;    /* Indices of roots in irootloc[0,ioffset[ndiranks]) have dups, implying theads working ... */ \
                                    /* ... on these roots in parallel may have data race. */                                       \
  PetscBool        remoterootdups   /* Indices of roots in irootloc[ioffset[ndiranks],ioffset[niranks]) have dups */

typedef struct {
  SFBASICHEADER;
} PetscSF_Basic;

PETSC_STATIC_INLINE PetscErrorCode PetscSFGetRootInfo_Basic(PetscSF sf,PetscInt *nrootranks,PetscInt *ndrootranks,const PetscMPIInt **rootranks,const PetscInt **rootoffset,const PetscInt **rootloc)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (nrootranks)  *nrootranks  = bas->niranks;
  if (ndrootranks) *ndrootranks = bas->ndiranks;
  if (rootranks)   *rootranks   = bas->iranks;
  if (rootoffset)  *rootoffset  = bas->ioffset;
  if (rootloc)     *rootloc     = bas->irootloc;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFGetLeafInfo_Basic(PetscSF sf,PetscInt *nleafranks,PetscInt *ndleafranks,const PetscMPIInt **leafranks,const PetscInt **leafoffset,const PetscInt **leafloc,const PetscInt **leafrremote)
{
  PetscFunctionBegin;
  if (nleafranks)  *nleafranks  = sf->nranks;
  if (ndleafranks) *ndleafranks = sf->ndranks;
  if (leafranks)   *leafranks   = sf->ranks;
  if (leafoffset)  *leafoffset  = sf->roffset;
  if (leafloc)     *leafloc     = sf->rmine;
  if (leafrremote) *leafrremote = sf->rremote;
  PetscFunctionReturn(0);
}

/* Get root locations either on Host or Device */
PETSC_STATIC_INLINE PetscErrorCode PetscSFGetRootIndicesWithMemType_Basic(PetscSF sf,PetscMemType mtype, const PetscInt **rootloc)
{
  PetscSF_Basic *bas = (PetscSF_Basic*)sf->data;
  PetscFunctionBegin;
  if (rootloc) {
    if (mtype == PETSC_MEMTYPE_HOST) *rootloc = bas->irootloc;
#if defined(PETSC_HAVE_CUDA)
    else if (mtype == PETSC_MEMTYPE_DEVICE) {
      if (!bas->irootloc_d) {
        cudaError_t    err;
        PetscErrorCode ierr;
        size_t         size = bas->ioffset[bas->niranks]*sizeof(PetscInt);
        err  = cudaMalloc((void **)&bas->irootloc_d,size);CHKERRCUDA(err);
        ierr = PetscMemcpyWithMemType(PETSC_MEMTYPE_DEVICE,PETSC_MEMTYPE_HOST,bas->irootloc_d,bas->irootloc,size);CHKERRQ(ierr);
      }
      *rootloc = bas->irootloc_d;
    }
#endif
    else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType in getting rootloc %d",(int)mtype);
  }
  PetscFunctionReturn(0);
}

/* Get leaf locations either on Host (CPU) or Device (GPU) */
PETSC_STATIC_INLINE PetscErrorCode PetscSFGetLeafIndicesWithMemType_Basic(PetscSF sf,PetscMemType mtype, const PetscInt **leafloc)
{
  PetscFunctionBegin;
  if (leafloc) {
    if (mtype == PETSC_MEMTYPE_HOST) *leafloc = sf->rmine;
#if defined(PETSC_HAVE_CUDA)
    else if (mtype == PETSC_MEMTYPE_DEVICE) {
      if (!sf->rmine_d) {
        cudaError_t    err;
        PetscErrorCode ierr;
        size_t         size = sf->roffset[sf->nranks]*sizeof(PetscInt);
        err  = cudaMalloc((void **)&sf->rmine_d,size);CHKERRCUDA(err);
        ierr = PetscMemcpyWithMemType(PETSC_MEMTYPE_DEVICE,PETSC_MEMTYPE_HOST,sf->rmine_d,sf->rmine,size);CHKERRQ(ierr);
      }
      *leafloc = sf->rmine_d;
    }
#endif
    else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType in getting leafloc %d",(int)mtype);
  }
  PetscFunctionReturn(0);
}

/* A convenience struct to provide info to the following (un)packing routines so that we can treat packings to self and to remote in one loop. */
typedef struct _n_PackInfo {
  PetscInt       count;  /* Number of entries to pack, unpack etc. */
  const PetscInt *idx;   /* Indices of the entries. NULL means contiguous indices [0,count) */
  PetscSFPackOpt opt;    /* Pack optimizations */
  char           *buf;   /* The contiguous buffer where we pack to or unpack from */
  PetscBool      atomic; /* Whether the unpack routine needs to use atomics */
  const void     *data;  /* Pointer to root/leafdata. Using pack as an example, we will copy data[idx[i]] to buf[i]. */
} PackInfo;

/* Utility routine to pack selected entries of rootdata into root buffer.
  Input Arguments:
  + sf       - The SF this packing works on.
  . link     - The PetscSFPack, which gives the memtype of the roots and also provides root buffer.
  . rootloc  - Indices of the roots, only meaningful if the root space is sparse
  . rootdata - Where to read the roots.
  - sparse   - Is the root space sparse (for SFBasic, SFNeighbor)  or dense (for SFAllgatherv etc)
 */
PETSC_STATIC_INLINE PetscErrorCode PetscSFPackRootData(PetscSF sf,PetscSFPack link,const PetscInt *rootloc,const void *rootdata,PetscBool sparse)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       i;
  PetscErrorCode (*Pack)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,const void*,void*);
  PackInfo       p[2];

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Pack,sf,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  /*  Shall we assume rootdata ready to use (instead of being computed by some asynchronous kernels)?
    Currently this assumption is not enforced in PETSc. MPI does not have a concept of stream. To be safe,
    I do synchronization here. Otherwise, if we do not sync now and call the Pack kernel directly on the
    default NULL stream (assume petsc objects are also computed on it), we have to sync the NULL stream
    before calling MPI routines. So, it looks a synchronization is inevitable. We do it now and put
    pack/unpack kernels to non-NULL streams.
   */

  /* When either self or remote has something to pack, we have to sync */
  if (link->rootmtype == PETSC_MEMTYPE_DEVICE && (link->selfbuflen || link->rootbuflen)) {cudaError_t err = cudaDeviceSynchronize();CHKERRCUDA(err);}
#endif
  if (sparse) {
    /* For SFBasic and SFNeighbor, whose root space is sparse and have separate buffers for self and remote. */
    p[0].count  = link->selfbuflen;
    p[0].idx    = rootloc;
    p[0].opt    = bas->selfrootpackopt;
    p[0].buf    = link->selfbuf[link->rootmtype];
    p[0].atomic = PETSC_FALSE;
    p[0].data   = rootdata;
    if (p[0].opt && p[0].opt->all_contiguous) { /* Adjust data/idx if indices are contiguous */
      p[0].data = (const char*)rootdata + p[0].opt->start_index*link->unitbytes;
      p[0].idx  = NULL; /* NULL means indices [0,count). We shifted rootdata above to access rootdata starting from p[0].data[0]. */
      p[0].opt  = NULL;
    }

    p[1].count  = link->rootbuflen;
    p[1].idx    = rootloc+bas->ioffset[bas->ndiranks];
    p[1].opt    = bas->rootpackopt;
    p[1].buf    = link->rootbuf[link->rootmtype];
    p[1].atomic = PETSC_FALSE;
    p[1].data   = rootdata;
    if (p[1].opt && p[1].opt->all_contiguous) {
      p[1].data = (const char*)rootdata + p[1].opt->start_index*link->unitbytes;
      p[1].idx  = NULL;
      p[1].opt  = NULL;
    }
  } else SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"SFAllgather etc should directly use rootdata instead of packing it");

  ierr = PetscSFPackGetPack(link,link->rootmtype,&Pack);CHKERRQ(ierr);
  /* Only do packing when count != 0 so that we can avoid invoking empty CUDA kernels */
  for (i=0; i<2; i++) {if (p[i].count) {ierr = (*Pack)(p[i].count,p[i].idx,link,p[i].opt,p[i].data,p[i].buf);CHKERRQ(ierr);}}

#if defined(PETSC_HAVE_CUDA)
  if (link->rootmtype == PETSC_MEMTYPE_DEVICE && link->rootbuflen) { /* We only care about remote, which involves MPI */
    /* Without use_gpu_aware_mpi, we need to copy rootbuf on device to rootbuf on host. The cudaStreamSynchronize()
       is to make sure rootbuf is ready before MPI communicatioin starts.
    */
    cudaError_t err;
    if (!use_gpu_aware_mpi) {
      err  = cudaMemcpyAsync(link->rootbuf[PETSC_MEMTYPE_HOST],link->rootbuf[PETSC_MEMTYPE_DEVICE],link->rootbuflen*link->unitbytes,cudaMemcpyDeviceToHost,link->stream);CHKERRCUDA(err);
      ierr = PetscLogGpuToCpu(link->rootbuflen*link->unitbytes);CHKERRQ(ierr);
    }
    err = cudaStreamSynchronize(link->stream);CHKERRCUDA(err); /* Make it ready to call MPI */
  }
#endif
  ierr = PetscLogEventEnd(PETSCSF_Pack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Utility routine to pack selected entries of leafdata into leaf buffer */
PETSC_STATIC_INLINE PetscErrorCode PetscSFPackLeafData(PetscSF sf,PetscSFPack link,const PetscInt *leafloc,const void *leafdata,PetscBool sparse)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscErrorCode (*Pack)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,const void*,void*);
  PackInfo       p[2];

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Pack,sf,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  if (link->leafmtype == PETSC_MEMTYPE_DEVICE && (link->selfbuflen || link->leafbuflen)) {cudaError_t err = cudaDeviceSynchronize();CHKERRCUDA(err);}
#endif
  if (sparse) {
    p[0].count  = link->selfbuflen;
    p[0].idx    = leafloc;
    p[0].opt    = sf->selfleafpackopt;
    p[0].buf    = link->selfbuf[link->leafmtype];
    p[0].atomic = PETSC_FALSE;
    p[0].data   = leafdata;
    if (p[0].opt && p[0].opt->all_contiguous) {
      p[0].data = (const char*)leafdata + p[0].opt->start_index*link->unitbytes;
      p[0].idx  = NULL; /* NULL means indices [0,count). That's why we need to shift leafdata above. */
      p[0].opt  = NULL;
    }

    p[1].count  = link->leafbuflen;
    p[1].idx    = leafloc+sf->roffset[sf->ndranks];
    p[1].opt    = sf->leafpackopt;
    p[1].buf    = link->leafbuf[link->leafmtype];
    p[1].atomic = PETSC_FALSE;
    p[1].data   = leafdata;
    if (p[1].opt && p[1].opt->all_contiguous) {
      p[1].data = (const char*)leafdata + p[1].opt->start_index*link->unitbytes;
      p[1].idx  = NULL;
      p[1].opt  = NULL;
    }
  } else SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"SFAllgather etc should directly use leafdata instead of packing it");

  ierr = PetscSFPackGetPack(link,link->leafmtype,&Pack);CHKERRQ(ierr);
  for (i=0; i<2; i++) {if (p[i].count) {ierr = (*Pack)(p[i].count,p[i].idx,link,p[i].opt,p[i].data,p[i].buf);CHKERRQ(ierr);}}
#if defined(PETSC_HAVE_CUDA)
  if (link->leafmtype == PETSC_MEMTYPE_DEVICE && link->leafbuflen) {
    cudaError_t err;
    if (!use_gpu_aware_mpi) {
      err  = cudaMemcpyAsync(link->leafbuf[PETSC_MEMTYPE_HOST],link->leafbuf[PETSC_MEMTYPE_DEVICE],link->leafbuflen*link->unitbytes,cudaMemcpyDeviceToHost,link->stream);CHKERRCUDA(err);
      ierr = PetscLogGpuToCpu(link->leafbuflen*link->unitbytes);CHKERRQ(ierr);
    }
    err = cudaStreamSynchronize(link->stream);CHKERRCUDA(err);
  }
#endif
  ierr = PetscLogEventEnd(PETSCSF_Pack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Utility routine to unpack data from root buffer and Op it into selected entries of rootdata */
PETSC_STATIC_INLINE PetscErrorCode PetscSFUnpackAndOpRootData(PetscSF sf,PetscSFPack link,const PetscInt *rootloc,void *rootdata,MPI_Op op,PetscBool sparse)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode (*UnpackAndOp)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PackInfo       p[2];

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  if (!use_gpu_aware_mpi && link->rootmtype == PETSC_MEMTYPE_DEVICE) { /* Copy roots from host to device when needed */
    cudaError_t err;
    if (!link->rootbuf[PETSC_MEMTYPE_DEVICE]) {ierr = PetscMallocWithMemType(PETSC_MEMTYPE_DEVICE,link->rootbuflen*link->unitbytes,(void**)&link->rootbuf[PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);}
    err  = cudaMemcpyAsync(link->rootbuf[PETSC_MEMTYPE_DEVICE],link->rootbuf[PETSC_MEMTYPE_HOST],link->rootbuflen*link->unitbytes,cudaMemcpyHostToDevice,link->stream);CHKERRCUDA(err);
    ierr = PetscLogCpuToGpu(link->rootbuflen*link->unitbytes);CHKERRQ(ierr);
  }
#endif

  if (sparse) {
    p[0].count  = link->selfbuflen;
    p[0].idx    = rootloc;
    p[0].opt    = bas->selfrootpackopt;
    p[0].buf    = link->selfbuf[link->rootmtype];
    p[0].atomic = bas->selfrootdups;
    p[0].data   = rootdata;
    if (p[0].opt && p[0].opt->all_contiguous) {
      p[0].data = (const char*)rootdata + p[0].opt->start_index*link->unitbytes;
      p[0].idx  = NULL; /* NULL means indices [0,count). That's why we need to shift rootdata above. */
      p[0].opt  = NULL;
    }

    p[1].count  = link->rootbuflen;
    p[1].idx    = rootloc+bas->ioffset[bas->ndiranks];
    p[1].opt    = bas->rootpackopt;
    p[1].buf    = link->rootbuf[link->rootmtype];
    p[1].atomic = bas->remoterootdups;
    p[1].data   = rootdata;
    if (p[1].opt && p[1].opt->all_contiguous) {
      p[1].data = (const char*)rootdata + p[1].opt->start_index*link->unitbytes;
      p[1].idx  = NULL;
      p[1].opt  = NULL;
    }
  } else {
    p[0].count  = 0;
    p[1].count  = sf->nroots;
    p[1].idx    = NULL;
    p[1].opt    = NULL;
    p[1].buf    = link->rootbuf[link->rootmtype]; /* Might just allocated by PetscMallocWithMemType() above */
    p[1].atomic = PETSC_FALSE;
    p[1].data   = rootdata;
  }

  for (i=0; i<2; i++) {
    if (p[i].count) {
      ierr = PetscSFPackGetUnpackAndOp(link,link->rootmtype,op,p[i].atomic,&UnpackAndOp);CHKERRQ(ierr);
      if (UnpackAndOp) {ierr = (*UnpackAndOp)(p[i].count,p[i].idx,link,p[i].opt,(void*)p[i].data,p[i].buf);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
      else {
        PetscInt    j;
        PetscMPIInt n;
        if (p[i].idx) {
          /* Note if done on GPU, this can be very slow due to the huge number of kernel calls. The op is likely user-defined. We must
             use link->unit (instead of link->basicunit) as the datatype and 1 (instead of link->bs) as the count in MPI_Reduce_local.
           */
          for (j=0; j<p[i].count; j++) {ierr = MPI_Reduce_local(p[i].buf+j*link->unitbytes,(char *)p[i].data+p[i].idx[j]*link->unitbytes,1,link->unit,op);CHKERRQ(ierr);}
        } else {
          ierr = PetscMPIIntCast(p[i].count,&n);CHKERRQ(ierr);
          ierr = MPI_Reduce_local(p[i].buf,(void*)p[i].data,n,link->unit,op);CHKERRQ(ierr);
        }
      }
#else
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No unpacking reduction operation for this MPI_Op");
#endif
    }
  }
#if defined(PETSC_HAVE_CUDA)
  /* Make sure rootdata is ready to use by SF client. If either self or remote has something done on the stream, we have to sync */
  if (link->rootmtype == PETSC_MEMTYPE_DEVICE && (p[0].count || p[1].count)) {cudaError_t err = cudaStreamSynchronize(link->stream);CHKERRCUDA(err);}
#endif
  ierr = PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Utility routine to unpack data from leaf buffer and Op it into selected entries of leafdata */
PETSC_STATIC_INLINE PetscErrorCode PetscSFUnpackAndOpLeafData(PetscSF sf,PetscSFPack link,const PetscInt *leafloc,void *leafdata,MPI_Op op,PetscBool sparse)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscErrorCode (*UnpackAndOp)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PackInfo       p[2];

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  if (!use_gpu_aware_mpi && link->leafmtype == PETSC_MEMTYPE_DEVICE) {
    cudaError_t err;
    if (!link->leafbuf[PETSC_MEMTYPE_DEVICE]) {ierr = PetscMallocWithMemType(PETSC_MEMTYPE_DEVICE,link->leafbuflen*link->unitbytes,(void**)&link->leafbuf[PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);}
    err  = cudaMemcpyAsync(link->leafbuf[PETSC_MEMTYPE_DEVICE],link->leafbuf[PETSC_MEMTYPE_HOST],link->leafbuflen*link->unitbytes,cudaMemcpyHostToDevice,link->stream);CHKERRCUDA(err);
    ierr = PetscLogCpuToGpu(link->leafbuflen*link->unitbytes);CHKERRQ(ierr);
  }
#endif

  if (sparse) {
    p[0].count  = link->selfbuflen;
    p[0].idx    = leafloc;
    p[0].opt    = sf->selfleafpackopt;
    p[0].buf    = link->selfbuf[link->leafmtype];
    p[0].atomic = sf->selfleafdups;
    p[0].data   = leafdata;
    if (p[0].opt && p[0].opt->all_contiguous) {
      p[0].data = (const char*)leafdata + p[0].opt->start_index*link->unitbytes;
      p[0].idx  = NULL; /* NULL means indices [0,count). That's why we need to shift leafdata above. */
      p[0].opt  = NULL;
    }

    p[1].count  = link->leafbuflen;
    p[1].idx    = leafloc+sf->roffset[sf->ndranks];
    p[1].opt    = sf->leafpackopt;
    p[1].buf    = link->leafbuf[link->leafmtype];
    p[1].atomic = sf->remoteleafdups;
    p[1].data   = leafdata;
    if (p[1].opt && p[1].opt->all_contiguous) {
      p[1].data = (const char*)leafdata + p[1].opt->start_index*link->unitbytes;
      p[1].idx  = NULL;
      p[1].opt  = NULL;
    }
  } else {
    p[0].count  = 0;
    p[1].count  = sf->nleaves;
    p[1].idx    = NULL;
    p[1].opt    = NULL;
    p[1].buf    = link->leafbuf[link->leafmtype];
    p[1].atomic = PETSC_FALSE;
    p[1].data   = leafdata;
  }

  for (i=0; i<2; i++) {
    if (p[i].count) {
      ierr = PetscSFPackGetUnpackAndOp(link,link->leafmtype,op,p[i].atomic,&UnpackAndOp);CHKERRQ(ierr);
      if (UnpackAndOp) {ierr = (*UnpackAndOp)(p[i].count,p[i].idx,link,p[i].opt,(void*)p[i].data,p[i].buf);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_MPI_REDUCE_LOCAL)
      else {
        PetscInt    j;
        PetscMPIInt n;
        if (p[i].idx) {
          for (j=0; j<p[i].count; j++) {ierr = MPI_Reduce_local(p[i].buf+j*link->unitbytes,(char *)p[i].data+p[i].idx[j]*link->unitbytes,1,link->unit,op);CHKERRQ(ierr);}
        } else {
          ierr = PetscMPIIntCast(p[i].count,&n);CHKERRQ(ierr);
          ierr = MPI_Reduce_local(p[i].buf,(void*)p[i].data,n,link->unit,op);CHKERRQ(ierr);
        }
      }
#else
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No unpacking reduction operation for this MPI_Op");
#endif
    }
  }
#if defined(PETSC_HAVE_CUDA)
  if (link->leafmtype == PETSC_MEMTYPE_DEVICE && (p[0].count || p[1].count)) {cudaError_t err = cudaStreamSynchronize(link->stream);CHKERRCUDA(err);}
#endif
  ierr = PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Utility routine to fetch and Op selected entries of rootdata */
PETSC_STATIC_INLINE PetscErrorCode PetscSFFetchAndOpRootData(PetscSF sf,PetscSFPack link,const PetscInt *rootloc,void *rootdata,MPI_Op op,PetscBool sparse)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscErrorCode (*FetchAndOp)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,void*);
  PackInfo       p[2];

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr); /* For simplicity, let's deem the rarely used FetchAndOp as Unpack */
#if defined(PETSC_HAVE_CUDA)
  if (!use_gpu_aware_mpi && link->rootmtype == PETSC_MEMTYPE_DEVICE) {
    cudaError_t err;
    if (!link->rootbuf[PETSC_MEMTYPE_DEVICE]) {ierr = PetscMallocWithMemType(PETSC_MEMTYPE_DEVICE,link->rootbuflen*link->unitbytes,(void**)&link->rootbuf[PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);}
    err = cudaMemcpyAsync(link->rootbuf[PETSC_MEMTYPE_DEVICE],link->rootbuf[PETSC_MEMTYPE_HOST],link->rootbuflen*link->unitbytes,cudaMemcpyHostToDevice,link->stream);CHKERRCUDA(err);
    ierr = PetscLogCpuToGpu(link->rootbuflen*link->unitbytes);CHKERRQ(ierr);
  }
#endif

  if (sparse) {
    /* For SFBasic and SFNeighbor, whose root space is sparse and have separate buffers for self and remote. */
    p[0].count  = link->selfbuflen;
    p[0].idx    = rootloc;
    p[0].opt    = bas->selfrootpackopt;
    p[0].buf    = link->selfbuf[link->rootmtype];
    p[0].atomic = bas->selfrootdups;
    p[0].data   = rootdata;
    if (p[0].opt && p[0].opt->all_contiguous) {
      p[0].data = (const char*)rootdata + p[0].opt->start_index*link->unitbytes;
      p[0].idx  = NULL; /* NULL means indices [0,count). That's why we need to shift rootdata above. */
      p[0].opt  = NULL;
    }

    p[1].count  = link->rootbuflen;
    p[1].idx    = rootloc+bas->ioffset[bas->ndiranks];
    p[1].opt    = bas->rootpackopt;
    p[1].buf    = link->rootbuf[link->rootmtype];
    p[1].atomic = bas->remoterootdups;
    p[1].data   = rootdata;
    if (p[1].opt && p[1].opt->all_contiguous) {
      p[1].data = (const char*)rootdata + p[1].opt->start_index*link->unitbytes;
      p[1].idx  = NULL;
      p[1].opt  = NULL;
    }
  } else {
    p[0].count  = 0;
    p[1].count  = sf->nroots;
    p[1].idx    = NULL;
    p[1].opt    = NULL;
    p[1].buf    = link->rootbuf[link->rootmtype];
    p[1].atomic = PETSC_FALSE;
  }

  for (i=0; i<2; i++) {
    if (p[i].count) {
      ierr = PetscSFPackGetFetchAndOp(link,link->rootmtype,op,p[i].atomic,&FetchAndOp);CHKERRQ(ierr);
      ierr = (*FetchAndOp)(p[i].count,p[i].idx,link,p[i].opt,(void*)p[i].data,p[i].buf);CHKERRQ(ierr);
    }
  }
#if defined(PETSC_HAVE_CUDA)
  if (link->rootmtype == PETSC_MEMTYPE_DEVICE && (p[0].count || p[1].count)) {cudaError_t err = cudaStreamSynchronize(link->stream);CHKERRCUDA(err);}
#endif
  ierr = PetscLogEventEnd(PETSCSF_Unpack,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFPackSetupOptimizations_Basic(PetscSF sf)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  ierr = PetscSFPackOptCreate(sf->ndranks,               sf->roffset,               sf->rmine,    &sf->selfleafpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackOptCreate(sf->nranks-sf->ndranks,    sf->roffset+sf->ndranks,   sf->rmine,    &sf->leafpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackOptCreate(bas->ndiranks,             bas->ioffset,              bas->irootloc,&bas->selfrootpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackOptCreate(bas->niranks-bas->ndiranks,bas->ioffset+bas->ndiranks,bas->irootloc,&bas->rootpackopt);CHKERRQ(ierr);

#if defined(PETSC_HAVE_CUDA)
  /* Check duplicates in irootloc[] so CUDA packing kernels can use cheaper regular operations
     instead of atomics to unpack data on leaves/roots, when they know there is not data race.
   */
  ierr = PetscCheckDupsInt(sf->roffset[sf->ndranks],                              sf->rmine,                                &sf->selfleafdups);CHKERRQ(ierr);
  ierr = PetscCheckDupsInt(sf->roffset[sf->nranks]-sf->roffset[sf->ndranks],      sf->rmine+sf->roffset[sf->ndranks],       &sf->remoteleafdups);CHKERRQ(ierr);
  ierr = PetscCheckDupsInt(bas->ioffset[bas->ndiranks],                           bas->irootloc,                            &bas->selfrootdups);CHKERRQ(ierr);
  ierr = PetscCheckDupsInt(bas->ioffset[bas->niranks]-bas->ioffset[bas->ndiranks],bas->irootloc+bas->ioffset[bas->ndiranks],&bas->remoterootdups);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFPackDestroyOptimizations_Basic(PetscSF sf)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  ierr = PetscSFPackOptDestroy(&sf->leafpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackOptDestroy(&sf->selfleafpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackOptDestroy(&bas->rootpackopt);CHKERRQ(ierr);
  ierr = PetscSFPackOptDestroy(&bas->selfrootpackopt);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  sf->selfleafdups    = PETSC_TRUE;
  sf->remoteleafdups  = PETSC_TRUE;
  bas->selfrootdups   = PETSC_TRUE; /* The default is assuming there are dups so that atomics are used. */
  bas->remoterootdups = PETSC_TRUE;
#endif
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFSetUp_Basic(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFView_Basic(PetscSF,PetscViewer);
PETSC_INTERN PetscErrorCode PetscSFReset_Basic(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFDestroy_Basic(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFBcastAndOpEnd_Basic  (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,      void*,      MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFReduceEnd_Basic      (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,      void*,      MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Basic(PetscSF,MPI_Datatype,PetscMemType,      void*,PetscMemType,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFCreateEmbeddedSF_Basic(PetscSF,PetscInt,const PetscInt*,PetscSF*);
PETSC_INTERN PetscErrorCode PetscSFCreateEmbeddedLeafSF_Basic(PetscSF,PetscInt,const PetscInt*,PetscSF*);
PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Basic(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**);
PETSC_INTERN PetscErrorCode PetscSFPackGet_Basic_Common(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,const void*,PetscInt,PetscInt,PetscSFPack*);
PETSC_INTERN PetscErrorCode PetscSFSetFromOptions_Basic(PetscOptionItems*,PetscSF);
#endif
