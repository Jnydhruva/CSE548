
#include <../src/mat/impls/aij/seq/aij.h>
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode MatLoad_AIJ_HDF5(Mat mat, PetscViewer viewer)
{
  hid_t           file_id, group_matrix_id;
  const PetscInt  *i_glob = NULL;
  PetscInt        *i = NULL;
  const PetscInt  *j = NULL;
  const PetscScalar *a = NULL;
  const char      *a_name = NULL, *i_name = NULL, *j_name = NULL, *mat_name = NULL, *c_name = NULL;
  PetscInt        p, m, M, N;
  PetscInt        bs = mat->rmap->bs;
  PetscBool       flg;
  IS              is_i = NULL, is_j = NULL;
  Vec             vec_a = NULL;
  PetscLayout     jmap = NULL;
  MPI_Comm        comm;
  PetscMPIInt     rank, size;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)mat,&mat_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetAIJNames(viewer,&i_name,&j_name,&a_name,&c_name);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"Options for loading matrix from HDF5","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    ierr = MatSetBlockSize(mat, bs);CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PushGroup(viewer,mat_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer,&file_id,&group_matrix_id);CHKERRQ(ierr);

  ierr = PetscViewerHDF5ReadAttribute(viewer,mat_name,c_name,PETSC_INT,&N);CHKERRQ(ierr);

  ierr = PetscViewerHDF5ReadSizes(viewer, i_name, NULL, &M);CHKERRQ(ierr);
  --M;  /* i has size M+1 as there is global number of nonzeros stored at the end */

  if (!mat->symmetric) {
    /* Swap row and columns layout for unallocated matrix. I want to avoid calling MatTranspose() just to transpose sparsity pattern and layout. */
    /* TODO: this should be needed only for MAT format and not HDF5 in general */
    if (!mat->preallocated) {
      PetscLayout tmp;
      tmp = mat->rmap; mat->rmap = mat->cmap; mat->cmap = tmp;
    } else SETERRQ(comm,PETSC_ERR_SUP,"Not for preallocated matrix - we would need to transpose it here which we want to avoid");
  }

  /* If global sizes are set, check if they are consistent with that given in the file */
  if (mat->rmap->N >= 0 && mat->rmap->N != M) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Inconsistent # of rows: Matrix in file has (%D) and input matrix has (%D)",mat->rmap->N,M);
  if (mat->cmap->N >= 0 && mat->cmap->N != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Inconsistent # of cols: Matrix in file has (%D) and input matrix has (%D)",mat->cmap->N,N);

  /* Determine ownership of all (block) rows and columns */
  mat->rmap->N = M;
  mat->cmap->N = N;
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);
  m = mat->rmap->n;

  /* Read array i (array of row indices) */
  ierr = PetscMalloc1(m+1, &i);CHKERRQ(ierr); /* allocate i with one more position for local number of nonzeros on each rank */
  if (rank == size-1) m++; /* in the loaded array i_glob, only the last rank has one more position with the global number of nonzeros */
  M++;
  ierr = ISCreate(comm,&is_i);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is_i,i_name);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(is_i->map,m);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(is_i->map,M);CHKERRQ(ierr);
  ierr = ISLoad(is_i,viewer);CHKERRQ(ierr);
  ierr = ISGetIndices(is_i,&i_glob);CHKERRQ(ierr);
  ierr = PetscMemcpy(i,i_glob,m*sizeof(PetscInt));CHKERRQ(ierr);

  /* Reset m and M to the matrix sizes */
  m = mat->rmap->n;
  M--;

  /* Determine offset and count of elements for reading local part of array data */
  /* Create PetscLayout for j and a vectors; construct ranges first */
  ierr = PetscLayoutCreate(comm,&jmap);CHKERRQ(ierr);
  ierr = PetscCalloc1(size+1, &jmap->range);CHKERRQ(ierr);
  ierr = MPI_Allgather(&i[0], 1, MPIU_INT, jmap->range, 1, MPIU_INT, comm);CHKERRQ(ierr);
  jmap->range[size] = i[m];
  ierr = MPI_Bcast(&jmap->range[size], 1, MPIU_INT, size-1, comm);CHKERRQ(ierr);
  for (p=size-1; p>0; p--) {
    if (!jmap->range[p]) jmap->range[p] = jmap->range[p+1]; /* for ranks with 0 rows, take the value from the next processor */
  }
  i[m] = jmap->range[rank+1];
  /* Deduce rstart, rend, n and N from the ranges */
  ierr = PetscLayoutSetUp(jmap);CHKERRQ(ierr);

  /* Convert global to local indexing of rows */
  for (p=1; p<m+1; ++p) i[p] -= i[0];
  i[0] = 0;

  /* Read array j (array of column indices) */
  ierr = ISCreate(comm,&is_j);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)is_j,j_name);CHKERRQ(ierr);
  ierr = PetscLayoutDuplicate(jmap,&is_j->map);CHKERRQ(ierr);
  ierr = ISLoad(is_j,viewer);CHKERRQ(ierr);
  ierr = ISGetIndices(is_j,&j);CHKERRQ(ierr);

  /* Read array a (array of values) */
  ierr = VecCreate(comm,&vec_a);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vec_a,a_name);CHKERRQ(ierr);
  ierr = PetscLayoutDuplicate(jmap,&vec_a->map);CHKERRQ(ierr);
  ierr = VecLoad(vec_a,viewer);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vec_a,&a);CHKERRQ(ierr);

  /* close group */
  PetscStackCallHDF5(H5Gclose,(group_matrix_id));
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  /* populate matrix */
  if (!((PetscObject)mat)->type_name) {
    ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocationCSR(mat,i,j,a);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocationCSR(mat,i,j,a);CHKERRQ(ierr);
  /*
  ierr = MatSeqBAIJSetPreallocationCSR(mat,bs,i,j,a);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocationCSR(mat,bs,i,j,a);CHKERRQ(ierr);
  */

  if (!mat->symmetric) {
    /* Transpose the input matrix back */
    /* TODO: this should be done only for MAT format and not HDF5 in general */
    ierr = MatTranspose(mat,MAT_INPLACE_MATRIX,&mat);CHKERRQ(ierr);
  }

  ierr = PetscLayoutDestroy(&jmap);CHKERRQ(ierr);
  ierr = PetscFree(i);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_i,&i_glob);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_j,&j);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vec_a,&a);CHKERRQ(ierr);
  ierr = ISDestroy(&is_i);CHKERRQ(ierr);
  ierr = ISDestroy(&is_j);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

