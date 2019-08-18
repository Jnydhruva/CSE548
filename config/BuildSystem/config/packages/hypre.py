import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version         = '2.16.0'
    self.minversion      = '2.14'
    self.versionname     = 'HYPRE_RELEASE_VERSION'
    self.versioninclude  = 'HYPRE_config.h'
    self.requiresversion = 1
    self.gitcommit       = 'v'+self.version
    self.download        = ['git://https://github.com/hypre-space/hypre','https://github.com/hypre-space/hypre/archive/'+self.gitcommit+'.tar.gz']
    self.functions       = ['HYPRE_IJMatrixCreate']
    self.includes        = ['HYPRE.h']
    self.liblist         = [['libHYPRE.a']]
    self.license         = 'https://computation.llnl.gov/casc/linear_solvers/sls_hypre.html'
    # Per hypre users guide section 7.5 - install manually on windows for MS compilers.
    self.downloadonWindows = 0
    self.precisions        = ['double']
    # HYPRE is supposed to work with complex number
    #self.complex           = 0
    self.hastests          = 1
    self.hastestsdatafiles = 1

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.openmp     = framework.require('config.packages.openmp',self)
    self.cxxlibs    = framework.require('config.packages.cxxlibs',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.scalar     = framework.require('PETSc.options.scalarTypes',self)
    self.cuda       = framework.require('config.packages.cuda',self)
    self.deps       = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib]
    self.odeps      = [self.cuda]

  def formGNUConfigureArgs(self):
    self.packageDir = os.path.join(self.packageDir,'src')
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Error: Hypre requires C++ compiler. None specified')
    if not hasattr(self.compilers, 'FC'):
      args.append('--disable-fortran')
    if self.mpi.include:
      # just use the first dir - and assume the subsequent one isn't necessary [relavant only on AIX?]
      args.append('--with-MPI-include="'+self.mpi.include[0]+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-MPI-lib-dirs="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append(ll[3:-2])
    libs = ' '.join(libs)
    args.append('--with-MPI-libs="'+libs+'"')

    if self.openmp.found:
      args.append('--with-openmp')
      self.usesopenmp = 'yes'
      # use OMP_NUM_THREADS to control the number of threads used

    blaslibs = self.libraries.toString(self.blasLapack.dlib)
    if self.cuda.found:
      if self.cuda.version_tuple < (8,0,0):
        raise RuntimeError('Hypre requires Cuda version 8.0.0 or higher, version '+self.cuda.foundversion+' is being used')
      self.usesgpu = 'Cuda'
      args.append('--with-cuda')
      args.append('--enable-unified-memory')
      args.append('CUDA_HOME="'+self.cuda.directory+'"')
      #HYPRE_CUDA_SM=60
      # Hypre changes the shared library linker to nvcc which cannot handle -Wl,-rpath
      blaslibs = blaslibs.split(' ')
      blaslibs = [x for x in blaslibs if not x.startswith('-Wl,-rpath')]
      blaslibs = ' '.join(blaslibs)

    # tell hypre configure not to look for blas/lapack [and not use hypre-internal blas]
    args.append('--with-blas-lib="'+blaslibs+'"')
    args.append('--with-lapack-lib=" "')
    args.append('--with-blas=no')
    args.append('--with-lapack=no')

    # explicitly tell hypre BLAS/LAPACK mangling since it may not match Fortran mangling
    if self.blasLapack.mangling == 'underscore':
      mang = 'one-underscore'
    elif self.blasLapack.mangling == 'caps':
      mang = 'caps-no-underscores'
    else:
      mang = 'no-underscores'
    args.append('--with-fmangle-blas='+mang)
    args.append('--with-fmangle-lapack='+mang)

    args.append('--without-mli')
    args.append('--without-fei')
    args.append('--without-superlu')

    if self.getDefaultIndexSize() == 64:
      args.append('--enable-bigint')

    if self.scalar.scalartype == 'complex':
      args.append('--enable-complex')

    # hypre configure assumes the AR flags are passed in with AR
    args = [arg for arg in args if not arg.startswith('AR')]
    args.append('AR="'+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'"')
    return args

  def consistencyChecks(self):
    config.package.GNUPackage.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('dgels'):
        raise RuntimeError('hypre requires the LAPACK routine dgels(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it')
      self.log.write('Found dgels() in Lapack library as needed by hypre\n')
    return

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    # check integers
    if self.defaultIndexSize == 64:
      code = '#if !defined(HYPRE_BIGINT) && !defined(HYPRE_MIXEDINT)\n#error HYPRE_BIGINT or HYPRE_MIXEDINT not defined!\n#endif'
      msg  = '--with-64-bit-indices option requires Hypre built with --enable-bigint or --enable-mixedint.\n'
    else:
      code = '#if defined(HYPRE_BIGINT)\n#error HYPRE_BIGINT defined!\n#endif\n#if defined(HYPRE_MIXEDINT)\n#error HYPRE_MIXEDINT defined!\n#endif\n'
      msg  = 'Hypre with --enable-bigint/--enable-mixedint appears to be specified for a default 32-bit-indices build of PETSc.\n'
    if not self.checkCompile('#include "HYPRE_config.h"',code):
      raise RuntimeError('Hypre specified is incompatible!\n'+msg+'Suggest using --download-hypre for a compatible hypre')
    self.compilers.CPPFLAGS = oldFlags

    # hypre puts HYPRE_USING_CUDA in HYPRE_config.h which causes all of the hypre includes to no longer simple
    # define the C API but instead have active Cuda code that cannot be compiled with PETSc; also provides prototypes
    # for VecScale(), VecSet(), and VecCopy() that conflict with PETSc's
    # This will not work if the hypre package has been installed in a sudo location
    if self.cuda.found:
      configfile = os.path.join(self.installDir,'include','HYPRE_config.h')
      try:
        fd = open(configfile,'r')
        f = fd.read()
        fd.close()
      except:
        raise RuntimeError('Unable open '+ configfile +' for reading to remove CUDA declaration')
      try:
        fd = open(configfile,'w')
        f = f.split('\n')
        for i in f:
          if i.find('HYPRE_USING_CUDA') > -1: continue
          if i.find('HYPRE_USING_GPU') > -1: continue
          fd.write(i+'\n')
        fd.close()
      except:
        raise RuntimeError('Unable open '+ configfile +' for reading to remove CUDA declaration')
