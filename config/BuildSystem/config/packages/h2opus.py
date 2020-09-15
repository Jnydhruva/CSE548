import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'origin/master'
    self.download               = ['git://https://github.com/wajihboukaram/h2opus']
    self.precisions             = ['single','double']
    self.skippackagewithoptions = 1
    self.cxx                    = 1
    self.requirescxx11          = 1
    self.liblist                = [['libh2opus.a']]
    self.includes               = ['h2opus.h']
    self.complex                = 0
    return

  # TODO
  #def setupHelp(self, help):
  #  config.package.Package.setupHelp(self, help)
  #  return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.scalartypes = framework.require('PETSc.options.scalarTypes',self)
    self.cuda        = framework.require('config.packages.cuda',self)
    self.magma       = framework.require('config.packages.magma',self)
    self.blas        = framework.require('config.packages.BlasLapack',self)
    self.kblas       = framework.require('config.packages.kblas',self)
    self.openmp      = framework.require('config.packages.openmp',self)
    self.mpi         = framework.require('config.packages.MPI',self)
    self.thrust      = framework.require('config.packages.thrust',self)
    self.deps        = [self.blas]
    self.odeps       = [self.openmp,self.cuda,self.kblas,self.magma,self.mpi,self.thrust]
    return

  def Install(self):
    import os

    if not self.blas.has_cheaders and not self.blas.mkl:
      raise RuntimeError('H2OPUS requires cblas.h and lapacke.h headers')

    if self.openmp.found:
      self.usesopenmp = 'yes'

    self.setCompilers.pushLanguage('Cxx')
    cxx = self.setCompilers.getCompiler()
    cxxflags = self.setCompilers.getCompilerFlags()
    cxxflags = cxxflags.replace('-fvisibility=hidden','')
    cxxflags = cxxflags.replace('-std=gnu++14','-std=c++11')
    self.setCompilers.popLanguage()

    with_gpu=False
    if self.cuda.found and self.magma.found and self.kblas.found:
      self.pushLanguage('CUDA')
      nvcc = self.getCompiler()
      nvopts = self.getCompilerFlags()
      self.popLanguage()
      self.getExecutable(nvcc,getFullPath=1,resultName='systemNvcc')
      if hasattr(self,'systemNvcc'):
        nvccDir = os.path.dirname(self.systemNvcc)
        cudaDir = os.path.split(nvccDir)[0]
      else:
        raise RuntimeError('Unable to locate CUDA NVCC compiler')
      with_gpu=True

    if not with_gpu and not (self.thrust.found or self.cuda.found):
      raise RuntimeError('Missing THRUST. Run with --download-thrust or specify the location of the package')

    if with_gpu:
      self.setCompilers.CUDAPPFLAGS += ' -std=c++11'

    with open(os.path.join(self.packageDir,'make.inc'),'w') as g:
      g.write('H2OPUS_INSTALL_DIR = '+self.installDir+'\n')
      g.write('CXX = '+cxx+'\n')
      g.write('CXXFLAGS = -DHLIB_PROFILING_ENABLED '+cxxflags+'\n')

      if self.blas.mkl:
        g.write('H2OPUS_USE_MKL = 1\n')

      g.write('CXXCPPFLAGS = '+self.headers.toString(self.blas.include)+'\n')
      g.write('H2OPUS_CBLAS_LIBS = '+self.libraries.toString(self.blas.lib)+'\n')

      if with_gpu:
        g.write('H2OPUS_USE_GPU = 1\n')
        g.write('NVCC = '+nvcc+'\n')
        g.write('NVCCFLAGS = '+nvopts+' -std=c++11 --expt-relaxed-constexpr\n')
        if self.cuda.gencodearch:
          g.write('H2OPUS_GENCODE_FLAGS = -gencode arch=compute_'+self.cuda.gencodearch+',code=sm_'+self.cuda.gencodearch+'\n')
        g.write('CXXCPPFLAGS += '+self.headers.toString(self.cuda.include)+'\n')
        g.write('CXXCPPFLAGS += '+self.headers.toString(self.magma.include)+'\n')
        g.write('CXXCPPFLAGS += '+self.headers.toString(self.kblas.include)+'\n')
        g.write('H2OPUS_CUDA_LIBS += '+self.libraries.toString(self.cuda.lib)+'\n')
        g.write('H2OPUS_MAGMA_LIBS += '+self.libraries.toString(self.magma.lib)+'\n')
        g.write('H2OPUS_KBLAS_LIBS += '+self.libraries.toString(self.kblas.lib)+'\n')
      else:
        if self.thrust.found:
          g.write('CXXCPPFLAGS += '+self.headers.toString(self.thrust.include)+'\n')
        elif self.cuda.found:
          g.write('CXXCPPFLAGS += '+self.headers.toString(self.cuda.include)+'\n')

      if self.scalartypes.precision == 'single':
        g.write('H2OPUS_USE_SINGLE_PRECISION = 1\n')

      if self.mpi.found and not self.mpi.usingMPIUni:
        g.write('H2OPUS_USE_MPI = 1\n')

      if self.openmp.found:
        g.write('LDFLAGS = '+self.openmp.ompflag+'\n')
        #g.write('LDFLAGS = '+self.openmp.ompflag+' $(PETSC_EXTERNAL_LIB_BASIC)\n')

    if self.installNeeded('make.inc'):
      try:
        output1,err1,ret1  = config.package.Package.executeShellCommand('make clean', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make clean on H2OPUS: '+str(e))
        raise RuntimeError('Error running make clean on H2OPUS')
      try:
        self.logPrintBox('Compiling H2OPUS; this may take several minutes')
        output2,err2,ret2 = config.package.Package.executeShellCommand('make', cwd=self.packageDir, timeout=2500, log = self.log)
        self.logPrintBox('Installing H2OPUS; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'make install', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make on H2OPUS: '+str(e))
        raise RuntimeError('Error running make on H2OPUS')
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir
