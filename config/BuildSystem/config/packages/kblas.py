import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version                = '4.0.0'
    self.gitcommit              = 'v'+self.version
    self.download               = ['git://https://github.com/ecrc/kblas-gpu.git']
    self.cxx                    = 1 # uses nvcc to compile everything
    self.functions              = ['kblasCreate']
    self.functionsCxx           = [1,'struct KBlasHandle; typedef struct KBlasHandle *kblasHandle_t;int kblasCreate(kblasHandle_t*);','kblasHandle_t h; kblasCreate(&h)']
    self.liblist                = [['libkblas.a']]
    self.includes               = ['kblas.h']
    return

  # TODO
  #def setupHelp(self, help):
  #  config.package.Package.setupHelp(self, help)
  #  return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cub    = framework.require('config.packages.cub',self)
    self.cuda   = framework.require('config.packages.cuda',self)
    self.magma  = framework.require('config.packages.magma',self)
    self.openmp = framework.require('config.packages.openmp',self)
    self.deps   = [self.cuda,self.magma]
    self.odeps  = [self.openmp,self.cub] #CUB is a building dependency only
    return

  def Install(self):
    import os

    if not self.cub.found:
      raise RuntimeError('Package kblas requested but dependency cub not requested. Perhaps you want --download-cub')
    if self.openmp.found:
      self.usesopenmp = 'yes'

    self.setCompilers.pushLanguage('Cxx')
    cxx = self.setCompilers.getCompiler()
    cxxflags = self.setCompilers.getCompilerFlags()
    self.setCompilers.popLanguage()

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

    with open(os.path.join(self.packageDir,'make.inc'),'w') as g:
      g.write('_SUPPORT_BLAS2_ = TRUE\n')
      g.write('_SUPPORT_BLAS3_ = TRUE\n')
      g.write('_SUPPORT_BATCH_TR_ = TRUE\n')
      g.write('_SUPPORT_TLR_ = TRUE\n')
      g.write('_SUPPORT_SVD_ = TRUE\n')
      g.write('_CUB_DIR_ = '+self.cub.directory+'/include\n')
      g.write('_USE_MAGMA_ = TRUE\n')
      g.write('_MAGMA_ROOT_ = '+self.magma.directory+'\n')
      g.write('_CUDA_ROOT_ = '+cudaDir+'\n')
      if self.cuda.gencodearch:
        gencodestr = '-DTARGET_SM='+self.cuda.gencodearch+' -arch sm_'+self.cuda.gencodearch
      else:
        # kblas as of v4.0.0 uses __ldg intrinsincs, available starting from 35
        gencodestr = '-DTARGET_SM=35 -arch sm_35'
      g.write('NVCC = '+nvcc+'\n')
      g.write('CC = '+cxx+'\n')
      g.write('CXX = '+cxx+'\n')
      g.write('LIB_KBLAS_NAME = kblas\n')
      g.write('COPTS = '+cxxflags+' -DUSE_MAGMA\n')
      g.write('NVOPTS = '+nvopts+' -DUSE_MAGMA\n')
      g.write('NVOPTS_2 = '+gencodestr+'\n')
      if self.openmp.found:
        g.write('NVOPTS_3 = '+gencodestr+' -Xcompiler '+self.openmp.ompflag+'\n')
      else:
        g.write('NVOPTS_3 = '+gencodestr+'\n')
      g.close()

    if self.installNeeded('make.inc'):
      try:
        output1,err1,ret1  = config.package.Package.executeShellCommand('make clean', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        pass
      try:
        self.logPrintBox('Compiling KBLAS; this may take several minutes')
        output2,err2,ret2 = config.package.Package.executeShellCommand('cd src && ' + self.make.make, cwd=self.packageDir, timeout=2500, log = self.log)
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        self.logPrintBox('Installing KBLAS; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommandSeq(
          [self.installSudo+'mkdir -p '+libDir+' '+includeDir,
           self.installSudo+'cp -f lib/*.* '+libDir+'/.',
           self.installSudo+'cp -f include/*.* '+includeDir+'/.'
          ], cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make on KBLAS: '+str(e))
        raise RuntimeError('Error running make on KBLAS')
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir
