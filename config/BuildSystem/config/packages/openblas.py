import config.package

#    We do not use CMAKE for OpenBLAS the cmake for OpenBLAS
#       does not have an install rule https://github.com/xianyi/OpenBLAS/issues/957
#       fails on mac due to argument list too long https://github.com/xianyi/OpenBLAS/issues/977
#       does not support 64 bit integers with INTERFACE64


class Configure(config.packages.BlasLapack.Configure):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version                = '0.3.7'
    self.gitcommit              = 'e7c4d6705a41910240dd19b9e7082a422563bf15'
    self.versionname            = 'OPENBLAS_VERSION'
    self.download               = ['git://https://github.com/xianyi/OpenBLAS.git','https://github.com/xianyi/OpenBLAS/archive/'+self.gitcommit+'.tar.gz']
    self.includes               = ['openblas_config.h']
    self.functions              = ['openblas_get_config']
    self.liblist                = [['libopenblas.a']]
    self.precisions             = ['single','double']
    self.fc                     = 1
    self.installwithbatch       = 1
    self.usespthreads           = 0

  def __str__(self):
    output  = config.package.Package.__str__(self)
    if self.usespthreads: output += '  using pthreads; use export OPENBLAS_NUM_THREADS=<p> to control the number of threads\n'
    return output

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('OpenBLAS', '-download-openblas-64-bit-blas-indices', nargs.ArgBool(None, 0, 'Use 64 bit integers for OpenBLAS (deprecated: use --with-64-bit-blas-indices'))
    help.addArgument('OpenBLAS', '-download-openblas-use-pthreads', nargs.ArgBool(None, 0, 'Use pthreads for OpenBLAS'))
    help.addArgument('OpenBLAS', '-download-openblas-make-options=<options>', nargs.Arg(None, None, 'additional options for building OpenBLAS'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.make            = framework.require('config.packages.make',self)
    self.openmp          = framework.require('config.packages.openmp',self)

  def generateGuessesLib(self,libDir):
    for (name,dir,lib,include) in config.packages.generateGuesses(self):
    if self.directory and libDir: raise RuntimeError('Using both --download-openblas and --with-blaslapack-dir not allowed')
    if self.directory: libDir = os.path.join(self.directory,'lib')
    yield (libDir, 'libopenblas.a',self.known64,self.usesopenmp)

  def configureLibrary(self):
    import os
    config.packages.BlasLapack.Configure.configureLibrary(self)
    if self.found:
      pass
      # TODO: Use openblas_get_config() or openblas_config.h to determine use of OpenMP and 64 bit indices for prebuilt OpenBLAS libraries
    return

  def versionToStandardForm(self,ver):
    '''Converts from " OpenBLAS 0.3.6<.dev> " to standard 0.3.6 format'''
    import re
    ver = re.match("\s*OpenBLAS\s*([0-9\.]+)\s*",ver).group(1)
    if ver.endswith('.'): ver = ver[0:-1]
    return ver

  def Install(self):
    import os
    # OpenBLAS handles its own compiler optimization options
    cmdline = 'CC='+self.compilers.CC+' '
    cmdline += 'FC='+self.compilers.FC+' '
    if self.argDB['download-openblas-64-bit-blas-indices'] or self.argDB['with-64-bit-blas-indices']:
      cmdline += " INTERFACE64=1 "
      self.known64 = 'yes'
    else:
      self.known64 = 'no'
    if 'download-openblas-make-options' in self.argDB and self.argDB['download-openblas-make-options']:
      cmdline+=" "+self.argDB['download-openblas-make-options']
    if not self.argDB['with-shared-libraries']:
      cmdline += " NO_SHARED=1 "
    cmdline += " MAKE_NB_JOBS="+str(self.make.make_np)+" "
    if self.openmp.found:
      cmdline += " USE_OPENMP=1 "
      self.usesopenmp = 'yes'
      # use the environmental variable OMP_NUM_THREADS to control the number of threads used
    else:
      cmdline += " USE_OPENMP=0 "
      self.usesopenmp = 'no'
      if 'download-openblas-use-pthreads' in self.argDB and self.argDB['download-openblas-use-pthreads']:
        self.usespthreads = 1
        cmdline += " USE_THREAD=1 "
        # use the environmental variable OPENBLAS_NUM_THREADS to control the number of threads used
      else:
        cmdline += " USE_THREAD=0 "
    cmdline += " NO_EXPRECISION=1 "
    libdir = self.libDir
    blasDir = self.packageDir
    g = open(os.path.join(blasDir,'tmpmakefile'),'w')
    g.write(cmdline)
    g.close()
    if not self.installNeeded('tmpmakefile'): return self.installDir
    try:
      self.logPrintBox('Compiling OpenBLAS; this may take several minutes')
      output1,err1,ret  = config.package.Package.executeShellCommand('cd '+blasDir+' && make '+cmdline, timeout=2500, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error running make on '+blasDir+': '+str(e))
      raise RuntimeError('Error running make on '+blasDir)
    try:
      self.logPrintBox('Installing OpenBLAS')
      self.installDirProvider.printSudoPasswordMessage()
      output2,err2,ret  = config.package.Package.executeShellCommand('cd '+blasDir+' && '+self.installSudo+' make PREFIX='+self.installDir+' install', timeout=30, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error moving '+blasDir+' libraries: '+str(e))
      raise RuntimeError('Error moving '+blasDir+' libraries')
    self.postInstall(output1+err1+output2+err2,'tmpmakefile')
    return self.installDir


