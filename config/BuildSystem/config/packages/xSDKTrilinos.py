import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'origin/master'
    self.download          = ['git://https://github.com/trilinos/xSDKTrilinos.git']
    self.downloaddirname   = 'xSDKTrilinos'
    self.includes          = []
    self.functions         = []
    self.cxx               = 1
    self.requirescxx11     = 1
    self.downloadonWindows = 0
    self.hastests          = 1
    self.linkedbypetsc     = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.arch            = framework.require('PETSc.options.arch', self.setCompilers)
    self.petscdir        = framework.require('PETSc.options.petscdir', self.setCompilers)
    self.installdir      = framework.require('PETSc.options.installDir',  self)
    self.trilinos        = framework.require('config.packages.Trilinos',self)
    self.hypre           = framework.require('config.packages.hypre',self)
    self.x               = framework.require('config.packages.X',self)
    self.ssl             = framework.require('config.packages.ssl',self)
    self.exodusii        = framework.require('config.packages.exodusii',self)
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  # the install is delayed until postProcess() since xSDKTrilinos requires PETSc
  def Install(self):
    return self.installDir

  def configureLibrary(self):
    ''' Just assume the downloaded library will work'''
    if self.framework.clArgDB.has_key('with-xsdktrilinos'):
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos; only --download-xsdktrilinos')
    if self.framework.clArgDB.has_key('with-xsdktrilinos-dir'):
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos-dir; only --download-xsdktrilinos')
    if self.framework.clArgDB.has_key('with-xsdktrilinos-include'):
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos-include; only --download-xsdktrilinos')
    if self.framework.clArgDB.has_key('with-xsdktrilinos-lib'):
      raise RuntimeError('Xsdktrilinos does not support --with-xsdktrilinos-lib; only --download-xsdktrilinos')

    self.checkDownload()
    self.include = [os.path.join(self.installDir,'include')]
    self.lib     = [os.path.join(self.installDir,'lib','libxsdktrilinos.a')]
    self.found   = 1
    self.dlib    = self.lib
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    args.append('-DTRILINOS_INSTALL_DIR='+os.path.dirname(self.trilinos.include[0]))
    args.append('-DTrilinos_INSTALL_DIR='+os.path.dirname(self.trilinos.include[0]))
    if self.hypre.found:
      args.append('-DTPL_ENABLE_HYPRE=ON')
      args.append('-DTPL_HYPRE_LIBRARIES="'+self.libraries.toStringNoDupes(self.hypre.lib)+'"')
      args.append('-DTPL_HYPRE_INCLUDE_DIRS='+self.headers.toStringNoDupes(self.hypre.include)[2:])

    args.append('-DTPL_ENABLE_PETSC=ON')
    # These are packages that PETSc may be using that Trilinos is not be using 
    plibs = self.exodusii.dlib+self.ssl.lib+self.x.lib

    if self.framework.argDB['prefix']:
       idir = os.path.join(self.installdir.dir,'lib')
    else:
       idir = os.path.join(self.petscdir.dir,self.arch,'lib')
    if self.framework.argDB['with-single-library']:
      plibs = self.libraries.toStringNoDupes(['-L'+idir,' -lpetsc']+plibs)
    else:
      plibs = self.libraries.toStringNoDupes(['-L'+idir,'-lpetscts -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetscsys']+plibs)

    args.append('-DTPL_PETSC_LIBRARIES="'+plibs+'"')
    args.append('-DTPL_PETSC_INCLUDE_DIRS='+os.path.join(self.petscdir.dir,'include'))

    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=DEBUG')
      args.append('-DxSDKTrilinos_ENABLE_DEBUG=YES')
    else:
      args.append('-DCMAKE_BUILD_TYPE=RELEASE')
      args.append('-DxSDKTrilinos_ENABLE_DEBUG=NO')

    args.append('-DxSDKTrilinos_EXTRA_LINK_FLAGS="'+self.libraries.toStringNoDupes(self.libraries.math+self.compilers.flibs+self.compilers.cxxlibs)+' '+self.compilers.LIBS+'"')
    args.append('-DxSDKTrilinos_ENABLE_TESTS=ON')
    return args

  def postProcess(self):
    self.compilePETSc()
    config.package.CMakePackage.Install(self)
    if not self.argDB['with-batch']:
      try:
        self.logPrintBox('Testing xSDKTrilinos; this may take several minutes')
        output,err,ret  = config.package.CMakePackage.executeShellCommand('cd '+os.path.join(self.packageDir,'build')+' && '+self.cmake.ctest,timeout=50, log = self.log)
        output = output+err
        self.log.write(output)
        if output.find('Failure') > -1:
          raise RuntimeError('Error running ctest on xSDKTrilinos: '+output)
      except RuntimeError, e:
        raise RuntimeError('Error running ctest on xSDKTrilinos: '+str(e))






