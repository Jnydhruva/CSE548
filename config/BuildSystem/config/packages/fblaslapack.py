import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit              = 'd9e9690419d1fbf80d5dc67f77341af2e771ee90' # barry/2022-10-21/fix-lapacke-always-included Oct 21, 11:22 AM Eastern Time
    self.download               = ['git://https://github.com/petsc/lapack','https://github.com/petsc/lapack/archive/'+self.gitcommit+'.tar.gz']
    self.includes               = []
    self.liblist                = [['liblapack.a','libblas.a']]
    self.precisions             = ['single','double']
    self.functionsFortran       = 1
    self.downloadonWindows      = 1
    self.skippackagewithoptions = 1
    self.buildLanguages         = ['FC']
    self.minCmakeVersion        = (3,0,0)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    return args

