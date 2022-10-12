import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit              = 'v3.10.1'
    self.download               = ['git://https://github.com/Reference-LAPACK/lapack','https://github.com/Reference-LAPACK/lapack/archive/'+self.gitcommit+'.tar.gz']
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

