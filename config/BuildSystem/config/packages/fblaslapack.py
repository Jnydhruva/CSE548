import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit              = 'acbdf3621104c6b9224e3e17e74d54509c2f9087' # barry/2022-10-21/fix-lapacke-always-included Oct 22, 12:02 PM Eastern Time
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
    # Remove compilers not needed by the LAPACK CMake since it tests them even
    # though it never uses them (why?) and they may fail
    # on some systems, such as Microsoft Windows with Microsoft Windows compilers
    args = self.rmArgsStartsWith(args,'-DCMAKE_C_COMPILER=')
    args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
    args = self.rmArgsStartsWith(args,'-DMPI_C_COMPILER=')
    args = self.rmArgsStartsWith(args,'-DMPI_CXX_COMPILER=')
    args = self.rmArgsStartsWith(args,'-DMPI_Fortran_COMPILER=')
    return args

  def Install(self):
    config.package.CMakePackage.Install(self)

    # LAPACK CMake cannot name the generated files with Microsoft compilers with .lib so need to rename them
    if self.framework.getCompiler().find('win32fe') > -1:
      import os
      from shutil import copyfile
      if os.path.isfile(os.path.join(self.installDir,self.libdir,'libblas.a')):
        copyfile(os.path.join(self.installDir,self.libdir,'libblas.a'),os.path.join(self.installDir,self.libdir,'libblas.lib'))
      if os.path.isfile(os.path.join(self.installDir,self.libdir,'liblapack.a')):
        copyfile(os.path.join(self.installDir,self.libdir,'liblapack.a'),os.path.join(self.installDir,self.libdir,'liblapack.lib'))

    return self.installDir
