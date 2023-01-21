#
#  Prototype - Provides a Pythonic binding for PETSc/Tao/SLEPc on top of the pets4py binding
#
#  Goals:
#  * be more pythonic
#  * require much less user boiler-plate code, for example no need for multiple steps to create and manage KSP, SNES, etc solver objects
#  * use optional arguments for controlling much of the solution process; requiring less use of setXXXX()
#  * leverage PETSc and petsc4py API as much as possible, avoid reproducing duplicate functionality or APIs
#  * allow direct interactions between PETSc objects and numpy objects in new API calls
#    (for example numpy arrays handled as Vec and Mat automatically)
#  * be "compatible" with other Python solver libraries like Scipy
#  * minimize the amount of new code that must be written
#
#  This is a prototype of the current PETSc functionality. It is not intended to include batching and other new features
#
#  Model: The user instantiates a problem object and that problem object "knows" how to solve itself
#    - KSP  -> LinearSystem
#    - SNES -> NonlinearSystem
#    - TS   -> ODE
#    - Tao  -> Optimization
#
#  Each solver can be used by
#    - subclassing a base PETSc solver [and Grid] class and overloading the needed function
#    - passing the needed functions and creation routines as keyward arguments into the constructor of a base PETSc solver class
#    - pass the needed functions and creation routines as keyward arguments into petsc.solve(), petsc.integrate(), petsc.optimize()
#
#  petsc.PETSc exposes the petsc4py API
#  petsc.xxx exposes the petsc solver API
#
#  TODO: Need simple example where user provided class functions are in C or Fortran
#
#  Question:  Could use Python properties to do a lazy evaluation of Mat and Solver objects? What would be the use case?
#
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import re

COMM_WORLD = PETSc.COMM_WORLD
COMM_SELF  = PETSc.COMM_SELF

def convertFromPetscTerminology(str):
  '''Converts petsc4py solver object names, ksp, etc to petsc names, ls, etc
  '''
  if not str: return ''
  for i,j in {'ksp' : 'ls' ,'snes': 'nls','ts' : 'ode','tao' : 'opt'}.items():
    str = re.sub('_'+i+'_','_'+j+'_',re.sub('-'+i+'_',j+'_',str))
    str = re.sub('\([A-Za-z0-9 ]*\)','',str)
  return str

def convertToPetscTerminology(str):
  '''Converts from petsc names to petsc4py solver object names, ls, etc to ksp etc
  '''
  for i,j in {'ls' : 'ksp','nls':'snes','ode':'ts','opt':'tao'}.items():
    str = re.sub('_'+i+'_','_'+j+'_',re.sub('^'+i+'_',j+'_',str))
  return str

def convertToVec(v,comm = PETSc.COMM_WORLD):
  '''Converts a numpy array to a PETSc Vec or returns a PETSc Vec if that is provided
  '''
  if isinstance(v,PETSc.Vec):return v
  return PETSc.Vec().createWithArray(v,comm = comm)

# if we delay everything in the init() to setup() we could inspect the object to make better decisions, for example is there a Grid/DM?
# what functions have been provide in the constructor and in the class definition?

# ***********************************
class Problem:
  '''The base object for all PETSc solvers. This is NEVER instantiated directly by users
  '''
  def __init__(self, comm = PETSc.COMM_WORLD, type = 'default', solution = None,matrix=None, createMatrix = None, pmatrix = None, createPMatrix = None, createLeftVec = None, createRightVec = None,**kwargs):
    '''This is never called directly by users

       solution - vector to hold the problem's solution
       matrix - two dimensiona numpy array that provides the numerical values in the matrix
              - a petsc4py.PETSc.Mat
       createMatrix - a routine that creates the matrix (does not fill its numerical values)
       createLeftVector - a routine that creates the vectors such as to contain function evaluations
       createRightVector - a routine that creates vectors such as to contain the solution
       kwargs - options for controlling the solver, use help(object) to display them
    '''
    # TODO: should this accept solution = None?
    import inspect
    if not isinstance(super(),object): super().__init__(comm = comm,**kwargs) # WTF needed for cooperative inheritance
    self.comm = comm
    if solution: self.solution = convertToVec(solution)
    if createMatrix: self.createMatrix = createMatrix
    if createLeftVec: self.createLeftVec = createLeftVec
    if createRightVec: self.createRightVec = createRightVec
    self.Mat = self.createMatrix(matrix=matrix,**kwargs) # TODO: should not always call this, depends on solver options
    self.Solver.create(self.comm)

  # cannot be merged with __init__() because it needs to be called after TSSetIFunction() which must be called after the __init__()
  def initialSetOptionsKwargs(self,**kwargs):
    ''' 1) sets the options for the solver from both the default options database and kwargs
        2) saves into the object's help string the kwargs supported by the object, that can be displayed with help(object)
    '''
    PETSc.petscHelpPrintfStringBegin(self.comm)
    self.Solver.setFromOptions()
    self.setOptionsFromKwargs(**kwargs)
    self.__doc__ = LinearSystem.__doc__ + convertFromPetscTerminology(PETSc.petscHelpPrintfStringEnd(self.comm))

  def setOptionsFromKwargs(self,**kwargs):
    '''Calls self.Solver.setFromOptions() with the given kwargs arguments
       For example: setOptionsFromKwargs(ls,ls_type='gmres',ls_monitor="")
       The keywoard options='a list of PETSc options database entries' is handled as a special case
       For example: setOptionsFromKwargs(ls,ls_type='gmres',petsc_options="-ksp_monitor -ksp_view")
    '''
    # Requiring the empty string for example with ls_monitor='' is annoying, would be nice not to need it
    options = ''
    for k,v in kwargs.items():
      if k == 'petsc_options':
        options = options + ' ' + v + ' '
      k = convertToPetscTerminology(k)
      options = options + '-' + k + ' ' + str(v) + ' '
    opts = PETSc.Options()
    opts.create()
    opts.insertString(options)
    opts.push()
    self.Solver.setFromOptions()
    opts.pop()

  def createMatrix(self,matrix=None,pmatrix=None,**kwargs):
    '''Often overloaded by subclasses
       Creates the matrix from the self if it is not provide via matrix and no creation function is provided.
       Arguments:
         * matrix - a numpy array/matrix or a PETSc.Mat
         * pmatrix - a numpy array/matrix or a PETSc.Mat that will be used to construct the preconditioner

       Needs to handle situations where no matrix is needed
    '''
    # if matrix not a PETSc Mat wrap it
    if isinstance(matrix,np.ndarray):
      # support for other formats
      if not matrix.ndim == 2: raise RuntimeError("Dimension of array must be 2")
      array = matrix
      matrix = PETSc.Mat()
      matrix.createDense(array.shape, array=array, comm=self.comm)
    return matrix

  def createLeftVector(self,vector=None,**kwargs):
    '''Sometimes overloaded by subclasses
       Creates the left vector from the self if it is not provided by the user and no creation class/function is provided?
    '''
    if vector == None:
      if hasattr(self,'Mat'):
        (f,vector) = self.Mat.createVecs()
    return vector

  def createRightVector(self,vector=None,**kwargs):
    '''Sometimes overloaded by subclasses
       Creates the right vector from the self if it is not provided by the user and no creation class/function is provided?
    '''
    if vector == None:
      if hasattr(self,'Mat'):
        (f,vector) = self.Mat.createVecs()
      elif hasattr(self,'solution'):
        vector = self.solution.duplicate()
    return vector

# ***********************************
class LinearSystem(Problem):
  '''Class for defining and solving linear systems with PETSc
  '''
  def __init__(self, comm = PETSc.COMM_WORLD,rhs = None, evalMatrix = None, evalRHS = None, **kwargs):
    '''
       solution - vector to hold the problem's solution
       matrix - two dimensiona numpy array that provides the numerical values in the matrix
              - a petsc4py.PETSc.Mat
       createMatrix - a routine that creates the matrix (does not fill its numerical values)
       evalMatrix - a routine that evaluates the values in the matrix
       rhs - right hand side vector
       createLeftVector - a routine that creates the right hand side vector (does not fill its numerical values)
       createRightVector - a routine that creates the right hand side vector (does not fill its numerical values)
       evalRHS - a routine that evaluates the right hand side vector
       kwargs - options for controlling the solver, use help(object) to display them
    '''
    if rhs: self.rhs = convertToVec(rhs)
    self.Solver = self.KSP = PETSc.KSP(comm)
    super().__init__(comm = comm,**kwargs)
    self.KSP.setOperators(self.Mat)
    self.initialSetOptionsKwargs(**kwargs)

  def solve(self,rhs=None,solution=None,**kwargs):
    # rhs and solution should be Vec or numpy array, or regular array or other types of arrays etc
    self.setOptionsFromKwargs(**kwargs)
    if rhs == None:
      if hasattr(self,'rhs'): rhs = self.rhs
      else: rhs = self.createLeftVector(**kwargs)
    rhs = convertToVec(rhs)
    self.evalRHS(self.KSP,rhs)  # when should this be called?
    if solution == None:
      if hasattr(self,'solution'): solution = self.solution
      else: solution = self.createRightVector(**kwargs)
    solution = convertToVec(solution)
    self.evalInitialGuess(self.KSP,solution)
    self.evalMatrix(self.Mat)
    self.KSP.solve(rhs, solution)
    return solution

  def evalRHS(self,ksp,rhs):
    '''Overloaded by user. This is option: one can pass the function to the LinearSystem() constructor or to the .solve() method
       Does nothing by default
    '''
    pass

  def evalInitialGuess(self,ksp,solution):
    '''Overloaded by user. This is option: one can pass the function to the LinearSystem() constructor or to the .solve() method
       Does nothing by default
    '''
    pass

  def evalMatrix(self,mat):
    '''Overloaded by user. This is option: one can pass the function to the LinearSystem() constructor or to the .solve() method
       Does nothing by default
    '''
    pass

# ***********************************
class NonlinearSystem(Problem):
  '''Class for defining and solving nonlinear systems with PETSc
  '''
  def __init__(self, comm = PETSc.COMM_WORLD,evalFunction = None, evalJacobian = None, rhs = None, evalRHS = None, evalInitialGuess = None, **kwargs):
    '''
       solution - vector to hold the problem's solution
       matrix - two dimensiona numpy array that will be used to store the computed Jacobians
              - a petsc4py.PETSc.Mat
       createMatrix - a routine that creates the matrix to be used for the Jacobian values (does not fill its numerical values)
       evalFunction - a routine that evaluates the nonlinear function to be solved with
       evalJacobian - a routine that evaluates the values in the matrix
       rhs - right hand side vector
       createLeftVector - a routine that creates the right hand side vector (does not fill its numerical values)
       evalRHS - a routine that evaluates the right hand side vector
       kwargs - options for controlling the solver, use help(object) to display them
    '''
    if rhs: self.rhs = convertToVec(rhs)
    self.Solver = self.SNES = PETSc.SNES(comm=comm)
    super().__init__(comm = comm, **kwargs)
    f = self.createRightVector()
    if evalInitialGuess: self.evalInitialGuess = evalInitialGuess  # Do these make sense?
    if evalRHS: self.evalRHS = evalRHS
    if evalFunction: self.evalFunction = evalFunction
    if evalJacobian: self.evalJacobian = evalJacobian
    self.SNES.setFunction(self.evalFunction,f)                 # if DM exists then use self.dmevalFunction ? See around line 386
    if hasattr(self,'evalJacobian'):
      self.SNES.setJacobian(self.evalJacobian,self.Mat)
    self.initialSetOptionsKwargs(**kwargs)

  def solve(self,solution=None,rhs=None,**kwargs):
    # b and solution should be Vec or numpy array, or regular array or other types of arrays etc
    self.setOptionsFromKwargs(**kwargs)
    if solution == None:
      solution = self.createRightVector()
    solution = convertToVec(solution)
    self.evalInitialGuess(self.SNES,solution) # TODO: under what conditions should this be called?
    self.SNES.solve(rhs, solution)
    return solution

  def evalRHS(self,x):
    '''Overloaded by user. This is option: one can pass the function to the NonlinearSystem() constructor or to the .solve() method
       Does nothing by default
    '''
    pass

  def evalInitialGuess(self,snes,x):
    '''Overloaded by user. This is option: one can pass the function to the NonlinearSystem() constructor or to the .solve() method
       Does nothing by default
    '''
    pass

  def evalFunction(self, snes, x, f):
    '''Overloaded by user. This is option: one can pass the function to the NonlinearSystem() constructor or to the .solve() method
    '''
    pass

#  def evalJacobian(self, snes, x, J, P):
#    '''Overloaded by user. This is option: one can pass the function to the NonlinearSystem() constructor or to the .solve() method
#    '''
#    pass

# using this solver does not require explicitly creating a PETSc objects
def solve(solution = None, rhs=None,evalFunction=None, **kwargs):
  '''This can be called with a Python/numpy native matrix
  '''
  # needs to handle matrix == petsc.PETSc Mat also
  if not evalFunction == None:
    nls = NonlinearSystem(evalFunction = evalFunction,**kwargs)
    return nls.solve(solution = solution,rhs=rhs)
  else:
    ls = LinearSystem(**kwargs)
    return ls.solve(rhs = rhs,solution=solution)

# ***********************************
class ODE(Problem):
  '''Class for defing and solving ODEs (and adjoints etc) with PETSc
  '''
  def __init__(self, comm = PETSc.COMM_WORLD, evalIFunction = None, evalIJacobian = None, evalRHSFunction = None, evalRHSJacobian = None, evalInitialConditions=None, **kwargs):
    '''
       solution - vector to hold the problem's solution
       matrix - two dimensiona numpy array that will be used to store the computed Jacobians
              - a petsc4py.PETSc.Mat
       createMatrix - a routine that creates the matrix to be used for the Jacobian values (does not fill its numerical values)
       evalInitialConditions - evaluates the initial conditions of the ODE
       evalRHSFunction - a routine that evaluates the nonlinear function that defines the right hand side of an explicitly defined ODE
       evalIRHSJacobian - a routine that evaluates the values in the Jacobian of evalRHSFunction()
       evalIFunction - a routine that evaluates the nonlinear function that defines an implicit ODE
       evalIJacobian - a routine that evaluates the values in the Jacobian of evalIFunction()
       createRightVector - a routine that creates work vectors such as the solution vector if needed
       createLeftVector - a routine that creates work vectors such as a location to store function evaluations
       kwargs - options for controlling the solver, use help(object) to display them
    '''
    self.Solver = self.TS = PETSc.TS(comm = comm)
    super().__init__(comm = comm,**kwargs)
    f = self.createRightVector()
    if evalInitialConditions: self.evalInitialConditions = evalInitialConditions  # Do these make sense?
    if evalRHSFunction: self.evalRHSFunction = evalRHSFunction
    if evalRHSJacobian: self.evalRHSJacobian = evalRHSJacobian
    if evalIFunction: self.evalIFunction = evalIFunction
    if evalIJacobian: self.evalIJacobian = evalIJacobian
    if hasattr(self,'evalIFunction'):
      self.TS.setIFunction(self.evalIFunction,f)
    if hasattr(self,'evalIJacobian'):
      self.TS.setIJacobian(self.evalIJacobian,self.Mat)  # TODO: matrix cannot be shared by IJacobian an RHSJacobian
    if hasattr(self,'evalRHSFunction'):
      self.TS.setRHSFunction(self.evalRHSFunction,f)
    if hasattr(self,'evalRHSJacobian'):
      self.TS.setRHSJacobian(self.evalRHSJacobian,self.Mat)
    self.initialSetOptionsKwargs(**kwargs)

  def integrate(self,solution=None,**kwargs):
    # solution should be Vec or numpy array, or regular array or other types of arrays etc
    self.setOptionsFromKwargs(**kwargs)
    if solution == None:
      solution = self.createRightVector()
    solution = convertToVec(solution)
    self.evalInitialConditions(self.TS,solution) # should this always be done?
    # check if explicit method being used and temporarily zero out IFunction?
    self.TS.solve(solution)

  def evalInitialConditions(self,ts,u):
    '''Overloaded by user. This is option: one can pass the function to the ODE() constructor or to the .integrate() method
       Does nothing by default
    '''
    pass

#  def evalIFunction(self, ts,t,u,du,F):
#    '''Overloaded by user. This is option: one can pass the function to the ODE() constructor or to the .integrate() method
#    '''
#    pass

#  def evalIJacobian(self,ts,t,u,du,a,J,P):
#    '''Overloaded by user. This is option: one can pass the Jacobian to the ODE() constructor or to the .integrate() method
#    '''
#    pass

#  def evalRHSFunction(self, ts,t,u,F):
#    '''Overloaded by user. This is option: one can pass the function to the ODE() constructor or to the .integrate() method
#    '''
#    pass

#  def evalRHSJacobian(self,ts,t,u,J,P):
#    '''Overloaded by user. This is option: one can pass the Jacobian to the ODE() constructor or to the .integrate() method
#    '''
#    pass

# using this integrator does not require explicitly creating an PETSc objects
def integrate(solution=None,**kwargs):
  '''This can be called with a Python/numpy native matrix
  '''
  ode = ODE(solution=solution,**kwargs)
  ode.integrate(solution = solution)

# ***********************************
class Optimization(Problem):
  '''Class for defining objective functions (and constraints etc) and optimizing them
  '''
  def __init__(self, solution = None,comm = PETSc.COMM_WORLD,evalObjective = None, evalGradient = None, evalHessian = None, evalInitialGuess = None, **kwargs):
    '''
       solution - vector to hold the problem's solution
       matrix - two dimensiona numpy array that will be used to store the computed Hessians
              - a petsc4py.PETSc.Mat
       createMatrix - a routine that creates the matrix to be used for the Hessian values (does not fill its numerical values)
       evalInitialGuess - evaluates the initial guess for the optimization problem
       evalObjective - a routine that evaluates the object function to be optimized
       evalGradient - a routine that evaluates the gradient of evalObjective()
       evalHessian - a routine that evaluates the Hessian of the evalObjective()
       createLeftVector - a routine that creates work vectors such as the gradient
       createRightVector - a routine that creates work vectors such as the solution
       kwargs - options for controlling the solver, use help(object) to display them
    '''
    self.Solver = self.Tao = PETSc.TAO(comm = comm)
    super().__init__(comm = comm,**kwargs)
    if evalInitialGuess: self.evalInitialGuess = evalInitialGuess # Do these make sense?
    if evalObjective: self.evalObjective = evalObjective
    if evalGradient: self.evalGradient = evalGradient
    if evalHessian: self.evalHessian = evalHessian
    try:
      f = self.createLeftVector()
    except:
      if solution: f = solution.duplicate()
      else: raise RuntimeError('Must provide createLeftVector() or createMatrix()')
    self.Tao.setObjective(self.evalObjective)
    if hasattr(self,'evalGradient'):
      self.Tao.setGradient(self.evalGradient,f)
    if hasattr(self,'evalHessian'):
      self.Tao.setHessian(self.evalHessian)
    self.initialSetOptionsKwargs(**kwargs)

  def optimize(self,solution = None,**kwargs):
    # solution should be Vec or numpy array, or regular array or other types of arrays etc
    self.setOptionsFromKwargs(**kwargs)
    if solution == None:
      solution = self.createRightVector()
    solution = convertToVec(solution)
    self.evalInitialGuess(self.Tao,solution)
    self.Tao.solve(solution)

  def evalInitialGuess(self,tao, x):
    '''Overloaded by user. This is option: one can pass the Objective function to the Optimization() constructor or to the .optimize() method
    '''
    pass

  def evalObjective(self,tao, x):
    '''Overloaded by user. This is option: one can pass the Objective function to the Optimization() constructor or to the .optimize() method
    '''
    pass

#  def evalGradiant(self, tao, x, g):
#    '''Overloaded by user. This is option: one can pass the Gradient to the Optimization() constructor or to the .optimize() method
#    '''
#    pass

#  def evalHessian(self): # TODO: add arguments
#    '''Overloaded by user. This is option: one can pass the Hessian to the Optimization() constructor or to the .optimize() method
#    '''
#    pass

  def EqualityConstraints(self): #TODO: add arguments
    '''Overloaded by user. This is option: one can pass the equality constraints to the Optimization() constructor or to the .optimize() method
    '''
    pass

# using this optimizer does not require explicitly creating an PETSc objects
def optimize(solution = None,**kwargs):
  '''This can be called with a Python/numpy native matrix
  '''
  opt = Optimization(solution=solution,**kwargs)
  return opt.optimize(solution=solution)

# ***********************************
# ***********************************
class Grid():
  '''The base object for all PETSc grid and discretization types; DM provides the matrices and vectors and possibly the adapts the function calls
  '''
  def __init__(self, comm = PETSc.COMM_WORLD, **kwargs):
    super().__init__(comm = comm,**kwargs)  # WTF needed for cooperative inheritance
    self.comm = comm

  # these overload the base operations provided in Problem
  def createMatrix(self,**kwargs):
    return self.DM.createMat()

  def createLeftVector(self,**kwargs):
    return self.DM.createGlobalVector()

  def createRightVector(self,**kwargs):
    return self.DM.createGlobalVector()

# ***********************************
class Structured(Grid):
  '''Structured grid object'''
  def __init__(self, dimensions=(2,2), comm = PETSc.COMM_WORLD, **kwargs):  # TODO: add support for passing the evalXXXLocal() functions here
    self.DM = self.da = PETSc.DMDA().create(dimensions,stencil_width = 1, comm = comm, **kwargs)
    super().__init__(comm=comm,**kwargs)

  def evalInitialGuess(self, snes, x):
    '''Dispatches to DMDA local function
    '''
    # if would be nice if we didn't need the Local visible to users; but this might require NonlinearSystem to know if DM exists when calling
    # setFunction() and friends see around line 186
    self.evalInitialGuessLocal(self.DM.getVecArray(x))

  def evalRHS(self,x):
    '''Dispatches to DMDA local function
    '''
    self.evalRHSLocal(self,self.da.getVecArray(x))

  def evalFunction(self, snes, x, f):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:  # actually need a smarter test for case with nontrivial boundary conditions that require local
      with self.da.globalToLocal(x) as xlocal:
        self.evalFunctionLocal(self.da.getVecArray(xlocal,readonly=1),self.da.getVecArray(f))
    else:
      self.evalFunctionLocal(self.da.getVecArray(x,readonly=1),self.da.getVecArray(f))

  def evalJacobian(self, snes, x, J, P):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(x) as xlocal:
        self.evalJacobianLocal(self.da.getVecArray(xlocal,readonly=1),P)  # inspect the user evalJacobianLocal to see if it takes both J and P?
    else:
      self.evalJacobianLocal(self.da.getVecArray(x,readonly=1),P)
    if not P == J: J.assemble()

  # the problem is that the DM needs to know about all the PETSc Problem types, KSP, SNES, ODE, ...
  # this would be true with PETSc C API if we actively support the Local versions for all solvers

  def evalInitialConditions(self,u):
    '''Dispatches to DMDA local function
    '''
    self.evalInitialConditionsLocal(self,self.da.getVecArray(u))

  def evalIFunction(self, ts,t,u,du,F):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocalVector(u),self.da.globalToLocal(du) as ulocal,dulocal:
        self.evalIFunctionLocal(self, t,self.da.getVecArray(ulocal,readonly=1),self.da.getVecArray(dulocal,readonly=1),self.da.getVecArray(F))
    else:
      self.evalIFunctionLocal(self, t,self.da.getVecArray(u,readonly=1),self.da.getVecArray(du,readonly=1),self.da.getVecArray(F))

  def evalIJacobian(self,ts,t,u,du,a,J,P):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
     with self.da.globalToLocal(u),self.da.globalToLocal(du) as ulocal,dulocal:
        self.evalIJacobianLocal(self, t,self.da.getVecArray(ulocal,readonly=1),self.da.getVecArray(dulocal,readonly=1),P)
    else:
      self.evalIJacobianLocal(self, t,self.da.getVecArray(u,readonly=1),self.da.getVecArray(du,readonly=1),P)
    if not P == J: J.assemble()

  def evalRHSFunction(self, ts,t,u,F):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(u) as ulocal:
        self.evalRHSFunctionLocal(self, t,self.da.getVecArray(ulocal,readonly=1),self.da.getVecArray(F))
    else:
      self.evalRHSFunctionLocal(self, t,self.da.getVecArray(u,readonly=1),self.da.getVecArray(F))

  def evalRHSJacobian(self,ts,t,u,J,P):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(u) as ulocal:
        self.evalRHSJacobianLocal(self, t,self.da.getVecArray(ulocal,readonly=1),P)
    else:
      self.evalRHSJacobianLocal(self, t,self.da.getVecArray(u,readonly=1),P)
    if not P == J: J.assemble()

  def evalObjective(self,tao, u):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(u) as ulocal:
        self.evalObjectiveLocal(self, self.da.getVecArray(ulocal))
    else:
      self.evalObjectiveLocal(self, self.da.getVecArray(u))

  def evalGradiant(self, tao, u, g):
    '''Dispatches to DMDA local function
    '''
    if self.da.comm.size > 1:
      with self.da.globalToLocal(u) as ulocal:
        self.evalGradiantLocal(self,  self.da.getVecArray(ulocal,readonly=1), self.da.getVecArray(g))
    else:
      self.evalGradiantLocal(self,  self.da.getVecArray(u,readonly=1), self.da.getVecArray(g))

  def evalHessian(self): # TODO: add arguments
    '''Dispatches to DMDA local function
    '''
    pass

# there could be a subclass of Structured where users only provide function evaluations for inside the inner loops of the
# function and Jacobian evaluation

# ***********************************
class Staggered(Grid):
  '''Staggered structured grid object'''  # could this share Structured code above it ds.getVecArray() worked for staggered grids?
  pass