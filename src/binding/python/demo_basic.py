import petsc
import numpy as np

# solve a linear system with a provided matrix and right hand side
ls = petsc.LinearSystem(matrix = np.array([[1, 2], [3, 4]]),comm = petsc.PETSc.COMM_SELF)

solution = ls.createLeftVector()
rhs = ls.createRightVector()
rhs.set(1)
ls.solve(rhs, solution = solution, type='gmres', ls_view='', ls_monitor='', ls_converged_reason='')
# ****************************8

# solve a linear system without explicitly creating a linear solver object
solution = petsc.solve(rhs = rhs, matrix = np.array([[1, 2], [3, 4]]), type='gmres', ls_view='', ls_monitor='', ls_converged_reason='',comm = petsc.PETSc.COMM_SELF)

# ****************************8
# solve a nonlinear system defined by subclassing a NonlinearSystem object
class NonlinearSystem(petsc.NonlinearSystem):
  def createMatrix(self,**kwargs):
    mat = petsc.PETSc.Mat()
    mat.createDense((2,2),comm = self.comm)
    mat.setUp()
    return mat

  def evalFunction(self, snes, x, f):
    f[0] = (x[0]*x[0] + x[0]*x[1] - 3.0).item()
    f[1] = (x[0]*x[1] + x[1]*x[1] - 6.0).item()
    f.assemble()

  def evalJacobian(self, snes, x, J, P):
    P[0,0] = (2.0*x[0] + x[1]).item()
    P[0,1] = (x[0]).item()
    P[1,0] = (x[1]).item()
    P[1,1] = (x[0] + 2.0*x[1]).item()
    P.assemble()
    if J != P: J.assemble()

nls = NonlinearSystem(comm = petsc.PETSc.COMM_SELF)
solution.setArray([2,3])
nls.solve(solution=solution,nls_monitor='',nls_view='')

# ****************************8
# solve a nonlinear system without explicitly creating a nonliner system object
def evalFunction(snes, x, f):
  f[0] = (x[0]*x[0] + x[0]*x[1] - 3.0).item()
  f[1] = (x[0]*x[1] + x[1]*x[1] - 6.0).item()
  f.assemble()

def evalJacobian(snes, x, J, P):
  P[0,0] = (2.0*x[0] + x[1]).item()
  P[0,1] = (x[0]).item()
  P[1,0] = (x[1]).item()
  P[1,1] = (x[0] + 2.0*x[1]).item()
  P.assemble()
  if J != P: J.assemble()

solution.setArray([2,3])
petsc.solve(solution = solution,matrix = np.array([[1, 2], [3, 4]]),evalFunction = evalFunction, evalJacobian = evalJacobian, comm = petsc.PETSc.COMM_SELF)

# ****************************8
# solve an ODE defined by subclassing an ODE object
class ODE(petsc.ODE):
  def createMatrix(self,**kwargs):
    mat = petsc.PETSc.Mat()
    mat.createDense((3,3),comm = self.comm)
    mat.setUp()
    return mat

  def evalIFunction(self, ts,t,u,du,F):
    f = du + u * u
    f.copy(F)

  def evalIJacobian(self,ts,t,u,du,a,J,P):
    P.zeroEntries()
    diag = a + 2 * u
    P.setDiagonal(diag)
    P.assemble()
    if J != P: J.assemble()

ode = ODE(comm = petsc.PETSc.COMM_SELF)
solution = ode.createLeftVector()
solution[0], solution[1], solution[2] = 1, 2, 3
ode.integrate(solution,ode_max_time=1,ode_monitor='',ode_view='')

# ****************************8
# solve a ODE without explicitly creating an ODE object
def createMatrix(comm = petsc.COMM_SELF,**kwargs):
  mat = petsc.PETSc.Mat()
  mat.createDense((3,3),comm = comm)
  mat.setUp()
  return mat

def evalIFunction(ts,t,u,du,F):
  f = du + u * u
  f.copy(F)

def evalIJacobian(ts,t,u,du,a,J,P):
  P.zeroEntries()
  diag = a + 2 * u
  P.setDiagonal(diag)
  P.assemble()
  if J != P: J.assemble()

solution[0], solution[1], solution[2] = 1, 2, 3
petsc.integrate(solution=solution,evalIFunction = evalIFunction, evalIJacobian = evalIJacobian, createMatrix = createMatrix, ode_max_time=1,ode_monitor='',ode_view='',comm = petsc.PETSc.COMM_SELF)
# ****************************
# solve an optimization problem by subclassing an Optimization object
class Optimization(petsc.Optimization):
  def createLeftVector(self,**kwargs):
    vec = petsc.PETSc.Vec().create(comm=petsc.COMM_SELF)
    vec.setSizes(2)
    vec.setUp()
    return vec

  def createRightVector(self,**kwargs):
    return self.createLeftVector(**kwargs)

  def evalObjective(self, tao, x):
    return (x[0] - 2.0)**2 + (x[1] - 2.0)**2 - 2.0*(x[0] + x[1])

  def evalGradient(self, tao, x, g):
    g[0] = 2.0*(x[0] - 2.0) - 2.0
    g[1] = 2.0*(x[1] - 2.0) - 2.0
    g.assemble()

opt = Optimization(comm = petsc.PETSc.COMM_SELF)
solution = opt.createLeftVector()
opt.optimize(solution,opt_monitor='',opt_view='',opt_view_solution='')
   # bug somewhere, tao_view does not work if provided here but tao_monitor does?
   # bug in Tao, view_solution acts like monitor_solution

# ****************************
# solve an optimization problem without explicitly creating an Optimization object
solution = petsc.PETSc.Vec().create(comm=petsc.COMM_SELF)
solution.setSizes(2)
solution.setUp()

def evalObjective(tao, x):
  return (x[0] - 2.0)**2 + (x[1] - 2.0)**2 - 2.0*(x[0] + x[1])

def evalGradient(tao, x, g):
  g[0] = 2.0*(x[0] - 2.0) - 2.0
  g[1] = 2.0*(x[1] - 2.0) - 2.0
  g.assemble()

solution.set(0.0)
petsc.optimize(solution=solution,evalObjective = evalObjective, evalGradient = evalGradient, opt_monitor='',opt_view='',opt_view_solution='',comm = petsc.PETSc.COMM_SELF)