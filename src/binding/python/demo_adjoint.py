import petsc
import numpy as np
from numpy import exp, sqrt

class Vanderpol(petsc.ODE):
  n = 2
  def __init__(self, mu_=1.0e3,**kwargs):
    super().__init__(**kwargs)
    self.mu_ = mu_

  def createMatrix(self,**kwargs):
    return petsc.PETSc.Mat().createDense([2,2], comm=self.comm).setUp()

  def evalInitialConditions(self,ts, u):
    u[0] = 2.0
    u[1] = -2.0/3.0 + 10.0/(81.0*self.mu_) - 292.0/(2187.0*self.mu_*self.mu_)
    u.assemble()

  def evalRHSFunction(self, ts, t, u, f):
    f[0] = u[1]
    f[1] = self.mu_*((1.-u[0]*u[0])*u[1]-u[0])
    f.assemble()

  def evalRHSJacobian(self, ts, t, u, A, B):
    A[0,0] = 0
    A[1,0] = -self.mu_*(2.0*u[1]*u[0]+1.)
    A[0,1] = 1.0
    A[1,1] = self.mu_*(1.0-u[0]*u[0])
    A.assemble()
    if A != B: B.assemble()

ode = Vanderpol(ts_problemtype=petsc.PETSc.TS.ProblemType.NONLINEAR)

ode.TS.setType(ode.TS.Type.RK)

ode.TS.setSaveTrajectory()
ode.TS.setTime(0.0)
ode.TS.setTimeStep(0.001)
ode.TS.setMaxTime(0.5)
ode.TS.setMaxSteps(1000)
ode.TS.setExactFinalTime(petsc.PETSc.TS.ExactFinalTime.MATCHSTEP)

u = ode.integrate(ts_monitor='')

