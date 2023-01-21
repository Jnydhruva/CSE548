#
#  Examples that use a petsc.Grid object to automatically manage the creation of Vecs, Mats, and simplify the user function evaluations
#
import petsc
import numpy as np
from numpy import exp, sqrt

class StructuredNonlinearSystem(petsc.Structured,petsc.NonlinearSystem):
  '''The nonlinear system is defined using finite differences on a simple structured grid
  '''
  def __init__(self,lambda_ = 6.0,**kwargs): # TODO: generate options database handling for these automatically
    super().__init__(**kwargs)
    self.lambda_ = lambda_

  def evalInitialGuessLocal(self, x):
    mx, my, mz = self.da.getSizes()
    hx, hy, hz = [1.0/(m-1) for m in [mx, my, mz]]
    scale = self.lambda_/(self.lambda_ + 1.0)
    #
    (xs, xe), (ys, ye), (zs, ze) = self.da.getRanges()
    for k in range(zs, ze):
       min_k = min(k,mz-k-1)*hz
       for j in range(ys, ye):
         min_j = min(j,my-j-1)*hy
         for i in range(xs, xe):
           min_i = min(i,mx-i-1)*hx
           if (i==0    or j==0    or k==0 or i==mx-1 or j==my-1 or k==mz-1):
             # boundary points
             x[i, j, k] = 0.0
           else:
             # interior points
             min_kij = min(min_i,min_j,min_k)
             x[i, j, k] = scale*sqrt(min_kij)

  def evalFunctionLocal(self, x, f):
    mx, my, mz = self.da.getSizes()
    hx, hy, hz = [1.0/m for m in [mx, my, mz]]
    hxhyhz  = hx*hy*hz
    hxhzdhy = hx*hz/hy;
    hyhzdhx = hy*hz/hx;
    hxhydhz = hx*hy/hz;
     #
    (xs, xe), (ys, ye), (zs, ze) = self.da.getRanges()
    for k in range(zs, ze):
      for j in range(ys, ye):
        for i in range(xs, xe):
          if (i==0    or j==0    or k==0 or i==mx-1 or j==my-1 or k==mz-1):
            f[i, j, k] = x[i, j, k] - 0
          else:
            u   = x[ i  ,  j  ,  k ] # center
            u_e = x[i+1 ,  j  ,  k ] # east
            u_w = x[i-1 ,  j  ,  k ] # west
            u_n = x[ i  , j+1 ,  k ] # north
            u_s = x[ i  , j-1 ,  k ] # south
            u_u = x[ i  ,  j  , k+1] # up
            u_d = x[ i  ,  j  , k-1] # down
            u_xx = (-u_e + 2*u - u_w)*hyhzdhx
            u_yy = (-u_n + 2*u - u_s)*hxhzdhy
            u_zz = (-u_u + 2*u - u_d)*hxhydhz
            f[i, j, k] = u_xx + u_yy + u_zz - self.lambda_*exp(u)*hxhyhz

  def evalJacobianLocal(self, x, P):
    mx, my, mz = self.da.getSizes()
    hx, hy, hz = [1.0/m for m in [mx, my, mz]]
    hxhyhz  = hx*hy*hz
    hxhzdhy = hx*hz/hy;
    hyhzdhx = hy*hz/hx;
    hxhydhz = hx*hy/hz;
    #
    P.zeroEntries()
    row = petsc.PETSc.Mat.Stencil()
    col = petsc.PETSc.Mat.Stencil()
    #
    (xs, xe), (ys, ye), (zs, ze) = self.da.getRanges()
    for k in range(zs, ze):
      for j in range(ys, ye):
         for i in range(xs, xe):
           row.index = (i,j,k)
           row.field = 0
           if (i==0    or j==0    or k==0 or i==mx-1 or j==my-1 or k==mz-1):
             P.setValueStencil(row, row, 1.0)
           else:
             u = x[i,j,k]
             diag = (2*(hyhzdhx+hxhzdhy+hxhydhz) - self.lambda_*exp(u)*hxhyhz)
             for index, value in [\
                                  ((i,j,k-1), -hxhydhz),\
                                  ((i,j-1,k), -hxhzdhy),\
                                  ((i-1,j,k), -hyhzdhx),\
                                  ((i, j, k), diag),\
                                  ((i+1,j,k), -hyhzdhx),\
                                  ((i,j+1,k), -hxhzdhy),\
                                  ((i,j,k+1), -hxhydhz),\
                                 ]:
               col.index = index
               col.field = 0
               P.setValueStencil(row, col, value)
    P.assemble()

snls = StructuredNonlinearSystem(dimensions = (4,4,4))
s = snls.solve(nls_monitor='',nls_view='')
