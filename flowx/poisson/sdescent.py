import numpy as np
import scipy as sp
from scipy import sparse
from numpy.linalg import matrix_rank

def solve_sdescent(grid,ivar,rvar,maxiter=10000,tol=1e-9,verbose=False):

    """
    This function solves Ax=b using Steepest Descent Method.

    Arguments
    ======================
    grid: Grid object which contains information for computational domain
    ivar: Solution vector - x
    rvar: Right hand side vector - b
    Maxiter: Maximum number of iterations

    Return
    =======================
    ivar: solution vector
    """

    nx = grid.nx
    ny = grid.ny
    dx = grid.dx
    dy = grid.dy
    x = grid.get_values(ivar)
    rhs = grid.get_values(rvar)

    coef1_y, coef1_x, coef1_c = 0.0, 0.0, 0.0

    if grid.bc_type[ivar][0]=='dirichlet':
        coef1_y = (-2/(dx*dx))+(-3/(dy*dy))
        coef1_x = (-3/(dx*dx))+(-2/(dy*dy))
        coef1_c = (-3/(dx*dx))+(-3/(dy*dy))
    elif grid.bc_type[ivar][0]=='neumann':
        coef1_y = (-2/(dx*dx))+(-1/(dy*dy))
        coef1_x = (-1/(dx*dx))+(-2/(dy*dy))
        coef1_c = (-1/(dx*dx))+(-1/(dy*dy))
       
    coef1_i = (-2/(dx*dx))+(-2/(dy*dy))
    coef2 = (1/(dx*dx))
    coef3 = (1/(dy*dy))

    diag_x = sp.sparse.diags([coef2,coef1_i,coef2],[-1,0,1],shape=(nx,ny)).toarray()
    diag_y = sp.sparse.diags([coef2,coef1_y,coef2],[-1,0,1],shape=(nx,ny)).toarray() 
    diag = sp.sparse.diags([coef3],[0],shape=(nx,ny)).toarray()
    diag_y[0,0] = coef1_c
    diag_y[-1,-1] = coef1_c
    diag_x[0,0] = coef1_x
    diag_x[-1,-1] = coef1_x
 
    A = np.zeros((nx*ny,nx*ny))
    A[0:nx,0:ny] = diag_y
    A[(nx-1)*nx:nx*nx,(ny-1)*ny:ny*ny] = diag_y
    A[nx:2*nx,0:ny] = diag
    A[0:nx,ny:2*ny] = diag  

    for i in range(1,nx-1):        
        A[nx*i:nx*(i+1),ny*i:ny*(i+1)] = diag_x
        A[nx*(i+1):nx*(i+2),nx*i:nx*(i+1)] = diag
        A[nx*i:nx*(i+1),nx*(i+1):nx*(i+2)] = diag
 
    k = 0
    b = np.zeros((nx*ny,1))
    xs = np.zeros((nx*ny,1))

    for i in range(nx):
        for j in range(ny):
            b[k] = rhs[i+1,j+1]
#            xs[k] = x[i+1,j+1]
            k = k+1

    err = 1.0
    it = 1
    xs = np.zeros((nx*ny,1))
    x_old = np.ones((nx*ny,1))

    while err>tol and it<maxiter:

        r = b-np.dot(A,x_old)
        alpha = np.dot(np.transpose(r),r)/np.dot(np.dot(np.transpose(r),A),r)
        xs = x_old+alpha*r
        it = it+1
        err = np.sqrt(np.sum((xs-x_old)**2)/(nx*ny))
        x_old = xs

    k = 0
    for i in range(nx):
        for j in range(ny):
            x[i+1,j+1] = xs[k]
            k = k+1
    
    if verbose:
        print("Steepest Descent Method:")
        print("The solution is obtained in %d iterations" %it)
