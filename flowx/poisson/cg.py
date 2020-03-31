import numpy as np
import scipy as sp
from scipy import sparse

def solve_cg(grid,ivar,rvar,maxiter=10000,tol=1e-9,verbose=False):

    """
    This function solves Ax=b using Conjugate Gradient Method.

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

    coef1_y = (-2/(dx*dx))+(-3/(dy*dy))
    coef1_x = (-3/(dx*dx))+(-2/(dy*dy))
    coef1_c = (-3/(dx*dx))+(-3/(dy*dy))
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

    b = np.zeros((nx*ny,1))
    k = 0 
    for i in range(nx):
        for j in range(ny):
            b[k] = rhs[i+1,j+1]
            k = k+1

    err = 1.0
    it = 1
    xs = np.zeros((nx*ny,1))
    x_old = 2*(np.ones((nx*ny,1)))

    r = b-np.dot(A,x_old)
    d = r

    while err>tol and it<maxiter:
        alpha = np.dot(np.transpose(r),r)/np.dot(np.transpose(d),np.dot(A,d))
        xs = x_old+alpha*d
        r_new = r-alpha*np.dot(A,d)
        beta = np.dot(np.transpose(r_new),r_new)/np.dot(np.transpose(r),r)
        d_new = r_new + beta*d

        r = r_new
        d = d_new
        err = np.sqrt(np.sum((xs-x_old)**2)/(nx*ny)) 
        x_old = xs
        it = it+1

    k = 0
    for i in range(nx):
        for j in range(ny):
            x[i+1,j+1] = xs[k]
            k = k+1

    if verbose:
        print("Conjugate Gradient Method:")
        print("The solution is obtained in %d iterations" %it)    
