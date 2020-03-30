"""Routine to solve the Poisson system with scipy Direct Solver."""

import numpy
import scipy
from scipy.sparse.linalg import dsolve
    """Solve the Poisson system using a Jacobi method.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
    rvar : string
        Name of the grid variable of the right-hand side.
    maxiter : integer, optional
        Maximum number of iterations;
        default: 3000
    tol : float, optional
        Exit-criterion tolerance;
        default: 1e-9

    Returns
    -------
    ites: integer
        Number of iterations computed.
    residual: float
        Final residual.
    verbose : bool, optional
        Set True to display convergence information;
        default: False.

    """
def solve_cg(grid, ivar, rvar,maxiter=3000, tol=1e-9,verbose=False):


    from scipy.sparse.linalg import dsolve
    ones = numpy.ones(nx*ny-nx)
    diag = -4*numpy.ones(nx*nx)
    mid_diags = numpy.ones(nx*nx)
    mid_diags[3:mid_diags.size:4] = 0.0 
    Mat  = [ones, mid_diags, diag, mid_diags, ones]
    k    = [nx,1,0,-1,-ny]
    matrix = scipy.sparse.diags(Mat, k)
    pyplot.spy(matrix)
    b_flatten=grid.get_values('rvar')[1:-1,1:-1].flatten()
    result=scipy.sparse.linalg.spsolve(matrix,b_flatten)
    result=numpy.resize(result, nx*nx).reshape(ny,nx)

    if verbose:
        print('Jacobi method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        print('- Final residual: {}'.format(residual))

    return ites, residual
