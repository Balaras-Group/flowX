"""Routine to solve the Poisson system with Gauss-Seidel."""

import numpy
import time

def solve_gauss_seidel(grid, ivar, rvar, maxiter=100000, tol=1e-9, verbose=False):
    """Solve the Poisson system using a Gauss_Seidel method.

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
    # Copy of variables
    u = grid.get_values(ivar)
    rhs = grid.get_values(rvar)
    asol = grid.get_values('asol')

    # Copy of necessary parameters
    nx = grid.nx
    ny = grid.ny
    dx = grid.dx
    dy = grid.dy

    coef1 = 1.0/((2.0/(dx*dx))+(2.0/(dy*dy)))
    coef2 = 1.0/(dx*dx)
    coef3 = 1.0/(dy*dy)

    err = 1.0
    it = 1
    err_list = numpy.zeros((maxiter,1))
    
    tic = time.perf_counter()
    while it < maxiter and err > tol:

        u_old = numpy.copy(u)

        for j in range(1,ny+1):
            for i in range(1,nx+1):
                u[i,j] =  coef1*coef2*(u_old[i+1,j]+u[i-1,j])+\
    			  coef1*coef3*(u_old[i,j+1]+u[i,j-1])-\
    			  coef1*rhs[i,j]
    
        grid.fill_guard_cells(ivar)
        err = (numpy.sqrt(numpy.sum((u - u_old)**2) /
              ((nx + 2) * (ny + 2))))
       
        max_err = numpy.max(numpy.absolute((asol - u)))
        err_list[it-1] = max_err
        it = it+1

    toc = time.perf_counter()

    if verbose:
        print('Gauss-Seidel Method:')
        if it==maxiter:
            print("Simulation reached maximum iteration")
        print("- Number of iterations: %d" %it)
        print('- Final residual: {}'.format(err))
        print(f'- Simulation time: {toc - tic:0.4f} seconds')
    return it,err_list
