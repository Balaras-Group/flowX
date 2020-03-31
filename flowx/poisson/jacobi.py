"""Routine to solve the Poisson system with Jacobi."""

import numpy as np
import time

def solve_jacobi(grid, ivar, rvar, maxiter=100000, tol=1e-9, verbose=False):
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
    u = grid.get_values(ivar)
    rhs = grid.get_values(rvar)
    dx, dy = grid.dx, grid.dy
    nx,ny = grid.nx, grid.ny
    asol = grid.get_values('asol')

    coef1 = 1.0/((2.0/(dx*dx))+(2.0/(dy*dy)))
    coef2 = 1.0/(dx*dx)
    coef3 = 1.0/(dy*dy)

    err = 1.0
    it = 1
    err_list = np.zeros((maxiter,1))

    #phi[1:-1, 1:-1] = (((phi_old[1:-1, :-2] +
    #                     phi_old[1:-1, 2:]) * dy**2 +
    #                    (phi_old[:-2, 1:-1] +
    #                     phi_old[2:, 1:-1]) * dx**2 -
    #                    b[1:-1, 1:-1] * dx**2 * dy**2) /
    #                   (2 * (dx**2 + dy**2)))

    tic = time.perf_counter() 
    while it < maxiter and err > tol:

        u_old = np.copy(u)

        for j in range(1,ny+1):
            for i in range(1,nx+1):

                u[i,j] =  coef1*coef2*(u_old[i+1,j]+u_old[i-1,j])+\
                          coef1*coef3*(u_old[i,j+1]+u_old[i,j-1])-\
                          coef1*rhs[i,j]

        grid.fill_guard_cells(ivar)
        err = (np.sqrt(np.sum((u - u_old)**2) /
              ((nx + 2) * (ny + 2))))

        max_err = np.max(np.absolute((asol - u)))
        
        err_list[it-1] = max_err
        it = it+1
   
    toc = time.perf_counter()

    if verbose:
        print('Jacobi method:')
        if it == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(it))
        print('- Final residual: {}'.format(err))
        print(f'- Simulation time: {toc - tic:0.4f} seconds')

    return it,err_list
