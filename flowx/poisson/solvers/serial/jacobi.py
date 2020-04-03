"""Routine to solve the Poisson system with Jacobi."""

import numpy
import time

def solve_serial_jacobi(grid, ivar, rvar, options):
    """Solve the Poisson system using a Jacobi method.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
    rvar : string
        Name of the grid variable of the right-hand side.
    options : dictionary

    Returns
    -------
    ites: integer
        Number of iterations computed.
    residual: float
        Final residual.
    """

    maxiter = options['maxiter']
    tol = options['tol']
    verbose = options['verbose']

    phi = grid.get_values(ivar)
    b = grid.get_values(rvar)
    dx, dy = grid.dx, grid.dy

    ites = 0
    residual = tol + 1.0
   
    t=time.time()

    while ites < maxiter and residual > tol:
        phi_old = numpy.copy(phi)  # previous solution
        phi[1:-1, 1:-1] = (((phi_old[1:-1, :-2] +
                             phi_old[1:-1, 2:]) * dy**2 +
                            (phi_old[:-2, 1:-1] +
                             phi_old[2:, 1:-1]) * dx**2 -
                            b[1:-1, 1:-1] * dx**2 * dy**2) /
                           (2 * (dx**2 + dy**2)))

        grid.fill_guard_cells(ivar)

        residual = (numpy.sqrt(numpy.sum((phi - phi_old)**2) /
                    ((grid.nx + 2) * (grid.ny + 2))))
        ites += 1
    
    t=time.time()-t

    if verbose:
        print('Jacobi method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        

        print('- Final residual: {}'.format(residual))

        print('- Elapsed time: {}'.format(t))
        
    return ites, residual
