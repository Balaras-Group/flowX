"""Routine to solve the Poisson system with Gauss-Seidel."""

import numpy
import time

def solve_serial_gauss(grid, ivar, rvar, options):
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
        for j in range (1,grid.ny+1):
            for i in range (1,grid.nx+1):
                phi[i,j] = (((phi[i,j-1] +
                             phi[i,j+1]) * dy**2 +
                            (phi[i-1,j] +
                             phi[i+1,j]) * dx**2 -
                            b[i,j] * dx**2 * dy**2) /
                           (2 * (dx**2 + dy**2)))

        grid.fill_guard_cells(ivar)

        residual = (numpy.sqrt(numpy.sum((phi - phi_old)**2) /
                    ((grid.nx + 2) * (grid.ny + 2))))
        ites += 1
    
    t=time.time()-t

    if verbose:
        print('Gauss  method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        

        print('- Final residual: {}'.format(residual))

        print('- Elapsed time: {}'.format(t))
        
    return ites, residual
