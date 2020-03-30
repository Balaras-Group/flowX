"""Routine to solve the Poisson system with Jacobi."""

import numpy
import time

def solve_jacobi(grid, ivar, rvar, maxiter=3000, tol=1e-9, verbose=False):
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
    difference: float
        Final difference.
    verbose : bool, optional
        Set True to display convergence information;
        default: False.

    """

    start = time.time()
    u = grid.get_values(ivar)
    b = grid.get_values(rvar)
    dx, dy = grid.dx, grid.dy

    ites = 0
    difference = tol + 1.0
    while ites < maxiter and difference > tol:
        u_old = numpy.copy(u)  # previous solution
        u[1:-1, 1:-1] = (((u_old[1:-1, :-2] +
                             u_old[1:-1, 2:]) * dy**2 +
                            (u_old[:-2, 1:-1] +
                             u_old[2:, 1:-1]) * dx**2 -
                            b[1:-1, 1:-1] * dx**2 * dy**2) /
                           (2 * (dx**2 + dy**2)))

        grid.fill_guard_cells(ivar)

        difference  = (numpy.sqrt(numpy.sum((u - u_old)**2) /
                    ((grid.nx + 2) * (grid.ny + 2))))
        ites += 1

    end=time.time()
    if verbose:
        print('Jacobi method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        print('- Final difference: {}'.format(difference))
        print('- Execution time: {}'.format(end - start))
    return ites, difference
