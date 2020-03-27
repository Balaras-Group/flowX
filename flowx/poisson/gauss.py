"""Routine to solve the Poisson system with Gauss-Seidel."""

import numpy


def solve_gauss(grid, ivar, rvar, maxiter=3000, tol=1e-9, verbose=False):
    """Solve the Poisson system using Gauss-Seidel method.

    Arguments
    ---------
    grid : grid object
        Grid containing data
    ivar : string
        Name of the grid variable of the numerical solution
    rvar : string
        Name of the grid variable of the right-hand side
    maxiter : integer, optional
        Maximum number of iterations;
        Default : 3000
    tol : float, optional
        Exit - Criterion tolerance;
        Default : 1e-9

    Returns
    -------
    ites : integer
        Number of iterations computed
    residual : float
        Final residual
    verbose : bool, optional
        Set True to display convergence information;
        Default : False

    """
    phi = grid.get_values(ivar)
    b = grid.get_values(rvar)
    dx, dy = grid.dx, grid.dy

    ites = 0
    residual = tol + 1.0
    while ites < maxiter and residual > tol:
        phi_old = numpy.copy(phi)  # previous solution
        for j in range(1, grid.ny + 1):
            for i in range(1, grid.nx + 1):
                phi[i, j] = (((phi[i, j - 1] + phi[i, j + 1]) * dy**2 +
                              (phi[i - 1, j] + phi[i + 1, j]) * dx**2 -
                              b[i, j] * dx**2 * dy**2) /
                             (2.0 * (dx**2 + dy**2)))

        grid.fill_guard_cells(ivar)

        residual = (numpy.sqrt(numpy.sum((phi - phi_old)**2) /
                    ((grid.nx + 2) * (grid.ny + 2))))

        ites += 1

    if verbose:
        print('Gauss - Seidel method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        print('- Final residual: {}'.format(residual))

    return ites, residual
