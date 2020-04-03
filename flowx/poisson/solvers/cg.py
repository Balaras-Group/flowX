"""Routine to solve the Poisson system with Conjugate-Gradient."""

import numpy


def solve_serial_cg(grid, ivar, rvar, options):
    """Solve the Poisson system using a conjugate-gradient method.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
    rvar : string
        Name of the grid variable of the right-hand side.

    options: dictionary

    Returns
    -------
    ites: integer
        Number of iterations computed.
    residual: float
        Final residual.
    """

    verbose = options['verbose']
    maxiter = options['maxiter']
    tol = options['tol']

    def A(p):
        return ((p[:-2, 1:-1] - 2 * p[1:-1, 1:-1] + p[2:, 1:-1]) / dx**2 +
                (p[1:-1, :-2] - 2 * p[1:-1, 1:-1] + p[1:-1, 2:]) / dy**2)

    def fill_guard_cells_neumann(x, bc_val, dx, dy):
        x[0, :] = bc_val * dx + x[1, :]
        x[-1, :] = bc_val * dx + x[-2, :]
        x[:, 0] = bc_val * dy + x[:, 1]
        x[:, -1] = bc_val * dy + x[:, -2]

    p = grid.get_values(ivar)  # initial guess
    b = grid.get_values(rvar)  # RHS of the system
    dx, dy = grid.dx, grid.dy  # cell widths

    r = b[1:-1, 1:-1] - A(p)  # initial residuals
    rk_norm = numpy.sum(r * r)  # inner product
    d = numpy.zeros_like(p)  # search direction
    d[1:-1, 1:-1] = r  # set direction to initial residual

    # Check for Neumann boundary conditions.
    bc_type, bc_val = grid.bc_type[ivar][0], grid.bc_val[ivar][0]
    if bc_type == 'neumann':
        # Apply Neumann boundary conditions to search direction.
        fill_guard_cells_neumann(d, bc_val, dx, dy)

    ites = 0  # iteration index
    res = rk_norm  # initial residual
    while ites < maxiter and res > tol:
        Ad = A(d)
        alpha = rk_norm / numpy.sum(d[1:-1, 1:-1] * Ad)  # step size
        p[1:-1, 1:-1] += alpha * d[1:-1, 1:-1]  # update solution
        r -= alpha * Ad  # update residuals
        r_norm = numpy.sum(r * r)  # inner product
        beta = r_norm / rk_norm
        rk_norm = r_norm
        d[1:-1, 1:-1] = r + beta * d[1:-1, 1:-1]  # update search direction
        if bc_type == 'neumann':
            fill_guard_cells_neumann(d, bc_val, dx, dy)
        res = r_norm
        ites += 1

    grid.fill_guard_cells(ivar)

    if verbose:
        print('CG method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        print('- Final residual: {}'.format(res))

    return ites, res
