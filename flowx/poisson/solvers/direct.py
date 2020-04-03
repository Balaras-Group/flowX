import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import spdiags, csr_matrix

def solve_serial_direct(grid, ivar, rvar, options):

    """Solve the Poisson system using a direct solver from the scipy library.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
    rvar : string
        Name of the grid variable of the right-hand side.

    options : dictionary

    """

    verbose = options['verbose']
    mtx = options['mtx']

    phi = grid.get_values(ivar)
    b = grid.get_values(rvar)
    dx, dy = grid.dx, grid.dy
    nx, ny = grid.nx, grid.ny

    sol = linsolve.spsolve(mtx, b[1:-1, 1:-1].flatten())

    residual = np.linalg.norm(mtx * sol - b[1:-1, 1:-1].flatten())

    phi[1:-1,1:-1] = np.reshape(sol,(nx,ny))
    grid.fill_guard_cells(ivar)

    if verbose:
        print('Direct Solver:')
        print('- Final residual: {}'.format(residual))

    return None, residual
