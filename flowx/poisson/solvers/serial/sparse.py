import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse import spdiags, csr_matrix
from scipy.sparse import linalg as sla

def build_serial_sparse(grid, ivar):

    """Solve the Poisson system using a direct solver from the scipy library.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.

   Returns
    -------
    mtx : CSR format matrix

    """

    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    matrix_length = nx*ny

    mtx = sps.lil_matrix((matrix_length, matrix_length), dtype=np.float64)

    counter = 0

    bc_type = grid.bc_type[ivar]

    if bc_type[0] is 'neumann':
        coeff_start = 0.0
        coeff_add = -1.0

    else:
        coeff_start = -8.0
        coeff_add = 1.0

    for i in range(1,nx+1):
        for j in range(1,ny+1):

            coeff = coeff_start/dx**2
    
            if(j > 1):
                mtx[counter,counter-1] = 1.0/(dx**2)
                coeff = coeff + coeff_add/(dx**2)

            if(j < ny):
                mtx[counter,counter+1] = 1.0/(dx**2)
                coeff = coeff + coeff_add/(dx**2)

            if(i > 1):
                mtx[counter,counter-ny] = 1.0/(dx**2)
                coeff = coeff + coeff_add/(dx**2)

            if(i < nx):
                mtx[counter,counter+ny] = 1.0/(dx**2)
                coeff = coeff + coeff_add/(dx**2)

            mtx[counter,counter] = coeff

            counter = counter + 1
        
    mtx = mtx.tocsr()

    lu = sla.splu(mtx.tocsc())

    return lu, mtx
