
"""Routine to build the matrix A of the sustem Au=b for the solution of the Poisson equation."""

import numpy as np
from scipy import sparse
import time


def build_matrix( grid, user_bc, verbose=False):
    """Build the matrix to solve Poisson equation.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.

    Returns
    -------
    A : matrix
        Matrix A with dimentions (nx*ny,nx*ny).
    end-start : float
        Time to build the matrix
    verbose : bool, optional
        Set True to display matrix information;
        default: False.

    """
    start = time.time()
    nx = grid.nx
    ny = grid.ny
    
    # Definition of the diagonals of the matrix
    diag_1 = np.ones(nx*ny) # nx-th lower/upper diagonal
 
    diag_2 = np.ones(nx*ny) # first lower diagonal
    diag_2[nx-1:((nx*ny)-1):nx] = 0 

    diag_3 = np.ones(nx*ny) # maine diagonal
    if(user_bc == 'dirichlet'):
        c1 = -6
        c2 = -5
    else: 
        c1 = -2
        c2 = -3
    diag_3[0:nx] = np.concatenate(( [c1], c2* np.ones(nx-2), [c1] ))
    diag_3[((nx*ny)-nx):(nx*ny)] = np.concatenate(( [c1], c2* np.ones(nx-2), [c1] ))
    i = nx
    while i<(nx*ny)-nx:
        diag_3[i:(nx+i)] = np.concatenate(( [c2], (-4)* np.ones(nx-2) ,[c2] ))
        i = i+nx

    diag_4 = np.ones(nx*ny) #first upper diagonal
    diag_4[nx:((nx*ny)-1):nx] = 0

    #Build matrix
    data =  np.array([ diag_1, diag_2, diag_3, diag_4, diag_1])
    diags = np.array([ -nx, -1, 0, 1, nx])
    A = sparse.spdiags( data, diags, nx*ny, nx*ny ).toarray()
    A = A.astype(np.float64)
    A = sparse.csc_matrix(A)

    end = time.time()

    if verbose:
        print('Build of Compressed Sparse Column  matrix A:')
        print('- Execution time: {}'.format(end-start))

    return A, (end-start) 
