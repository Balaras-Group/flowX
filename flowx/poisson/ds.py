
import numpy
from scipy.sparse.linalg import dsolve
import scipy.sparse as sps

def solve_direct( grid, ivar, rvar ):
    
    """Solve the Poisson system using a direct solver.

    Arguments
    ---------
    grid : Grid object
        Grid containing data.
    ivar : string
        Name of the grid variable of the numerical solution.
    rvar : string
        Name of the grid variable of the right-hand side.

    """
    
    phi = grid.get_values( 'ivar' )
    dx, dy = grid.dx, grid.dy
    N = phi.shape[ 0 ] * phi.shape[ 1 ]
    D = numpy.diag( -4 * numpy.ones( N ) )
    L = numpy.diag( numpy.ones( N - 1 ), k = -1 )
    U = numpy.diag( numpy.ones( N - 1 ), k = 1 )
    UU = numpy.diag( numpy.ones( N - phi.shape[ 0 ] ), k = phi.shape[ 0 ] )
    LL = numpy.diag( numpy.ones( N - phi.shape[ 0 ] ), k = -phi.shape[ 0 ] )
    A = ( LL + L + D + U + UU ) * ( 1 / dx**2 )

    b = grid.get_values( 'rvar' ).flatten()
    a = sps.lil_matrix( A.shape, dtype = numpy.float64 )
    a[ :, : ] = A.copy()
    a = a.tocsr()
    x = dsolve.spsolve( a, b )
    phi[ :, : ] = numpy.reshape( x.transpose(), phi.shape )