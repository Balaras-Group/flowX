import numpy


def solve_conjugate_gradient( grid, ivar, rvar, maxiter = 3000, tol = 1e-9, verbose = False ):
    
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
    
    phi = grid.get_values( ivar )
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    b = grid.get_values( rvar )
    
    def Laplace( phi ):
        return ( -4.0 * phi[ 1:-1, 1:-1 ] + phi[ 1:-1, :-2 ] + phi[ 1:-1, 2: ] + phi[ :-2, 1:-1 ] + phi[ 2:, 1:-1 ] ) / dx**2
    
    r = numpy.zeros_like( phi )
    Ad = numpy.zeros_like( phi )
    residual = tol + 1
    ites = 0
    r[ 1:-1, 1:-1 ] = b[ 1:-1, 1:-1 ] - Laplace( phi )
    d = r.copy()
    
    while residual > tol and ites < maxiter:
        phi_old = phi.copy()
        rk = r.copy()
        Ad[ 1:-1, 1:-1 ] = Laplace( d )
        alpha = numpy.sum( r * r ) / numpy.sum( d * Ad )
        phi = phi_old + alpha * d
        r = rk - alpha * Ad
        beta = numpy.sum( r * r ) / numpy.sum( rk * rk )
        d = r + beta * d
        residual = ( numpy.sqrt( numpy.sum( ( phi - phi_old )**2 ) / ( ( grid.nx + 2 ) * ( grid.ny + 2 ) ) ) )
        
        grid.fill_guard_cells( ivar )
        ites += 1
    
    if verbose:
        print('Conjugate Gradient method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        print('- Final residual: {}'.format(residual))
        
    return ites, residual