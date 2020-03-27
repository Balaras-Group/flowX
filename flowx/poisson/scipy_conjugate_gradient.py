"""Routine to solve the Poisson system with Conjugate Gradient."""
import numpy


def solve_CG(grid, ivar, rvar, maxiter=3000, tol=1e-9, verbose=False):
    """Solve the Poisson system using a Conjugate Gradient method.

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
    phi = grid.get_values(ivar)
    b = grid.get_values(rvar)
    dx, dy = grid.dx, grid.dy

    ites = 0
    r = numpy.zeros()
    p = r 
    while ites < maxiter and r > tol:

   
    
        for j in range(1, grid.ny+1):
            for i in range (1, grid.nx+1):
                q[i,j] = (p[i+1,j] - 2.0*p[i,j] + p[i-1,j])/(dx**2) + (p[i,j+1] - 2.0*p[i,j] + p[i,j-1])/(dy**2)

        aa, bb = 0.0, 0.0
        for j in range(1, grid.ny+1):
            for i in range (1, grid.nx+1):
                aa = aa + r[i,j]*r[i,j] 
                bb = bb + q[i,j]*q[i,j]

        alpha = aa/bb 
        for j in range(1, grid.ny+1):
            for i in range (1, grid.nx+1):
                phi[i,j] = phi[i,j] + alpha*p[i,j]

        bb, aa = 0.0, 0.0
        for j in range(1, grid.ny+1):
            for i in range (1, grid.nx+1):
                r[i,j] = r[i,j] - alpha*q[i,j]
                aa = aa + r[i,j]*r[i,j]

        alpha = aa/bb
        for j in range(1, grid.ny+1):
            for i in range (1, grid.nx+1):
                p[i,j] = r[i,j] + alpha*p[i,j]

        for j in range(1, grid.ny+1):
            for i in range (1, grid.nx+1):
                d2udx2 =(phi[i+1,j] - 2.0*phi[i,j] + phi[i-1,j])/dx**2
                d2udy2 =(phi[i+1,j] - 2.0*phi[i,j] + phi[i-1,j])/dy**2
                r[i,j] = rvar - d2dx2 -d2dy2

   

        grid.fill_guard_cells(ivar)

        residual = (numpy.sqrt(numpy.sum((phi - phi_old)**2) /
                    ((grid.nx + 2) * (grid.ny + 2))))
        ites += 1

    if verbose:
        print('Conjugate Gradient method:')
        if ites == maxiter:
            print('Warning: maximum number of iterations reached!')
        print('- Number of iterations: {}'.format(ites))
        print('- Final residual: {}'.format(residual))

    return ites, residual
