"""User-defined module for simulation."""

import numpy
from scipy.integrate import quad

def set_initial_velocity(gridc, gridx, gridy, ivar, pres):
    """Set the initial velocity field.

    The x- and y-components of the velocity are set to 1.0 and 0.0,
    respectively.

    Arguments
    ---------
    gridx : flowx.Grid object
        Grid containing x-face data.
    gridy : flowx.Grid object
        Grid containing y-face data.
    ivar : string
        Name of the velocity variable on the grid.

    """

    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)
    p = gridc.get_values(pres)

    u[:, :] = 1.0
    v[:, :] = 0.0
    p[:, :] = 0.0

    return


def get_qin(grid, ivar, bc_type):
    """Compute and return the mass getting in the domain.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    ivar : string
        Name of the velocity variable on the grid.
    bctype : dictionary
        Type of boundary conditions for the variable `ivar`.

    Returns
    -------
    Qin : float
        Mass getting in the domain.

    """
    vel = grid.get_values(ivar)
    dx, dy = grid.dx, grid.dy

    Qin = 0.0

    if grid.type_ is 'x-face':
        if bc_type[0] is not 'outflow':
            Qin += numpy.sum(vel[0, 1:-1]) * dy
        if bc_type[1] is not 'outflow':
            Qin -= numpy.sum(vel[-1, 1:-1]) * dy
    elif grid.type_ is 'y-face':
        if bc_type[2] is not 'outflow':
            Qin += numpy.sum(vel[1:-1, 0]) * dx
        if bc_type[3] is not 'outflow':
            Qin -= numpy.sum(vel[1:-1, -1]) * dx

    return Qin


def get_qout(grid, ivar, bc_type):
    """Compute and return the mass getting out the domain.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    ivar : string
        Name of the velocity variable on the grid.
    bctype : dictionary
        Type of boundary conditions for the variable `ivar`.

    Returns
    -------
    Qout : float
        Mass getting out the domain.

    """
    vel = grid.get_values(ivar)
    dx, dy = grid.dx, grid.dy

    Qout = 0.0

    if grid.type_ is 'x-face':
        if bc_type[0] is 'outflow':
            Qout -= numpy.sum(vel[0, 1:-1]) * dy
        if bc_type[1] is 'outflow':
            Qout += numpy.sum(vel[-1, 1:-1]) * dy
    elif grid.type_ is 'y-face':
        if bc_type[2] is 'outflow':
            Qout -= numpy.sum(vel[1:-1, 0]) * dx
        if bc_type[3] is 'outflow':
            Qout += numpy.sum(vel[1:-1, -1]) * dx

    return Qout


def rescale_velocity(grid, ivar, bc_type, Qin, Qout):
    """Rescale velocity.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    ivar : string
        Name of the velocity variable on the grid.
    bctype : dictionary
        Type of boundary conditions for the variable `ivar`.
    Qin : float
        Mass in.
    Qout : float
        Mass out.

    """
    vel = grid.get_values(ivar)

    Qinout = 1.0
    if Qout > 0.0:
        Qinout = Qin/Qout

    if grid.type_ is 'x-face':
        if bc_type[0] is 'outflow':
            vel[0, 1:-1] *= Qinout
        if bc_type[1] is 'outflow':
            vel[-1, 1:-1] *= Qinout

    if grid.type_ is 'y-face':
        if bc_type[2] is 'outflow':
            vel[1:-1, 0] *= Qinout
        if bc_type[3] is 'outflow':
            vel[1:-1, -1] *= Qinout

    return


def get_convvel(grid, ivar):
    """Get convective outflow velocity.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    ivar : string
        Name of the velocity variable on the grid.

    Returns
    -------
    convvel : float
        Variable containing outflow velocity.

    """
    vel = grid.get_values(ivar)

    convvel = numpy.mean(vel[-1, :])

    return convvel


def update_outflow_bc(grid, ivar, dt, convvel=None):
    """Update the value of the velocity at the right boundary.

    The function uses a linear convective equation in the x-direction
    where the convective velocity is defined as the mean x-velocity
    along the right boundary.

    Parameters
    ----------
    grid : flowx.GridFaceX object
        The grid for the velocity.
    ivar : string
        Name of the variable in the Grid structure.
    dt : float
       Time-step size.
    convvel : float (optional)
        Convective velocity;
        default: None (will compute the convective velocity).

    """
    vel = grid.get_values(ivar)
    dx = grid.dx

    if convvel is None:
        convvel = get_convvel(grid, ivar)

    bc_val = grid.bc_val[ivar]
    bc_val[1] = vel[-1, :] - convvel * dt * (vel[-1, :] - vel[-2, :]) / dx
    grid.update_bc_val({ivar: bc_val})

    return

def tag( grid, D ):
    """Tag the indices of the boundary neighbor points.

    Parameters
    ----------
    grid : flowx.GridFaceX object
        The grid for the velocity.
    D : float
        The diameter of the cylinder
    
    Returns
    -------
    idxs : list
        The indices of the fluid boundary grid pionts to the cylinder.
    idxnot : numpy.array
        The indices of the grid points inside the cylinder boundary.

    """
    X, Y = numpy.meshgrid( grid.x, grid.y )
    X, Y = X.T, Y.T
    r = D / 2
    mask = numpy.ones_like( Y )
    dist = numpy.sqrt( X**2 + Y**2 )
    idxnot = numpy.where( abs( dist ) < r )
    mask[ idxnot[ 0 ], idxnot[ 1 ] ] = 0

    idxYpos = []; idxXpos = []; idxYneg = []; idxXneg = []

    for i in range( 1, mask.shape[ 0 ] - 1 ):
        for j in range( 1, mask.shape[ 1 ] - 1 ):
            if mask[ i, j ] == 0 and mask[ i, j + 1 ] == 1.:
                idxYpos.append( [ i, j + 1 ] )
            if mask[ i, j ] == 0 and mask[ i + 1, j ] == 1.:
                idxXpos.append( [ i + 1, j ] )
            if mask[ i, j ] == 0 and mask[ i, j - 1 ] == 1.:
                idxYneg.append( [ i, j - 1 ] )
            if mask[ i, j ] == 0 and mask[ i - 1, j ] == 1.:
                idxXneg.append( [ i - 1, j ] )
#     idxYpos.insert( -1, [ idxYpos[ -1 ][ 0 ] + 1, idxYpos[ -1 ][ 1 ] ] )
#     idxXpos.insert( 0, [ idxXpos[ 0 ][ 0 ], idxXpos[ 0 ][ 1 ] - 1 ] )
#     idxYneg.insert( 0, [ idxYneg[ 0 ][ 0 ] - 1, idxYneg[ 0 ][ 1 ] ] )
#     idxXneg.insert( -1, [ idxXneg[ -1 ][ 0 ], idxXpos[ -1 ][ 1 ] + 1 ] )
    idxs = [ idxXpos, idxXneg, idxYpos, idxYneg ]
    return idxs, idxnot

def vfrac( grid, idxs, D ):
    """Calculates the volume fraction of the cylinder in boundary cells.

    Parameters
    ----------
    grid : flowx.GridFaceX object
        The grid for the velocity.
    idxs : list
        The indices of the fluid boundary grid pionts to the cylinder.
    D : float
        The diameter of the cylinder
    
    Returns
    -------
    vfrac: numpy.array
        The volume fractions of the cylinder in the boundary grid points for the top and right side of the cylinder.

    """
    X, Y = numpy.meshgrid( grid.x, grid.y )
    X, Y = X.T, Y.T
    dx, dy = grid.dx, grid.dy
    r = D / 2

    def abstrapz( y, d ):
        y = numpy.asanyarray( y )
        ret = ( d * ( y[ 1: ] + y[ :-1 ] ) / 2.0 )
        return ret[ ret > 0 ].sum()

    def funcY( X, Y, idxs, i, n, r ):
        x = numpy.linspace( X[ idxs[ 2 ][ i ][ 0 ], idxs[ 2 ][ i ][ 1 ] ], X[ idxs[ 2 ][ i + 1 ][ 0 ], idxs[ 2 ][ i + 1 ][ 1 ] ], n )
        return numpy.sqrt( abs( r**2 - x**2 ) ) - Y[ idxs[ 2 ][ i ][ 0 ], idxs[ 2 ][ i ][ 1 ] - 1 ]

    def funcX( X, Y, idxs, i, n, r ):
        x = numpy.linspace( Y[ idxs[ 0 ][ i ][ 0 ], idxs[ 0 ][ i ][ 1 ] ], Y[ idxs[ 0 ][ i + 1 ][ 0 ], idxs[ 0 ][ i + 1 ][ 1 ] ], n )
        return numpy.sqrt( abs( r**2 - x**2 ) ) - X[ idxs[ 0 ][ i ][ 0 ] - 1, idxs[ 0 ][ i ][ 1 ] ]

    y_a = []
    x_a = []
    n = 100
    for j in range( len( idxs[ 0 ] ) - 1 ):
        x_a.append( abstrapz( funcX( X, Y, idxs, j, n, r ), dy / n ) )
    for i in range( len( idxs[ 2 ] ) - 1 ):
        y_a.append( abstrapz( funcY( X, Y, idxs, i, n, r ), dx / n ) )


    def vfrac1( a, dx, dy ):
        volfrac = 1 - numpy.array( a ) / ( dx * dy )
        volfrac = numpy.insert( volfrac, 0, 1 )
        volfrac = numpy.append( volfrac, 1 )
        volfrac = ( volfrac[ 1: ] + volfrac[ :-1 ] ) / 2
        return volfrac

    vfracx = vfrac1( x_a, dx, dy )
    vfracy = vfrac1( y_a, dx, dy )
    vfrac = numpy.array( [ vfracx, vfracy ] )
    
    return vfrac

def apply_ibm( grid, idxs, idxnot, vfrac ):
   
    velc = grid.get_values( 'velc' )   
    velc[ idxnot[ 0 ], idxnot[ 1 ] ] = 0


# def apply_ibm( grid, idxs, idxnot, vfrac ):
#     """Applies forcing to the velocities at the fluid boundary grid points.

#     Parameters
#     ----------
#     grid : flowx.GridFaceX object
#         The grid for the velocity.
#     idxs : list
#         The indices of the fluid boundary grid pionts to the cylinder.
#     idxnot : numpy.array
#         The indices of the grid points inside the cylinder boundary.
#     vfrac: numpy.array
#         The volume fractions of the cylinder in the boundary grid points for the top and right side of the cylinder.
 
#     """
#     velc = grid.get_values( 'velc' )   
#     for i in range( len( idxs ) ):
#         for j in range( len( idxs[ i ] ) ):
#             if i < 2:
#                 velc[ idxs[ i ][ j ][ 0 ], idxs[ i ][ 1 ] ] *= vfrac[ 0 ][ j ]
#             else:
#                 velc[ idxs[ i ][ j ][ 0 ], idxs[ i ][ 1 ] ] *= vfrac[ 1 ][ j ]
#     velc[ idxnot[ 0 ], idxnot[ 1 ] ] = 0