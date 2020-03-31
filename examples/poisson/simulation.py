"""User defined module for simulation."""

import numpy
from flowx.poisson.jacobi import solve_jacobi 
from flowx.poisson.gauss_seidel import solve_gauss_seidel
from flowx.poisson.sdescent import solve_sdescent
from flowx.poisson.cg import solve_cg
from flowx.poisson.cg_scipy import solve_cg_scipy
from flowx.poisson.ds_scipy import solve_ds_scipy

def get_analytical(grid, asol, user_bc):
    """Compute and set the analytical solution.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    asol : string
        Name of the variable on the grid.

    """
    X, Y = numpy.meshgrid(grid.x, grid.y)

    if(user_bc == 'dirichlet'):
        values = numpy.sin(2 * numpy.pi * X) * numpy.sin(2 * numpy.pi * Y)
    else:
        values = numpy.cos(2 * numpy.pi * X) * numpy.cos(2 * numpy.pi * Y)

    grid.set_values(asol, values.transpose())

def get_numerical(grid,ivar,rvar,user_num_type):

    it = 0
    err_list = [0]

    if user_num_type == 'Jacobi':
        it,err_list = solve_jacobi(grid,ivar,rvar,verbose=True)
    elif user_num_type == 'Gauss-Seidel':
        it,err_list = solve_gauss_seidel(grid,ivar,rvar,verbose=True)
    elif user_num_type == 'SDescent':
        solve_sdescent(grid,ivar,rvar,verbose=True)
    elif user_num_type == 'CG':
        solve_cg(grid,ivar,rvar,verbose=True)
    elif user_num_type == 'CG_scipy':
        solve_cg_scipy(grid,ivar,rvar,verbose=True)
    elif user_num_type == 'Direct Solver':
        solve_ds_scipy(grid,ivar,rvar,verbose=True)

    x = list(range(0,it))
    y = err_list[0:it]
    plt_data = zip(x,y)

    return plt_data

def write_to_file(user_num_type,user_bc,plt_data,nx):
    
    if user_bc == 'neumann':
        fname = user_num_type + '_' + 'N' + '_' + str(nx) + '.txt'
    elif user_bc == 'dirichlet':
        fname = user_num_type + '_' + str(nx) + '.txt'
    
    if user_num_type == 'Jacobi' or user_num_type == 'Gauss-Seidel':
        with open(fname, "w") as file:
            for x in plt_data:
                file.write("{0}\t{1}\n".format(*x))    


def get_rhs(grid, rvar, user_bc):
    """Compute and set the right-hand side of the Poisson system.

    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    rvar : string
        Name of the variable on the grid.

    """
    X, Y = numpy.meshgrid(grid.x, grid.y)

    if(user_bc == 'dirichlet'):
        values = (-8 * numpy.pi**2 *
                  numpy.sin(2 * numpy.pi * X) * numpy.sin(2 * numpy.pi * Y))
    else:
        values = (-8 * numpy.pi**2 *
                  numpy.cos(2 * numpy.pi * X) * numpy.cos(2 * numpy.pi * Y))

    grid.set_values(rvar, values.transpose())
