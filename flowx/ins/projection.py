"""Routine to compute the predictor, corrector, and divergence."""

import numpy

from .operators import diffusion, convective_facex, convective_facey

def predictor(gridx, gridy, ivar, hvar, Re, ifac):
    """Velocity prediction step in x and y direction.

    Arguments
    ---------
    gridx : grid object (x-direction)
        Grid containing data in x-direction.
    gridy : grid object (y-direction)
        Grid containing data in y-direction.
    ivar : string
        Name of the grid variable of the velocity solution.
    hvar : string
        Name of the grid variable to store convective + diffusion terms.
    Re : float
        Reynolds number.
    ifac : float
        Time-step size.

    """
    hx = gridx.get_values(hvar)
    hy = gridy.get_values(hvar)

    hx[1:-1, 1:-1] = (convective_facex(gridx, gridy, ivar) +
                      diffusion(gridx, ivar, 1 / Re))
    hy[1:-1, 1:-1] = (convective_facey(gridx, gridy, ivar) +
                      diffusion(gridy, ivar, 1 / Re))

    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)

    u[1:-1, 1:-1] = u[1:-1, 1:-1] + ifac * hx[1:-1, 1:-1]
    v[1:-1, 1:-1] = v[1:-1, 1:-1] + ifac * hy[1:-1, 1:-1]

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

def predictor_rk3(gridx, gridy, ivar, hvar, Re, ifac, i=0):
    """Velocity prediction step in x and y direction.

    Arguments
    ---------
    gridx : grid object (x-direction)
        Grid containing data in x-direction.
    gridy : grid object (y-direction)
        Grid containing data in y-direction.
    ivar : string
        Name of the grid variable of the velocity solution.
    hvar : string
        Name of the grid variable to store convective + diffusion terms.
    Re : float
        Reynolds number.
    ifac : float
        Time-step size.
    i : integer
        Number of stage.
    """

    #coefficient for h
    c1 = numpy.array([0, -5/9, -153/128])

    hx = gridx.get_values(hvar)
    hy = gridy.get_values(hvar)

    hx[1:-1, 1:-1] = c1[i] * hx[1:-1, 1:-1] + (convective_facex(gridx, gridy, ivar) +
                      diffusion(gridx, ivar, 1 / Re))
    hy[1:-1, 1:-1] = c1[i] * hy[1:-1, 1:-1] + (convective_facey(gridx, gridy, ivar) +
                      diffusion(gridy, ivar, 1 / Re))

    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)

    #coefficient for u, v
    c2 = numpy.array([1/3, 15/16, 8/15])

    u[1:-1, 1:-1] = u[1:-1, 1:-1] + c2[i] * ifac * hx[1:-1, 1:-1]
    v[1:-1, 1:-1] = v[1:-1, 1:-1] + c2[i] * ifac * hy[1:-1, 1:-1]

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

def divergence(gridc, gridx, gridy, ivar, dvar, ifac=1.0):
    """Compute the divergence of the variable tagged "ivar".

    Arguments
    ---------
    gridc : grid object
        Grid containing data in on cell centers.
    gridx : grid object (x-direction)
        Grid containing data in x-direction.
    gridy : grid object (y-direction)
        Grid containing data in y-direction.
    ivar : string
        Name of the face-centered grid variable for velocity.
    dvar : string
        Name of the cell-centered grid variable to store divergence.
    ifac : float (optional)
        Multiplying factor for time-step; default: 1.0.

    """
    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)

    div = gridc.get_values(dvar)

    dx, dy = gridc.dx, gridc.dy

    div[1:-1, 1:-1] = ((u[1:, 1:-1] - u[:-1, 1:-1]) / dx +
                       (v[1:-1, 1:] - v[1:-1, :-1]) / dy) / ifac

    gridc.fill_guard_cells(dvar)

    return

def divergence_rk3(gridc, gridx, gridy, ivar, dvar, ifac=1.0, i=0):
    """Compute the divergence of the variable tagged "ivar".

    Arguments
    ---------
    gridc : grid object
        Grid containing data in on cell centers.
    gridx : grid object (x-direction)
        Grid containing data in x-direction.
    gridy : grid object (y-direction)
        Grid containing data in y-direction.
    ivar : string
        Name of the face-centered grid variable for velocity.
    dvar : string
        Name of the cell-centered grid variable to store divergence.
    ifac : float (optional)
        Multiplying factor for time-step; default: 1.0.
    i : integer
        Number of stage.

    """
    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)

    div = gridc.get_values(dvar)

    dx, dy = gridc.dx, gridc.dy

    #coefficient for u, v
    c = numpy.array([3, 12/5, 4])

    div[1:-1, 1:-1] = ((u[1:, 1:-1] - u[:-1, 1:-1]) / dx +
                       (v[1:-1, 1:] - v[1:-1, :-1]) / dy) / (ifac / c[i])

    gridc.fill_guard_cells(dvar)

    return

def corrector(gridc, gridx, gridy, ivar, pvar, ifac):
    """Velocity correction in x and y direction.

    Arguments
    ---------
    gridc : grid object (cell center)
        Grid contaning data in cell center.
    gridx : grid object (x-direction)
        Grid containing data in x-direction.
    gridy : grid object  (y-direction)
        Grid containing data in y-direction.
    ivar : string
        Name of the grid variable of the velocity solution.
    pvar : string
        Name of the grid variable of the pressure solution.
    ifac : float
        Time-step size.

    """
    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)
    p = gridc.get_values(pvar)

    dx, dy = gridc.dx, gridc.dy

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - ifac * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dx
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - ifac * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dy

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

def corrector_rk3(gridc, gridx, gridy, ivar, pvar, ifac, i=0):
    """Velocity correction in x and y direction.

    Arguments
    ---------
    gridc : grid object (cell center)
        Grid contaning data in cell center.
    gridx : grid object (x-direction)
        Grid containing data in x-direction.
    gridy : grid object  (y-direction)
        Grid containing data in y-direction.
    ivar : string
        Name of the grid variable of the velocity solution.
    pvar : string
        Name of the grid variable of the pressure solution.
    ifac : float
        Time-step size.
    i : integer
        Number of stage.

    """
    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)
    p = gridc.get_values(pvar)

    dx, dy = gridc.dx, gridc.dy

    #coefficient for u, v
    c = numpy.array([3, 12/5, 4])

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - (ifac / c[i]) * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dx
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - (ifac / c[i]) * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dy

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return
