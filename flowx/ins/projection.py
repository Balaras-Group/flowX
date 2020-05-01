"""Routine to compute the predictor, corrector, and divergence for Euler."""

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


###############################################################################
"""Routine to compute the RK3 predictor, corrector, and divergence segments."""
###############################################################################

def predictor_step1(gridx, gridy, ivar, hvar, Re, ifac):
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

    u[1:-1, 1:-1] = u[1:-1, 1:-1] + (ifac/3) * hx[1:-1, 1:-1]
    v[1:-1, 1:-1] = v[1:-1, 1:-1] + (ifac/3) * hy[1:-1, 1:-1]

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

def divergence_step1(gridc, gridx, gridy, ivar, dvar, ifac=1.0):
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
                       (v[1:-1, 1:] - v[1:-1, :-1]) / dy) / (ifac/3.0)

    gridc.fill_guard_cells(dvar)

    return

def corrector_step1(gridc, gridx, gridy, ivar, pvar, ifac):
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

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - (ifac/3) * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dx
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - (ifac/3) * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dy

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

""" Going from t+t/3 to t+3t/4."""

def predictor_step2(gridx, gridy, ivar, hvar, Re, ifac):
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

    hx[1:-1, 1:-1] = ((-5/9)*hx[1:-1, 1:-1] + convective_facex(gridx, gridy, ivar) +
                      diffusion(gridx, ivar, 1 / Re))
    hy[1:-1, 1:-1] = ((-5/9)*hy[1:-1, 1:-1] + convective_facey(gridx, gridy, ivar) +
                      diffusion(gridy, ivar, 1 / Re))

    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)

    u[1:-1, 1:-1] = u[1:-1, 1:-1] + (15/16)* ifac * hx[1:-1, 1:-1]
    v[1:-1, 1:-1] = v[1:-1, 1:-1] + (15/16)* ifac * hy[1:-1, 1:-1]

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

def divergence_step2(gridc, gridx, gridy, ivar, dvar, ifac=1.0):
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
                       (v[1:-1, 1:] - v[1:-1, :-1]) / dy) / (ifac * 5/12)

    gridc.fill_guard_cells(dvar)

    return

def corrector_step2(gridc, gridx, gridy, ivar, pvar, ifac):
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

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - (5/12)* ifac * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dx
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - (5/12)* ifac * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dy

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return
""" Going from t+3t/4 to t+1."""

def predictor_step3(gridx, gridy, ivar, hvar, Re, ifac):
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

    hx[1:-1, 1:-1] = ((-153/128)*hx[1:-1, 1:-1] + convective_facex(gridx, gridy, ivar) +
                      diffusion(gridx, ivar, 1 / Re))
    hy[1:-1, 1:-1] = ((-153/128)*hy[1:-1, 1:-1] + convective_facey(gridx, gridy, ivar) +
                      diffusion(gridy, ivar, 1 / Re))

    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)

    u[1:-1, 1:-1] = u[1:-1, 1:-1] + (8/15) * ifac * hx[1:-1, 1:-1]
    v[1:-1, 1:-1] = v[1:-1, 1:-1] + (8/15) * ifac * hy[1:-1, 1:-1]

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

def divergence_step3(gridc, gridx, gridy, ivar, dvar, ifac=1.0):
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
                       (v[1:-1, 1:] - v[1:-1, :-1]) / dy) / (ifac/4)

    gridc.fill_guard_cells(dvar)

    return

def corrector_step3(gridc, gridx, gridy, ivar, pvar, ifac):
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

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - 0.25 * ifac * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dx
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - 0.25 * ifac * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dy

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

##########################################################################################
"""Different Routine to compute the RK3 predictor, corrector, and divergence segments."""
##########################################################################################

def predictor_rk3(gridx, gridy, ivar, hvar, Re, ifac, cnst1, cnst2, cnst3):
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
                      diffusion(gridx, ivar, 1 / Re) + cnst1*hx[1:-1, 1:-1])
    hy[1:-1, 1:-1] = (convective_facey(gridx, gridy, ivar) +
                      diffusion(gridy, ivar, 1 / Re) + cnst1*hy[1:-1, 1:-1])

    u = gridx.get_values(ivar)
    v = gridy.get_values(ivar)

    u[1:-1, 1:-1] = u[1:-1, 1:-1] + ifac * cnst2 * hx[1:-1, 1:-1]
    v[1:-1, 1:-1] = v[1:-1, 1:-1] + ifac * cnst2 * hy[1:-1, 1:-1]

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return

def divergence_rk3(gridc, gridx, gridy, ivar, dvar, ifac, cnst1, cnst2, cnst3):
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
                       (v[1:-1, 1:] - v[1:-1, :-1]) / dy) / (ifac*cnst3)

    gridc.fill_guard_cells(dvar)

    return

def corrector_rk3(gridc, gridx, gridy, ivar, pvar, ifac, cnst1, cnst2, cnst3):
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

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - ifac * cnst3 * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dx
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - ifac * cnst3 * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dy

    gridx.fill_guard_cells(ivar)
    gridy.fill_guard_cells(ivar)

    return
