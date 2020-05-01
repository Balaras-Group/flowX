"""Rk3 time advancement routine"""

from .projection import predictor_step1, corrector_step1, divergence_step1
from .projection import predictor_step2, corrector_step2, divergence_step2
from .projection import predictor_step3, corrector_step3, divergence_step3
from .stats import stats
from .projection import predictor_rk3, corrector_rk3, divergence_rk3
from .projection import predictor, corrector, divergence

def advance_rk3(gridc, gridx, gridy, scalars, grid_var_list, predcorr):

    """
    Subroutine for the fractional step rk3 explicit time advancement of Navier Stokes equations
 
    Arguments
    ---------
    gridc : object
          Grid object for cell centered variables

    gridx : object
          Grid object for x-face variables

    gridy : object
          Grid object for y-face variables

    scalars: object
           Scalars object to access time-step and Reynold number

    grid_var_list : list
           List containing variable names for velocity, RHS term from the previous time-step, divergence and pressure

    predcorr : string
           Flag for the fractional step method equations - 'predictor', 'divergence', 'corrector'

    """

    velc = grid_var_list[0]
    hvar = grid_var_list[1]
    divv = grid_var_list[2]
    pres = grid_var_list[3]

        
    # t to t+t/3
    if(predcorr == 'predictor_step1'):
        predictor_step1(gridx, gridy, velc, hvar, scalars.variable['Re'],scalars.variable['dt'])
    if(predcorr == 'divergence_step1'):
        divergence_step1(gridc, gridx, gridy, velc, divv, ifac = scalars.variable['dt'])
    if(predcorr == 'corrector_step1'):
        corrector_step1(gridc, gridx, gridy, velc, pres, scalars.variable['dt'])

        
    # t+t/3 to t+3t/4
    if(predcorr == 'predictor_step2'):
        predictor_step2(gridx, gridy, velc, hvar, scalars.variable['Re'],scalars.variable['dt'])
    if(predcorr == 'divergence_step2'):
        divergence_step2(gridc, gridx, gridy, velc, divv, ifac = scalars.variable['dt'])
    if(predcorr == 'corrector_step2'):
        corrector_step2(gridc, gridx, gridy, velc, pres, scalars.variable['dt'])      
 

    # t+3t/4 to t+1
    if(predcorr == 'predictor_step3'):
        predictor_step3(gridx, gridy, velc, hvar, scalars.variable['Re'],scalars.variable['dt'])
    if(predcorr == 'divergence_step3'):
        divergence_step3(gridc, gridx, gridy, velc, divv, ifac = scalars.variable['dt'])
    if(predcorr == 'corrector_step3'):
        corrector_step3(gridc, gridx, gridy, velc, pres, scalars.variable['dt']) 

    
    scalars.stats.update(stats(gridc, gridx, gridy, velc, pres, divv))
    """

    if(predcorr == 'predictor'):
        # Calculate predicted velocity: u* = dt*H(u^n)
        predictor_rk3(gridx, gridy, velc, hvar, scalars.variable['Re'], scalars.variable['dt'], cnst1, cnst2, cnst3)

    if(predcorr == 'divergence'):    
        # Calculate RHS for the pressure Poission solver div(u)/dt
        divergence_rk3(gridc, gridx, gridy, velc, divv,  scalars.variable['dt'], cnst1, cnst2, cnst3)

    elif(predcorr == 'corrector'):

        # Calculate corrected velocity u^n+1 = u* - dt * grad(P) 
        corrector_rk3(gridc, gridx, gridy, velc, pres, scalars.variable['dt'], cnst1, cnst2, cnst3)
    
        # Calculate divergence of the corrected velocity to display stats
        divergence(gridc, gridx, gridy, velc, divv)
    
        # Calculate stats
        scalars.stats.update(stats(gridc, gridx, gridy, velc, pres, divv))

    """

