"""Interface for Poisson solver module"""

from flowx.poisson.poisson_interface import poisson_interface

class poisson_main(poisson_interface):

    def __init__(self, grid=None, poisson_vars=[None]*2, poisson_info=None):

        """
        Constructor for the Poisson unit

        Arguments
        ---------

        grid : object

             Grid object where the poisson equation needs to be solved

        poisson_vars : list
                List of string for field variables required by poisson unit
               
                poisson_vars[0] --> Phi (numerical solution)
                poisson_vars[1] --> RHS

        poisson_info : Dictionary of keyword arguments

        'poisson_solver' keyword refers to the type of solver to be used
        poisson_info['poisson_solver'] = 'serial_cg' --> default
                                       = 'serial_jacobi'

        poisson_info['maxiter'] = maximum number of iterations --> default 2000
       
        poisson_info['tol']  = minimum tolerance of the residuals --> default 1e-9
 
        poisson_info['verbose'] = bool to displacy poisson stats or not --> default False

        """

        from flowx.poisson.solvers.serial.jacobi import solve_serial_jacobi
        from flowx.poisson.solvers.serial.cg import solve_serial_cg
        from flowx.poisson.solvers.serial.direct import solve_serial_direct
        from flowx.poisson.solvers.serial.sparse import build_serial_sparse
        from flowx.poisson.solvers.serial.superlu import solve_serial_lu
        from flowx.poisson.solvers.serial.stub import solve_serial_stub

        self._grid = grid

        self._ivar, self._rvar = poisson_vars

        self._options = {'poisson_solver' : 'serial_cg', 'maxiter': 2000, 'tol' : 1e-9, 'verbose' : False}

        if poisson_info: 
            for key in poisson_info: self._options[key] = poisson_info[key]

        self._iterative_solvers = {'serial_cg' : solve_serial_cg, 'serial_jacobi': solve_serial_jacobi}
        self._direct_solvers = {'serial_direct' : solve_serial_direct, 'serial_lu' : solve_serial_lu}

        self._solve_poisson = {**self._iterative_solvers, **self._direct_solvers}[self._options['poisson_solver']]

        if grid and poisson_vars: self._options['lu'], self._options['mtx'] = build_serial_sparse(self._grid, self._ivar)

        if not grid or None in poisson_vars:
            self._solve_poisson = solve_serial_stub
            print('Warning: Poisson unit is a stub') 
 
        return

    def solve_poisson(self):
        """ Subroutine to solve poisson equation

        """

        ites, residual = self._solve_poisson(self._grid, self._ivar, self._rvar, self._options)

        return ites, residual
