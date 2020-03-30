"""Direct and iterative solvers of the Au=b for uniform grids."""

import time
import flowx
import numpy as np

def solvers(grid, user_bc, rvar, ivar, method, error, asol,  maxiter=3000, tol=1e-9, verbose=False):
    """	Direct and iterative solvers that compute and set the solution.
 
    Arguments
    ---------
    grid : flowx.Grid object
        Grid containing data.
    rvar : string
        Name of the grid variable of the righ-hand side.
    ivar : string 
        Name of the grid variable of the solution.
    method : string
        Name of the method used to solve the system.
    error : string 
        Name of the grid variable of the error between the analytical and numerical solution.
    asol : string
        Name of the grid variable of the analytical solution. 
    maxiter : integer, optional
        Maximum number of iterations for the iterative methods;
        default: 3000
    tol : float, optional
        Exit-criterion tolerance for the iterative methods;
        default: 1e-9

    Returns
    -------
    ites : integer 
        Number of the iterations computed in case of iterative solver.
    difference : float
        Final difference between the solutions at the current and previous step in case of iterative solver.
    exec_time : float
        Execution time for the solution.
    exec_time_ A : float
        Time to build and set the matrix.
    max_error : float
        Maximum error in every step.
    step_ites : np.ndarray
        1d array of integers.
        Number of iterations in each step.
    verbose : bool, optional
        Set True to display convergence information;
        default: False.

    """

    if method == 'lu' or method == 'direct_inversion' or method == 'conjugate_gradient':
       
        from scipy.sparse.linalg import splu, spsolve,  cg

        #Build/Get matrix
        start_A = time.time()
        A, dur = flowx.poisson.build_matrix(grid,user_bc)
        end_A = time.time()
        exec_time_A=end_A-start_A
        
        #Conversion of the rhs from 2D array to vector b 
        rhs = (grid.get_values(rvar)) * grid.dx**2
        b = rhs[1:-1,1:-1].flatten()
    
        #Solve system
       
        if (method == 'lu'):
      
            start = time.time()
            lu = splu(A)
            u = lu.solve(b) #Direct Solver using LU decomposition
            end = time.time()

        elif (method == 'direct_inversion'):
    
            start = time.time()
            u = spsolve(A, b) #Direct Solver
            end = time.time()

        else:
            start = time.time()
            u = cg(A, b, maxiter = maxiter, tol = tol) #Iterative Solver based on the Conjugate Gradient method.
            u = u[0:-1]
            end = time.time()
       
        #Get residual and set solution
        u = np.reshape(u,(grid.nx, grid.ny))
        u1 = grid.get_values(ivar)
        u1[1:-1,1:-1] = u
        grid.fill_guard_cells(ivar)
       
        exec_time = end-start    
    
        if verbose:
            print('Build and get matrix A:')
            print('-Execution time:{}'.format(end_A-start_A)) 
            if (method == 'lu'):
                print('Direct Solver based on LU decomposition:')
            elif (method == 'direct_inversion'):
                print('Direct Solver:')
            else:
                print('Iterative Solver based on Conjugate Gradient method:')
            print('-Execution time:{}'.format(exec_time))
        return exec_time_A, exec_time
    else:
        
        u = grid.get_values(ivar)
        b = grid.get_values(rvar)

        ites = 0
        difference = tol + 1.0
        max_error = np.zeros(maxiter)
        step_ites = np.zeros(maxiter)


        #Solve1:
        if method == 'jacobi':
            start = time.time()
            while ites < maxiter and difference > tol:
                u_old = np.copy(u)  # previous solution
                u[1:-1, 1:-1] = (((u_old[1:-1, :-2] +
                                     u_old[1:-1, 2:]) *grid.dy**2 +
                                    (u_old[:-2, 1:-1] +
                                     u_old[2:, 1:-1]) *grid.dx**2 -
                                    b[1:-1, 1:-1] * grid.dx**2 * grid.dy**2) /
                                   (2 * (grid.dx**2 + grid.dy**2)))

                grid.fill_guard_cells(ivar)

                difference  = (np.sqrt(np.sum((u - u_old)**2) /
                                    ((grid.nx + 2) * (grid.ny + 2))))
                grid.get_error(error, ivar, asol)
                max_error[ites] = grid.get_l_max_norm(error)
                ites += 1
                step_ites[ites-1] = ites
            end = time.time()
            exec_time = end-start

        else:  #Solution based on Gauss-Seidel method
            start = time.time()
            while ites < maxiter and difference > tol:
                u_old = np.copy(u)  # previous solution
                for i in range(1, grid.ny + 1):
                    for j in range(1, grid.nx + 1):
                        u[j, i] = (((u[j, i - 1] + u[j, i + 1]) * grid.dy**2 +
                                  (u[j - 1, i] + u[j + 1, i]) *grid.dx**2 -
                                  b[j, i] * grid.dx**2 * grid.dy**2) /
                                 (2 * (grid.dx**2 + grid.dy**2)))

                grid.fill_guard_cells(ivar)

                difference  = (np.sqrt(np.sum((u - u_old)**2) /
                                    ((grid.nx + 2) * (grid.ny + 2))))
                grid.get_error(error, ivar, asol)
                max_error[ites] = grid.get_l_max_norm(error)
                ites += 1
                step_ites[ites] = ites

            end = time.time()
            exec_time = end-start

        if verbose:
            if method == 'jacobi':
                print('Iterative Solver based on the Jacobi method:')
            else: 
                 print('Iterative Solver based on the Gauss-Seidel method:')
            if ites == maxiter:
                print('Warning: maximum number of iterations reached!')
            print('- Number of iterations: {}'.format(ites))
            print('- Final difference: {}'.format(difference))
            print('- Execution time: {}'.format(exec_time))
   
        
        return ites, exec_time, max_error, step_ites

