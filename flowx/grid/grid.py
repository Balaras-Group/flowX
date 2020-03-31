"""Module with implementation of the Grid classes."""

import numpy


class GridBase(object):
    """Base class for the Grid."""

    type_ = 'base'

    def __init__(self, var_names, nx, ny, xmin, xmax, ymin, ymax,
                 user_num_solver=None, user_bc_type=None, user_bc_val=None):
        """Initialize the Grid object and allocate the data.

        Parameters
        ----------
        var_names : list of strings
            List of names for the variables to create.
        nx : integer
            Number of cells in the x-direction.
        ny : integer
            Number of cells in the y-direction.
        xmin : float
            Domain limit at the left side.
        xmax : float
            Domain limit at the right side.
        ymin : float
            Domain limit at the bottom side.
        ymax : float
            Domain limit at the top side.
        user_bc_type : dictionary of (string, list) items
            User-defined boundary types to overwrite default ones.
        user_bc_val : dictionary of (string, list) items
            User-defined boundary values to overwrite default ones.

        """
        # Domain information
        self.nx, self.ny = nx, ny  # number of cells in each direction
        self.xmin, self.xmax = xmin, xmax  # domain limits (x direction)
        self.ymin, self.ymax = ymin, ymax  # domain limits (y direction)
        self.dx = abs(self.xmax - self.xmin) / nx  # cell-width (x direction)
        self.dy = abs(self.ymax - self.ymin) / ny  # cell-width (y direction)
        self.set_gridline_coordinates()

        # Initialize the data
        self.num = len(var_names)
        self.vars = dict(zip(var_names, range(self.num)))
        self.data = None
        self.initialize_data()

        # Boundary condition information
        self.set_default_bc()
        if user_bc_type is not None and user_bc_val is not None:
            self.set_user_bc(user_bc_type, user_bc_val)

        self.set_default_numsolver()
        if user_num_solver is not None:
            self.set_user_num_solver(user_num_solver)

        self.fill_guard_cells(var_names)

    def __repr__(self):
        """Return a representation of the object."""
        return ("Grid:\n" +
                " - type: {}\n".format(type(self)) +
                " - size: {} x {}\n".format(self.nx, self.ny) +
                " - domain: [{}, {}] x [{}, {}]\n".format(self.xmin,
                                                          self.xmax,
                                                          self.ymin,
                                                          self.ymax) +
                " - number of variables: {}\n".format(self.num))

    def set_gridline_coordinates(self):
        """Set the gridline coordinates.

        Method implemented in the child class.

        """
        raise NotImplementedError()

    def initialize_data(self):
        """Initialize the data with zeros.

        Method implemented in child classes.

        """
        raise NotImplementedError()

    def set_default_numsolver(self):
        """
        Set default numerical solver to Gauss-Seidel
        """
        var_name = list(self.vars.keys())
        num_var = var_name[1]
        default_num_solver = 'Gauss-Seidel'
        self.num_solver = {num_var:default_num_solver}

    def set_default_bc(self):
        """Set default boundary conditions (homogeneous Neumann)."""
        var_names = list(self.vars.keys())
        num = len(var_names)
        default_bc_type = 4 * ['neumann']
        self.bc_type = dict(zip(var_names, num * [default_bc_type]))
        default_bc_val = 4 * [0.0]
        self.bc_val = dict(zip(var_names, num * [default_bc_val]))

    def set_user_bc(self, user_bc_type, user_bc_val):
        """Overwrite default boundary conditions with user-provided ones.

        Parameters
        ----------
        user_bc_type : dictionary of (string, list) items
            User-defined boundary types.
        user_bc_val : dictionary of (string, list) items
            User-defined boundary values.

        """
        # Overwrite default boundary types
        self.bc_type = {**self.bc_type, **user_bc_type}
        # Overwrite default boundary values
        self.bc_val = {**self.bc_val, **user_bc_val}

    def set_user_num_solver(self,user_num_solver):

        """
        Overwrite default numerical solver with user-provided solver

        Parameters
        ----------
        user_num_solver: Dictionary of (string,string) items
        		 User-defined numerical solvers
        """
        # Replace num_solver with user_num_solver
        # ** is used to unpack and replace two dictionaries
        self.num_solver = {**self.num_solver, **user_num_solver}

    def update_bc_val(self, user_bc_val):
        """Overwrite boundary condition values with user-provided ones.

        Parameters
        ----------
        user_bc_val : dictionary of (string, list) items
            User-defined boundary values.

        """
        self.bc_val = {**self.bc_val, **user_bc_val}

    def get_variable_indices(self, *var_names):
        """Get the grid index of given variable names.

        Parameters
        ----------
        var_names : tuple of strings
            The name of the variable(s).

        Returns
        -------
        indices : integer or list of integers
            Index of the grid variables.

        """
        indices = [self.vars[name] for name in var_names]
        return indices[0] if len(indices) == 1 else indices

    def set_values(self, var_name, values):
        """Set the values of a variable.

        Parameters
        ----------
        var_name : string
            Name of the variable.
        values : numpy.ndarray
            2D array with the values to set.

        """
        idx = self.vars[var_name]
        self.data[:, :, idx] = values

    def get_values(self, var_name):
        """Get the data of a variable (as a copy).

        Parameters
        ----------
        var_name : string
            Name of the variable.

        Returns
        -------
        data : numpy.ndarray
            Variable's data as a 2D array of floats.

        """
        idx = self.vars[var_name]
        data = self.data[:, :, idx]
        return data

    def set_value(self, var_name, i, j, value):
        """Set a value using (i, j) indexation.

        Parameters
        ----------
        var_name : string
            Name of the variable.
        i : integer
            Index in the x-direction.
        j : integer
            Index in the y-direction.
        value : float
            Value to set.

        """
        idx = self.vars[var_name]
        self.data[i, j, idx] = value

    def get_value(self, var_name, i, j):
        """Get a value using (i, j) indexation.

        Parameters
        ----------
        var_name : string
            Name of the variable.
        i : integer
            Index in the x-direction.
        j : integer
            Index in the y-direction.

        Returns
        -------
        value : float
            Value of the variable at index (i, j).

        """
        idx = self.vars[var_name]
        value = self.data[i, j, idx]
        return value

    def get_error(self, eror, ivar, asol):
        """Compute the error between the numerical and analytical solutions.

        Error is defined as the absolute difference between the two solutions.

        Arguments
        ---------
        eror : string
            Name of the grid variable of the error.
        ivar : string
            Name of the grid variable of the numerical solution.
        asol : string
            Name of the grid variable of the analytical solution.

        """
        i_eror, i_ivar, i_asol = self.get_variable_indices(eror, ivar, asol)
        self.data[:, :, i_eror] = numpy.abs(self.data[:, :, i_ivar] -
                                            self.data[:, :, i_asol])

    def get_l2_norm(self, eror):
        """Compute the L2 norm for a given variable.

        Arguments
        ---------
        eror : string
            Name of the grid variable for which norm is desired

        Returns
        -------
        l2_norm : float
            The L2-norm.

        """
        i_eror = self.get_variable_indices(eror)

        l2_norm = (numpy.sqrt(numpy.sum(self.data[:, :, i_eror]**2)) /
                   ((self.nx + 2) * (self.ny + 2)))

        return l2_norm

    def fill_guard_cells(self, var_names):
        """Fill value at guard cells for given variable names.

        Parameters
        ----------
        var_names : string or list of strings
            Name of variables to update.

        """
        # Convert single string to a list
        if type(var_names) is str:
            var_names = [var_names]
        # Set locations and delta value (either dx or dy)
        locs = ['left', 'right', 'bottom', 'top']
        deltas = [self.dx, self.dx, self.dy, self.dy]
        # Fill guard cells for each variable
        for name in var_names:
            bc_types = self.bc_type[name]
            bc_vals = self.bc_val[name]
            # Fill guard cells for each boundary
            for loc, delta, bc_type, bc_val in zip(locs, deltas,
                                                   bc_types, bc_vals):
                if bc_type == 'neumann':
                    self.fill_guard_cells_neumann(name, loc, bc_val, delta)
                elif bc_type == 'dirichlet':
                    self.fill_guard_cells_dirichlet(name, loc, bc_val)
                elif bc_type == 'outflow':
                    self.fill_guard_cells_dirichlet(name, loc, bc_val)
                else:
                    raise ValueError('Boundary type "{}" not implemented'
                                     .format(bc_type))

    def fill_guard_cells_dirichlet(self, var_name, loc, bc_val):
        """Fill guard cells using a Dirichlet condition.

        Method implemented in child classes.

        Parameters
        ----------
        var_name : string
            Name of the variable to update.
        loc : string
            Boundary location;
            choices: ['left', 'right', 'bottom', 'top'].
        bc_val : float
            Neumann boundary value.

        """
        raise NotImplementedError()

    def fill_guard_cells_neumann(self, var_name, loc, bc_val, delta):
        """Fill guard cells using a Neumann condition.

        Parameters
        ----------
        var_name : string
            Name of the variable to update.
        loc : string
            Boundary location;
            choices: ['left', 'right', 'bottom', 'top'].
        bc_val : float
            Neumann boundary value.
        delta : float
            Grid-cell width.

        """
        var = self.get_values(var_name)
        if loc == 'left':
            var[0, :] = bc_val * delta + var[1, :]
        elif loc == 'right':
            var[-1, :] = bc_val * delta + var[-2, :]
        elif loc == 'bottom':
            var[:, 0] = bc_val * delta + var[:, 1]
        elif loc == 'top':
            var[:, -1] = bc_val * delta + var[:, -2]
        else:
            raise ValueError('Unknown boundary location "{}"'.format(loc))


class GridCellCentered(GridBase):
    """Class for a cell-centered grid."""

    type_ = 'cell-centered'

    def __init__(self, *args, **kwargs):
        """Call the constructor of the base class."""
        super(GridCellCentered, self).__init__(*args, **kwargs)

    @classmethod
    def check_gridtype(cls, gridtype):
        """Check if grid type if 'cell-centered'."""
        return gridtype == 'cell-centered'

    def set_gridline_coordinates(self):
        """Set the gridline coordinates."""
        self.x = numpy.linspace(self.xmin - self.dx / 2,
                                self.xmax + self.dx / 2,
                                num=self.nx + 2)
        self.y = numpy.linspace(self.ymin - self.dy / 2,
                                self.ymax + self.dy / 2,
                                num=self.ny + 2)

    def initialize_data(self):
        """Initialize the data with zeros."""
        self.data = numpy.zeros((self.nx + 2, self.ny + 2, self.num))

    def fill_guard_cells_dirichlet(self, var_name, loc, bc_val):
        """Fill guard cells using a Dirichlet condition.

        Parameters
        ----------
        var_name : string
            Name of the variable to update.
        loc : string
            Boundary location;
            choices: ['left', 'right', 'bottom', 'top'].
        bc_val : float
            Neumann boundary value.

        """
        var = self.get_values(var_name)
        if loc == 'left':
            var[0, :] = 2 * bc_val - var[1, :]
        elif loc == 'right':
            var[-1, :] = 2 * bc_val - var[-2, :]
        elif loc == 'bottom':
            var[:, 0] = 2 * bc_val - var[:, 1]
        elif loc == 'top':
            var[:, -1] = 2 * bc_val - var[:, -2]
        else:
            raise ValueError('Unknown boundary location "{}"'.format(loc))


class GridFaceX(GridBase):
    """Class for a x-face centered grid."""

    type_ = 'x-face'

    def __init__(self, *args, **kwargs):
        """Call the constructor of the base class."""
        super(GridFaceX, self).__init__(*args, **kwargs)

    @classmethod
    def check_gridtype(cls, gridtype):
        """Check if grid type if 'x-face'."""
        return gridtype == 'x-face'

    def set_gridline_coordinates(self):
        """Set the gridline coordinates."""
        self.x = numpy.linspace(self.xmin, self.xmax, num=self.nx + 1)
        self.y = numpy.linspace(self.ymin - self.dy / 2,
                                self.ymax + self.dy / 2,
                                num=self.ny + 2)

    def initialize_data(self):
        """Initialize the data with zeros."""
        self.data = numpy.zeros((self.nx + 1, self.ny + 2, self.num))

    def fill_guard_cells_dirichlet(self, var_name, loc, bc_val):
        """Fill guard cells using a Dirichlet condition.

        Parameters
        ----------
        var_name : string
            Name of the variable to update.
        loc : string
            Boundary location;
            choices: ['left', 'right', 'bottom', 'top'].
        bc_val : float
            Neumann boundary value.

        """
        var = self.get_values(var_name)
        if loc == 'left':
            var[0, :] = bc_val
        elif loc == 'right':
            var[-1, :] = bc_val
        elif loc == 'bottom':
            var[:, 0] = 2 * bc_val - var[:, 1]
        elif loc == 'top':
            var[:, -1] = 2 * bc_val - var[:, -2]
        else:
            raise ValueError('Unknown boundary location "{}"'.format(loc))


class GridFaceY(GridBase):
    """Class for a y-face centered grid."""

    type_ = 'y-face'

    def __init__(self, *args, **kwargs):
        """Call the constructor of the base class."""
        super(GridFaceY, self).__init__(*args, **kwargs)

    @classmethod
    def check_gridtype(cls, gridtype):
        """Check if grid type if 'y-face'."""
        return gridtype == 'y-face'

    def set_gridline_coordinates(self):
        """Set the gridline coordinates."""
        self.x = numpy.linspace(self.xmin - self.dx / 2,
                                self.xmax + self.dx / 2,
                                num=self.nx + 2)
        self.y = numpy.linspace(self.ymin, self.ymax, num=self.ny + 1)

    def initialize_data(self):
        """Initialize the data with zeros."""
        self.data = numpy.zeros((self.nx + 2, self.ny + 1, self.num))

    def fill_guard_cells_dirichlet(self, var_name, loc, bc_val):
        """Fill guard cells using a Dirichlet condition.

        Parameters
        ----------
        var_name : string
            Name of the variable to update.
        loc : string
            Boundary location;
            choices: ['left', 'right', 'bottom', 'top'].
        bc_val : float
            Neumann boundary value.

        """
        var = self.get_values(var_name)
        if loc == 'left':
            var[0, :] = 2 * bc_val - var[1, :]
        elif loc == 'right':
            var[-1, :] = 2 * bc_val - var[-2, :]
        elif loc == 'bottom':
            var[:, 0] = bc_val
        elif loc == 'top':
            var[:, -1] = bc_val
        else:
            raise ValueError('Unknown boundary location "{}"'.format(loc))


def Grid(gridtype, *args, **kwargs):
    """Return an instance of the GridBase child class based on type.

    Parameters
    ----------
    gridtype : string
        Type of grid;
        choices: ['cell-centered', 'x-face', 'y-face'].
    var_names : list of strings
            List of names for the variables to create.
    nx : integer
        Number of cells in the x-direction.
    ny : integer
        Number of cells in the y-direction.
    xmin : float
        Domain limit at the left side.
    xmax : float
        Domain limit at the right side.
    ymin : float
        Domain limit at the bottom side.
    ymax : float
        Domain limit at the top side.
    user_bc_type : dictionary of (string, list) items
        User-defined boundary types to overwrite default ones.
    user_bc_val : dictionary of (string, list) items
        User-defined boundary values to overwrite default ones.

    Returns
    -------
    obj : instance of the GridBase child
        The grid object.

    """
    for cls in GridBase.__subclasses__():
        if cls.check_gridtype(gridtype):
            return cls(*args, **kwargs)
    raise ValueError('Parameter "gridtype" should be either '
                     '"cell-centered", "x-face", or "y-face"')
