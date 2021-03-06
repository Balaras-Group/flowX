"""Module with implementation of the Grid classes."""

import numpy
from .gridBase import GridBase

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
