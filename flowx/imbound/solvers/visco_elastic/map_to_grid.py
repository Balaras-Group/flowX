import numpy
import time

def map_to_grid_visco(gridc, particles, ibmf, ibmx, ibmy, search_function, options):

    """
    Subroutine to compute IB mapping on grid using the level set function
 
    Arguments
    ---------
    gridc : object
      Grid object for x-face variables

    particles: object
       Object containing immersed boundary information

    ibmf : string for forcing variable

    search_function : 
         Search function

    """

    x, y = numpy.meshgrid(gridc.x, gridc.y)

    nx, ny = gridc.nx, gridc.ny
    dx, dy = gridc.dx, gridc.dy

    x = x.transpose()
    y = y.transpose()

    phi = gridc.get_values(ibmf)
    lmx = gridc.get_values(ibmx)
    lmy = gridc.get_values(ibmy)

    lmx[:,:] = x
    lmy[:,:] = y

    for particle in particles:

        points =  particle.x[1:,:]
        np = particle.nnp-1
        options['max_panel_length'] = particle.max_panel_length

        ites, phi[:,:] = search_function(x, y, points, nx+2, ny+2, np, options)

    if (options['verbose']):
        print("Mapping Iterations: ", ites)

    return ites
