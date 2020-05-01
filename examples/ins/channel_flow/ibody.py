'''Function to creat the immersed body. Here we creat a circle with n points'''

import numpy
from matplotlib import pyplot

def ib(xc,yc,r,Nc):
    theta = numpy.linspace(0,2*numpy.pi,Nc)
    x = xc + r*numpy.cos(theta)
    y = yc + r*numpy.sin(theta)
    
    size = 4
    pyplot.figure(figsize=(size, size))
    pyplot.grid()
    pyplot.xlabel('x', fontsize=16)
    pyplot.ylabel('y', fontsize=16)
    pyplot.plot(x, y, color='b', linestyle='-', linewidth=2)
    pyplot.xlim(-min(x)-1, max(x)+1)
    pyplot.ylim(-min(y)-1, max(y)+1);
    return x,y