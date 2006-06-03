"""
PyNIO enables NetCDF-like access for
NetCDF (rw), HDF (rw), HDFEOS (r)(optional), GRIB (r), and CCM (r) data files.

import Nio

Class NioFile:

f = Nio.open_file(filepath, mode='r', options=None, history='')

To see summary of file contents, including dimensions, attributes, and variables:
   print f
attributes:
   dimensions -- a dictionary with dimension names as keys and dimension lengths as values
   variables -- a dictionary with variable names as keys and the variable objects as values
   __dict__ --  contains the global attributes associated with the file
methods:
   close(history='')
   create_dimension(name, length)
   create_variable(name, type,dimensions)
For more detailed information:
    print f.__doc__

Class NioOptions

opt = Nio.options()

To set format-specific options assign option names and settings as attributes and
values of 'opt'. Then pass 'opt' as the optional options argument to Nio.open_file.
To see valid options:
    print opt.__doc__ 

Class NioVariable

v = f.variables['varname']

To see summary of variable contents including all dimensions,
associated coordinate variables, and attributes:
    print v 
Attributes:
    rank -- a scalar value indicating the number of dimensions
    shape -- a tuple containing the number of elements in each dimension
    dimensions -- a tuple containing the dimensions names in order
    __dict__ -- a dictionary containing the variable attributes
Methods:
    assign_value(value) -- assign a value to a variable in the file.
    get_value() -- retrieve the value of a variable in the file.
    typecode() -- return a character code representing the variable's type.
For more detailed information:
    print v.__doc__

For complete documentation see:

	http://www.pyngl.ucar.edu/Nio.html
"""

from nio import *
from nio import _C_API

import pynio_version
__version__ = pynio_version.version
del pynio_version

#open_file.__doc__ = """
#
#f = open_file(filepath, mode='r', options=None, history='')
#
#"""
