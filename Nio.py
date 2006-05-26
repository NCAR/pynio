'''
PyNIO enables NetCDF-like access for
NetCDF (rw), HDF (rw), HDFEOS (r)(optional), GRIB (r), and CCM (r) data files.

import Nio

Class NioFile constructor:
open_file(filepath, mode='r', options=None, history='')

attributes:
   dimensions -- a dictionary with dimension names as keys and dimension lengths as values
   variables -- a dictionary with variable names as keys and the variable objects as values
   __dict__ --  contains the global attributes associated with the file
methods:
   close(history='')
   create_dimension(name, length)
   create_variable(name, type,dimensions)

For more information see:

	http://www.pyngl.ucar.edu/Nio.html
'''

from nio import *
from nio import _C_API

import pynio_version
__version__ = pynio_version.version
del pynio_version
