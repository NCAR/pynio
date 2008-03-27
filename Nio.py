"""
PyNIO enables NetCDF-like access for
NetCDF (rw), HDF (rw), HDFEOS (r)(optional), GRIB (r), and CCM (r) data files.

import Nio

Class NioFile:

f = Nio.open_file(filepath, mode='r', options=None, history='')

To see summary of file contents, including all dimensions, attributes,
and variables:
   print f
attributes:
   dimensions -- dimension names (keys), dimension lengths (values)
   variables -- variable names (keys), variable objects (values)
   __dict__ --  contains the global attributes associated with the file
methods:
   close(history='')
   create_dimension(name, length)
   create_variable(name, type,dimensions)
For more detailed information:
    print f.__doc__

Class NioOptions

opt = Nio.options()

To set format-specific options assign option names and settings as attributes
and values of 'opt'. Then pass 'opt' as the optional options argument to
Nio.open_file.
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

#
#  Get version number and flag for NumPy compatibility.
#
#  Also, get the __array_module__  and __array_module_version__
#  attributes.
#
import pynio_version
__version__              = pynio_version.version
__array_module__         = pynio_version.array_module
__array_module_version__ = pynio_version.array_module_version
del pynio_version

def pyniopath_ncarg():
#
#  Find the root directory that contains the supplemental PyNIO files,
#  in particular, the grib2 codetables. For now the default is to look
#  in site-packages/PyNGL/ncarg. Otherwise, check the PYNGL_NCARG
#  environment variable. This may change if the grib2 codetables 
#  are moved into the PyNIO tree.
#  
#
  import sys
  pkgs_path = None
  for path in sys.path:
    slen = len('site-packages')
    i = path.rfind('site-packages')
    if i > -1 and i + slen == len(path):
      pkgs_path = path
      break

  pyngl1_dir  = os.path.join(pkgs_path,"PyNGL","ncarg")
  pyngl2_dir  = os.environ.get("PYNGL_NCARG")
  ncarg_dir  = os.environ.get("NCARG_ROOT")

  if pyngl2_dir != None and os.path.exists(pyngl2_dir):
    pyngl_ncarg = pyngl2_dir
  elif os.path.exists(pyngl1_dir):
    pyngl_ncarg = pyngl1_dir
  else:
    if os.path.exists(ncarg_dir):
	pyngl_ncarg = os.path.join(ncarg_dir,"lib","ncarg")
	if not os.path.exists(pyngl_ncarg):
	    print "pynglpath: directory " + pyngl1_dir + \
        	  "\n           does not exist and " + \
          	  "environment variable PYNGL_NCARG is not set and " + \
		  "no usable NCARG installation found"
            sys.exit()
    else:
	sys.exit()

  return pyngl_ncarg

#
# Set the NCARG_NCARG environment variable.
# This should allow the grib2_codetables directory to be found without
# requiring any environment variables to be set by the user

import os
os.environ["NCARG_NCARG"] = pyniopath_ncarg()
del pyniopath_ncarg
del os
