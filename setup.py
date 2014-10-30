#
# This script builds PyNIO from source. Some environment variables
# may be required. See the following comments.
#

import os, sys, commands

# Test to make sure we actually have NumPy.
try:
  import numpy
except ImportError:
  print "Error: Cannot import NumPy. Can't continue."
  sys.exit()

#
# At a minimum, you must NetCDF (3.6.0 or later) installed.
#
# You must set the environment variables:
#
#    NETCDF_PREFIX
#
# to the parent locations of the NetCDF-3 installations.
#
# Note: you must at least set one XXXX_PREFIX environment variable. 
# If all the other software is installed in this same root directory, 
# then you don't need to set any of the other XXXX_PREFIX variables.
#
# You can optionally build PyNIO with NetCDF-4, HDF4, HDF-EOS2, HDF-EOS5, HDF5,
# GRIB2 and/or shapefile (using the GDAL library) support.  To do this, 
# the corresponding environment variables:
#
#    HAS_NETCDF4
#    HAS_HDF4
#    HAS_HDFEOS
#    HAS_HDFEOS5
#    HAS_GRIB2
#    HAS_GDAL
#    HAS_HDF5
#
# must be set to 1. In addition, the corresponding environment variables:
#
#    NETCDF4_PREFIX
#    HDF4_PREFIX
#    HDF5_PREFIX
#    HDFEOS_PREFIX
#    HDFEOS5_PREFIX
#    GRIB2_PREFIX 
#    GDAL_PREFIX
#
# must be set to the root location of that software, unless they are
# the same as a previous setting, such as HDF_PREFIX.
#
# If you are linking against NetCDF version 4, this script assumes
# that OPeNDAP support has been included since NetCDF 4.1.1 turns
# this on by default. If you built NetCDF-4 with OPeNDAP support
# turned off (--disable-dap), then set the environment variable
# HAS_OPENDAP to 0.
#
# If you want support for any file types that depend on the HDF4
# or HDF5 libraries and either of them were built with support
# for SZIP compression then additionally you need to set environment
# variables for SZIP in a similar fashion:
#    HAS_SZIP
# (set to 1)
#    SZIP_PREFIX
# (if it resides in a location of its own)
# Note: NetCDF4, HDFEOS5, and HDF5 depend on the HDF5 libraries.
# HDFEOS and HDF4 depend on the HDF4 libraries.
#
# Finally, you may need to include Fortran system libraries
# (like "-lgfortran" or "-lf95") to resolve undefined symbols.
#
# Use F2CLIBS and F2CLIBS_PREFIX for this. For example, if you
# need to include "-lgfortran", and this library resides in /sw/lib:
#
#  F2CLIBS gfortran
#  F2CLIBS_PREFIX /sw/lib
#

from os.path import join

#
# This proceure tries to figure out which extra libraries are
# needed with curl.
#
def set_curl_libs():
  curl_libs = commands.getstatusoutput('curl-config --libs')
  if curl_libs[0] == 0:
#
# Split into individual lines so we can loop through each one.
#
    clibs = curl_libs[1].split()
#
# Check if this is a -L or -l string and do the appropriate thing.
#
    for clibstr in clibs:
      if clibstr[0:2] == "-L":
        LIB_DIRS.append(clibstr.split("-L")[1])
      elif clibstr[0:2] == "-l":
        LIBRARIES.append(clibstr.split("-l")[1])
  else:
#
# If curl-config doesn't produce valid output, then try -lcurl.
#
    LIBRARIES.append('curl')

  return

# End of set_curl_libs

formats = {}
LIB_MACROS        =  [ ('NeedFuncProto',None), ('NIO_LIB_ONLY' , None), ('_POSIX_SOURCE',None) ]

if sys.byteorder == 'little':
  LIB_MACROS.append(('ByteSwapped', None))

LIB_EXCLUDE_SOURCES = []
LIB_DIRS   = ['libsrc']
INC_DIRS   = ['libsrc']


# These are the required NIO, HDF4, and NetCDF libraries.
LIBRARIES = ['nio', 'jpeg', 'png', 'z', 'netcdf']

# Check for XXXX_PREFIX environment variables.
try:
  LIB_DIRS.append(os.path.join(os.environ["NETCDF_PREFIX"],"lib"))
  INC_DIRS.append(os.path.join(os.environ["NETCDF_PREFIX"],"include"))
except:
  pass

try:
  HAS_HDF4 = int(os.environ["HAS_HDF4"])
  if HAS_HDF4 > 0:
    LIBRARIES.append('mfhdf')
    LIBRARIES.append('df')
    LIB_MACROS.append(('BuildHDF4', None))
    try:
      LIB_DIRS.append(os.path.join(os.environ["HDF4_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["HDF4_PREFIX"],"include"))
    except:
      pass
  else:
    LIB_EXCLUDE_SOURCES.append('NclHDF.c')
except:
  HAS_HDF4 = 0
  LIB_EXCLUDE_SOURCES.append('NclHDF.c')

formats['hdf4'] = HAS_HDF4

HAS_OPENDAP = 0
try:
  HAS_NETCDF4 = int(os.environ["HAS_NETCDF4"])
  if HAS_NETCDF4 > 0:
    LIBRARIES.append('hdf5_hl')
    LIBRARIES.append('hdf5')
    LIB_MACROS.append(('USE_NETCDF4', None))
    LIB_MACROS.append(('USE_NETCDF4_FEATURES', None))
    try:
      HAS_OPENDAP = int(os.environ["HAS_OPENDAP"])
    except:    
      HAS_OPENDAP = 1
    if HAS_OPENDAP > 0:
      set_curl_libs()
    try:
      LIB_DIRS.append(os.path.join(os.environ["NETCDF4_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["NETCDF4_PREFIX"],"include"))
    except:
      pass
except:
  HAS_NETCDF4 = 0

formats['netcdf4'] = HAS_NETCDF4
formats['opendap'] = 0
if HAS_NETCDF4 > 0:
    formats['opendap'] = HAS_OPENDAP

try:
  HAS_HDFEOS = int(os.environ["HAS_HDFEOS"]) and int(os.environ["HAS_HDF4"])
  if HAS_HDFEOS > 0:
    LIB_MACROS.append(('BuildHDFEOS', None))
    LIBRARIES.append('hdfeos')
    LIBRARIES.append('Gctp')
    if HAS_HDF4 == 0:
      LIBRARIES.append('mfhdf')
      LIBRARIES.append('df')
    try:
      LIB_DIRS.append(os.path.join(os.environ["HDFEOS_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["HDFEOS_PREFIX"],"include"))
    except:
      pass
  else:
      LIB_EXCLUDE_SOURCES.append('NclHDFEOS.c')
except:
  HAS_HDFEOS = 0
  LIB_EXCLUDE_SOURCES.append('NclHDFEOS.c')

formats['hdfeos'] = HAS_HDFEOS

try:
  HAS_GRIB2 = int(os.environ["HAS_GRIB2"])
  if HAS_GRIB2 > 0:
    LIBRARIES.append('grib2c')
    LIBRARIES.append('jasper')   # png is needed again, b/c it 
    LIBRARIES.append('png')      # must come after jasper
    LIB_MACROS.append(('BuildGRIB2', None))
    # This should test whether the system is 64 bits or not
    if sys.maxint > 2**32:
	LIB_MACROS.append(('__64BIT__',None))
    try:
      LIB_DIRS.append(os.path.join(os.environ["GRIB2_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["GRIB2_PREFIX"],"include"))
    except:
      pass
  else:
      LIB_EXCLUDE_SOURCES.append('NclGRIB2.c')
except:
  HAS_GRIB2 = 0
  LIB_EXCLUDE_SOURCES.append('NclGRIB2.c')

formats['grib2'] = HAS_GRIB2

try:
  HAS_HDF5 = int(os.environ["HAS_HDF5"])
  if HAS_HDF5 > 0:
    LIB_MACROS.append(('BuildHDF5', None))
    LIBRARIES.append('hdf5_hl')
    LIBRARIES.append('hdf5')

    try:
      LIB_DIRS.append(os.path.join(os.environ["HDF5_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["HDF5_PREFIX"],"include"))
    except:
      pass
  else:
    LIB_EXCLUDE_SOURCES.append('NclNewHDF5.c')
    LIB_EXCLUDE_SOURCES.append('NclHDF5.c')
    LIB_EXCLUDE_SOURCES.append('h5reader.c')
    LIB_EXCLUDE_SOURCES.append('h5writer.c')
except:
  HAS_HDF5 = 0
  LIB_EXCLUDE_SOURCES.append('NclNewHDF5.c')
  LIB_EXCLUDE_SOURCES.append('NclHDF5.c')
  LIB_EXCLUDE_SOURCES.append('h5reader.c')
  LIB_EXCLUDE_SOURCES.append('h5writer.c')

formats['hdf5'] = HAS_HDF5

try:
  HAS_HDFEOS5 = int(os.environ["HAS_HDFEOS5"]) and int(os.environ["HAS_HDF5"])
  if HAS_HDFEOS5 > 0:
    LIB_MACROS.append(('BuildHDFEOS5', None))
    LIBRARIES.append('he5_hdfeos')
    LIBRARIES.append('Gctp')
    try:
      LIB_DIRS.append(os.path.join(os.environ["HDFEOS5_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["HDFEOS5_PREFIX"],"include"))
    except:
      pass
  else:
    LIB_EXCLUDE_SOURCES.append('NclHDFEOS5.c')
except:
  HAS_HDFEOS5 = 0
  LIB_EXCLUDE_SOURCES.append('NclHDFEOS5.c')

formats['hdfeos5'] = HAS_HDFEOS5

try:
  HAS_GDAL = int(os.environ["HAS_GDAL"])
  if HAS_GDAL > 0:
    LIBRARIES.append('gdal')
    LIBRARIES.append('proj') 
    LIBRARIES.append('iconv') 
    LIBRARIES.append('jasper')   # png is needed again, b/c it 
    LIBRARIES.append('png')      # must come after jasper
    LIB_MACROS.append(('BuildGDAL', None))
    try:
      LIB_DIRS.append(os.path.join(os.environ["GDAL_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["GDAL_PREFIX"],"include"))
    except:
      pass
  else:
    LIB_EXCLUDE_SOURCES.append('NclOGR.c')
except:
  HAS_GDAL = 0
  LIB_EXCLUDE_SOURCES.append('NclOGR.c')

formats['shapefile'] = HAS_GDAL

try:
  try:
    HAS_ZLIB = int(os.environ["HAS_ZLIB"])
  except:
    if HAS_NETCDF4 > 0 or HAS_HDFEOS5 > 0:
      HAS_ZLIB = 1
  if HAS_ZLIB > 0:
    LIBRARIES.append('z')
    try:
      LIB_DIRS.append(os.path.join(os.environ["ZLIB_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["ZLIB_PREFIX"],"include"))
    except:
      pass
except:
  HAS_ZLIB = 0

try:
  try:
    HAS_SZIP = int(os.environ["HAS_SZIP"])
  except:
    if HAS_NETCDF4 > 0 or HAS_HDFEOS5 > 0:
      HAS_SZIP = 1
  if HAS_SZIP > 0:
    LIBRARIES.append('sz')
    try:
      LIB_DIRS.append(os.path.join(os.environ["SZIP_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["SZIP_PREFIX"],"include"))
    except:
      pass
except:
  HAS_SZIP = 0

# Depending on what Fortran compiler was used to build, we may need
# additional library paths or libraries.
try:
  f2clibs = os.environ["F2CLIBS"].split()
  for lib in f2clibs:
    LIBRARIES.append(lib)
except:
  pass

try:
  LIB_DIRS.append(os.environ["F2CLIBS_PREFIX"])
except:
  pass

try:
  EXTRA_OBJECTS = [os.environ["EXTRA_OBJECTS"]]
except:
  EXTRA_OBJECTS = ""

#
# Done with environment variables.
#

#
# Function for getting list of needed GRIB2 code tables and
# copying them over for the PyNIO installation.
#
def get_grib2_codetables():
  data_files = []
  plat_dir = os.path.join("build","lib."+get_platform()+"-"+sys.version[:3], \
                          "PyNIO")

  ncarg_dirs    = os.path.join("ncarg","grib2_codetables")


# Walk through each directory and copy some data files.
  for root, dirs, files in os.walk(ncarg_dirs):
      for name in files:
          data_files.append(os.path.join(root,name))

  return data_files


# Main code.

import platform
from numpy.distutils.core import setup
from distutils.util import get_platform
from distutils.sysconfig import get_python_lib

#
# Initialize some variables.
#
pynio_vfile = "pynio_version.py"         # PyNIO version file.
pkgs_pth    = get_python_lib()

#
# These variables are temporarily defined for readability.
# The default FORTRAN_CALLING_METHOD is APPEND_UNDERSCORE.
#

APPEND_UNDERSCORE = 1
NO_APPEND_UNDERSCORE = 2
CAPS_NO_APPEND_UNDERSCORE = 3

if sys.platform == "linux2" and os.uname()[-1] == "x86_64" and \
    platform.python_compiler()[:5] == "GCC 4":
    LIBRARIES.append('gfortran')

elif sys.platform == "irix6-64":
    LIBRARIES.append('ftn')
    LIBRARIES.append('fortran')
    LIBRARIES.append('sz')

elif sys.platform == "sunos5":
    LIBRARIES.append('fsu')
    LIBRARIES.append('sunmath')

elif sys.platform == "aix5":
    os.putenv('OBJECT_MODE',"64")
    LIBRARIES.append('xlf90')
    LIB_MACROS.append(('FORTRAN_CALLING_METHOD', NO_APPEND_UNDERSCORE))

del(APPEND_UNDERSCORE)
del(NO_APPEND_UNDERSCORE)
del(CAPS_NO_APPEND_UNDERSCORE)
    
#----------------------------------------------------------------------
#
# Set some variables.
#
#----------------------------------------------------------------------
from numpy import __version__ as array_module_version

# I read somewhere that distutils doesn't update this file properly
# when the contents of directories change.

if os.path.exists('MANIFEST'): os.remove('MANIFEST')

pynio_pkg_name = 'PyNIO'
pynio_pth_file = ['Nio.pth']
DMACROS        =  [ ('NeedFuncProto',None), ('NIO_LIB_ONLY', None) ]

INC_DIRS.insert(0,numpy.get_include())


#----------------------------------------------------------------------
#
# Create version file that contains version and array module info.
#
#----------------------------------------------------------------------
if os.path.exists(pynio_vfile):
  os.system("/bin/rm -rf " + pynio_vfile)

pynio_version = open('version','r').readlines()[0].strip('\n')

vfile = open(pynio_vfile,'w')
vfile.write("version = '%s'\n" % pynio_version)
vfile.write("array_module = 'numpy'\n")
vfile.write("array_module_version = '%s'\n" % array_module_version)
vfile.write("python_version = '%s'\n" % sys.version[:5])
vfile.write("formats = {}\n")
for key in formats.keys():
  vfile.write("formats['%s'] = %d\n" % (key, formats[key]))
vfile.close()

#----------------------------------------------------------------------
#
# Here are the instructions for compiling the "nio.so" file.
#
#----------------------------------------------------------------------
print '====> Installing Nio to the "'+pynio_pkg_name+'" site packages directory.'


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(pynio_pkg_name,parent_package, top_path)

    files  = os.listdir('libsrc')
    sources = [ file for file in files if file.endswith('.c') or file.endswith('.f') ]

    for file in LIB_EXCLUDE_SOURCES: 
      sources.remove(file)
    sources = [ join('libsrc', file) for file in sources ]

    config.add_library('nio',sources,
                       include_dirs=INC_DIRS,
                       macros=LIB_MACROS,
                       extra_compiler_args = [ '-O0 -g', '-w' ]
#                       extra_compiler_args = [ '-O2', '-w' ]
                       )
    
    sources = ['niomodule.c']

    config.add_extension('nio',
                         sources=sources,
                         libraries=LIBRARIES,
                         include_dirs=INC_DIRS,
                         define_macros = DMACROS,
                         library_dirs  = LIB_DIRS,
                         extra_objects = EXTRA_OBJECTS,
                         language = 'c++'
                         )
    return config


if HAS_GRIB2 > 0:
  data_files = get_grib2_codetables()
else:
  data_files = []

if os.environ.has_key('PYTHONPATH'):
  print "\n\n\nOld pkgs_pth = ", pkgs_pth

  print "FORCED pkgs_pth to the first of PYTHONPATH"
  pythonpaths = os.environ["PYTHONPATH"].split(':')
  print "pythonpaths = ", pythonpaths
  pkgs_pth = pythonpaths[0]
  print "\n\n\nNew pkgs_pth = ", pkgs_pth
  print "\n\n\n"
 
#print data_files
setup (version      = pynio_version,
       description  = 'Multi-format data I/O package',
       author       = 'David I. Brown',
       author_email = 'dbrown@ucar.edu',
       url          = 'http://www.pyngl.ucar.edu/Nio.shtml',
       long_description = '''
       Enables NetCDF-like access for NetCDF (rw), HDF (rw), HDF-EOS2 (r), HDF-EOS5, GRIB (r), and CCM (r) data files
       ''',
       package_data = { pynio_pkg_name : data_files },
       data_files   = [(pkgs_pth, pynio_pth_file)],
       **configuration().todict())
#
# Cleanup: remove the pynio_version.py file.
#
if os.path.exists(pynio_vfile):
  os.system("/bin/rm -rf " + pynio_vfile)

print "\n\n\nNew pkgs_pth = ", pkgs_pth
print "\n\n\n"
 
