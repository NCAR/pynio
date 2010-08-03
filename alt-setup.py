#
# This script builds PyNIO from source. Some environment variables
# may be required. See the following comments.
#

# Test to make sure we actually have NumPy.
try:
  import numpy
except ImportError:
  print "Error: Cannot import NumPy. Can't continue."
  sys.exit()

#
# At a minimum, you must have NCL, NetCDF (3.6.0 or later), and 
# HDF-4 (4.1 or later) installed.
#
# You must set the environment variables:
#
#    NCARG_ROOT
#    NETCDF_PREFIX
#    HDF_PREFIX
#
# to the parent locations of the NCL, NetCDF-3, and HDF-4 installations.
#
# Note: at a minimum, you must have NCARG_ROOT set. If all the other
# software is installed in this same root directory, then you don't
# need to set any of the previous or following XXXX_PREFIX variables.
#
# You can optionally build PyNIO with NetCDF-4, HDF-EOS2, HDF-EOS5,
# GRIB2 and/or shapefile (using the GDAL library) support.  To do this, 
# the corresponding environment variables:
#
#    HAS_NETCDF4
#    HAS_HDFEOS
#    HAS_HDFEOS5
#    HAS_GRIB2
#    HAS_GDAL
#
# must be set to 1. In addition, the corresponding environment variables:
#
#    NETCDF4_PREFIX
#    HDFEOS_PREFIX
#    HDFEOS5_PREFIX
#    GRIB2_PREFIX 
#    GDAL_PREFIX
#
# must be set to the root location of that software, unless they are
# all the same as a previous setting, like NCARG_ROOT.
#
#
# If you are linking against NetCDF version 4, this script assumes
# that OPeNDAP support has been included since NetCDF 4.1.1 turns
# this on by default. If you built NetCDF-4 with OPeNDAP support
# turned off (--disable-dap), then set the environment variable
# HAS_OPENDAP to 0.
#
# If your HDF4 library was built with support for SZIP compression or
# if you want to include NETCDF4 and/or HDFEOS5 support and the HDF5 
# libraries on which they depend have SZIP support included, then you
# additionally need to set environment variables for SZIP in a similar
# fashion:
#    HAS_SZIP
# (set to 1)
#    SZIP_PREFIX
# (if it resides in a location of its own)
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

import os, sys, commands

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

try:
  ncarg_root = os.environ["NCARG_ROOT"]
  LIB_DIRS   = [os.path.join(ncarg_root,'lib') ]
  INC_DIRS   = [os.path.join(ncarg_root,'include/ncarg/nio')]
  #LIB_DIRS   = ['libsrc']
  #INC_DIRS   = ['libsrc']

except:
  print "NCARG_ROOT is not set; can't continue!"
  sys.exit()

# These are the required NCL, HDF4, and NetCDF libraries.
LIBRARIES = ['nio', 'mfhdf', 'df', 'jpeg', 'png', 'z', 'netcdf']
# Check for XXXX_PREFIX environment variables.
try:
  LIB_DIRS.append(os.path.join(os.environ["NETCDF_PREFIX"],"lib"))
  INC_DIRS.append(os.path.join(os.environ["NETCDF_PREFIX"],"include"))
except:
  pass

try:
  LIB_DIRS.append(os.path.join(os.environ["HDF_PREFIX"],"lib"))
  INC_DIRS.append(os.path.join(os.environ["HDF_PREFIX"],"include"))
except:
  pass

try:
  HAS_HDFEOS5 = int(os.environ["HAS_HDFEOS5"])
  if HAS_HDFEOS5 > 0:
    LIBRARIES.append('he5_hdfeos')
    LIBRARIES.append('Gctp')
    try:
      LIB_DIRS.append(os.path.join(os.environ["HDFEOS5_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["HDFEOS5_PREFIX"],"include"))
    except:
      pass
except:
  HAS_HDFEOS5 = 0

try:
  HAS_NETCDF4 = int(os.environ["HAS_NETCDF4"])
  if HAS_NETCDF4 > 0:
    LIBRARIES.append('hdf5_hl')
    LIBRARIES.append('hdf5')
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

try:
  HAS_HDFEOS = int(os.environ["HAS_HDFEOS"])
  if HAS_HDFEOS > 0:
    LIBRARIES.append('hdfeos')
    LIBRARIES.append('Gctp')
    try:
      LIB_DIRS.append(os.path.join(os.environ["HDFEOS_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["HDFEOS_PREFIX"],"include"))
    except:
      pass
except:
  HAS_HDFEOS = 0

try:
  HAS_GRIB2 = int(os.environ["HAS_GRIB2"])
  if HAS_GRIB2 > 0:
    LIBRARIES.append('grib2c')
    LIBRARIES.append('jasper')   # png is needed again, b/c it 
    LIBRARIES.append('png')      # must come after jasper
    try:
      LIB_DIRS.append(os.path.join(os.environ["GRIB2_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["GRIB2_PREFIX"],"include"))
    except:
      pass
except:
  HAS_GRIB2 = 0

try:
  HAS_GDAL = int(os.environ["HAS_GDAL"])
  if HAS_GRIB2 > 0:
    LIBRARIES.append('gdal')
    LIBRARIES.append('proj') 
    try:
      LIB_DIRS.append(os.path.join(os.environ["GDAL_PREFIX"],"lib"))
      INC_DIRS.append(os.path.join(os.environ["GDAL_PREFIX"],"include"))
    except:
      pass
except:
  HAS_GDAL = 0

try:
  HAS_SZIP = int(os.environ["HAS_SZIP"])
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
  plat_dir = os.path.join("build","lib."+get_platform()+"-"+sys.version[:3], \
                          "PyNIO")

  ncl_lib       = os.path.join(ncarg_root,'lib')
  ncl_ncarg_dir = os.path.join(ncl_lib,'ncarg')
  ncarg_dirs    = ["grib2_codetables"]

  cwd = os.getcwd()          # Retain current directory.
  if not os.path.exists('ncarg'):
    os.mkdir('ncarg')          # make a directory to copy files to
  os.chdir(ncl_ncarg_dir)    # cd to $NCARG_ROOT/lib/ncarg

# Walk through each directory and copy some data files.
  for ncarg_dir in ncarg_dirs:
    for root, dirs, files in os.walk(ncarg_dir):
      dir_to_copy_to = os.path.join(cwd,'ncarg',root)
      if not os.path.exists(dir_to_copy_to):
        os.mkdir(dir_to_copy_to)
      for name in files:
        file_to_copy = os.path.join(ncl_ncarg_dir,root,name)
        cmd = "cp " + file_to_copy + " " + dir_to_copy_to
        os.system(cmd)
        data_files.append(os.path.join('ncarg',root,name))

  os.chdir(cwd)    # cd back to original directory

  return


# Main code.

import platform
from distutils.core import setup, Extension
from distutils.util import get_platform
from distutils.sysconfig import get_python_lib

#
# Initialize some variables.
#
pynio_vfile = "pynio_version.py"         # PyNIO version file.
pkgs_pth    = get_python_lib()

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
DMACROS        =  [ ('NeedFuncProto','1'), ('NIO_LIB_ONLY', '1') ]

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
vfile.write("python_version = '%s'\n" % sys.version[:3])
vfile.close()

#----------------------------------------------------------------------
#
# Here are the instructions for compiling the "nio.so" file.
#
#----------------------------------------------------------------------
print '====> Installing Nio to the "'+pynio_pkg_name+'" site packages directory.'

module1 = [Extension('nio',
                    define_macros = DMACROS,
                    include_dirs  = INC_DIRS,
                    libraries     = LIBRARIES,
                    library_dirs  = LIB_DIRS,
                    extra_objects = EXTRA_OBJECTS,
                    sources       = ['niomodule.c']
                    )]

data_files = []
if HAS_GRIB2 > 0:
  get_grib2_codetables()

setup (name         = 'PyNIO',
       version      = pynio_version,
       description  = 'Multi-format data I/O package',
       author       = 'David I. Brown',
       author_email = 'dbrown@ucar.edu',
       url          = 'http://www.pyngl.ucar.edu/Nio.shtml',
       long_description = '''
       Enables NetCDF-like access for NetCDF (rw), HDF (rw), HDFEOS (r), GRIB (r), and CCM (r) data files
       ''',
       package_dir  = { pynio_pkg_name : '.' },
       packages     = [ pynio_pkg_name ],
       ext_modules  = module1,
       ext_package  = pynio_pkg_name,
       package_data = { pynio_pkg_name : data_files },
       data_files   = [(pkgs_pth, pynio_pth_file)])

#
# Cleanup: remove the pynio_version.py file.
#
if os.path.exists(pynio_vfile):
  os.system("/bin/rm -rf " + pynio_vfile)
