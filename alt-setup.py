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
# You can optionally build PyNIO with NetCDF-4, HDF-EOS 2,
# and/or GRIB 2 support.  To do this, the corresponding environment
# variables:
#
#    HAS_NETCDF4
#    HAS_HDFEOS
#    HAS_GRIB2
#
# must be set to 1. In addition, the corresponding environment variables:
#
#    NETCDF4_PREFIX
#    HDFEOS_PREFIX
#    GRIB2_PREFIX 
#
# must be set to the root location of that software, unless they are
# all the same as a previous setting, like NCARG_ROOT.
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

import os, sys

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
    LIBRARIES.append('sz')
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
