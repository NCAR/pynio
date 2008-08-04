import os,sys,platform
from distutils.core import setup, Extension

try:
  HAS_NETCDF4 = int(os.environ["HAS_NETCDF4"])
except:
  HAS_NETCDF4 = 1

try:
  HAS_HDFEOS = int(os.environ["HAS_HDFEOS"])
except:
  HAS_HDFEOS = 1

try:
  HAS_GRIB2 = int(os.environ["HAS_GRIB2"])
except:
  HAS_GRIB2 = 1

try:
  path = os.environ["PYNIO2PYNGL"]
  pynio2pyngl = True
except:
  pynio2pyngl = False

#
# This script used to build both a Numeric and NumPy version of PyNIO.
# As of October 2007, it only builds a NumPy version.
#
#  The NumPy version will be installed to the site-packages directory
#  "PyNIO".

#
# Test to make sure we actually have NumPy.
#
try:
  import numpy
except ImportError:
  print "Cannot find NumPy; good-bye!"
  sys.exit()

#
# Initialize some variables.
#
pynio_vfile = "pynio_version.py"      # PyNIO version file.

ncarg_root = os.getenv("NCARG_ROOT")
lib_paths = [ os.path.join(ncarg_root,'lib'),'/Users/haley/lib' ]

ncl_src_dir = '../ni/src/ncl/'
pkgs_pth  = os.path.join(sys.exec_prefix, 'lib', 'python'+sys.version[:3],
            'site-packages')

LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','png','z','netcdf']

if HAS_NETCDF4 > 0:
    LIBRARIES.append('hdf5_hl')
    LIBRARIES.append('hdf5')
    LIBRARIES.append('sz')

if HAS_HDFEOS > 0:
    LIBRARIES.append('hdfeos')
    LIBRARIES.append('Gctp')

if HAS_GRIB2 > 0:
    LIBRARIES.append('grib2c')
    LIBRARIES.append('jasper')
    LIBRARIES.append('png')
    LIBRARIES.append('z')
    
LIBRARIES.append('g2c')   # Put on the end.

#
# The long path below is for the g95 compiler on the Intel Mac.
#
if sys.platform == "darwin":
    dirs = ['/sw/lib','/Users/haley/lib/gcc-lib/i386-apple-darwin8.9.1/4.0.3']
    for dir in dirs:
      if(os.path.exists(dir)):
#        print "appending",dir
        lib_paths.append(dir)
#
# Special test for Intel Mac platform, which is using the g95 compiler
# and doesn't need g2c loaded.
#
    if "i386" in os.uname():
      LIBRARIES.remove('g2c')
      LIBRARIES.append('f95')

if sys.platform == "irix6-64":
    LIBRARIES.remove('g2c')
    LIBRARIES.append('ftn')
    LIBRARIES.append('fortran')
    LIBRARIES.append('sz')

if sys.platform == "linux2" and os.uname()[-1] == "x86_64" and \
    platform.python_compiler() == "GCC 4.1.1":
    print("Using gcc4 compiler, thus removing g2c...")
    LIBRARIES.remove('g2c')
    LIBRARIES.append('gfortran')

if sys.platform == "sunos5":
    LIBRARIES.remove('g2c')
    LIBRARIES.append('fsu')
    LIBRARIES.append('sunmath')

if sys.platform == "aix5":
    os.putenv('OBJECT_MODE',"64")
    LIBRARIES.remove('g2c')
    LIBRARIES.append('xlf90')
    
#
# Special test for Intel Mac platform, which is using the g95 compiler
# and needs f95 loaded.
#
if sys.platform == "darwin":
    dir = '/Users/haley/lib/gcc-lib/i386-apple-darwin8.6.1/4.0.3'
    if dir in lib_paths:
      LIBRARIES.remove('g2c')
      LIBRARIES.append('f95')

INCLUDE_DIRS = [ncl_src_dir, os.path.join(ncarg_root,'include'),'/Users/haley/include']

#----------------------------------------------------------------------
#
# Set some variables.
#
#----------------------------------------------------------------------
from numpy import __version__ as array_module_version

if pynio2pyngl:
  pynio_pkg_name = 'PyNGL'
  pynio_pth_file = []
  pynio_files    = ['Nio.py',pynio_vfile]
  pynio_files    = ['Nio.py',pynio_vfile,'coordsel.py','_xarray.py']
else:
  pynio_pkg_name = 'PyNIO'
  pynio_pth_file = [pynio_pkg_name + '.pth']
  pynio_files    = ['Nio.py', '__init__.py','test/nio_demo.py',pynio_vfile,'coordsel.py','xarray.py']

DMACROS =  [ ('NeedFuncProto','1') ]

INCLUDE_DIRS.insert(0,os.path.join(pkgs_pth,"numpy/core/include"))

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
                    include_dirs  = INCLUDE_DIRS,
                    libraries     = LIBRARIES,
                    library_dirs  = lib_paths,
                    sources       = ['niomodule.c']
                    )]

DATA_FILES  = [(pkgs_pth, pynio_pth_file),
               (os.path.join(pkgs_pth,pynio_pkg_name), pynio_files)]

setup (name         = 'Nio',
       version      = pynio_version,
       description  = 'Multi-format data I/O package',
       author       = 'David I. Brown',
       author_email = 'dbrown@ucar.edu',
       url          = 'http://www.pyngl.ucar.edu/Nio.shtml',
       long_description = '''
       Enables NetCDF-like access for NetCDF (rw), HDF (rw), HDFEOS (r), GRIB (r), and CCM (r) data files
       ''',
       package_dir = {pynio_pkg_name : ''},
       ext_modules = module1,
       ext_package = pynio_pkg_name,
       data_files  = DATA_FILES)

#
# Cleanup: remove the pynio_version.py file.
#
if os.path.exists(pynio_vfile):
  os.system("/bin/rm -rf " + pynio_vfile)
