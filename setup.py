import os,sys
from distutils.core import setup, Extension

try:
  HAS_HDFEOS = int(os.environ["HAS_HDFEOS"])
except:
  HAS_HDFEOS = 1

try:
  path = os.environ["PYNIO2PYNGL"]
  pynio2pyngl = True
except:
  pynio2pyngl = False

#
# Determine whether we want to build a Numeric and/or Numpy version
# of PyNIO.  If the environment variable USE_NUMPY is set, it will
# try to build a NumPy version. USE_NUMPY doesn't need to be set to
# any value; it just has to be set.  If USE_NUMERPY is set, then
# both versions of PyNIO will be built, and the Numeric version will
# be put in package PyNIO, and the numpy version in package PyNIO_numpy.
#
# HAS_NUM will be set by this script depending on USE_NUMPY and USE_NUMERPY.
#
# HAS_NUM = 3 --> install both numpy and Numeric versions of module
# HAS_NUM = 2 --> install numpy version of module
# HAS_NUM = 1 --> install Numeric version of module
# HAS_NUM = 0 --> You're hosed, you have neither module
#
try:
  path = os.environ["USE_NUMERPY"]
  HAS_NUM = 3
except:
  try:
    path = os.environ["USE_NUMPY"]
    HAS_NUM = 2
  except:
    HAS_NUM = 1

#
# Test to make sure we actually the Numeric and/or numpy modules
# that we have requested.
#
if HAS_NUM > 1:
  try:
    import numpy
  except ImportError:
    print "Cannot find numpy; we'll try Numeric."
    HAS_NUM = 1

if HAS_NUM == 1 or HAS_NUM == 3:
  try:
    import Numeric
  except ImportError:
    print "Cannot find Numeric."
    HAS_NUM = HAS_NUM-1

if HAS_NUM == 3:
  array_modules = ['Numeric','numpy']
elif HAS_NUM == 2:
  array_modules = ['numpy']
elif HAS_NUM == 1:
  array_modules = ['Numeric']
else:
  print "Cannot find Numeric or numpy; good-bye!"
  exit

#
# Initialize some variables.
#
pynio_vfile = "pynio_version.py"      # PyNIO version file.

ncarg_root = os.getenv("NCARG_ROOT")
lib_paths = [ os.path.join(ncarg_root,'lib') ]

ncl_src_dir = '../ni/src/ncl/'
pkgs_pth  = os.path.join(sys.exec_prefix, 'lib', 'python'+sys.version[:3],
            'site-packages')

if HAS_HDFEOS > 0:
    LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','hdfeos','Gctp','grib2c','jasper','g2c']
else:
    LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','grib2c','jasper','g2c']
    
#
# The long path below is for the g95 compiler on the Intel Mac.
#
if sys.platform == "darwin":
    dirs = ['/sw/lib','/Users/haley/lib/gcc-lib/i386-apple-darwin8.6.1/4.0.3']
    for dir in dirs:
      if(os.path.exists(dir)):
        lib_paths.append(dir)

if sys.platform == "irix6-64":
    LIBRARIES.remove('g2c')
    LIBRARIES.append('ftn')
    LIBRARIES.append('fortran')
    LIBRARIES.append('sz')

if sys.platform == "sunos5":
    os.environ["CC"]="/opt/SUNWspro/bin/cc"
    LIBRARIES.remove('g2c')
    LIBRARIES.append('f77compat')
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

INCLUDE_DIRS = [ncl_src_dir, os.path.join(ncarg_root,'include')]

#----------------------------------------------------------------------
#
# Loop through the modules for which we want to create versions of PyNIO.
#
#----------------------------------------------------------------------

for array_module in array_modules:
#----------------------------------------------------------------------
#
# Set some variables based on whether we're doing a numpy or Numeric
# build.
#
#----------------------------------------------------------------------
  if array_module == 'Numeric':
    from Numeric import  __version__ as array_module_version

    if pynio2pyngl:
      pynio_pkg_name = 'PyNGL'
      pynio_files    = ['Nio.py',pynio_vfile]
      pynio_pth_file = []
    else:
      pynio_pkg_name = 'PyNIO'
      pynio_pth_file = [pynio_pkg_name + '.pth']
      pynio_files    = ['Nio.py', '__init__.py','test/nio_demo.py',pynio_vfile]

    DMACROS =  [ ('NeedFuncProto','1') ]

  else:
    from numpy import __version__ as array_module_version

    if pynio2pyngl:
      pynio_pkg_name = 'PyNGL_numpy'
      pynio_files    = ['Nio.py',pynio_vfile]
      pynio_pth_file = []
    else:
      pynio_pkg_name = 'PyNIO_numpy'
      pynio_files    = ['Nio.py', '__init__.py',pynio_vfile]
      pynio_pth_file = []

    DMACROS =  [ ('USE_NUMPY','1'), ('NeedFuncProto','1') ]

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
  vfile.write("array_module = '%s'\n" % array_module)

#
# The Ngl.py and Nio.py files use HAS_NUM to tell whether to use
# Numeric or numpy specific operations.
#
  if array_module == 'Numeric':
    vfile.write("HAS_NUM = 1\n")
  else:
    vfile.write("HAS_NUM = 2\n")

  vfile.write("array_module_version = '%s'\n" % array_module_version)
  vfile.close()

#----------------------------------------------------------------------
#
# Here are the instructions for compiling the "nio.so" file.
#
#----------------------------------------------------------------------
  print '====> Installing the',array_module,'version of Nio to the',pynio_pkg_name,'package directory.'

  module1 = [Extension('nio',
                      define_macros = DMACROS,
                      include_dirs  = INCLUDE_DIRS,
                      libraries     = LIBRARIES,
                      library_dirs  = lib_paths,
                      sources       = ['niomodule.c']
                      )]

  DATA_FILES  = [(pkgs_pth, pynio_pth_file),
                 (os.path.join(pkgs_pth,pynio_pkg_name), pynio_files)]

#----------------------------------------------------------------------
#
# Clean *.o files if doing multiple builds here.
#
#----------------------------------------------------------------------
  if len(array_modules) > 1:
    print "====> Removing build's *.o and *.so files..."
    os.system("find build -name '*.o' -exec /bin/rm {} \;")
    os.system("find build -name '*.so' -exec /bin/rm {} \;")

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
