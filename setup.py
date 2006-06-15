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

try:
  path = os.environ["USE_NUMPY"]
  use_numpy = True
except:
  use_numpy = False

if use_numpy:
    try:
        import numpy
        HAS_NUM = 2
    except ImportError:
        try:
            print 'cannot find NumPy; defaulting to Numeric'
            import Numeric
            HAS_NUM = 1
        except ImportError:
            HAS_NUM = 0
else:
        try:
            import Numeric
            HAS_NUM = 1
        except ImportError:
            HAS_NUM = 0

#
# Create pynio_version.py file that contains version and
# array module info.
#
os.system("/bin/rm -rf pynio_version.py")

pynio_version = open('version','r').readlines()[0].strip('\n')
vfile = open('pynio_version.py','w')
vfile.write("version = '%s'\n" % pynio_version)
vfile.write("HAS_NUM = %d\n" % HAS_NUM)

if HAS_NUM == 2:
    print '====> building with numpy/arrayobject.h'
    from numpy import __version__ as array_module_version
    vfile.write("array_module = 'numpy'\n")
elif HAS_NUM == 1:
    print '====> building with Numeric/arrayobject.h'
    from Numeric import  __version__ as array_module_version
    vfile.write("array_module = 'Numeric'\n")
else:
    print '====> cannot find NumPy or Numeric: cannot proceed'
    exit

vfile.write("array_module_version = '%s'\n" % array_module_version)
vfile.close()

ncarg_root = os.getenv("NCARG_ROOT") + '/'
lib_paths = [ ncarg_root + 'lib' ]

if sys.platform == "darwin":
    lib_paths.append('/sw/lib')

ncl_src_dir = '../ni/src/ncl/'
pkgs_pth  = os.path.join(sys.exec_prefix, 'lib', 'python'+sys.version[:3],
            'site-packages')

if HAS_HDFEOS > 0:
    LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','hdfeos','Gctp','g2c']
else:
    LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','g2c']
    
if sys.platform == "irix6-64":
    LIBRARIES.remove('g2c')
    LIBRARIES.append('ftn')
    LIBRARIES.append('fortran')
    LIBRARIES.append('sz')

if sys.platform == "aix5":
    os.putenv('OBJECT_MODE',"64")
    LIBRARIES.remove('g2c')
    LIBRARIES.append('xlf90')
    
include_paths = [ncl_src_dir, ncarg_root + 'include']

if HAS_NUM == 2:
    DMACROS =  [ ('USE_NUMPY','1'), ('NeedFuncProto','1') ]
    include_paths.insert(0,os.path.join(pkgs_pth,"numpy/core/include"))
elif HAS_NUM == 1:
    DMACROS =  [ ('NeedFuncProto','1') ]
else:
    print "error can't proceed"
    exit
    
if pynio2pyngl:
    print '====> installing to PyNGL directory'
    module1 = Extension('PyNGL/nio',
                        define_macros = DMACROS,
                        include_dirs = include_paths,
                        libraries = LIBRARIES,
                        library_dirs = lib_paths,
                        sources = ['niomodule.c']
                        )
    setup (name = 'Nio',
           version = pynio_version,
           description = 'Multi-format data I/O package',
           author = 'David I. Brown',
           author_email = 'dbrown@ucar.edu',
           url = 'http://www.pyngl.ucar.edu/Nio.shtml',
           long_description = '''
           Enables NetCDF-like access for NetCDF (rw), HDF (rw), HDFEOS (r), GRIB (r), and CCM (r) data files
           ''',
           package_dir = {'PyNGL' : ''},
           ext_modules = [module1],
           data_files = [ (pkgs_pth + '/PyNGL', ["Nio.py"]),
                          (pkgs_pth + '/PyNGL', ["pynio_version.py"])
                          ]
           )
else:
    print '====> installing to PyNIO directory'
    module1 = Extension('PyNIO/nio',
                        define_macros = DMACROS,
                        include_dirs = include_paths,
                        libraries = LIBRARIES,
                        library_dirs = lib_paths,
                        sources = ['niomodule.c']
                        )
    setup (name = 'Nio',
           version = pynio_version,
           description = 'Multi-format data I/O package',
           author = 'David I. Brown',
           author_email = 'dbrown@ucar.edu',
           url = 'http://www.pyngl.ucar.edu/Nio.html',
           long_description = '''
           Enables NetCDF-like access for NetCDF (rw), HDF (rw), HDFEOS (r), GRIB (r), and CCM (r) data files
           ''',
           package_dir = {'PyNIO' : ''},
           ext_modules = [module1],
           data_files = [ (pkgs_pth, ["PyNIO.pth"]),
                          (pkgs_pth + '/PyNIO', ["Nio.py"]),
                          (pkgs_pth + '/PyNIO', ["pynio_version.py"]),
                          (pkgs_pth + '/PyNIO/test', ["test/nio_demo.py"])
                          ]
           )

