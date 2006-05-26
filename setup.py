import os,sys
from distutils.core import setup, Extension

HAS_HDFEOS = 1

use_numpy = os.environ.get('USE_NUMPY')
pynio2pyngl = os.environ.get('PYNIO2PYNGL')

# Get version info.

execfile('pynio_version.py')
pynio_version = version

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

if HAS_NUM == 2:
    print '====> building with numpy/arrayobject.h'
elif HAS_NUM == 1:
    print '====> building with Numeric/arrayobject.h'
else:
    print '====> cannot find NumPy or Numeric: cannot proceed'
    exit

ncarg_root = os.getenv("NCARG_ROOT") + '/'
lib_paths = [ ncarg_root + 'lib' ]

ncl_src_dir = '../ni/src/ncl/'
pkgs_pth  = os.path.join(sys.exec_prefix, 'lib', 'python'+sys.version[:3],
            'site-packages')

if HAS_HDFEOS > 0:
    LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','hdfeos','Gctp','g2c']
else:
    LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','g2c']
    

if HAS_NUM == 2:
    DMACROS =  [ ('USE_NUMPY','1'), ('NeedFuncProto','1') ]
elif HAS_NUM == 1:
    DMACROS =  [ ('NeedFuncProto','1') ]
else:
    print "error can't proceed"
    exit
    
if pynio2pyngl:
    print '====> installing to PyNGL directory'
    module1 = Extension('PyNGL/nio',
                        define_macros = DMACROS,
                        include_dirs = [ncl_src_dir,
                                        ncarg_root + 'include'],
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
                        include_dirs = [ncl_src_dir,
                                        ncarg_root + 'include'],
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

