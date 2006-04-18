import os,sys
from distutils.core import setup, Extension

DEBUG = 0
HAS_HDFEOS = 1

try:
    import numpy
    HAS_NUM = 2
except ImportError:
    try:
        import Numeric
        HAS_NUM = 1
    except ImportError:
        HAS_NUM = 0
        pass

# force Numeric for now

if HAS_NUM > 0:
    HAS_NUM = 1

#end of code to force Numeric

extra_lib_paths = '/sw/lib'
ncarg_root = os.getenv("NCARG_ROOT") + '/'
#python_include = os.path.join(sys.prefix,'include','python'+sys.version[:3]) + '/'
ncl_src_dir = '../ni/src/ncl/'
pkgs_pth  = os.path.join(sys.exec_prefix, 'lib', 'python'+sys.version[:3],
            'site-packages')


if HAS_HDFEOS > 0:
    LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','hdfeos','Gctp','g2c']
else:
    LIBRARIES = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','g2c']
    
if DEBUG:
    if HAS_NUM == 2:
        DMACROS =  [ ('CCOPTIONS','-g'), ('NDEBUG','0'), ('NUMPY','1'), ('NeedFuncProto','1') ]
    elif HAS_NUM == 1:
        DMACROS =  [ ('CCOPTIONS','-g'), ('NDEBUG','0'), ('NeedFuncProto','1') ]
    else:
        print "error can't proceed"
        exit
    module1 = Extension('PyNIO/Nio',
                        extra_link_args = [ '-g' ],
                        extra_compile_args = [ '-O0' ],
                        define_macros = DMACROS,
                        include_dirs = [ncl_src_dir,
                                        ncarg_root + 'include'],
                        libraries = LIBRARIES,
                        library_dirs = [ncarg_root + 'lib',
                                        extra_lib_paths],
                        sources = ['niomodule.c']
                        )
else:
    if HAS_NUM == 2:
        DMACROS =  [ ('NUMPY','1'), ('NeedFuncProto','1') ]
    elif HAS_NUM == 1:
        DMACROS =  [ ('NeedFuncProto','1') ]
    else:
        print "error can't proceed"
        exit
    module1 = Extension('PyNIO/Nio',
                        define_macros = DMACROS,
                        include_dirs = [ncl_src_dir,
                                        ncarg_root + 'include'],
                        libraries = LIBRARIES,
                        library_dirs = [ncarg_root + 'lib',
                                        extra_lib_paths],
                        sources = ['niomodule.c']
                        )

setup (name = 'Nio',
       version = '0.1.1b1',
       description = 'Multi-format data I/O package',
       author = 'David I. Brown',
       author_email = 'dbrown@ucar.edu',
       url = 'http://www.pyngl.ucar.edu',
       long_description = '''
Enables NetCDF-like access for NetCDF (rw), HDF (rw), GRIB (r), and CCM (r) data files
''',
       package_dir = {'PyNIO' : ''},
       ext_modules = [module1],
       data_files = [ (pkgs_pth, ["PyNIO.pth"]),
                      (pkgs_pth + '/PyNIO/test', ["test/nio_demo.py"])
                      ]
       )

