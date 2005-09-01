import os,sys
from distutils.core import setup, Extension

extra_lib_paths = '/sw/lib'
ncarg_root = os.getenv("NCARG_ROOT") + '/'
python_include = os.path.join(sys.prefix,'include','python'+sys.version[:3]) + '/'
ncl_src_dir = '../ni/src/ncl/'

module1 = Extension('Nio',
                    define_macros = [ ('CCOPTIONS','-g'), ('NDEBUG','0') ],
                    extra_link_args = [ '-g' ],
		    extra_compile_args = [ '-O0' ],
                    include_dirs = [ncl_src_dir,
                                    ncarg_root + 'include',
                                    python_include +  'Numeric'],
                    libraries = ['nio','mfhdf', 'df', 'jpeg','z','netcdf','g2c'],
                    library_dirs = [ncarg_root + 'lib',
                                    extra_lib_paths],
                    sources = ['niomodule.c'])

setup (name = 'Nio',
       version = '0.1.1b1',
       description = 'Multi-format data I/O package',
       author = 'David I. Brown',
       author_email = 'dbrown@ucar.edu',
       url = 'http://www.pyngl.ucar.edu',
       long_description = '''
Enables NetCDF-like access for NetCDF (rw), HDF (rw), GRIB (r), and CCM (r) data files
''',
       ext_modules = [module1])
