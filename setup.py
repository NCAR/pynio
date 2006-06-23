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
  path = os.environ["USE_NUMERPY"]
  HAS_NUM = 3
except:
  try:
    path = os.environ["USE_NUMPY"]
    HAS_NUM = 2
  except:
    HAS_NUM = 1

#
# Test to make sure we actually have what we say we have.
#
if HAS_NUM > 1:
  try:
    import numpy
  except ImportError:
    try:
      print "Cannot find numpy; we'll try Numeric"
      HAS_NUM = 1
    except ImportError:
      print "Cannot find Numeric or numpy; good-bye!"
      exit

if HAS_NUM == 1 or HAS_NUM == 3:
  try:
    import Numeric
  except ImportError:
    HAS_NUM = HAS_NUM-1
    if HAS_NUM == 0:
      print "Cannot find Numeric or numpy; good-bye!"
      exit

#
# Create pynio_version.py file that contains version and
# array module info.
#
pynio_vfile = "pynio_version.py"
os.system("/bin/rm -rf " + pynio_vfile)

pynio_version = open('version','r').readlines()[0].strip('\n')

vfile = open(pynio_vfile,'w')
vfile.write("version = '%s'\n" % pynio_version)

if HAS_NUM == 2:
    vfile.write("HAS_NUM = 2\n")
    print '====> building with numpy/arrayobject.h'
    from numpy import __version__ as array_module_version
    vfile.write("array_module = 'numpy'\n")
else:
  vfile.write("HAS_NUM = 1\n")
  from Numeric import  __version__ as array_module_version
  vfile.write("array_module = 'Numeric'\n")

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
    print '====> building with numpy/arrayobject.h'
    DMACROS =  [ ('USE_NUMPY','1'), ('NeedFuncProto','1') ]
    include_paths.insert(0,os.path.join(pkgs_pth,"numpy/core/include"))
elif HAS_NUM == 1:
    print '====> building with Numeric/arrayobject.h'
    DMACROS =  [ ('NeedFuncProto','1') ]
else:
    print '====> building with Numeric and numpy arrayobject.h'
    DMACROS =  [ ('NeedFuncProto','1') ]
    DNUMPYMACROS =  [ ('USE_NUMPY','1'), ('NeedFuncProto','1') ]
    include_numpy_paths = [os.path.join(pkgs_pth,"numpy/core/include"),
                           ncl_src_dir, ncarg_root + 'include']
    
if pynio2pyngl:
    if HAS_NUM == 1 or HAS_NUM == 3:
      ext_dir = 'PyNGL'
    elif HAS_NUM == 2:
      ext_dir = 'PyNGL_numpy'

    print '====> installing to ' + ext_dir + ' directory'
    module1 = Extension(ext_dir + '/nio',
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
           package_dir = {ext_dir : ''},
           ext_modules = [module1],
           data_files = [ (pkgs_pth + '/' + ext_dir, ["Nio.py"]),
                          (pkgs_pth + '/' + ext_dir, ["__init__.py"]),
                          (pkgs_pth + '/' + ext_dir, [pynio_vfile])
                          ]
           )
else:
    if HAS_NUM == 1 or HAS_NUM == 3:
      ext_dir = 'PyNIO'
    elif HAS_NUM == 2:
      ext_dir = 'PyNIO_numpy'

    print '====> installing to ' + ext_dir + ' directory'
    module1 = Extension(ext_dir + '/nio',
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
           package_dir = {ext_dir : ''},
           ext_modules = [module1],
           data_files = [ (pkgs_pth, ["PyNIO.pth"]),
                          (pkgs_pth + '/' + ext_dir, ["Nio.py"]),
                          (pkgs_pth + '/' + ext_dir, ["__init__.py"]),
                          (pkgs_pth + '/' + ext_dir, [pynio_vfile]),
                          (pkgs_pth + '/' + ext_dir + '/test', ["test/nio_demo.py"])
                          ]
           )

#
# if HAS_NUM is 3, then this means we just created a Numeric
# version of PyNIO, and now we need to create a numpy version.
#

if HAS_NUM == 3:
#
# Create a new pynio_version.py file that contains version and
# array module info for numpy.
#
  os.system("/bin/rm -rf " + pynio_vfile)

  vfile = open(pynio_vfile,'w')
  vfile.write("version = '%s'\n" % pynio_version)
  vfile.write("HAS_NUM = 2\n")
  from numpy import __version__ as array_module_version
  vfile.write("array_module = 'numpy'\n")
  vfile.write("array_module_version = '%s'\n" % array_module_version)
  vfile.close()

#
# Start with fresh build.
#
  os.system("find build -name '*.o' -exec /bin/rm {} \;")

  if pynio2pyngl:
    ext_dir = 'PyNGL_numpy'

    print '====> installing to ' + ext_dir + ' directory'
    module1 = Extension(ext_dir + '/nio',
                        define_macros = DNUMPYMACROS,
                        include_dirs = include_numpy_paths,
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
           package_dir = {ext_dir : ''},
           ext_modules = [module1],
           data_files = [ (pkgs_pth + '/' + ext_dir, ["Nio.py"]),
                          (pkgs_pth + '/' + ext_dir, ["__init__.py"]),
                          (pkgs_pth + '/' + ext_dir, [pynio_vfile])
                          ]
           )
  else:
    ext_dir = 'PyNIO_numpy'

    print '====> installing to ' + ext_dir + ' directory'
    module1 = Extension(ext_dir + '/nio',
                        define_macros = DNUMPYMACROS,
                        include_dirs = include_numpy_paths,
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
           package_dir = {ext_dir : ''},
           ext_modules = [module1],
           data_files = [ (pkgs_pth + '/' + ext_dir, ["Nio.py"]),
                          (pkgs_pth + '/' + ext_dir, [pynio_vfile]),
                          (pkgs_pth + '/' + ext_dir, ["__init__.py"]),
                          (pkgs_pth + '/' + ext_dir + '/test', ["test/nio_demo.py"])
                          ]
           )

