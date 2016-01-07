#!/bin/csh -f
#$Id: mysetup.csh 16178 2015-04-12 16:54:47Z huangwei $

 set echo

 setenv HAS_ZLIB	1
 setenv HAS_SZIP	1
 setenv HAS_HDF5	1
 setenv USE_GFORTRAN	1
 setenv HAS_NETCDF4	1
 setenv HAS_HDF4	1
 setenv HAS_HDFEOS	1
 setenv HAS_HDFEOS5	1
 setenv HAS_GRIB2	1
 setenv HAS_GDAL	1

 setenv CFLAGS		"-O0 -g"

 ~/bin/cleanmyipy

 set mn = `uname -n | cut -c -2`
 switch($mn)
     case ys:
     case ge:
         #module refresh gnu/4.7.2
         #setenv ZLIB_PREFIX	/glade/p/work/huangwei/lib/zlib/1.2.8
         #setenv SZIP_PREFIX	/glade/p/work/huangwei/lib/szip/2.1
         #setenv HDF5_PREFIX	/glade/p/work/huangwei/lib/hdf5/1.8.11

          setenv ZLIB_PREFIX	/glade/p/work/haley/dev/external/gnu/4.7.2
          setenv SZIP_PREFIX	/glade/p/work/haley/dev/external/gnu/4.7.2
          setenv HDF5_PREFIX	/glade/p/work/haley/dev/external/gnu/4.7.2
	  setenv INSTALL_DIR	/glade/p/work/huangwei

          breaksw
     default:
          setenv ZLIB_PREFIX	/usr/local
          setenv SZIP_PREFIX	/usr/local
          setenv HDF5_PREFIX	/usr/local

	  setenv ZLIB_INCDIR	/usr/local/include
	  setenv ZLIB_LIBDIR	/usr/local/lib
	  setenv SZIP_INCDIR	/usr/local/include
	  setenv SZIP_LIBDIR	/usr/local/lib
	  setenv HDF5_INCDIR	/usr/local/include
	  setenv HDF5_LIBDIR	/usr/local/lib
	  setenv F2CLIBS_PREFIX	/opt/local/lib/gcc49
	  setenv F2CLIBS	gfortran

 	  python setup.py install --prefix=/usr/local
          breaksw
 endsw

