#!/bin/csh -f
#$Id: mysetup.csh 15291 2014-05-09 21:31:48Z huangwei $

 set echo

 setenv HAS_ZLIB	1
 setenv HAS_SZIP	1
 setenv HAS_HDF5	1
 setenv HAS_NETCDF4	1
 setenv USE_GFORTRAN	1

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
          setenv SZIP_PREFIX	/opt/local
          setenv HDF5_PREFIX	/usr/local
          breaksw
 endsw

 setenv ZLIB_INCDIR	${PREFIX}/include
 setenv ZLIB_LIBDIR	${PREFIX}/lib
 setenv SZIP_INCDIR	${PREFIX}/include
 setenv SZIP_LIBDIR	${PREFIX}/lib
 setenv HDF5_INCDIR	${PREFIX}/include
 setenv HDF5_LIBDIR	${PREFIX}/lib

 setenv F2CLIBS_PREFIX	/opt/local/lib
 setenv F2CLIBS		gfortran

 python setup.py install \
	--prefix=${PREFIX}

