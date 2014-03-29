#!/bin/csh -f
#$Id$

 set echo

 setenv HAS_ZLIB	1
 setenv HAS_SZIP	1
 setenv HAS_HDF5	1
 setenv HAS_NETCDF4	1

 setenv CFLAGS		"-O0 -g"

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
         #module refresh gnu/4.7.2
          setenv ZLIB_PREFIX	/Users/huangwei/ncl/lib/zlib/1.2.8
          setenv SZIP_PREFIX	/Users/huangwei/ncl/lib/szip/2.1
          setenv HDF5_PREFIX	/Users/huangwei/ncl/lib/hdf5/1.8.11
          setenv DYLD_LIBRARY_PATH "/Users/huangwei/ncl/lib/szip/2.1/lib:$DYLD_LIBRARY_PATH"
          breaksw
 endsw

 setenv ZLIB_INCDIR	${ZLIB_PREFIX}/include
 setenv ZLIB_LIBDIR	${ZLIB_PREFIX}/lib
 setenv SZIP_INCDIR	${SZIP_PREFIX}/include
 setenv SZIP_LIBDIR	${SZIP_PREFIX}/lib
 setenv HDF5_INCDIR	${HDF5_PREFIX}/include
 setenv HDF5_LIBDIR	${HDF5_PREFIX}/lib

 python setup.py install \
	--prefix=${INSTALL_DIR}


