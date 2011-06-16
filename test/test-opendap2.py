# 
# This is a test for OPeNDAP. It will only work on systems in which NCL
# has OPenDAP capabilities built in.
#
# I'm using what looks like a test opendap server. See:
#
# http://test.opendap.org/opendap/data/nc/contents.html
#
import sys, Nio

if not Nio.__formats__['opendap']:
    print '==========================='
    print 'Optional format OPeNDAP is not enabled in this version of PyNIO'
    print '==========================='
    sys.exit()

url      = "http://test.opendap.org/opendap/data/nc/"
filename = "123.nc"
 
f = Nio.open_file(url + filename)
variables = f.variables.keys()

print f
variables.sort()
print "variables",variables
#
# Note: the variables on the file used to include a coordinate arrays
# "i" and a "j".
#
# The web page for 123.nc indicates that the DAP version doesn't have
# an "i" and "j", but if you download the 123.nc file and do an
# ncl_filedump on it, you see the "i" and "j". I wrote to the Unidata
# folks about this, and they said it was a problem the server, and
# not the NetCDF software.
#
#vars_out = (/"l","cross","aloan","shot","order","bears"/)


