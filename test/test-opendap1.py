import sys, Nio

# 
# This is a test for OPeNDAP. It will only work on systems in which 
# PyNIO has OPenDAP capabilities built in.

if not Nio.__formats__['opendap']:
    print '==========================='
    print 'Optional format OPeNDAP is not enabled in this version of PyNIO'
    print '==========================='
    sys.exit()

#
# url = "http://apdrc.soest.hawaii.edu:80/dods/public_data/ERA-40/"
# filename = "daily-pressure"
#
# The URL is so long, break it into two pieces.
#
url      = "http://apdrc.soest.hawaii.edu:80/dods/public_data/ERA-40/"
filename = "daily-surface"
 
f = Nio.open_file(url + filename)
variables = f.variables.keys()
variables.sort()
print "variables",variables


