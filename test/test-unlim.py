
import numpy
import Nio 
import os

#
#  Creating a NetCDF file named "nio04.nc".  If there is already
#  a file with that name, delete it first.
#
filename = "test-unlim.nc"

if (os.path.exists(filename)):
  os.remove(filename)

#
#  Open a NetCDF file for writing file and specify a 
#  global history attribute.
#
file       = Nio.open_file(filename, "w")
file.title = "Unlimited dimension test file"

#
#  Create some dimensions.
#
file.create_dimension("time", None)     # unlimited dimension

print file

#
#  Create a variable of type double with three dimemsions.
#
var = file.create_variable("var", 'd', ("time",))
var._FillValue = -999.0

data = numpy.arange(10)
# assign 5 elements of the unlimited dimension
var[:] = data[0:4]
print len(var[:]),var[:]
# this does not work with NIO 5.0.0
#var[::-1] = data
#print len(var[:]),var[:]
# add 10 elements starting at element 3 - total 13
var[3:] = data
print len(var[:]),var[:]
# prepend single element dimensions to the data array
# but it should still work
data = data[numpy.newaxis,numpy.newaxis,:]
# adding 10 from element 10 = 20
var[10:] = data
print len(var[:]),var[:]

# this does not work with NIO 5.0.0
#var[25:15:-1] = data
#print len(var[:]),var[:]
# add 10 elements spaced 2 elements apart -- resulting in some missing value elements --
# and 39 total elements
var[20::2] = data
print len(var[:]),var[:]

print "After assigning elements of unlimited dimension:"
print file

print "Closing '" + filename + "' file...\n"
file.close()

