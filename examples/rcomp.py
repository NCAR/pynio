import numpy
import Nio
import time, os

#open a file
fn = "pynio_compound.nc"
file = Nio.open_file(fn, mode="r")

print "file <%s>:" %(fn)
print file

station_data = file.variables['station_data'][:]

print "station_data:"
print  station_data

file.close()

