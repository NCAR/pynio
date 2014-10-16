import numpy
import Nio
import time, os

#open a file
#fn = "examples/vlen.nc"
fn = "pynio_vlen.nc"
file = Nio.open_file(fn, "r")

print "file:"
print file

var = file.variables['vlen_var'][:]

print "var:"
print var

file.close()

