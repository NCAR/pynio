import numpy
import Nio
import time, os

#open a file
fn = "vlen.h5"
file = Nio.open_file(fn, "r")

print "file:"
print file

print file.variables['vlen_var']
var = file.variables['vlen_var'][:]

print "var:"
print var

file.close()

