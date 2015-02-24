import numpy
import Nio
import time, os

#open a file
fn = "strings.nc"
file = Nio.open_file(fn, mode="r")

print "file <%s>:" %(fn)
print file

strs = file.variables['universal_declaration_of_human_rights'][:]

print "strs:"
print  strs

file.close()

