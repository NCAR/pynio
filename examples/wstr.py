from __future__ import print_function, division
import numpy
import Nio
import time, os

#open a file
fn = "strings.nc"
file = Nio.open_file(fn, mode="r")

print("file <%s>:" %(fn))
print(file)

varname = 'universal_declaration_of_human_rights'
#strs = file.variables['universal_declaration_of_human_rights'][:]
strs = file.variables[varname].get_value()

#print "strs:"
#print  strs

file.close()

opt = Nio.options()
opt.Format = 'NetCDF4'

#print opt.Format

#create a file
fatt = "Created at " + time.ctime(time.time())
fn = "new_strings.nc"
if(os.path.isfile(fn)):
    os.remove(fn)

strf = Nio.open_file(fn, options=opt, history=fatt, mode="w")
strf.source   = "Nio created NetCDF4 string variable"
#setattr(file, 'source', "Nio test file")
strf.history = "Created " + time.ctime(time.time())

#print "strf after add attributes:"
#print strf

print("len(strs) = ", len(strs))

nstrings = strf.create_dimension('nstrings', len(strs))

newstrs = strf.create_variable("newstrs", "S1", ("nstrings", ))

newstrs[:] = strs[:]

#print "strf after add strs:"
#print strf

strf.close()

