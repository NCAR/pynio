import numpy as np
import Nio

fn = "ncl_wrt_string.h5"
opt = Nio.options()
opt.FileStructure = 'advanced'
f = Nio.open_file(fn, "r", options=opt)
print f

h5str = f.variables['h5_string'][:]
print h5str
#print h5str[:]

