import numpy as np
import Nio

fn = "MSG3-SEVI-MSG15-0100-NA-20130521001244.164000000Z-1074164.hdf5"
opt = Nio.options()
opt.FileStructure = 'advanced'
f = Nio.open_file(fn, "r", options=opt)
#print f

#print f.groups
#n = 0
#for key in f.groups.keys():
#    n += 1
#    print "groun %d: <%s>" %(n, key)

g = f.groups['U_MARF/MSG/Level1_5/DATA/Channel_07']
#print g

#palette = g.variables['Palette']
#print palette

lsid = g.variables['LineSideInfo_DESCR'][:]
#print lsid
dims = lsid.shape
print "lsid.shape = ", lsid.shape
print "dims[0] = ", dims[0]

for n in xrange(dims[0]):
    name = str(lsid[:][n][0])
    value = str(lsid[:][n][1])

    print "name %d: <%s>, value %d: <%s>" %(n, name, n, value)
