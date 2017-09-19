from __future__ import print_function, division
import numpy as np
import Nio

fn = "MSG3-SEVI-MSG15-0100-NA-20130521001244.164000000Z-1074164.h5"
opt = Nio.options()
opt.FileStructure = 'advanced'
f = Nio.open_file(fn, "r", options=opt)
#f = Nio.open_file(fn)
print(list(f.variables.keys()))

#print f.groups
#n = 0
#for key in f.groups.keys():
#    n += 1
#    print "groun %d: <%s>" %(n, key)

#g = f.groups['/U_MARF/MSG/Level1_5/DATA/Channel_07']
g = f.groups['U-MARF/MSG/Level1.5/DATA/Channel 07']
print(g)

palette = g.variables['Palette']
print(palette)

print("\nLineSideInfo_DESCR:")
lsid = g.variables['LineSideInfo_DESCR'][:]
print(lsid[:])
dims = lsid.shape
for n in range(dims[0]):
    name = str(lsid[:][n][0])
    value = str(lsid[:][n][1])

    print("name %3d: %40s, value %3d: %20s" %(n, name, n, value))

print("\nPacketHeader_DESCR:")
phd = g.variables['PacketHeader_DESCR']
print(phd)
dims = phd.shape
for n in range(dims[0]):
    name = str(phd[:][n][0])
    value = str(phd[:][n][1])

    print("name %3d: %25s, value %3d: %40s" %(n, name, n, value))

print("\nPacketHeader_DESCR:")
pha = g.variables['PacketHeader_ARRAY']
print(pha)
dims = pha.shape
for n in range(0, dims[0], 200):
    name = str(pha[:][n][0])
    value = str(pha[:][n][1])

    print("name %5d: %25s, value %5d: %40s" %(n, name, n, value))

lsia = g.variables['LineSideInfo_ARRAY']
print(lsia)
dims = lsia.shape
for n in range(0, dims[0], 200):
    field_0 = str(lsia[:][n][0])
    field_1 = str(lsia[:][n][1])
    field_2 = str(lsia[:][n][2])

    print("No. %5d: field_0: %20s, field_1: %20s, field_2: %20s" %(n, field_0, field_1, field_2))

