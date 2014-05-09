import numpy
import Nio
import time, os

opt = Nio.options()
opt.Format = 'NetCDF4'

print opt.Format

#create a file
hatt = "Created at " + time.ctime(time.time())
fn = "pynio_vlen.nc"
if(os.path.isfile(fn)):
    os.remove(fn)
file = Nio.open_file(fn, options=opt, history=hatt, mode="w")

#create global attributes
file.source   = "Nio created NetCDF4 vlen file"
#setattr(file, 'source', "Nio test NetCDF file")
file.history = "Created " + time.ctime(time.time())

print "file after add attributes:"
print file

nx = 3
ny = 4

x = file.create_dimension('x', nx)
y = file.create_dimension('y', ny)

print "file after add dimensions:"
print file

data = numpy.empty(ny*nx, object)
m = 0
for n in range(ny*nx):
    m += 1
    data[n] = numpy.arange(m, dtype='int32')+1
data = numpy.reshape(data,(ny, nx))

print "data", data

vlvar = file.create_vlen('vlen_var', 'i', ('y','x'))
#vlvar[:, :] = data
vlvar[:] = data

print "file after add vlen:"
print file

#print vlvar
#print 'vlen variable =\n',vlvar[:]
#print file
#print file.variables['phony_vlen_var']
#print file.vltypes['phony_vlen']

file.close()

