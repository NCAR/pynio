import numpy 
import Nio 
import time, os

opt = Nio.options()
opt.Format = 'NetCDF4'

print opt.Format

#create a file
hatt = "Created at " + time.ctime(time.time())
fn = "pynio_created.nc"
if(os.path.isfile(fn)):
    os.remove(fn)
file = Nio.open_file(fn, options=opt, history=hatt, mode="w")

#create global attributes
file.source   = "Nio created NetCDF4 file"
#setattr(file, 'source', "Nio test NetCDF file")
file.history = "Created " + time.ctime(time.time())

#Create some groups.
forecast = file.create_group("forecast")
analysis = file.create_group("analysis")

fmdl1 = forecast.create_group("model1")
fmdl2 = forecast.create_group("model2")
amdl1 = analysis.create_group("model1")
amdl2 = analysis.create_group("model2")

ntimes = 5
nlevels = 10
nlats = 73
nlons = 144
#Create some dimensions.
fmdl1.create_dimension("time", None)
fmdl1.create_dimension("level", None)
fmdl1.create_dimension("lat", nlats)
fmdl1.create_dimension("lon", nlons)

mtimes = 1
mlevels = (nlevels+1)/2
mlats = (nlats+1)/2
mlons = (nlons+1)/2

print "mtimes: %d, mlevels: %d, mlats: %d, mlons: %d" %(mtimes, mlevels, mlats, mlons)

#Create chunk dimensions.
fmdl1.create_chunk_dimension("time", mtimes)
fmdl1.create_chunk_dimension("level", mlevels)
fmdl1.create_chunk_dimension("lat", mlats)
fmdl1.create_chunk_dimension("lon", mlons)

#Create some variables.
time  = fmdl1.create_variable("time",  "d", ("time",))
level = fmdl1.create_variable("level", "i", ("level",))
lat   = fmdl1.create_variable("lat",   "f", ("lat",))
lon   = fmdl1.create_variable("lon",   "f", ("lon",))
temp  = fmdl1.create_variable("temp" , "f", ("time", "level", "lat", "lon"))
#temp  = fmdl1.create_variable("temp" , "d", ("time", "level", "lat", "lon"))

print "prepare to add attributes:"

#Specify attributes.
time.units = "hours since 0001-01-01 00:00:00.0"
print "prepare to add attributes: 2"
time.calendar = "gregorian"
level.units = "hPa"
print "prepare to add attributes: 3"
lat.units = "degrees north"
print "prepare to add attributes: 4"
lon.units = "degrees east"
print "prepare to add attributes: 5"
temp.units = "K"
#setattr(fmdl1.variables['temp'], 'units', 'K')

#fill in variables.
time[:] = [0.0, 12.0, 24.0, 36.0, 48.0]
level[:] =  [1000, 850, 700, 500, 300, 250, 200, 150, 100, 50]
latvalues = numpy.arange(-90, 91, 2.5, 'float32')
lonvalues = numpy.arange(-180, 180, 2.5, 'float32')
#print('lat =\n',latvalues[:])
#print('lon =\n',lonvalues[:])

#print('time shape = ', time.shape)
#print('level shape = ', level.shape)
tshape = time.shape
lshape = level.shape
#print('time shape = ', tshape[0])
#print('level shape = ', lshape[0])

#fmdl1.set_dimension("time", tshape[0])
#fmdl1.set_dimension("level", lshape[0])

from numpy.random.mtrand import uniform # random number generator.
tempvalues = uniform(200.0, 300.0, size=(ntimes, nlevels, nlats, nlons))
#print('temp shape = ', tempvalues.shape)
#print('temp dtype = ', tempvalues.dtype)
#print tempvalues
ftempvalues = numpy.ndarray(shape=(ntimes, nlevels, nlats, nlons), dtype=numpy.float32)
#ftempvalues = numpy.array(tempvalues, dtype=numpy.float)
#ftempvalues = tempvalues.view('float32')
#ftempvalues = tempvalues

print('ftemp dtype = ', ftempvalues.dtype)
ftempvalues[:,:,:,:] = tempvalues[:,:,:,:]

print('ftemp shape = ', ftempvalues.shape)
print('ftemp dtype = ', ftempvalues.dtype)
print "max: %f, min: %f" %(numpy.amax(tempvalues), numpy.amin(tempvalues))
print "max: %f, min: %f" %(numpy.amax(ftempvalues), numpy.amin(ftempvalues))

#fmdl1.variables['time'][:] = time
#fmdl1.variables['level'][:] = level
fmdl1.variables['time'].assign_value(time)
fmdl1.variables['level'].assign_value(level)
fmdl1.variables['lat'][:] = latvalues
fmdl1.variables['lon'][:] = lonvalues
#fmdl1.variables['lat'].assign_value(latvalues)
#fmdl1.variables['lon'].assign_value(lonvalues)
fmdl1.variables['temp'][:,:,:,:] = ftempvalues
#fmdl1.variables['temp'].assign_value(ftempvalues)

#test data types
nx = 2
#Create dimension.
amdl1.create_dimension("x", nx)

#Create some variables.
vdbl = amdl1.create_variable("vdbl", "d", ("x",))
vdbl[:] = numpy.array([0.1234456789, 1.2345678901], numpy.double)
amdl1.variables['vdbl'][:] = vdbl

vflt = amdl1.create_variable("vflt", "f", ("x",))
vflt[:] = [0.1234456, 1.234567]
amdl1.variables['vflt'][:] = vdbl

vint64 = amdl1.create_variable("vint64", "q", ("x",))
vint64[:] = numpy.array([-12344567890, 12345678901234], numpy.int64)
amdl1.variables['vint64'][:] = vint64

vuint64 = amdl1.create_variable("vuint64", "Q", ("x",))
vuint64[:] = numpy.array([1234456789, 1234567890123], numpy.uint64)
amdl1.variables['vuint64'][:] = vuint64

vlong = amdl1.create_variable("vlong", "l", ("x",))
vlong[:] = numpy.array([-1234567, 7654321], numpy.int64)
amdl1.variables['vlong'][:] = vlong

vulong = amdl1.create_variable("vulong", "L", ("x",))
vulong[:] = numpy.array([1234567, 87654321], numpy.uint64)
amdl1.variables['vulong'][:] = vulong

vint = amdl1.create_variable("vint", "i", ("x",))
vint[:] = numpy.array([-1234456, 1234567], numpy.int32)
amdl1.variables['vint'][:] = vint

vuint = amdl1.create_variable("vuint", "I", ("x",))
vuint[:] = numpy.array([1234456, 234567], numpy.uint32)
amdl1.variables['vuint'][:] = vuint

vshort = amdl1.create_variable("vshort", "h", ("x",))
vshort[:] = numpy.array([-12345, 12345], numpy.int16)
amdl1.variables['vshort'][:] = vshort

vushort = amdl1.create_variable("vushort", "H", ("x",))
vushort[:] = numpy.array([12345, 23451], numpy.uint16)
amdl1.variables['vushort'][:] = vushort

vbyte = amdl1.create_variable("vbyte", "b", ("x",))
vbyte[:] = numpy.array([-123, 123], numpy.int8)
amdl1.variables['vbyte'][:] = vbyte

vubyte = amdl1.create_variable("vubyte", "B", ("x",))
vubyte[:] = numpy.array([123, 234], numpy.uint8)
amdl1.variables['vubyte'][:] = vubyte

#vchar = amdl1.create_variable("vchar", "c", ("x",))
#vchar[:] = numpy.chararray([23, 123], numpy.chararray)
#amdl1.variables['vchar'][:] = vchar

vlogical = amdl1.create_variable("vlogical", "?", ("x",))
#vlogical[:] = numpy.array([0, 1], numpy.int32)
vlogical[:] = [0, 1]
amdl1.variables['vlogical'][:] = vlogical

vstring = amdl1.create_variable("vstring", "S1", ("x",))
vstring[:] = ["abcdef", "XYZ"]
amdl1.variables['vstring'][:] = vstring

file.close()

