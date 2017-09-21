from __future__ import print_function, division
import numpy
import Nio
import time, os

opt = Nio.options()
opt.Format = 'NetCDF4'

print(opt.Format)

#create a file
hatt = "Created at " + time.ctime(time.time())
fn = "pynio_compound.nc"
if(os.path.isfile(fn)):
    os.remove(fn)
file = Nio.open_file(fn, options=opt, history=hatt, mode="w")

#create global attributes
file.source   = "Nio created NetCDF4 compound variable"
#setattr(file, 'source', "Nio test file")
file.history = "Created " + time.ctime(time.time())

print("file after add attributes:")
print(file)

#create an unlimited  dimension call 'station'
station = file.create_dimension('station', None)
nstations = 10

print("station:")
print(station)

print("nstations:")
print(nstations)

#define a compound data type (can contain arrays, or nested compound types).
NUMCHARS = 80 # number of characters to use in fixed-length strings.
station_datatype = numpy.array([['latitude', 'f'],
                                ['longitude', 'f'],
                                #['location_name', 'S1', NUMCHARS],
                                ['location_name', 'c', NUMCHARS],
                                ['speed', 'f'],
                                ['direction', 'i'],
                                ['temp_sounding', 'f', 10],
                                ['press_sounding', 'i', 10]])

print("len(station_datatype) = ", len(station_datatype))
print("station_datatype:")
print(station_datatype)


#now that station_datatype is defined, create the station data type.
station_data = file.create_compound('station_data', station_datatype, ('station',))

#create nested compound data types to hold the units variable attribute.
station_units = numpy.array([['latitude', 'S1', NUMCHARS],
                             ['longitude', 'S1', NUMCHARS],
                             ['location_name','S1', NUMCHARS],
                             ['speed','S1', NUMCHARS],
                             ['direction','S1', NUMCHARS],
                             ['temp_sounding','S1', NUMCHARS],
                             ['press_sounding','S1', NUMCHARS]])

#create the wind_data_units type first, since it will nested inside
#the station_data_units data type.
#station_units_t = file.create_compoundtype(station_units, 'station_data_units')

#create a numpy structured array, assign data to it.
#data = numpy.empty(len(station_datatype), object)
data = {}
data['latitude'] = 40.
data['longitude'] = -105.
data['location_name'] = "Boulder, Colorado, USA'"
data['speed'] = 12.5
data['direction'] = 270
data['temp_sounding'] = [280.3,272.,270.,269.,266.,258.,254.1,250.,245.5,240.]
data['press_sounding'] = list(range(800,300,-50))

station = numpy.empty(2, "object")

#assign structured array to variable slice.
#station[0] = data

#or just assign a tuple of values to variable slice
#(will automatically be converted to a structured array).
station[0] = [[40.],[-105.],
              ["Boulder, Colorado, USA"],
              [12.5],[270],
              [280.3,272.,270.,269.,266.,258.,254.1,250.,245.5,240.],
              list(range(800,300,-50))]
station[1] = [[40.78],[-73.99],
              ["New York, New York, USA"],
              [-12.5],[90],
              [290.2,282.5,279.,277.9,276.,266.,264.1,260.,255.5,243.],
              list(range(900,400,-50))]
#print(f.cmptypes)
#windunits = numpy.empty(1,winddtype_units)
#stationobs_units = numpy.empty(1,statdtype_units)
#windunits['speed'] = stringtoarr('m/s',NUMCHARS)
#windunits['direction'] = stringtoarr('degrees',NUMCHARS)
#stationobs_units['latitude'] = stringtoarr('degrees north',NUMCHARS)
#stationobs_units['longitude'] = stringtoarr('degrees west',NUMCHARS)
#stationobs_units['surface_wind'] = windunits
#stationobs_units['location_name'] = stringtoarr('None', NUMCHARS)
#stationobs_units['temp_sounding'] = stringtoarr('Kelvin',NUMCHARS)
#stationobs_units['press_sounding'] = stringtoarr('hPa',NUMCHARS)
#statdat.units = stationobs_units
print (station)
station_data[:] = station

print("file after add compounden:")
print(file)

print("station_data:")
print(station)

file.close()

