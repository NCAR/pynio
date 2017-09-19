from __future__ import print_function, division
#
#  File:
#    nio04.py
#
#  Synopsis:
#    Gives some examples of using Nio's extended selection mechanism
#
#  Category:
#    Processing.
#
#  Author:
#    Dave Brown
#
#  Date of original publication:
#    August, 2008
#
#  Description:
#    This example reads an example NetCDF file "nio-ex5.nc" and
#    performs a number of subselections 
#
#  Effects illustrated:
#    Use of various features of the extended selection mechanism
#
#  Output:
#    None
#
#  Notes:
#

import numpy
import Nio

f = Nio.open_file("nio-ex5.nc")
print(f)

# get reference to temperature variable

tmp = f.variables['tmp']

# Print enough of the coordinate variables to get the spacing and direction,
print(f.variables['lon'][:4])
print(f.variables['lat'][:4])

# All of the level and time values,
print(f.variables['lev'][:])
print(f.variables['time'][:])
print("")

# Get temperature for the first time step and levels 1000 and 100000, latitude 60 and longitude 100-120.
# Use positiional syntax; the closest value to specified coordinate is selected.

print("tmp['i0 1000,100000 60 100:120']")
print(tmp['i0 1000,100000 60 100:120'])
print("")

# Same thing but specify the dimensions by name.

print("tmp['time|i0 lev|1000,100000 lat|60 lon|100:120']")
print(tmp['time|i0 lev|1000,100000 lat|60 lon|100:120'])
print("")

# Now rearrange the dimension order;
# For the longitudes 100 and 120 get all the level values as the rightmost dimension

print("tmp['time|i0 lat|60 lon|100,120 lev|:']")
print(tmp['time|i0 lat|60 lon|100,120 lev|:'])
print("")

# Suppose you need to set some of the values programmatically using variables that
# have been defined in your code. Although you cannot directly introduce variables into
# the specification string, you can use Python's string formatting syntax to put the values
# in the correct location in the string. For instance, suppose you have variables minlon
# and maxlon that you want to insert into the previous example. Here is one way: 

print("minlon = 100")
minlon = 100
print("maxlon = 120")
maxlon = 120        
print("tmp['time|i0 lat|60 lon|%f,%f lev|:' % (minlon,maxlon)]")
print(tmp['time|i0 lat|60 lon|%f,%f lev|:' % (minlon,maxlon)])

# Interpolate the level values from 0 to 100000 in steps of 10000.
# Use 'k' as a short form multiplier of 1000.
# Note that a minor amount of extrapolation can occur near the limits of the coordinate range.

print("tmp['time|i0 lat|60 lon|100,120 lev|0:100k:10ki']")
print(tmp['time|i0 lat|60 lon|100,120 lev|0:100k:10ki'])
print("")

# Same as above, but reverse the level axis 

print("tmp['time|i0 lat|60 lon|100,120 lev|100k:0:-10ki']")
print(tmp['time|i0 lat|60 lon|100,120 lev|100k:0:-10ki'])
print("")

# Interpolate the level values from 0 to 120000 in steps of 10000.
# Values outside the extrapolation range get set to the bounding value.

print("tmp['time|i0 lat|60 lon|100,120 lev|0:120k:10ki']")
print(tmp['time|i0 lat|60 lon|100,120 lev|0:120k:10ki'])
print("")

# Interpolate the level values from 0 to 120000 in steps of 10000
# Use the 'm' flag to indicate that values outside the bounding array
# should be set to missing values. Show the mask.

print("tmp['time|i0 lat|60 lon|100,120 lev|0:120k:10kmi']")
print(tmp['time|i0 lat|60 lon|100,120 lev|0:120k:10kmi'])
print(tmp['time|i0 lat|60 lon|100,120 lev|0:120k:10kmi'].mask)
print("")

# Insted of interpolating get the nearest level values from 0 to 100000 in steps of 10000
# This does not work well because of the uneven spacing of the level coordinate values.
# What seems to happen is that the spacing of the first 2 coordinate values and the
# specified spacing is used to calculate a constant stride in index space. What should
# happen is repeated values when adjacent steps resolve to the same element. This
# is a bug.

print("tmp['time|i0 lat|60 lon|100,120 lev|0:100k:10k']")
print(tmp['time|i0 lat|60 lon|100,120 lev|0:100k:10k'])
print("")

# An long-winded workaround would be to use vector subscripting

print("tmp['time|i0 lat|60 lon|100,120 lev|0k,10k,20k,30k,40k,50k,60k,70k,80k,90k,100k']")
print(tmp['time|i0 lat|60 lon|100,120 lev|0k,10k,20k,30k,40k,50k,60k,70k,80k,90k,100k'])
print("")

# Get temperature for the first time step and level, latitudes 30 - 40 and longitude 100.
# Use positiional syntax - note lat coordinates are in descending order north to south.
# The default stride is the coordinate spacing between the first 2 elements -- if the 
# coordinate values are descending # the default spacing is negative.

print("tmp['i0 i0 40:30 100']")
print(tmp['i0 i0 40:30 100'])
print("")

# Make the latitude values go south to north.
# Since the spacing is known to be 3 degrees (since the default spacing is negative a
# positive spacing is used to step in the opposite direction)

print("tmp['i0 i0 30:40:3 100']")
print(tmp['i0 i0 30:40:3 100'])
print("")

# Or alternatively use a negative index step (prefixing the step value with 'i'). In
# index space reversing the order always means a negative step.

print("tmp['i0 i0 30:40:i-1 100']")
print(tmp['i0 i0 30:40:i-1 100'])
print("")

# Use the geopotential height variable in the file to get temperature at constant geopotential height
# 2 time steps
print("tmp['time|0,3 lev|hgt|1500 lat|50,60 lon|237:252']")
print(tmp['time|0,3 lev|hgt|1500 lat|50,60 lon|237:252'])
print("")

# Indirect indexing uses interpolation by default; use 'n' suffix to turn off interpolation.
# This is useful to make sure it is selecting the correct values.
# If you examine the height and temperature variable carefully you will note that 
# 1500 meters (geo height) shifts from closer level 750000 to closer to 90000 between longitudes 240 and 243
# 
# 2 time steps
print("tmp['time|0,3 lev|hgt|1500n lat|60 lon|237:252']")
print(tmp['time|0,3 lev|hgt|1500n lat|60 lon|237:252'])
print("")

