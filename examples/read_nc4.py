from __future__ import print_function, division
import numpy
import Nio

file = Nio.open_file("nc4_withgroup.nc", "r")

#print "file.__dict__:"
#print file.__dict__

print("file.attributes:")
#print file.attributes

n = 0
for name in file.attributes:
    print("Global attr %d: <%s>" %(n, getattr(file, name)))
    n += 1

print("\n\n")

print("file.dimensions:")
#print file.dimensions

n = 0
for name in list(file.dimensions.keys()):
    print("Dim %d name: <%s>, size: %d" %(n, name, file.dimensions[name]))
    n += 1

print("\n\n")

print("file.variables:")
#print file.variables

n = 0
for name in list(file.variables.keys()):
    print("Var %d: <%s>" %(n, name))
    n += 1
print("\n\n")

n = 0
for name in list(file.groups.keys()):
    print("Group %d: <%s>" %(n, name))
    n += 1
print("\n\n")

print("file:")
print(file)

forecasts = file.groups['forecasts']

print("forecasts:\n")
print(forecasts)

mdl1 = forecasts.groups['model1']
temp = mdl1.variables['temp'][:]

print("temp:")
print(temp)

file.close()

