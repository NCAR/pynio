from __future__ import print_function, division
#
#  File:
#    nio01.py
#
#  Synopsis:
#    Creates a NetCDF with scalar and array versions of all permissible types and then reads it
#    printing typecodes and other info
#
#  Category:
#    Processing.
#
#  Author:
#    Dave Brown (modelled after an example of Konrad Hinsen).
#  

import numpy 
import Nio 
import time
import os
import pwd

#
#  Function to retrieve the user's name.
#
def getUserName():
    pwd_entry = pwd.getpwuid(os.getuid())
    raw_name = pwd_entry[4]
    name = raw_name.split(",")[0].strip()
    if name == '':
        name = pwd_entry[0]
        
    return name

#
#  Creating a NetCDF file named "test-types.nc".  If there is already
#  a file with that name, delete it first.
#
if (os.path.exists("test-types.nc")):
  os.system("/bin/rm -f test-types.nc")

#
#  Specify a global history attribute and open a NetCDF file
#  for writing.
#
hatt = "Created " + time.ctime(time.time()) + " by " + getUserName()
file = Nio.open_file("test-types.nc", "w", None, hatt)

#
#  Create some global attributes.
#
file.title   = "Nio test NetCDF file"
file.series  = [ 1, 2, 3, 4, 5,6 ]
file.version = 45

#
#  Create some dimensions.
#
file.create_dimension("array",    3)
#file.create_dimension("strlen",    6)
file.create_dimension("strlen",    10)
file.create_dimension("dim1", 2)
file.create_dimension("dim2", 1)
file.create_dimension("dim3",4)

#
#  Create some variables.
#
print("creating and assigning scalar double")
v1 = file.create_variable("v1", 'd', ())
v1.assign_value(42.0)

print("creating and assigning scalar float")
v2 = file.create_variable("v2", 'f', ())
v2.assign_value(52.0)

print("creating and assigning scalar integer")
v3 = file.create_variable("v3", 'i', ())
v3.assign_value(42)

print("creating and assigning scalar long")
v4 = file.create_variable("v4", 'l', ())
v4.assign_value(42)

print("creating and assigning scalar short")
v5 = file.create_variable("v5", 'h', ())
v5.assign_value(42)

print("creating and assigning scalar byte")
v6 = file.create_variable("v6", 'b', ())
v6.assign_value(42)

print("creating and assigning scalar char")
v7 = file.create_variable("v7", 'S1', ())
v7.assign_value('x')

print("creating and assigning array double")
v11 = file.create_variable("v11", 'd', ('array',))
v11.assign_value([42.0,43.0,44.0])

print("creating and assigning array float")
v22 = file.create_variable("v22", 'f', ('array',))
v22.assign_value([52.0,53.0,54.0])

print("creating and assigning array integer")
v33 = file.create_variable("v33", 'i', ('array',))
v33.assign_value([42,43,44])

print("creating and assigning array long")
v44 = file.create_variable("v44", 'l', ('array',))
a = numpy.array([42,43,44],'l')
v44.assign_value(a)

print("creating and assigning array short")
v55 = file.create_variable("v55", 'h', ('array',))
v55.assign_value([42,43,44])

print("creating and assigning array byte")
v66 = file.create_variable("v66", 'b', ('array',))
v66.assign_value([42,43,44])

print("creating and assigning array char")
v77 = file.create_variable("v77", 'S1', ('array','strlen'))
v77.assign_value(['bcdef','uvwxyz','ijklmnopqr'])
#v77.assign_value(['ab','uv','ij'])
#v77.assign_value(['a','u','i'])

#v77[1] = v77[1,::-1]

print(v77[:])

v_single = file.create_variable("v_single",'f',("dim1","dim2","dim3"))
print(v_single)
# type mismatch (double created then assigned to float variable)
a = numpy.array([1.0,2,3,4,5,6,7,8])
a.shape = (2,1,4)
print(a)
try:
   v_single.assign_value(a)
   print(v_single[:])
except:
   print("type mismatch in assignment")
# now do it right
a = numpy.array([1.0,2,3,4,5,6,7,8],'f')
a.shape = (2,1,4)
print(a)
v_single.assign_value(a)
print(v_single[:])
v_single[1,0,2] = 11.0
v_single[:,0,2] = [11.0,12.0]

vars = list(file.variables.keys())
print("printing PyNIO var summaries")
print(vars)

for var in vars:
  v = file.variables[var]
  print(v)
  
print("printing var info derived from PyNIO calls")
print(vars)
for var in vars:
  v = file.variables[var]
  print("  " + str(var))
  print("  " + str(v.get_value()))
  print("  dimensions: " + str(v.dimensions))
  print("  rank: " + str(v.rank))
  print("  typecode: " + str(v.typecode()))
  print("")

file.close()

#
#  Read the file we just created.
#
file = Nio.open_file("test-types.nc", "r")

print("\nOn read, a summary of the file contents:")
print(file)
print("\nOn read, file dimensions:")
print("  " + str(file.dimensions))
print("On read, file variables:")
print("  " + str(list(file.variables.keys())))

vars = list(file.variables.keys())
print("printing PyNIO var summaries")
for var in vars:
  v = file.variables[var]
  print(v)
  
print("printing var info derived from PyNIO calls")
for var in vars:
  v = file.variables[var]
  print("  " + str(var))
  print("  " + str(v.get_value()))
  print("  dimensions: " + str(v.dimensions))
  print("  rank: " + str(v.rank))
  print("  typecode: " + str(v.typecode()))
  print("")

file.close()
