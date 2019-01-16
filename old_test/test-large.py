from __future__ import print_function, division
import numpy as np
import Nio
import time, os

#
# Creating a file
#
init_time = time.clock()
ncfile = 'test-large.nc'
if (os.path.exists(ncfile)):
  os.system("/bin/rm -f " + ncfile)
opt = Nio.options()
opt.Format = "LargeFile"
opt.PreFill = False
file = Nio.open_file(ncfile, 'w', options=opt)

file.title = "Testing large files and dimensions"

file.create_dimension('big', 2500000000)

bigvar = file.create_variable('bigvar', "b", ('big',))
print("created bigvar")
# note it is incredibly slow to write a scalar to a large file variable
# so create an temporary variable x that will get assigned in steps

x = np.empty(1000000,dtype = 'int8')
#print x
x[:] = 42
t = list(range(0,2500000000,1000000))
ii = 0
for i in t:
   if (i == 0):
    continue
   print(t[ii],i)
   bigvar[t[ii]:i] = x[:]
   ii += 1
x[:] = 84
bigvar[2499000000:2500000000] = x[:]

bigvar[-1] = 84
bigvar.units = "big var units"
#print bigvar[-1]
print(bigvar.dimensions)

# check unlimited status
for dim in list(file.dimensions.keys()):
  print(dim, " unlimited: ",file.unlimited(dim))
print(file)
print("closing file")
print('elapsed time: ',time.clock() - init_time)
file.close()
#quit()
#
# Reading a file
#
print('opening file for read')
print('elapsed time: ',time.clock() - init_time)
file = Nio.open_file(ncfile, 'r')

print('file is open')
print('elapsed time: ',time.clock() - init_time)
print(file.dimensions)
print(list(file.variables.keys()))
print(file)
print("reading variable")
print('elapsed time: ',time.clock() - init_time)
x = file.variables['bigvar']
print(x[0],x[1000000],x[249000000],x[2499999999])
print("max and min")
min = x[:].min()
max = x[:].max()
print(min, max)
print('elapsed time: ',time.clock() - init_time)

# check unlimited status
for dim in list(file.dimensions.keys()):
  print(dim, " unlimited: ",file.unlimited(dim))

print("closing file")
print('elapsed time: ',time.clock() - init_time)
file.close()
