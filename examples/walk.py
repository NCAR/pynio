from __future__ import print_function, division

def walktree(top):
     values = list(top.groups.values()),list(top.groups.keys())
#     print values
     yield values
     for value in list(top.groups.values()):
         for children in walktree(value):
             yield children
import sys
import Nio

if(len(sys.argv) > 1):
    name = sys.argv[1]
else:
    print("Usage: %s filename" %(sys.argv[0]))
    exit(-1)

f = Nio.open_file(name) 
#print f
print("-----------------------")
print("all groups in the file:")
print("-----------------------")
print(f.groups)
print("--------------------------")
print("traversing the group tree:")
print("--------------------------")
if len(f.groups) == 0:
  print(name, "contains no groups")
  exit(0)
rgroup = f.groups['/']
print("root group:",rgroup.name)
print("\tgroup:",  rgroup.name, "contains", len(rgroup.groups), "groups", len(rgroup.variables), "variables,", len(rgroup.dimensions), "dimensions, and", len(rgroup.attributes), "group attributes")
for children,keys in walktree(f.groups['/']):
    if len(children) > 0:
        print("groups:", keys)
    for child in children:
        print("\tgroup:",  child.name, "contains", len(child.groups), "groups", len(child.variables), "variables,", len(child.dimensions), "dimensions, and", len(child.attributes), "group attributes")
      
      #print child
print("file contents =======")
print(f)
#lat = f.variables['lat'][:]
#print lat
#latb = lat > 0
#print(latb)
#latsub = f.variables['lat'][(latb)]
#print latsub
