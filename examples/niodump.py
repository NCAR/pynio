#$Id: niodump.py 16153 2015-04-03 18:06:42Z huangwei $

from __future__ import print_function, division
import sys
import Nio

class NIOdump:
    def __init__(self, name=None):
        self.name = name
        if(None != name):
           self.file = Nio.open_file(name, "r") 

    def dump(self):
        print(self.file)

    def close(self):
        self.file.close()

if __name__ == '__main__':
    if(len(sys.argv) > 1):
        name = sys.argv[1]
    else:
        print("Usage: %s filename" %(sys.argv[0]))
        exit(-1)

    app = NIOdump(name=name)
    app.dump()
    app.close()

