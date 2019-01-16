from __future__ import print_function, division
import numpy
import Nio 
import os
import numpy.testing as nt
import unittest as ut
import tempfile

class Test(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        global filename
        filename = tempfile.mktemp(prefix="test_", suffix=".nc")

    @classmethod
    def tearDownClass(cls):
        global filename
        if os.path.exists(filename):
            os.remove(filename)

    def setUp(self):
        global filename
        self.filename = filename

    def test_unlim_create(self):
        #
        #  Open a NetCDF file for writing file and specify a 
        #  global history attribute.
        #
        f       = Nio.open_file(self.filename, "w")
        f.title = "Unlimited dimension test file"

        #
        #  Create some dimensions.
        #
        f.create_dimension("time", None)     # unlimited dimension

        nt.assert_equal(f.name, os.path.basename(self.filename))
        nt.assert_equal(f.attributes, {'title': "Unlimited dimension test file"})
        nt.assert_equal(f.title, "Unlimited dimension test file")
        nt.assert_equal(f.dimensions, {'time': None})
        # check for unlimited status
        nt.assert_equal(f.unlimited('time'), True)
        nt.assert_equal(f.variables, {})

        #
        #  Create a variable of type float with three dimemsions.
        #
        var = f.create_variable("var", 'f', ("time",))
        var._FillValue = numpy.array(-999.,dtype = numpy.float32)

        data = numpy.arange(10,dtype='f')
        # assign 5 elements of the unlimited dimension
        var[:] = data[0:4]
        nt.assert_equal(len(var[:]), 4)
        nt.assert_equal(var[:], [0, 1, 2, 3])
        var.assign_value(data[5:])
        nt.assert_equal(f.dimensions, {'time': 5})

        # make sure this was actually written to the file
        f.close()

    def test_unlim_open(self):
        #filename = "test_unlim.nc"
        data = numpy.arange(10,dtype='f')

        f       = Nio.open_file(self.filename, "w")
        nt.assert_equal(f.dimensions, {'time': 5})
        # check for unlimited status
        nt.assert_equal(f.unlimited('time'), True)
        var = f.variables['var']
        nt.assert_equal(len(var[:]), 5)
        nt.assert_equal(var[:], [5, 6, 7, 8, 9])

        # this does not work with NIO 5.0.0
        var[::-1] = data
        nt.assert_equal(len(var[:]), 10)
        nt.assert_equal(var[:], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        # add 10 elements starting at element 3 - total 13
        var[3:] = data
        nt.assert_equal(f.dimensions, {'time': 13})
        nt.assert_equal(len(var[:]), 13)
        nt.assert_equal(var[:], [9, 8, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # prepend single element dimensions to the data array
        # but it should still work
        data = data[numpy.newaxis,numpy.newaxis,:]

        # adding 10 from element 10 = 20
        var[10:] = data
        nt.assert_equal(f.dimensions, {'time': 20})
        nt.assert_equal(len(var[:]), 20)
        nt.assert_equal(var[:], [9, 8, 7, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # this does not work with NIO 5.0.0
        #var[25:15:-1] = data
        #print len(var[:]),var[:]
        # add 10 elements spaced 2 elements apart -- resulting in some missing value elements --
        # and 39 total elements
        var[20::2] = data
        nt.assert_equal(len(var[:]), 39)
        nt.assert_equal(var[:], [9, 8, 7, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, None, 1, None, 2, None, 3, None, 4, None, 5, None, 6, None, 7, None, 8, None, 9])

        nt.assert_equal(f.name, os.path.basename(self.filename))
        nt.assert_equal(f.attributes, {'title': "Unlimited dimension test file"})
        nt.assert_equal(f.title, "Unlimited dimension test file")
        nt.assert_equal(f.dimensions, {'time': 39})
        # check for unlimited status
        nt.assert_equal(f.unlimited('time'), True)
        nt.assert_equal(list(f.variables.keys()), ['var'])
        nt.assert_equal(f.variables['var']._FillValue, -999)
        nt.assert_equal(f.variables['var'].dimensions, ('time',))

        f.close()
        #os.system('ncdump %s' % self.filename)

if __name__ == "__main__":
    ut.main()
