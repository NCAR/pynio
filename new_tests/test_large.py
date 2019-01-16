from __future__ import print_function, division
import numpy as np
import Nio
import time, os
import numpy.testing as nt
import unittest as ut
import tempfile

class Test(ut.TestCase):
    def setUp(self):
        #self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        self.filename = tempfile.mktemp(prefix=__file__, suffix=".nc")

    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_large(self):
        #
        # Creating a file
        #
        #init_time = time.clock()
        opt = Nio.options()
        opt.Format = "LargeFile"
        opt.PreFill = False
        f = Nio.open_file(self.filename, 'w', options=opt)

        f.title = "Testing large files and dimensions"

        f.create_dimension('big', 2500000000)

        bigvar = f.create_variable('bigvar', "b", ('big',))
        #print("created bigvar")
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
           #print(t[ii],i)
           bigvar[t[ii]:i] = x[:]
           ii += 1
        x[:] = 84
        bigvar[2499000000:2500000000] = x[:]

        bigvar[-1] = 84
        bigvar.units = "big var units"
        #print bigvar[-1]
        #print(bigvar.dimensions)

        # check unlimited status

        #print(f)
        nt.assert_equal(bigvar.dimensions, ('big',))
        nt.assert_equal(f.unlimited('big'), False)
        nt.assert_equal(f.attributes, {'title': 'Testing large files and dimensions'})
        nt.assert_equal(f.dimensions, {'big': 2500000000})
        nt.assert_equal(list(f.variables.keys()), ['bigvar'])
        #print("closing file")
        #print('elapsed time: ',time.clock() - init_time)
        f.close()
        #quit()
        #
        # Reading a file
        #
        #print('opening file for read')
        #print('elapsed time: ',time.clock() - init_time)
        f = Nio.open_file(self.filename, 'r')

        #print('file is open')
        #print('elapsed time: ',time.clock() - init_time)
        nt.assert_equal(f.attributes, {'title': 'Testing large files and dimensions'})
        nt.assert_equal(f.dimensions, {'big': 2500000000})
        nt.assert_equal(list(f.variables.keys()), ['bigvar'])
        #print(f.dimensions)
        #print(list(f.variables.keys()))
        #print(f)
        #print("reading variable")
        #print('elapsed time: ',time.clock() - init_time)
        x = f.variables['bigvar']
        #print(x[0],x[1000000],x[249000000],x[2499999999])
        nt.assert_equal((x[0],x[1000000],x[249000000],x[2499999999]), (42, 42, 42, 84))
        #print("max and min")
        min = x[:].min()
        max = x[:].max()
        nt.assert_equal((x[:].min(), x[:].max()), (42, 84))

        # check unlimited status
        nt.assert_equal(f.variables['bigvar'].dimensions, ('big',))
        nt.assert_equal(f.unlimited('big'), False)

        #print("closing file")
        #print('elapsed time: ',time.clock() - init_time)
        f.close()


if __name__ == '__main__':
    ut.main()
