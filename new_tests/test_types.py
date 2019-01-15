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

import numpy as np
import numpy.testing as nt
import Nio 
import time
import os
import pwd
import unittest as ut
import tempfile

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

class Test(ut.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp(prefix='test_', suffix='.nc')

    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_types(self):
        #
        #  Specify a global history attribute and open a NetCDF file
        #  for writing.
        #
        hatt = "Created " + time.ctime(time.time()) + " by " + getUserName()
        f = Nio.open_file(self.filename, "w", None, hatt)

        #
        #  Create some global attributes.
        #
        f.title   = "Nio test NetCDF file"
        f.series  = [1, 2, 3, 4, 5, 6]
        f.version = 45

        file_attributes.update({'history': hatt})

        #
        #  Create some dimensions.
        #
        f.create_dimension("array",    3)
        #f.create_dimension("strlen",    6)
        f.create_dimension("strlen",    10)
        f.create_dimension("dim1", 2)
        f.create_dimension("dim2", 1)
        f.create_dimension("dim3",4)

        #
        #  Create some variables.
        #
        #print("creating and assigning scalar double")
        v1 = f.create_variable("v1", 'd', ())
        v1.assign_value(42.0)

        #print("creating and assigning scalar float")
        v2 = f.create_variable("v2", 'f', ())
        v2.assign_value(52.0)

        #print("creating and assigning scalar integer")
        v3 = f.create_variable("v3", 'i', ())
        v3.assign_value(42)

        #print("creating and assigning scalar long")
        v4 = f.create_variable("v4", 'l', ())
        v4.assign_value(42)

        #print("creating and assigning scalar short")
        v5 = f.create_variable("v5", 'h', ())
        v5.assign_value(42)

        #print("creating and assigning scalar byte")
        v6 = f.create_variable("v6", 'b', ())
        v6.assign_value(42)

        #print("creating and assigning scalar char")
        v7 = f.create_variable("v7", 'S1', ())
        v7.assign_value('x')

        #print("creating and assigning array double")
        v11 = f.create_variable("v11", 'd', ('array',))
        v11.assign_value([42.0,43.0,44.0])

        #print("creating and assigning array float")
        v22 = f.create_variable("v22", 'f', ('array',))
        v22.assign_value([52.0,53.0,54.0])

        #print("creating and assigning array integer")
        v33 = f.create_variable("v33", 'i', ('array',))
        v33.assign_value([42,43,44])

        #print("creating and assigning array long")
        v44 = f.create_variable("v44", 'l', ('array',))
        a = np.array([42,43,44],'l')
        v44.assign_value(a)

        #print("creating and assigning array short")
        v55 = f.create_variable("v55", 'h', ('array',))
        v55.assign_value([42,43,44])

        #print("creating and assigning array byte")
        v66 = f.create_variable("v66", 'b', ('array',))
        v66.assign_value([42,43,44])

        #print("creating and assigning array char")
        v77 = f.create_variable("v77", 'S1', ('array','strlen'))
        v77.assign_value(['bcdef','uvwxyz','ijklmnopqr'])
        #v77.assign_value(['ab','uv','ij'])
        #v77.assign_value(['a','u','i'])

        #v77[1] = v77[1,::-1]

        #print(v77[:])

        v_single = f.create_variable("v_single",'f',("dim1","dim2","dim3"))
        #print(v_single)
        # type mismatch (double created then assigned to float variable)
        a = np.array([1.0,2,3,4,5,6,7,8], dtype=np.float64)
        a.shape = (2,1,4)
        #print(a)
        with nt.assert_raises(SystemError):
            v_single.assign_value(a)

        # now do it right
        a = np.array([1.0,2,3,4,5,6,7,8], dtype=np.float32)
        a.shape = (2,1,4)
        #print(a)
        v_single.assign_value(a)
        #print(v_single[:])
        v_single[1,0,2] = 11.0
        v_single[:,0,2] = [11.0,12.0]

        var_names = list(f.variables.keys())

        nt.assert_equal(set(var_names), file_variables)

        for var in var_names:
            v = f.variables[var]
            nt.assert_equal(v.dimensions, var_dimensions[var])
            nt.assert_equal(v.attributes, {})
            nt.assert_equal(v.get_value(), var_values[var])

        f.close()

        #
        #  Read the file we just created.
        #
        f = Nio.open_file(self.filename, "r")

        nt.assert_equal(f.attributes, file_attributes)
        nt.assert_equal(f.dimensions, file_dimensions)
        nt.assert_equal(set(f.variables.keys()), file_variables)

        for var in var_names:
            v = f.variables[var]
            nt.assert_equal(v.dimensions, var_dimensions[var])
            nt.assert_equal(v.attributes, {})
            nt.assert_equal(v.get_value(), var_values[var])

        f.close()

file_attributes = {'title': 'Nio test NetCDF file', 'series': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32), 'version': np.array([45], dtype=np.int32)}
file_dimensions = {'array': 3, 'strlen': 10, 'dim1': 2, 'dim2': 1, 'dim3': 4}
file_variables = set(['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v11', 'v22', 'v33', 'v44', 'v55', 'v66', 'v77', 'v_single'])
var_dimensions = {'v1': (),
                  'v2': (),
                  'v3': (),
                  'v4': (),
                  'v5': (),
                  'v6': (),
                  'v7': (),
                  'v11': ('array',),
                  'v22': ('array',),
                  'v33': ('array',),
                  'v44': ('array',),
                  'v55': ('array',),
                  'v66': ('array',),
                  'v77': ('array', 'strlen'),
                  'v_single': ('dim1', 'dim2', 'dim3')}
var_values = {'v1': np.float64(42.0),
              'v2': np.float32(52.0),
              'v3': np.int32(42),
              'v4': np.int64(42),
              'v5': np.int16(42),
              'v6': np.int8(42),
              'v7': b'x',
              'v11': np.array([42,43,44], np.float64),
              'v22': np.array([52,53,54], np.float32),
              'v33': np.array([42,43,44], np.int32),
              'v44': np.array([42,43,44], np.int64),
              'v55': np.array([42,43,44], np.int16),
              'v66': np.array([42,43,44], np.int8),
              'v77': np.array([[b'b', b'c', b'd', b'e', b'f', b'', b'', b'', b'', b''],
                               [b'u', b'v', b'w', b'x', b'y', b'z', b'', b'', b'', b''],
                               [b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r']],
                              dtype='|S1'),
              'v_single': np.array([[[1., 2., 11., 4.]],
                                    [[5., 6., 12., 8.]]],
                                   dtype=np.float32)
             }


if __name__ == '__main__':
    ut.main()
