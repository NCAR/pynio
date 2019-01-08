import Nio
import numpy as np
import numpy.testing as nt
import os
import unittest as ut

file_to_test = '../ncarg/data/hdf5/num-types.h5'

class Test(ut.TestCase):
    def setUp(self):
        self.f = Nio.open_file(os.path.realpath(os.path.join(os.path.dirname(__file__), file_to_test)))

    def test_hdf5_groups(self):
        nt.assert_equal(set(self.f.groups.keys()), set(file_groups))

    def test_hdf5_variables(self):
        nt.assert_equal(set(self.f.variables.keys()), set(file_variables))

    def test_hdf5_attributes(self):
        nt.assert_equal(self.f.attributes, file_attributes)

    def test_hdf5_dimensions(self):
        nt.assert_equal(self.f.dimensions, file_dimensions)

    def test_hdf5_var_shapes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.shape, var_shapes[var])

    def test_hdf5_var_attributes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.attributes, var_attributes[var])

    def test_hdf5_var_dimensions(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.dimensions, var_dimensions[var])

    def test_hdf5_var_coordinates(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(get_coord_dims(v), var_coordinates[var])

    def test_hdf5_var_values(self):
        for var in var_values.keys():
            v = self.f.variables[var]
            val = v.get_value()
            nt.assert_almost_equal((val.min(), val.max(), val.mean(), np.ma.count_masked(val)), var_values[var])


file_attributes = {}
file_dimensions = {'DIM_000': 4, 'DIM_001': 7}
file_groups = ['/']
file_variables = ['c_data', 'double_data', 'float_data', 'i_data', 'l_data', 'll_data', 's_data', 'uc_data', 'ui_data', 'ul_data', 'ull_data', 'us_data']

var_attributes = {'c_data': {},
                  'double_data': {},
                  'float_data': {},
                  'i_data': {},
                  'l_data': {},
                  'll_data': {},
                  's_data': {},
                  'uc_data': {},
                  'ui_data': {},
                  'ul_data': {},
                  'ull_data': {},
                  'us_data': {}}
var_coordinates = {'c_data': [], 'double_data': [], 'float_data': [], 'i_data': [], 'l_data': [], 'll_data': [], 's_data': [], 'uc_data': [], 'ui_data': [], 'ul_data': [], 'ull_data': [], 'us_data': []}
var_dimensions = {'c_data': ('DIM_000', 'DIM_001'), 'double_data': ('DIM_000', 'DIM_001'), 'float_data': ('DIM_000', 'DIM_001'), 'i_data': ('DIM_000', 'DIM_001'), 'l_data': ('DIM_000', 'DIM_001'), 'll_data': ('DIM_000', 'DIM_001'), 's_data': ('DIM_000', 'DIM_001'), 'uc_data': ('DIM_000', 'DIM_001'), 'ui_data': ('DIM_000', 'DIM_001'), 'ul_data': ('DIM_000', 'DIM_001'), 'ull_data': ('DIM_000', 'DIM_001'), 'us_data': ('DIM_000', 'DIM_001')}
var_shapes = {'c_data': (4, 7), 'double_data': (4, 7), 'float_data': (4, 7), 'i_data': (4, 7), 'l_data': (4, 7), 'll_data': (4, 7), 's_data': (4, 7), 'uc_data': (4, 7), 'ui_data': (4, 7), 'ul_data': (4, 7), 'ull_data': (4, 7), 'us_data': (4, 7)}
var_values = {'c_data': (np.int8(-128), np.int8(127), np.float64(-23.3571428571), 0), 'll_data': (np.int64(-9223372036854775808), np.int64(9223372036854775807), np.float64(-1.976436865040309e+18), 0), 'float_data': (np.float32(-1.79820001125), np.float32(1.66650002561e+34), np.float32(2.14264286796e+33), 0), 'uc_data': (np.uint8(0), np.uint8(255), np.float64(31.5), 0), 'us_data': (np.uint16(0), np.uint16(65535), np.float64(7025.78571429), 0), 'l_data': (np.int32(0), np.int32(2147483647), np.float64(920350137.0), 0), 'ull_data': (np.uint64(0), np.uint64(18446744073709551615), np.float64(3.952873730080618e+18), 0), 'ul_data': (np.uint32(0), np.uint32(4294967295), np.float64(1840700271.857143), 0), 'ui_data': (np.uint32(0), np.uint32(4294967295), np.float64(460175071.5), 0), 'double_data': (np.float64(-0.0009), np.float64(1.11e+33), np.float64(4.757142857142857e+32), 0), 'i_data': (np.int32(-2147483648), np.int32(2147483647), np.float64(-460175063.35714287), 0), 's_data': (np.int16(-32768), np.int16(32767), np.float64(-7017.64285714), 0)}

def get_coord_dims(var):
    return [dim for dim in var.dimensions if dim in var.file.variables.keys()]

if __name__ == '__main__':
    ut.main()
