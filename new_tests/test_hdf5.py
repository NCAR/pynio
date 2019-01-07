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

def get_coord_dims(var):
    return [dim for dim in var.dimensions if dim in var.file.variables.keys()]

if __name__ == '__main__':
    ut.main()
