import Nio
import numpy as np
import numpy.testing as nt
import os
import unittest as ut

file_to_test = '../ncarg/data/hdf/avhrr.hdf'

class Test(ut.TestCase):
    def setUp(self):
        self.f = Nio.open_file(os.path.realpath(os.path.join(os.path.dirname(__file__), file_to_test)))

    def test_hdf4_variables(self):
        nt.assert_equal(set(self.f.variables.keys()), set(file_variables))

    def test_hdf4_attributes(self):
        nt.assert_equal(self.f.attributes, file_attributes)

    def test_hdf4_dimensions(self):
        nt.assert_equal(self.f.dimensions, file_dimensions)

    def test_hdf4_var_shapes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.shape, var_shapes[var])

    def test_hdf4_var_attributes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.attributes, var_attributes[var])

    def test_hdf4_var_dimensions(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.dimensions, var_dimensions[var])

    def test_hdf4_var_coordinates(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(get_coord_dims(v), var_coordinates[var])

    def test_hdf4_var_values(self):
        for var in var_values.keys():
            v = self.f.variables[var]
            val = v.get_value()
            nt.assert_almost_equal((val.min(), val.max(), val.mean(), np.ma.count_masked(val)), var_values[var])


file_attributes = {}
file_dimensions = {'fakeDim0': 180, 'fakeDim1': 360}
file_variables = ['fakeDim0', 'fakeDim1', 'Data_Set_2']

var_attributes = {'fakeDim0': {'hdf_name': 'fakeDim0'},
                  'fakeDim1': {'hdf_name': 'fakeDim1'},
                  'Data_Set_2': {'coordsys': 'Interrrupted Goode Homolosine', 'valid_max': np.array([253], dtype=np.uint8), 'valid_min': np.array([3], dtype=np.uint8), 'scale_factor': np.array([0.008]), 'scale_factor_err': np.array([-9.]), 'add_offset': np.array([128.]), 'add_offset_err': np.array([-9.]), 'calibrated_nt': np.array([21], dtype=np.int32), 'long_name': 'NDVI', 'units': 'n/a', 'format': ' ', 'hdf_name': 'Data-Set-2'}}
var_coordinates = {'fakeDim0': ['fakeDim0'], 'fakeDim1': ['fakeDim1'], 'Data_Set_2': ['fakeDim0', 'fakeDim1']}
var_dimensions = {'fakeDim0': ('fakeDim0',), 'fakeDim1': ('fakeDim1',), 'Data_Set_2': ('fakeDim0', 'fakeDim1')}
var_shapes = {'fakeDim0': (180,), 'fakeDim1': (360,), 'Data_Set_2': (180, 360)}
var_values = {'fakeDim1': (np.uint8(129), np.uint8(129), np.float64(129.0), 0), 'Data_Set_2': (np.uint8(0), np.uint8(214), np.float64(39.0547376543), 0), 'fakeDim0': (np.uint8(129), np.uint8(129), np.float64(129.0), 0)}

def get_coord_dims(var):
    return [dim for dim in var.dimensions if dim in var.file.variables.keys()]

if __name__ == '__main__':
    ut.main()
