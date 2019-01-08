import Nio
import numpy as np
import numpy.testing as nt
import os
import unittest as ut

file_to_test = '../ncarg/data/netcdf/pop.nc'

class Test(ut.TestCase):
    def setUp(self):
        self.f = Nio.open_file(os.path.realpath(os.path.join(os.path.dirname(__file__), file_to_test)))

    def test_netcdf_variables(self):
        nt.assert_equal(set(self.f.variables.keys()), set(file_variables))

    def test_netcdf_attributes(self):
        nt.assert_equal(self.f.attributes, file_attributes)

    def test_netcdf_dimensions(self):
        nt.assert_equal(self.f.dimensions, file_dimensions)

    def test_netcdf_var_shapes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.shape, var_shapes[var])

    def test_netcdf_var_attributes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.attributes, var_attributes[var])

    def test_netcdf_var_dimensions(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.dimensions, var_dimensions[var])

    def test_netcdf_var_coordinates(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(get_coord_dims(v), var_coordinates[var])

    def test_netcdf_var_values(self):
        for var in var_values.keys():
            v = self.f.variables[var]
            val = v.get_value()
            nt.assert_almost_equal((val.min(), val.max(), val.mean(), np.ma.count_masked(val)), var_values[var])


file_attributes = {}
file_dimensions = {'nlat': 384, 'nlon': 320}
file_variables = ['urot', 'vrot', 't', 'lat2d', 'lon2d']

var_attributes = {'urot': {'missing_value': np.array([9.96921e+36], dtype=np.float32), '_FillValue': np.array([9.96921e+36], dtype=np.float32), 'cell_methods': 'time: mean', 'coordinates': 'lat2d lon2d', 'units': 'centimeter/s', 'long_name': 'Zonal Velocity', 'time': np.array([365031.]), 'z_t': np.array([500.622], dtype=np.float32)},
                  'vrot': {'missing_value': np.array([9.96921e+36], dtype=np.float32), '_FillValue': np.array([9.96921e+36], dtype=np.float32), 'cell_methods': 'time: mean', 'coordinates': 'lat2d lon2d', 'units': 'centimeter/s', 'long_name': 'Meridional Velocity', 'time': np.array([365031.]), 'z_t': np.array([500.622], dtype=np.float32)},
                  't': {'z_t': np.array([500.622], dtype=np.float32), 'time': np.array([365031.]), 'long_name': 'Potential Temperature', 'units': 'degC', 'coordinates': 'lat2d lon2d', 'cell_methods': 'time: mean', '_FillValue': np.array([9.96921e+36], dtype=np.float32), 'missing_value': np.array([9.96921e+36], dtype=np.float32)},
                  'lat2d': {'long_name': 'array of u-grid latitudes', 'units': 'degrees_north'},
                  'lon2d': {'long_name': 'array of u-grid longitudes', 'units': 'degrees_east'}}
var_coordinates = {'urot': [], 'vrot': [], 't': [], 'lat2d': [], 'lon2d': []}
var_dimensions = {'urot': ('nlat', 'nlon'), 'vrot': ('nlat', 'nlon'), 't': ('nlat', 'nlon'), 'lat2d': ('nlat', 'nlon'), 'lon2d': ('nlat', 'nlon')}
var_shapes = {'urot': (384, 320), 'vrot': (384, 320), 't': (384, 320), 'lat2d': (384, 320), 'lon2d': (384, 320)}
var_values = {'urot': (np.float32(-105.20891571), np.float32(116.921279907), np.float64(-1.4017843907), 33499), 'lat2d': (np.float32(-78.9528961182), np.float32(89.9773406982), np.float32(2.99511003494), 0), 'lon2d': (np.float32(0.00727074081078), np.float32(359.99621582), np.float32(185.568359375), 0), 'vrot': (np.float32(-64.6780166626), np.float32(77.3827590942), np.float64(1.04894356742), 33499), 't': (np.float32(-2.32870078087), np.float32(31.1261768341), np.float64(16.8203470598), 36526)}

def get_coord_dims(var):
    return [dim for dim in var.dimensions if dim in var.file.variables.keys()]

if __name__ == '__main__':
    ut.main()
