import Nio
import numpy as np
import numpy.testing as nt
import os
import unittest as ut

file_to_test = '../ncarg/data/grib/19580101.ice125.grb'

class Test(ut.TestCase):
    def setUp(self):
        self.f = Nio.open_file(os.path.realpath(os.path.join(os.path.dirname(__file__), file_to_test)))

    def test_grib_variables(self):
        nt.assert_equal(set(self.f.variables.keys()), set(file_variables))

    def test_grib_attributes(self):
        nt.assert_equal(self.f.attributes, file_attributes)

    def test_grib_dimensions(self):
        nt.assert_equal(self.f.dimensions, file_dimensions)

    def test_grib_var_shapes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.shape, var_shapes[var])

    def test_grib_var_attributes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.attributes, var_attributes[var])

    def test_grib_var_dimensions(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.dimensions, var_dimensions[var])

    def test_grib_var_coordinates(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(get_coord_dims(v), var_coordinates[var])

    def test_grib_var_values(self):
        for var in var_values.keys():
            v = self.f.variables[var]
            val = v.get_value()
            nt.assert_equal(np.ma.is_masked((val.min(), val.max(), val.mean(), np.ma.count_masked(val))), np.ma.is_masked(var_values[var]))
            nt.assert_almost_equal((val.min(), val.max(), val.mean(), np.ma.count_masked(val)), var_values[var])


file_attributes = {}
file_dimensions = {'initial_time0_hours': 5, 'forecast_time1': 2, 'g0_lat_2': 145, 'g0_lon_3': 288}
file_variables = ['ICEC_GDS0_SFC', 'initial_time0_hours', 'initial_time0_encoded', 'g0_lat_2', 'g0_lon_3', 'forecast_time1', 'initial_time0']

var_attributes = {'ICEC_GDS0_SFC': {'sub_center': '241', 'center': 'Japanese Meteorological Agency - Tokyo (RSMC)', 'long_name': 'Ice cover (1 = ice, 0 = no ice)', 'units': 'Proportion', '_FillValue': np.array([1.e+20], dtype=np.float32), 'level_indicator': np.array([1], dtype=np.int32), 'gds_grid_type': np.array([0], dtype=np.int32), 'parameter_table_version': np.array([200], dtype=np.int32), 'parameter_number': np.array([91], dtype=np.int32)},
                  'initial_time0_hours': {'long_name': 'initial time', 'units': 'hours since 1800-01-01 00:00'},
                  'initial_time0_encoded': {'long_name': 'initial time encoded as double', 'units': 'yyyymmddhh.hh_frac'},
                  'g0_lat_2': {'long_name': 'latitude', 'GridType': 'Cylindrical Equidistant Projection Grid', 'units': 'degrees_north', 'Dj': np.array([1.25], dtype=np.float32), 'Di': np.array([1.25], dtype=np.float32), 'Lo2': np.array([-1.25], dtype=np.float32), 'La2': np.array([-90.], dtype=np.float32), 'Lo1': np.array([0.], dtype=np.float32), 'La1': np.array([90.], dtype=np.float32)},
                  'g0_lon_3': {'long_name': 'longitude', 'GridType': 'Cylindrical Equidistant Projection Grid', 'units': 'degrees_east', 'Dj': np.array([1.25], dtype=np.float32), 'Di': np.array([1.25], dtype=np.float32), 'Lo2': np.array([-1.25], dtype=np.float32), 'La2': np.array([-90.], dtype=np.float32), 'Lo1': np.array([0.], dtype=np.float32), 'La1': np.array([90.], dtype=np.float32)},
                  'forecast_time1': {'long_name': 'Forecast offset from initial time', 'units': 'hours'},
                  'initial_time0': {'long_name': 'Initial time of first record', 'units': 'mm/dd/yyyy (hh:mm)'}}
var_coordinates = {'ICEC_GDS0_SFC': ['initial_time0_hours', 'forecast_time1', 'g0_lat_2', 'g0_lon_3'], 'initial_time0_hours': ['initial_time0_hours'], 'initial_time0_encoded': ['initial_time0_hours'], 'g0_lat_2': ['g0_lat_2'], 'g0_lon_3': ['g0_lon_3'], 'forecast_time1': ['forecast_time1'], 'initial_time0': ['initial_time0_hours']}
var_dimensions = {'ICEC_GDS0_SFC': ('initial_time0_hours', 'forecast_time1', 'g0_lat_2', 'g0_lon_3'), 'initial_time0_hours': ('initial_time0_hours',), 'initial_time0_encoded': ('initial_time0_hours',), 'g0_lat_2': ('g0_lat_2',), 'g0_lon_3': ('g0_lon_3',), 'forecast_time1': ('forecast_time1',), 'initial_time0': ('initial_time0_hours',)}
var_shapes = {'ICEC_GDS0_SFC': (5, 2, 145, 288), 'initial_time0_hours': (5,), 'initial_time0_encoded': (5,), 'g0_lat_2': (145,), 'g0_lon_3': (288,), 'forecast_time1': (2,), 'initial_time0': (5,)}
var_values = {'g0_lat_2': (np.float32(-90.0), np.float32(90.0), np.float32(0.0), 0), 'forecast_time1': (np.int32(3), np.int32(6), np.float64(4.5), 0), 'initial_time0_encoded': (np.float64(1957123118.0), np.float64(1958010118.0), np.float64(1957832710.8), 0), 'ICEC_GDS0_SFC': (np.float32(-5.88999320428e-08), np.float32(0.999999940395), np.float64(0.174019816407), 196888), 'g0_lon_3': (np.float32(0.0), np.float32(358.75), np.float32(179.375), 0), 'initial_time0_hours': (np.float64(1384986.0), np.float64(1385010.0), np.float64(1384998.0), 0)}

def get_coord_dims(var):
    return [dim for dim in var.dimensions if dim in var.file.variables.keys()]

if __name__ == '__main__':
    ut.main()
