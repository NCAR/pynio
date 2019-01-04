import Nio
import numpy as np
import numpy.testing as nt
import os
import unittest as ut

file_to_test = '../ncarg/data/shapefile/states.shp'

class Test(ut.TestCase):
    def setUp(self):
        self.f = Nio.open_file(os.path.realpath(os.path.join(os.path.dirname(__file__), file_to_test)))

    def test_shapefile_variables(self):
        nt.assert_equal(set(self.f.variables.keys()), set(file_variables))

    def test_shapefile_attributes(self):
        nt.assert_equal(self.f.attributes, file_attributes)

    def test_shapefile_dimensions(self):
        nt.assert_equal(self.f.dimensions, file_dimensions)

    def test_shapefile_var_shapes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.shape, var_shapes[var])

    def test_shapefile_var_attributes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.attributes, var_attributes[var])

    def test_shapefile_var_dimensions(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.dimensions, var_dimensions[var])

    def test_shapefile_var_coordinates(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(get_coord_dims(v), var_coordinates[var])


file_attributes = {'layer_name': 'states', 'geometry_type': 'polygon', 'geom_segIndex': np.array([0], dtype=np.int32), 'geom_numSegs': np.array([1], dtype=np.int32), 'segs_xyzIndex': np.array([0], dtype=np.int32), 'segs_numPnts': np.array([1], dtype=np.int32)}
file_dimensions = {'geometry': 2, 'segments': 2, 'num_features': 49, 'num_segments': 95, 'num_points': 11481}
file_variables = ['geometry', 'segments', 'x', 'y', 'STATE_NAME', 'STATE_FIPS', 'SUB_REGION', 'STATE_ABBR', 'LAND_KM', 'WATER_KM', 'PERSONS', 'FAMILIES', 'HOUSHOLD', 'MALE', 'FEMALE', 'WORKERS', 'DRVALONE', 'CARPOOL', 'PUBTRANS', 'EMPLOYED', 'UNEMPLOY', 'SERVICE', 'MANUAL', 'P_MALE', 'P_FEMALE', 'SAMP_POP']

var_attributes = {'geometry': {},
                  'segments': {},
                  'x': {},
                  'y': {},
                  'STATE_NAME': {},
                  'STATE_FIPS': {},
                  'SUB_REGION': {},
                  'STATE_ABBR': {},
                  'LAND_KM': {},
                  'WATER_KM': {},
                  'PERSONS': {},
                  'FAMILIES': {},
                  'HOUSHOLD': {},
                  'MALE': {},
                  'FEMALE': {},
                  'WORKERS': {},
                  'DRVALONE': {},
                  'CARPOOL': {},
                  'PUBTRANS': {},
                  'EMPLOYED': {},
                  'UNEMPLOY': {},
                  'SERVICE': {},
                  'MANUAL': {},
                  'P_MALE': {},
                  'P_FEMALE': {},
                  'SAMP_POP': {}}
var_coordinates = {'geometry': ['geometry'], 'segments': ['segments'], 'x': [], 'y': [], 'STATE_NAME': [], 'STATE_FIPS': [], 'SUB_REGION': [], 'STATE_ABBR': [], 'LAND_KM': [], 'WATER_KM': [], 'PERSONS': [], 'FAMILIES': [], 'HOUSHOLD': [], 'MALE': [], 'FEMALE': [], 'WORKERS': [], 'DRVALONE': [], 'CARPOOL': [], 'PUBTRANS': [], 'EMPLOYED': [], 'UNEMPLOY': [], 'SERVICE': [], 'MANUAL': [], 'P_MALE': [], 'P_FEMALE': [], 'SAMP_POP': []}
var_dimensions = {'geometry': ('num_features', 'geometry'), 'segments': ('num_segments', 'segments'), 'x': ('num_points',), 'y': ('num_points',), 'STATE_NAME': ('num_features',), 'STATE_FIPS': ('num_features',), 'SUB_REGION': ('num_features',), 'STATE_ABBR': ('num_features',), 'LAND_KM': ('num_features',), 'WATER_KM': ('num_features',), 'PERSONS': ('num_features',), 'FAMILIES': ('num_features',), 'HOUSHOLD': ('num_features',), 'MALE': ('num_features',), 'FEMALE': ('num_features',), 'WORKERS': ('num_features',), 'DRVALONE': ('num_features',), 'CARPOOL': ('num_features',), 'PUBTRANS': ('num_features',), 'EMPLOYED': ('num_features',), 'UNEMPLOY': ('num_features',), 'SERVICE': ('num_features',), 'MANUAL': ('num_features',), 'P_MALE': ('num_features',), 'P_FEMALE': ('num_features',), 'SAMP_POP': ('num_features',)}
var_shapes = {'geometry': (49, 2), 'segments': (95, 2), 'x': (11481,), 'y': (11481,), 'STATE_NAME': (49,), 'STATE_FIPS': (49,), 'SUB_REGION': (49,), 'STATE_ABBR': (49,), 'LAND_KM': (49,), 'WATER_KM': (49,), 'PERSONS': (49,), 'FAMILIES': (49,), 'HOUSHOLD': (49,), 'MALE': (49,), 'FEMALE': (49,), 'WORKERS': (49,), 'DRVALONE': (49,), 'CARPOOL': (49,), 'PUBTRANS': (49,), 'EMPLOYED': (49,), 'UNEMPLOY': (49,), 'SERVICE': (49,), 'MANUAL': (49,), 'P_MALE': (49,), 'P_FEMALE': (49,), 'SAMP_POP': (49,)}

def get_coord_dims(var):
    return [dim for dim in var.dimensions if dim in var.file.variables.keys()]

if __name__ == '__main__':
    ut.main()
