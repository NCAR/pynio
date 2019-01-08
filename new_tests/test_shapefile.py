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

    def test_shapefile_var_values(self):
        for var in var_values.keys():
            v = self.f.variables[var]
            val = v.get_value()
            nt.assert_almost_equal((val.min(), val.max(), val.mean(), np.ma.count_masked(val)), var_values[var])


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
var_values = {'FAMILIES': (np.float64(119825.0), np.float64(7139394.0), np.float64(1307638.0612244897), 0), 'SERVICE': (np.float64(65498.0), np.float64(3664771.0), np.float64(632147.8979591837), 0), 'WORKERS': (np.float64(164561.0), np.float64(11306576.0), np.float64(1863683.8775510204), 0), 'HOUSHOLD': (np.float64(168839.0), np.float64(10381206.0), np.float64(1864097.3265306123), 0), 'SAMP_POP': (np.float64(72696.0), np.float64(3792553.0), np.float64(782455.632653), 0), 'WATER_KM': (np.float64(17.991), np.float64(30456.797), np.float64(5128.5664898), 0), 'PUBTRANS': (np.float64(971.0), np.float64(2113133.0), np.float64(122876.857143), 0), 'segments': (np.int32(0), np.int32(11461), np.float64(3180.87368421), 0), 'UNEMPLOY': (np.float64(13112.0), np.float64(996502.0), np.float64(158057.632653), 0), 'DRVALONE': (np.float64(106694.0), np.float64(9982242.0), np.float64(1706986.469387755), 0), 'MALE': (np.float64(227007.0), np.float64(14897627.0), np.float64(2455152.0816326533), 0), 'LAND_KM': (np.float64(159.055), np.float64(688219.07), np.float64(156361.47251020407), 0), 'FEMALE': (np.float64(226581.0), np.float64(14862394.0), np.float64(2583244.93877551), 0), 'EMPLOYED': (np.float64(207868.0), np.float64(13996309.0), np.float64(2343349.163265306), 0), 'geometry': (np.int32(0), np.int32(92), np.float64(24.3775510204), 0), 'CARPOOL': (np.float64(28109.0), np.float64(2036025.0), np.float64(310404.142857), 0), 'MANUAL': (np.float64(22407.0), np.float64(1798201.0), np.float64(348944.734694), 0), 'P_MALE': (np.float64(0.466), np.float64(0.509), np.float64(0.487367346939), 0), 'PERSONS': (np.float64(453588.0), np.float64(29760021.0), np.float64(5038397.020408163), 0), 'P_FEMALE': (np.float64(0.491), np.float64(0.534), np.float64(0.512632653061), 0), 'y': (np.float64(24.955967), np.float64(49.371735), np.float64(38.2179739868), 0), 'x': (np.float64(-124.731422), np.float64(-66.969849), np.float64(-91.4683404153), 0)}

def get_coord_dims(var):
    return [dim for dim in var.dimensions if dim in var.file.variables.keys()]

if __name__ == '__main__':
    ut.main()
