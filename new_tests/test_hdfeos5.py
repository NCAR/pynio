import Nio
import numpy as np
import numpy.testing as nt
import os
import unittest as ut

file_to_test = '../ncarg/data/hdfeos5/OMI-Aura_L3-OMAERUVd_2010m0131_v003-2010m0202t014811.he5'

class Test(ut.TestCase):
    def setUp(self):
        self.f = Nio.open_file(os.path.realpath(os.path.join(os.path.dirname(__file__), file_to_test)))

    def test_hdfeos5_groups(self):
        nt.assert_equal(set(self.f.groups.keys()), set(file_groups))

    def test_hdfeos5_variables(self):
        nt.assert_equal(set(self.f.variables.keys()), set(file_variables))

    def test_hdfeos5_attributes(self):
        nt.assert_equal(self.f.attributes, file_attributes)

    def test_hdfeos5_dimensions(self):
        nt.assert_equal(self.f.dimensions, file_dimensions)

    def test_hdfeos5_var_shapes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.shape, var_shapes[var])

    def test_hdfeos5_var_attributes(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.attributes, var_attributes[var])

    def test_hdfeos5_var_dimensions(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(v.dimensions, var_dimensions[var])

    def test_hdfeos5_var_coordinates(self):
        for var in self.f.variables.keys():
            v = self.f.variables[var]
            nt.assert_equal(get_coord_dims(v), var_coordinates[var])


file_attributes = {'Aerosol NearUV Grid/GCTPProjectionCode': np.array([0], dtype=np.int32), 'Aerosol NearUV Grid/Projection': 'Geographic', 'Aerosol NearUV Grid/GridOrigin': 'Center', 'Aerosol NearUV Grid/GridSpacing': '(1.0,1.0)', 'Aerosol NearUV Grid/GridSpacingUnit': 'deg', 'Aerosol NearUV Grid/GridSpan': '(-180,180,-90,90)', 'Aerosol NearUV Grid/GridSpanUnit': 'deg', 'Aerosol NearUV Grid/NumberOfLongitudesInGrid': np.array([360], dtype=np.int32), 'Aerosol NearUV Grid/NumberOfLatitudesInGrid': np.array([180], dtype=np.int32), 'OrbitNumber': np.array([29485, 29486, 29487, 29488, 29489, 29490, 29491, 29492, 29493,
       29494, 29495, 29496, 29497, 29498, 29499, 29500, 29501, 29502,
       29503, 29504, 29505, 29506, 29507, 29508, 29509, 29510, 29511,
       29512, 29513, 29514, 29515, 29516, 29517, 29518, 29519, 29520,
       29521, 29522, 29523, 29524, 29525, 29526, 29527, 29528],
      dtype=np.int32), 'OrbitPeriod': np.array([5934., 5933., 5933., 5934., 5933., 5934., 5933., 5934., 5933.,
       5933., 5934., 5933., 5934., 5933., 5934., 5933., 5933., 5934.,
       5933., 5934., 5933., 5933., 5934., 5933., 5933., 5934., 5933.,
       5934., 5933., 5934., 5933., 5933., 5934., 5933., 5934., 5933.,
       5934., 5933., 5933., 5934., 5933., 5934., 5933., 5934.]), 'InstrumentName': 'OMI', 'ProcessLevel': '3', 'GranuleMonth': np.array([1], dtype=np.int32), 'GranuleDay': np.array([31], dtype=np.int32), 'GranuleYear': np.array([2010], dtype=np.int32), 'GranuleDayOfYear': np.array([31], dtype=np.int32), 'TAI93At0zOfGranule': np.array([5.39049607e+08]), 'PGEVersion': '"0.9.50"', 'StartUTC': '2010-01-30T12:15:00.000000Z', 'EndUTC': '2010-02-01T11:45:00.000000Z', 'Period': 'Daily'}
file_dimensions = {'Aerosol NearUV Grid/XDim': 360, 'Aerosol NearUV Grid/YDim': 180}
file_groups = ['Aerosol NearUV Grid', '/']
file_variables = ['Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth388', 'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth388', 'Aerosol NearUV Grid/FinalAerosolSingleScattAlb388', 'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth500', 'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth500', 'Aerosol NearUV Grid/FinalAerosolSingleScattAlb500']

var_attributes = {'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth388': {'_FillValue': np.array([-1.2676506e+30], dtype=np.float32), 'Units': 'NoUnits', 'Title': 'Final Aerosol Absorption Optical Depth at 388 nm', 'UniqueFieldDefinition': 'OMI-Specific', 'ScaleFactor': np.array([1.]), 'Offset': np.array([0.]), 'MissingValue': np.array([-1.2676506e+30], dtype=np.float32), 'projection': 'Geographic', 'long_name': 'FinalAerosolAbsOpticalDepth388'},
                  'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth388': {'_FillValue': np.array([-1.2676506e+30], dtype=np.float32), 'Units': 'NoUnits', 'Title': 'Final Aerosol Extinction Optical Depth at 388 nm', 'UniqueFieldDefinition': 'OMI-Specific', 'ScaleFactor': np.array([1.]), 'Offset': np.array([0.]), 'MissingValue': np.array([-1.2676506e+30], dtype=np.float32), 'projection': 'Geographic', 'long_name': 'FinalAerosolExtOpticalDepth388'},
                  'Aerosol NearUV Grid/FinalAerosolSingleScattAlb388': {'_FillValue': np.array([-1.2676506e+30], dtype=np.float32), 'Units': 'NoUnits', 'Title': 'Final Aerosol Single Scattering Albedo at 388 nm', 'UniqueFieldDefinition': 'OMI-Specific', 'ScaleFactor': np.array([1.]), 'Offset': np.array([0.]), 'MissingValue': np.array([-1.2676506e+30], dtype=np.float32), 'projection': 'Geographic', 'long_name': 'FinalAerosolSingleScattAlb388'},
                  'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth500': {'_FillValue': np.array([-1.2676506e+30], dtype=np.float32), 'Units': 'NoUnits', 'Title': 'Final Aerosol Absorption Optical Depth at 500 nm', 'UniqueFieldDefinition': 'OMI-Specific', 'ScaleFactor': np.array([1.]), 'Offset': np.array([0.]), 'MissingValue': np.array([-1.2676506e+30], dtype=np.float32), 'projection': 'Geographic', 'long_name': 'FinalAerosolAbsOpticalDepth500'},
                  'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth500': {'_FillValue': np.array([-1.2676506e+30], dtype=np.float32), 'Units': 'NoUnits', 'Title': 'Final Aerosol Extinction Optical Depth at 500 nm', 'UniqueFieldDefinition': 'OMI-Specific', 'ScaleFactor': np.array([1.]), 'Offset': np.array([0.]), 'MissingValue': np.array([-1.2676506e+30], dtype=np.float32), 'projection': 'Geographic', 'long_name': 'FinalAerosolExtOpticalDepth500'},
                  'Aerosol NearUV Grid/FinalAerosolSingleScattAlb500': {'_FillValue': np.array([-1.2676506e+30], dtype=np.float32), 'Units': 'NoUnits', 'Title': 'Final Aerosol Single Scattering Albedo at 500 nm', 'UniqueFieldDefinition': 'OMI-Specific', 'ScaleFactor': np.array([1.]), 'Offset': np.array([0.]), 'MissingValue': np.array([-1.2676506e+30], dtype=np.float32), 'projection': 'Geographic', 'long_name': 'FinalAerosolSingleScattAlb500'}}
var_coordinates = {'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth388': [], 'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth388': [], 'Aerosol NearUV Grid/FinalAerosolSingleScattAlb388': [], 'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth500': [], 'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth500': [], 'Aerosol NearUV Grid/FinalAerosolSingleScattAlb500': []}
var_dimensions = {'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth388': ('YDim', 'XDim'), 'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth388': ('YDim', 'XDim'), 'Aerosol NearUV Grid/FinalAerosolSingleScattAlb388': ('YDim', 'XDim'), 'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth500': ('YDim', 'XDim'), 'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth500': ('YDim', 'XDim'), 'Aerosol NearUV Grid/FinalAerosolSingleScattAlb500': ('YDim', 'XDim')}
var_shapes = {'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth388': (180, 360), 'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth388': (180, 360), 'Aerosol NearUV Grid/FinalAerosolSingleScattAlb388': (180, 360), 'Aerosol NearUV Grid/FinalAerosolAbsOpticalDepth500': (180, 360), 'Aerosol NearUV Grid/FinalAerosolExtOpticalDepth500': (180, 360), 'Aerosol NearUV Grid/FinalAerosolSingleScattAlb500': (180, 360)}

def get_coord_dims(var):
    return [dim for dim in var.dimensions if dim in var.file.variables.keys()]

if __name__ == '__main__':
    ut.main()
