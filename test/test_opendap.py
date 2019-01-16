from __future__ import print_function, division
import sys, Nio
import numpy.testing as nt
import unittest as ut

class Test(ut.TestCase):
    def test_opendap1(self):
    # 
    # This is a test for OPeNDAP. It will only work on systems in which 
    # PyNIO has OPenDAP capabilities built in.

        if not Nio.__formats__['opendap']:
            print('===========================')
            print('Optional format OPeNDAP is not enabled in this version of PyNIO')
            print('===========================')
            sys.exit(1)

        #
        # url = "http://apdrc.soest.hawaii.edu:80/dods/public_data/ERA-40/"
        # filename = "daily-pressure"
        #
        # The URL is so long, break it into two pieces.
        #
        url      = "http://apdrc.soest.hawaii.edu:80/dods/public_data/ERA-40/"
        filename = "daily-surface"
 
        f = Nio.open_file(url + filename)
        variables = sorted(list(f.variables.keys()))
        file_variables = ['lat', 'lon', 'param01', 'param02', 'param03', 'param04', 'param05', 'param06', 'param07', 'param08', 'param09', 'param10', 'param11', 'param12', 'param13', 'param14', 'param15', 'param16', 'param17', 'param18', 'param19', 'param20', 'param21', 'param22', 'param23', 'param24', 'param25', 'param26', 'param27', 'param28', 'param29', 'param30', 'param31', 'param32', 'param33', 'param34', 'param35', 'param36', 'param37', 'param38', 'param39', 'param40', 'param41', 'time']
        nt.assert_equal(variables, file_variables)

    def test_opendap2(self):
        if not Nio.__formats__['opendap']:
            print('===========================')
            print('Optional format OPeNDAP is not enabled in this version of PyNIO')
            print('===========================')
            sys.exit()

        url      = "http://test.opendap.org/opendap/data/nc/"
        filename = "bears.nc"

        f = Nio.open_file(url + filename)
        variables = sorted(list(f.variables.keys()))

        file_variables = ['aloan', 'bears', 'cross', 'i', 'j', 'l', 'order', 'shot']


if __name__ == '__main__':
    ut.main()
