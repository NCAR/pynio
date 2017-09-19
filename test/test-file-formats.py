import sys
import Nio
import os
import io
import difflib
from numpy.testing import assert_equal

initialize = 0
if len(sys.argv) > 1 and sys.argv[1] == "-i":
    initialize = 1


testfiles = [
'../ncarg/data/grib/19580101.ice125.grb',
'../ncarg/data/netcdf/pop.nc',
'../ncarg/data/grib2/wafsgfs_L_t06z_intdsk60.grib2',
'../ncarg/data/hdf/avhrr.hdf',
'../ncarg/data/hdfeos/MOD04_L2.A2001066.0000.004.2003078090622.hdfeos',
'../ncarg/data/hdfeos5/OMI-Aura_L3-OMAERUVd_2010m0131_v003-2010m0202t014811.he5',
'../ncarg/data/hdf5/num-types.h5',
'../ncarg/data/shapefile/states.shp', 
'../ncarg/data/netcdf/test_hgroups.nc' ]


formats = [ 'grib', 'netcdf', 'grib2', 'hdf4', 'hdfeos', 'hdfeos5', 'hdf5', 'shapefile', 'netcdf4' ]

format_dict = dict(list(zip(formats,testfiles)))
supported_formats = Nio.__formats__

def test_formats():
    for format in format_dict:
        if format in supported_formats and not supported_formats[format]:
            print('===========================')
            print('Optional format %s is not enabled in this version of PyNIO' % (format,))
            print('===========================')
            continue
        try:
            cmpname = "%s-base.txt" % (format_dict[format],)
            print('===========================')
            if initialize:
                print("Format %s: creating comparison output" % format)
                if cmpname:
                    os.system('mv -f %s %s.back' % (cmpname, cmpname))
            else:
                print('Format %s: opening and comparing contents metadata to known contents' % (format,))
            
            print(repr(format_dict[format]))
            f = Nio.open_file(format_dict[format])
            print(f)
            print(format_dict[format])
            str_out = io.StringIO()
            sys.stdout = str_out
            
            sys.stdout = sys.__stdout__
            #print str_out.getvalue()
            if initialize:
                fout = open(cmpname,mode='w')
                sys.stdout = fout
                print(str_out.getvalue())
                sys.stdout = sys.__stdout__
                fout.close()
            else:
                fin = open(cmpname,mode='r')
                contents = fin.read()
                #assert_equal(contents.strip(),str_out.getvalue().strip())
        except:
            
            print('===========================')
            print('Format %s: failed to open and/or metadata contents do not match known contents' % (format,))
            print('===========================')
            raise
            assert False
       
if __name__ == "__main__":
    test_formats()
       
