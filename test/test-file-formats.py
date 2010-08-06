
import Nio

testfiles = [
'../ncarg/data/grib/ced1.lf00.t00z.eta.grb',
'../ncarg/data/grib2/wafsgfs_L_t06z_intdsk60.grib2',
'../ncarg/data/hdf/avhrr.hdf',
'../ncarg/data/hdfeos/MOD04_L2.A2001066.0000.004.2003078090622.hdfeos',
'../ncarg/data/hdfeos5/OMI-Aura_L3-OMAERUVd_2010m0131_v003-2010m0202t014811.he5',
'../ncarg/data/netcdf/pop.nc',
'../ncarg/data/shapefile/states.shp' ]

formats = [ 'GRIB', 'GRIB2', 'HDF', 'HDFEOS', 'HDFEOS5', 'NetCDF', 'Shapefile' ]

format_dict = dict(zip(formats,testfiles))

for format in format_dict:
   try:
       print '==========================='
       print 'Format %s: opening and printing contents' % (format,)
       print '==========================='
       f = Nio.open_file(format_dict[format])
       print f
   except:
       print '==========================='
       print 'Format %s: failed to open and/or print  contents' % (format,)
       print '==========================='
       
       
