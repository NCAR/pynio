import unittest as ut
from numpy.testing import assert_equal

import Nio
import numpy as N
from numpy import ma
import os
from xarray import xArray

verbose = True
filename = os.tempnam(None, 'test_')
filename += '.nc'
#print 'Creating temporary file: ', filename

def do_setup(filename):
    if os.path.exists(filename): os.remove(filename)
    f = Nio.open_file(filename, 'c')
    (nx, ny, nz, nt) = (21, 21, 12, 10)
    (dx, dy, dz, dt) = (1000., 1000., 400., 3600.)
    f.create_dimension('xc', nx)
    f.create_dimension('yc', ny)
    f.create_dimension('zc', nz)
    f.create_dimension('time', nt)
    f.Conventions = 'CF-1.0'
    f.source = 'ARPS'

    var = f.create_variable('xc', 'f', ('xc',))
    setattr(var, 'axis', 'X')
    var = f.create_variable('yc', 'f', ('yc',))
    setattr(var, 'axis', 'Y')
    var = f.create_variable('zc', 'f', ('zc',))
    setattr(var, 'axis', 'Z')
    var = f.create_variable('time', 'f', ('time',))
    setattr(var, 'axis', 'T')
    setattr(var, 'units', 'seconds since 2007-03-21 06:00:00')
    var = f.create_variable('PT', 'f', ('time', 'zc', 'yc', 'xc'))
    var = f.create_variable('ZP', 'f', ('zc', 'yc', 'xc'))
    var = f.create_variable('TOPO', 'f', ('yc', 'xc'))
    var = f.create_variable('lon', 'f', ('yc','xc'))
    var = f.create_variable('lat', 'f', ('yc','xc'))

    xc = N.arange(nx, dtype='float32')*dx
    yc = N.arange(ny, dtype='float32')*dy
    zc = N.arange(nz, dtype='float32')*dz
    f.variables['xc'][:] = xc
    f.variables['yc'][:] = yc
    f.variables['zc'][:] = zc
    f.variables['time'][:] = N.arange(nt, dtype='float32')*dt
    a = N.arange(nt*nz*ny*nx,dtype = 'float32')
    #a = N.array(N.random.randn(nt,nz,ny,nx), dtype='float32')
    a = a.reshape(nt,nz,ny,nx)
    print a.shape
    mask = N.zeros(a.shape,N.bool_)
    mask[:,3,:,:] = 1
    # tests adding a fill value

    am = ma.array(a,mask=mask)
    f.variables['PT'][:] = am[:]
    #if verbose: print f.variables['PT']
    H = 5000.
    topo = 1000*N.cos(2*N.pi*(xc-10000.)/20000.)+1000.
    zp = zc[:,N.newaxis]*(1-topo[N.newaxis,:]/H) + topo[N.newaxis,:]
    topof = N.zeros((ny, nx), dtype='float32')
    topof[:,:] = topo[N.newaxis,:]
    zpf = N.zeros((nz,ny,nx), dtype='float32')
    zpf[:] = zp[:,N.newaxis,:]
    f.variables['ZP'][:] = zpf
    f.variables['TOPO'][:] = topof
    f.variables['lon'][:] = N.cos(0.1)*xc[N.newaxis,:] - N.sin(0.1)*yc[:,N.newaxis]
    f.variables['lat'][:] = N.sin(0.1)*xc[N.newaxis,:] + N.cos(0.1)*yc[:,N.newaxis]
    f.close()

class test_masked_default(ut.TestCase):
    def setUp(self):
        #print 'Creating temporary file'
        do_setup(filename)
        self.f = Nio.open_file(filename)

    def test_masked_default(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing MaskedArrayMode default'
	v = file.variables['PT']
	assert_equal(v._FillValue,1e20)
	vm = v[0,0]
	assert_equal(N.array([vm._fill_value],dtype='f'),N.array([1e20],dtype='f'))
	if verbose: print vm[0]
        file.close()

class test_masked_if_fill_att(ut.TestCase):
    def setUp(self):
        #print 'Creating temporary file'
        do_setup(filename)
	opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedIfFillAtt'
        self.f = Nio.open_file(filename,options=opt)

    def test_masked_if_fill_att(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing MaskedArrayMode MaskedIfFillAtt'
	v = file.variables['lat']
	assert_equal(hasattr(v,'_FillValue'),False)
	vm = v[:]
	assert_equal(ma.isMaskedArray(vm),False)
	print type(vm),vm[0].__repr__()
	v = file.variables['PT']
	assert_equal(v._FillValue,1e20)
	vm = v[0,0]
	assert_equal(ma.isMaskedArray(vm),True)
	assert_equal(N.array([vm._fill_value],dtype='f'),N.array([1e20],dtype='f'))
	if verbose: print type(vm),vm[0].__repr__()
        file.close()


class test_masked_always(ut.TestCase):
    def setUp(self):
        #print 'Creating temporary file: ', filename
        do_setup(filename)
	opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedAlways'
        self.f = Nio.open_file(filename,options=opt)

    def test_masked_always(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing MaskedArrayMode MaskedAlways'
	v = file.variables['lat']
	assert_equal(hasattr(v,'_FillValue'),False)
	vm = v[:]
	assert_equal(N.array([vm.get_fill_value()],dtype='f'),N.array([1e20],dtype='f'))
	if verbose: print vm[1].__repr__
        file.close()

class test_masked_never(ut.TestCase):
    def setUp(self):
        do_setup(filename)
	opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedNever'
        self.f = Nio.open_file(filename,options=opt)

    def test_masked_never(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing MaskedArrayMode MaskedNever'
	v = file.variables['PT']
	assert_equal(v._FillValue,1e20)
	vm = v[0,3:5,0]
	if verbose: print type(vm),vm
	assert_equal(ma.isMaskedArray(vm),False)
        file.close()
	
class test_masked_if_att_and_val(ut.TestCase):
    def setUp(self):
        do_setup(filename)
	opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedIfFillAttAndValue'
        self.f = Nio.open_file(filename,options=opt)

    def test_masked_if_att_and_val(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing MaskedArrayMode MaskedIfFillAttAndValue'
	v = file.variables['PT']
	assert_equal(v._FillValue,1e20)
	vm = v[0,3:5,0]
	if verbose: print type(vm),vm
	assert_equal(ma.isMaskedArray(vm),True)
	assert_equal(N.array([vm._fill_value],dtype='f'),N.array([1e20],dtype='f'))
	vm = v[0,4:6,0]
	if verbose: print type(vm),vm
	assert_equal(ma.isMaskedArray(vm),False)
        file.close()

class test_masked_explicit(ut.TestCase):
    def setUp(self):
        do_setup(filename)
	opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedExplicit'
        self.f = Nio.open_file(filename,options = opt)

    def test_masked_explicit(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing MaskedArrayMode MaskedExplicit'
	v = file.variables['PT']
	assert_equal(v._FillValue,1e20)
	vm = v[0,3:5,0]
	if verbose: print type(vm),vm
	assert_equal(ma.isMaskedArray(vm),False)
        file.set_option('MaskedArrayMode','maskediffillatt')
        #setting explicitfillvalues sets maskedarraymode to 'maskedexplicit'
        file.set_option('explicitFillValues',1e20)
	vm = v[0,3:5,0]
	if verbose: print type(vm),vm
	assert_equal(ma.isMaskedArray(vm),True)
        file.set_option('MaskBelowValue',1770)
	vm = v[0,3:5,0]
	if verbose: print type(vm),vm
	assert_equal(ma.isMaskedArray(vm),True)
        file.set_option('MaskAboveValue',1780)
	vm = v[0,3:5,0]
	if verbose: print type(vm),vm
	assert_equal(ma.isMaskedArray(vm),True)
        file.set_option('MaskAboveValue',1770)
        file.set_option('MaskBelowValue',1780)
	vm = v[0,3:5,0]
	if verbose: print type(vm),vm
	assert_equal(ma.isMaskedArray(vm),True)
        file.close()
	


if __name__ == "__main__":
     ut.main()
     if os.path.exists(filename): os.remove(filename)
