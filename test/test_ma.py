from __future__ import print_function, division
import unittest as ut
from numpy.testing import assert_equal

import Nio
import numpy as N
from numpy import ma
import os
import tempfile

verbose = False

#print ('Creating temporary file: ', filename)

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
    if verbose: print(a.shape)
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
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup(self.filename)
        self.f = Nio.open_file(self.filename)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_masked_default(self):
        #if verbose: print self.f
        if verbose: print('testing MaskedArrayMode default')
        v = self.f.variables['PT']
        assert_equal(v._FillValue,1e20)
        vm = v[0,0]
        try:
            assert_equal(vm.get_fill_value(),N.array([1e20],dtype='f')) # numpy 1.6.1rc1
        except:
            assert_equal(N.array([vm._fill_value],dtype='f'),N.array([1e20],dtype='f')) # numpy 1.4.1
        if verbose: print(vm[0])

class test_masked_if_fill_att(ut.TestCase):
    def setUp(self):
        #print 'Creating temporary file'
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup(self.filename)
        opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedIfFillAtt'
        self.f = Nio.open_file(self.filename,options=opt)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_masked_if_fill_att(self):
        #if verbose: print self.f
        if verbose: print('testing MaskedArrayMode MaskedIfFillAtt')
        v = self.f.variables['lat']
        assert_equal(hasattr(v,'_FillValue'),False)
        vm = v[:]
        assert_equal(ma.isMaskedArray(vm),False)
        if verbose: print(type(vm),vm[0].__repr__())
        v = self.f.variables['PT']
        assert_equal(v._FillValue,1e20)
        vm = v[0,0]
        assert_equal(ma.isMaskedArray(vm),True)
        try:
            assert_equal(vm.get_fill_value(),N.array([1e20],dtype='f')) # numpy 1.6.1rc1
        except:
            assert_equal(N.array([vm._fill_value],dtype='f'),N.array([1e20],dtype='f')) # numpy 1.4.1
        if verbose: print(type(vm),vm[0].__repr__())


class test_masked_always(ut.TestCase):
    def setUp(self):
        #print 'Creating temporary file: ', filename
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup(self.filename)
        opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedAlways'
        self.f = Nio.open_file(self.filename,options=opt)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_masked_always(self):
        #if verbose: print self.f
        if verbose: print('testing MaskedArrayMode MaskedAlways')
        v = self.f.variables['lat']
        assert_equal(hasattr(v,'_FillValue'),False)
        vm = v[:]
        try:
            assert_equal(vm.get_fill_value(),N.array([1e20],dtype='f')) # numpy 1.6.1rc1
        except:
            assert_equal(N.array([vm._fill_value],dtype='f'),N.array([1e20],dtype='f')) # numpy 1.4.1
        if verbose: print(vm[1].__repr__)

class test_masked_never(ut.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup(self.filename)
        opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedNever'
        self.f = Nio.open_file(self.filename,options=opt)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_masked_never(self):
        #if verbose: print self.f
        if verbose: print('testing MaskedArrayMode MaskedNever')
        v = self.f.variables['PT']
        assert_equal(v._FillValue,1e20)
        vm = v[0,3:5,0]
        if verbose: print(type(vm),vm)
        assert_equal(ma.isMaskedArray(vm),False)

class test_masked_if_att_and_val(ut.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup(self.filename)
        opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedIfFillAttAndValue'
        self.f = Nio.open_file(self.filename,options=opt)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_masked_if_att_and_val(self):
        #if verbose: print self.f
        if verbose: print('testing MaskedArrayMode MaskedIfFillAttAndValue')
        v = self.f.variables['PT']
        assert_equal(v._FillValue,1e20)
        vm = v[0,3:5,0]
        if verbose: print(type(vm),vm)
        assert_equal(ma.isMaskedArray(vm),True)
        try:
            assert_equal(vm.get_fill_value(),N.array([1e20],dtype='f')) # numpy 1.6.1rc1
        except:
            assert_equal(N.array([vm._fill_value],dtype='f'),N.array([1e20],dtype='f')) # numpy 1.4.1
        vm = v[0,4:6,0]
        if verbose: print(type(vm),vm)
        assert_equal(ma.isMaskedArray(vm),False)

class test_masked_explicit(ut.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup(self.filename)
        opt = Nio.options()
        opt.MaskedArrayMode = 'MaskedExplicit'
        self.f = Nio.open_file(self.filename,options = opt)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_masked_explicit(self):
        #if verbose: print self.f
        if verbose: print('testing MaskedArrayMode MaskedExplicit')
        v = self.f.variables['PT']
        assert_equal(v._FillValue,1e20)
        vm = v[0,3:5,0]
        if verbose: print(type(vm),vm)
        assert_equal(ma.isMaskedArray(vm),False)
        self.f.set_option('MaskedArrayMode','maskediffillatt')
            #setting explicitfillvalues sets maskedarraymode to 'maskedexplicit'
        self.f.set_option('explicitFillValues',1e20)
        vm = v[0,3:5,0]
        if verbose: print(type(vm),vm)
        assert_equal(ma.isMaskedArray(vm),True)
        self.f.set_option('MaskBelowValue',1770)
        vm = v[0,3:5,0]
        if verbose: print(type(vm),vm)
        assert_equal(ma.isMaskedArray(vm),True)
        self.f.set_option('MaskAboveValue',1780)
        vm = v[0,3:5,0]
        if verbose: print(type(vm),vm)
        assert_equal(ma.isMaskedArray(vm),True)
        self.f.set_option('MaskAboveValue',1770)
        self.f.set_option('MaskBelowValue',1780)
        vm = v[0,3:5,0]
        if verbose: print(type(vm),vm)
        assert_equal(ma.isMaskedArray(vm),True)


if __name__ == "__main__":
     ut.main()
