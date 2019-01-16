from __future__ import print_function, division
import unittest as ut
from numpy.testing import assert_equal

import Nio
import numpy as N
from numpy import ma
import os
import tempfile
N.set_printoptions(precision=4)
verbose = False

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
    #a = N.array(N.random.randn(nt,nz,ny,nx), dtype='float32')
    a = N.arange(nt*nz*ny*nx,dtype = 'float32')
    a = a.reshape(nt,nz,ny,nx)
    f.variables['PT'][:] = a
    #a = N.zeros((nz,ny,nx))
    #a[:] = N.arange(nz)[:,N.newaxis,N.newaxis]
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


def do_setup_nocrd(filename):
    if os.path.exists(filename): os.remove(filename)
    f = Nio.open_file(filename, 'c')
    (nx, ny, nz, nt) = (20, 25, 5, 10)
    (dx, dy, dz, dt) = (1000., 1000., 800., 3600.)
    f.create_dimension('xc', nx)
    f.create_dimension('yc', ny)
    f.create_dimension('zc', nz)
    f.create_dimension('time', nt)
    f.Conventions = 'CF-1.0'
    f.source = 'ARPS'

    var = f.create_variable('time', 'f', ('time',))
    setattr(var, 'axis', 'T')
    setattr(var, 'units', 'seconds since 2007-03-21 06:00:00')
    var = f.create_variable('PT', 'f', ('time', 'zc', 'yc', 'xc'))
    var = f.create_variable('ZP', 'f', ('zc', 'yc', 'xc'))

    xc = N.arange(nx, dtype='float32')*dx
    yc = N.arange(ny, dtype='float32')*dy
    f.variables['time'][:] = N.arange(nt, dtype='float32')*dt
    #a = N.array(N.random.randn(nt,nz,ny,nx), dtype='float32')
    a = N.arange(nt*nz*ny*nx,dtype = 'float32')
    a = a.reshape(nt,nz,ny,nx)
    f.variables['PT'][:] = a
    a = N.zeros((nz,ny,nx))
    a[:] = N.arange(nz)[:,N.newaxis,N.newaxis]
    f.variables['ZP'][:] = N.array(a, dtype='float32')
    f.close()


class Test(ut.TestCase):
    def setUp(self):
        #print 'Creating temporary file: ', filename
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup(self.filename)
        self.f = Nio.open_file(self.filename)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_basic(self):
        # check inp2xsel
        xsel = Nio.inp2xsel(self.f, 'PT', 'time|i9 zc|i0 yc|i0 xc|i10:20:2')
        xsel = Nio.inp2xsel(self.f, 'PT', 'time|3600')

        xc_orig = self.f.variables['xc'][:]
        pt_orig = self.f.variables['PT'][:]
        if verbose: print('xc: ', xc_orig)
        if verbose: print('pt.shape: ', pt_orig.shape)
        if verbose: print()

        xsel_list = (5, slice(5,8), slice(None), slice(None,None,4))
        for xsel in xsel_list:
            if verbose: print('xsel: ', xsel)
            xc = self.f.variables['xc'][xsel]
            if verbose: print('xc[xsel]: ', xc)
            if verbose: print()
            assert_equal(xc, xc_orig[xsel])

        ptsel_list = ((1,1,1,1), (1,slice(None),0,0), (1,3,slice(5,8)), (slice(None),3,slice(None),1))
        for ptsel in ptsel_list:
            if verbose: print('ptsel: ', ptsel)
            pt = self.f.variables['PT'][ptsel]
            if verbose: print('pt[ptsel].shape: ', pt.shape)
            if verbose: print()
            assert_equal(pt.shape, pt_orig[ptsel].shape)

    def test_scalar(self):
        xc_orig = self.f.variables['xc'][:]
        pt_orig = self.f.variables['PT'][:]
        if verbose: print('xc: ', xc_orig)
        if verbose: print('pt.shape: ', pt_orig.shape)
        if verbose: print()

        xsel_list = ('1500', '1500i', 'xc|i5', 'xc|i5.5', 'xc|i5.5i', 'xc|1500' , 'xc|1500i')
        results = (2000, 1500, 5000, 6000, 5500, 2000, 1500)
        for (xsel, res) in zip(xsel_list, results):
            if verbose: print('xsel: ', xsel)
            xc = self.f.variables['xc'][xsel]
            if verbose: print('xc[xsel]: ', xc)
            if verbose: print()
            assert_equal(xc, res)

    def test_slice(self):
        xc_orig = self.f.variables['xc'][:]
        pt_orig = self.f.variables['PT'][:]
        if verbose: print('xc: ', xc_orig)
        if verbose: print('pt.shape: ', pt_orig.shape)
        if verbose: print()

        xsel_list = ('2000:4000', '1410:3900', '1500:3500:1000i', 'xc|1500:3500:1000', 'xc|i5:9', \
            'xc|i5.2:7.9', 'xc|i3:9:1.5i', 'xc|i3:9:1.5', '9k::4k', ':6k:3k', '::10k')
        results = ((2000,3000,4000), (2000, 3000), (1500,2500,3500), (2000,3000), \
                (5000,6000,7000,8000,9000), (6000, 7000), \
                (3000, 4500, 6000, 7500, 9000), (3000,5000,7000,9000), (9000,13000,17000), \
                (0,3000,6000), (0,10000,20000))
        for (xsel, res) in zip(xsel_list, results):
            if verbose: print('xsel: ', xsel)
            xc = self.f.variables['xc'][xsel]
            if verbose: print('xc[xsel]: ', xc)
            if verbose: print()
            assert_equal(xc, res)

        # Errors
        xsel_list = ('1500:3500i', '9.5k::1.5ki')

    def test_vector(self):
        xc_orig = self.f.variables['xc'][:]
        pt_orig = self.f.variables['PT'][:]
        if verbose: print('xc: ', xc_orig)
        if verbose: print('pt.shape: ', pt_orig.shape)
        if verbose: print()

        xsel_list = ('2000,4000', '1410,3900', '1500,3500i', 'xc|1500,i', 'xc|i5,9', \
            'xc|i5.2,7.9', 'xc|i3,9.2i')
        results = ((2000,4000), (1000, 4000), (1500,3500), (1500,), \
                (5000,9000), (5000, 8000), (3000, 9200))
        for (xsel, res) in zip(xsel_list, results):
            if verbose: print('xsel: ', xsel)
            xc = self.f.variables['xc'][xsel]
            if verbose: print('xc[xsel]: ', xc)
            if verbose: print()
            assert_equal(xc, res)

        # Errors
        xsel_list = ('1500,3500,1000')

    def test_extended(self):
        # basic case
        cstr_list = ('time|i0:6:3 zc|:3k:1k yc|i5:8:1 xc|0k:10k:2k', \
                'time|i0:6:3 zc|0:3k:500i yc|i5.5:8:0.5i xc|0k:10k:2k', \
                'time|i0:6:3 zc|ZP|2.5 yc|i5.5:8:0.5i xc|0k:10k:2k', \
                'time|i0:6:3 zc|ZP|2.5,3.5 yc|i5.5:8:0.5i xc|0k:10k:2k')
        results = ((3,4,4,6), (3,7,6,6), (3,6,6), (3,2,6,6))

        for (cstr, res) in zip(cstr_list, results):
            if verbose: print(cstr)
            pt = self.f.variables['PT'][cstr]
            if verbose: print(pt.shape)
            assert_equal(pt.shape, res)
 
        # ERROR:
        #cstr = 'xc|10k yc|i5.5:8:0.5i zc|ZP|2.5,3.5 time|i0:6:3'
        #if verbose: print cstr
        #pt = self.f.variables['PT'][cstr]
        #if verbose: print pt.shape

    def test_old(self): 
        var = self.f.variables['PT']

        if verbose: print("var[2,3,0,5:10]      # Nio selection works as usual")
        pt = var[2,3,0,5:10]
        if verbose: print(pt)
        assert_equal(pt.shape, (5,))

        cstr_list = ('time|i3 zc|i0 yc|i0 xc|0k:10k:2k', \
                    'i3 i0 i0 0k:10k:2k', \
                    'time|i3 zc|i0 yc|i0 xc|0', \
                    'time|i3:6 zc|i1 yc|i0 xc|2', \
                    'time|i0:6:6 zc|i1 yc|i0 xc|2', \
                    'time|i2:8:2 zc|i0 yc|i0 xc|i0',\
                    'zc|200,300,450,600', \
                    'time|i6 zc|200,300,450,600 yc|i0 xc|i0', \
                    'time|i6 zc|i0,2,3 yc|i0 xc|i0', \
                    'time|i6 zc|50, yc|i0 xc|i0', \
                    'time|i6 zc|i5, yc|i0 xc|i0',\
                    'time|i6 zc|50,100,175,350 yc|i1 xc|i0', \
                    'zc|50,100,175,350 yc|i1 xc|i0 time|i5:9',\
                    'time|i6:9 zc|50,100,175,350 yc|i1 xc|i0', \
                    'time|i6 zc|ZP|1.5 yc|i1:3 xc|i0:3', \
                    'time|i6 zc|ZP|1.5 yc|i1 xc|i1.5i', \
                    'time|i6 zc|ZP|1.5 yc|i1,1 xc|i0:3', \
                    'time|i6 zc|ZP|1.5,2.5 yc|i1 xc|i0', \
                    'time|i6 zc|ZP|1.5:2.5:0.5 yc|i1 xc|i0', \
                    'time|i6 zc|ZP|1.5:2.5:0.5 yc|: xc|:', \
                    'zc|ZP|1.5:2.5:0.5 time|i3:6 yc|: xc|:', \
                    )
        results = ((6,), (6,), (), (4,), (2,), (4,), (10,4,21,21), \
                (4,), (3,), (1,), (1,), (4,), (4,5), (4,4), (3,4), (), \
                (2,4), (2,), (3,), (3,21,21), (3,4,21,21))

        for (cstr, res) in zip(cstr_list, results):
            if verbose: print(cstr)
            fld = var[cstr]
            if verbose: print(fld.shape)
            assert_equal(fld.shape, res)

    def test_lonlat(self):
        var = self.f.variables['PT']

        cstr_list = ('time|i3 zc|i0 yc|lat|20k xc|0', \
                    'time|i3 zc|i0 yc|i0 xc|lon|10k', \
                    #'time|i3 zc|i0 yc|lat|20k xc|lon|10k', \
                    #'time|i3 zc|i0:2 yc|lat|20k xc|lon|10k', \
                    #'time|i6 zc|ZP|1.5 yc|lat|i1:3 xc|lon|i0:3', \
                    )
        results = ((), (), (), (3,), (3,4))

        for (cstr, res) in zip(cstr_list, results):
            if verbose: print(cstr)
            fld = var[cstr]
            if verbose: print(fld.shape)
            assert_equal(fld.shape, res)

    def test_topo(self):
        # basic case
        cstr_list = ('time|i0 zc|ZP|2500 yc|i5 xc|:', \
                'time|i0 zc|ZP|2500m yc|i5 xc|:', \
                'time|i0 zc|ZP|1500m yc|i5 xc|:', \
                'time|i0 zc|ZP|1000,1500m yc|i5.5 xc|:')
        results = ((21,), (21,), (21,), (2,21))

        for (cstr, res) in zip(cstr_list, results):
            if verbose:
                print(cstr)
                print("in test_topo")
            xsel = Nio.inp2xsel(self.f, 'PT', cstr)
            pt = self.f.variables['PT'][cstr]
            #pt = self.f.variables['ZP'][:]
            if verbose: print(pt.shape)
            if verbose: 
                if ma.isMA(pt):
                    print(N.asarray(pt.filled()))
                else:
                    print(pt)
            assert_equal(pt.shape, res)

        # ERROR:
        #cstr = 'xc|10k yc|i5.5:8:0.5i zc|ZP|2.5,3.5 time|i0:6:3'
        #if verbose: print cstr
        #pt = self.f.variables['PT'][cstr]
        #if verbose: print pt.shape


class test_cf_extended(ut.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup(self.filename)
        opt = Nio.options()
        opt.UseAxisAttribute = True
        self.f = Nio.open_file(self.filename, options = opt)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_cf_extended(self):
        # basic case
        cstr_list = ('t|i0:6:3 z|:3k:1k y|i5:8:1 x|0k:10k:2k', \
                't|i0:6:3 z|0:3k:500i y|i5.5:8:0.5i x|0k:10k:2k', \
                't|i0:6:3 z|ZP|2500 y|i5.5:8:0.5i x|0k:10k:2k', \
                't|i0:6:3 z|ZP|2500,3500 y|i5.5:8:0.5i x|0k:10k:2k')
        results = ((3,4,4,6), (3,7,6,6), (3,6,6), (3,2,6,6))

        for (cstr, res) in zip(cstr_list, results):
            if verbose: print(cstr)
            pt = self.f.variables['PT'][cstr]
            if verbose: print(pt.shape)
            assert_equal(pt.shape, res)

        # ERROR:
        #cstr = 'xc|10k yc|i5.5:8:0.5i zc|ZP|2.5,3.5 time|i0:6:3'
        #if verbose: print cstr
        #pt = file.variables['PT'][cstr]
        #if verbose: print pt.shape


class test_nocrd(ut.TestCase):
    def setUp(self):
        #filename = 'dat/test_nocrd.nc'
        self.filename = tempfile.mktemp(prefix="test_", suffix=".nc")
        do_setup_nocrd(self.filename)
        self.f = Nio.open_file(self.filename)

    def tearDown(self):
        self.f.close()
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def test_nocrd(self):
        var = self.f.variables['PT']

        if verbose: print("var[2,3,0,5:10]      # Nio selection works as usual")
        pt = var[2,3,0,5:10]
        if verbose: print(pt)
        assert_equal(pt.shape, (5,))

        cstr_list = ('time|i9 zc|i0 yc|i0 xc|i10:18:2', \
                    'i9 i0 i0 i10:18:2', \
                    'time|i2:8:2 zc|i0 yc|i0 xc|i0',\
                    )
        results = ((5,), (5,), (4,))

        for (cstr, res) in zip(cstr_list, results):
            if verbose: print(cstr)
            fld = var[cstr]
            if verbose: print(fld.shape)
            assert_equal(fld.shape, res)


if __name__ == "__main__":
     ut.main()
