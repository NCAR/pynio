from numpy.testing import *
#import mfio as Nio
import Nio
import numpy as N
from numpy import ma
import os
from xarray import xArray

verbose = True
filename = os.tempnam(None, 'test_')
filename += '.nc'
print 'Creating temporary file: ', filename

def do_setup(filename):
    if os.path.exists(filename): os.remove(filename)
    f = Nio.open_file(filename, 'c')
    (nx, ny, nz, nt,ns) = (21, 21, 12, 10,1)
    (dx, dy, dz, dt) = (1000., 1000., 400., 3600.)
    f.create_dimension('xc', nx)
    f.create_dimension('yc', ny)
    f.create_dimension('zc', nz)
    f.create_dimension('time', nt)
    f.create_dimension('single', ns)
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
    var = f.create_variable('PTS', 'f', ('single','time', 'zc', 'yc', 'xc'))
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
    #print a.shape
    mask = N.zeros(a.shape,N.bool_)
    mask[:,3,:,:] = 1
    # tests adding a fill value

    am = ma.array(a,mask=mask)
    f.variables['PT'][:] = am[:]
    f.variables['PTS'][:] = am[:]
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

class test_sub_1_assign(NumpyTestCase):
    def setUp(self):
        print 'Creating temporary file: ', filename
        do_setup(filename)
        self.f = Nio.open_file(filename,mode='w')

    def check_sub_1_assign(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing sub 1 assign'
	v = file.variables['PT']
	vm = v[0,0]
	v[0,0,5,:] = vm[0,:] 
        file.close()
	

class test_sub_2_assign(NumpyTestCase):
    def setUp(self):
        print 'Creating temporary file: ', filename
        do_setup(filename)
        self.f = Nio.open_file(filename,mode='w')

    def check_test_sub_2_assign(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing sub 2 assign'
	v = file.variables['PT']
	vm = v[0,:,0,:]
	v[0,1:4,4,:] = vm[3:6,:]
        file.close()

class test_sub_3_assign(NumpyTestCase):
    def setUp(self):
        print 'Creating temporary file: ', filename
        do_setup(filename)
        self.f = Nio.open_file(filename,mode='w')

    def check_test_sub_3_assign(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing sub 3 assign'
	v = file.variables['PT']
	vm = v[0,:,:,:]
	vm = vm.reshape([1,1,vm.shape[0],vm.shape[1],vm.shape[2]])
	v[0,1:4,4,:] = vm[:,:,3:6,9,:]
        file.close()

class test_sub_4_assign(NumpyTestCase):
    def setUp(self):
        print 'Creating temporary file: ', filename
        do_setup(filename)
        self.f = Nio.open_file(filename,mode='w')

    def check_test_sub_4_assign(self):
        file = self.f

        #if verbose: print file
	if verbose: print 'testing sub 4 assign'
	v = file.variables['PTS']
	vm = v[0,0,:,:,:]
	vm = vm.reshape([1,1,vm.shape[0],vm.shape[1],vm.shape[2]])
	v[:,0,1:4,4,:] = vm[:,:,3:6,9,:]
	v[:,0,1:4,4,:] = vm[0,:,3:6,9,:]
        file.close()
	

if __name__ == "__main__":
    NumpyTest().test(level=11, all=False)
    #NumpyTest().test(testcase_pattern='test_nocrd')
    if os.path.exists(filename): os.remove(filename)
