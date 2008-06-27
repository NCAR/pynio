
#@+leo-ver=4-thin
#@+node:schmidli.20080321230001.1:@thin xarray.py
#@@language python
#@<< xarray documentation >>
#@+node:schmidli.20080321230412:<< xarray documentation >>
"""xarray module 

Defines the class xArray and some helper functions.

xArray supports multilinear interpolation and masked indexing arrays
in addition to numpy's basic slicing and advanced selection mechanisms.

"""
#@-node:schmidli.20080321230412:<< xarray documentation >>
#@nl
#@<< xarray declarations >>
#@+node:schmidli.20080321230001.2:<< xarray declarations >>
import numpy as N
from numpy import asarray

__version__ = '0.1.1'
__all__ = ['xArray', 'intp', 'rindex', 'rindex2', 'rline', '__version__']

#@-node:schmidli.20080321230001.2:<< xarray declarations >>
#@nl
#@+others
#@+node:schmidli.20080321230001.3:class xArray
class xArray(N.ndarray):
    """ Extends numpy's ndarray class to support multilinear 
    interpolation and masked indexing arrays. In other words the
    elements of the selection tuble may contain floating point 
    numbers and masked arrays.

    """
    #@	@+others
    #@+node:schmidli.20080321230001.4:__new__
    def __new__(subtype, data, dtype=None, copy=True):
        """
            Returns an xArray object.
        """
        if isinstance(data, N.ma.MaskedArray):
            mask = data.mask
            masked = True
            fill_value = data._fill_value
            data = data.filled(fill_value)
        else:
            mask = False
            masked = False
            fill_value = None
        arr = N.array(data, dtype=dtype, copy=copy)
        shape = arr.shape
        ret = N.ndarray.__new__(subtype, shape, arr.dtype, buffer=arr)
        ndim = arr.ndim
        ret.masked = masked
        ret.mask = mask
        ret.fill_value = fill_value
        return ret
    #@-node:schmidli.20080321230001.4:__new__
    #@+node:schmidli.20080321230001.5:__array_finalize__
    def __array_finalize__(self, obj):
        """ """
        self.masked = False
        self.mask = False
        self.fill_value = None
    #@-node:schmidli.20080321230001.5:__array_finalize__
    #@+node:schmidli.20080321230001.6:__copy__
    def __copy__(self):
        """ """
        ret = N.ndarray.__copy__(self)
        ret.masked = self.masked
        ret.mask = self.mask
        ret.fill_value = self.fill_value
        return ret
    #@-node:schmidli.20080321230001.6:__copy__
    #@+node:schmidli.20080321230001.7:__getitem__
    def __getitem__(self, index):
        """x.__getitem__(y) <==> x[y]
        """
        ret = intp(asarray(self), index)
        return ret
    #@nonl
    #@-node:schmidli.20080321230001.7:__getitem__
    #@+node:schmidli.20080321230001.8:__str__
    def __str__(self):
        """ informal string representation """

        if self.masked:
            r = 'Masked: '
        else:
            r = ''
        r += str(self.__array__())
        return r
    #@-node:schmidli.20080321230001.8:__str__
    #@+node:schmidli.20080321230001.9:__repr__
    def __repr__(self):
        """ full string representation """

        r = 'xArray(' + self.__str__() + ')'
        return r
    #@-node:schmidli.20080321230001.9:__repr__
    #@+node:schmidli.20080321230001.10:getA
    def getA(self):
        """ base array """
        return asarray(self)

    A = property(getA)
    #@nonl
    #@-node:schmidli.20080321230001.10:getA
    #@-others
#@nonl
#@-node:schmidli.20080321230001.3:class xArray
#@+node:schmidli.20080321230001.12:intp
def intp(ar, sltup):
    """
        Returns the result of applying an extended selection
        tuple to the array. 

        In addition to numpy's basic slicing and advanced selection,
        masked indexing arrays and multi-linear interpolation are
        supported.

        Linear interpolation along a specific axis is triggered
        when the corresponding member of the selection tuple contains
        an object (scalar, list, array) with non-integer numbers.

        Examples, using shorthand notation:
            v = ar[5]
            v = ar[2.2,:,(4,5,6)]
            v = ar[[2.5,6.2],0,:]
            v = ar[:,[[2,2.2],[3,mv]],[[2],[3]],[[4,5]]]
    """
    if not isinstance(sltup, tuple):
        sltup = (sltup,)
    if len(sltup) != ar.ndim:
        raise ValueError, 'length of selection tuple does not match array dimension'

    # determine first axis which requires interpolation
    axis = None
    for i in range(len(sltup)):
        if isinstance(sltup[i], (N.ndarray, N.ma.MaskedArray)):
            if sltup[i].dtype in (N.single, N.double):
                axis = i
                break
        elif isinstance(sltup[i], slice):
            pass
        elif N.array(sltup[i]).dtype in (N.single, N.double):
            axis = i
            break

    # if no interpolation required: apply selection and quit
    if axis is None:
        if len(sltup) == 0:
            return ar[sltup]
        elif ismasked_tuple(sltup):
            sltup = list(sltup)
            axis_mask = 0
            for n in range(len(sltup)):
                if isinstance(sltup[n], N.ma.MaskedArray):
                    mask = sltup[n].mask
                    sltup[n] = sltup[n].filled(0)
                    axis_mask = n
            sltup = tuple(sltup)
            r = ar[sltup]
            #print "r.shape, mask.shape:", r.shape, mask.shape, axis_mask
            if axis_mask > 0:
                new_mask = N.zeros(r.shape, dtype='bool')
                sl = (None,)*axis_mask + (slice(None),)
                new_mask[:] = mask[sl]
                mask = new_mask
            if isinstance(r, N.ma.MaskedArray): 
                return N.ma.array(r, mask=mask | r.mask)
            else:
                return N.ma.array(r, mask=mask)
        else:
            return ar[sltup]

    # calculate indices and weights
    rj = sltup[axis]
    j = N.floor(rj).astype(N.int)
    j = N.clip(j, 0, ar.shape[axis]-2)
    jp = j+1
    f0j = rj-j
    f1j = 1-f0j
    sl = list(sltup)
    slp = list(sltup)
    sl[axis] = j
    slp[axis] = jp
    sl = tuple(sl); slp = tuple(slp)

    res = intp(ar, sl)*f1j + intp(ar, slp)*f0j
    return res
#@-node:schmidli.20080321230001.12:intp
#@+node:schmidli.20080321230916:rindex
def rindex(ar, val, axis=0, ep=0.5, clip=True, round=False, squeeze=True):
    """
        Assuming that ar is sorted in ascending order along axis, this
        function returns real indices of where val would fit (using linear
        interpolation).

        ar[j,I]     where I is a multi-dimensional index
        val[nv]     values
        ep          width of extrapolation zone
        clip        True: clip resulting indices to lie within a specified
                    range, i.e. -ep <= rj <= (n-1)+ep
                    False: set rj.mask = rj < -ep or rj > (n-1)+ep
        res[nv,I]

print rindex(N.arange(10), 5.5)
        [ 5.5]
print rindex(N.arange(10), -1)
        [-0.5]
print rindex(N.arange(10), [2,4.5])
        [ 2.   4.5]
    """

    # transpose input array if necessary
    if axis != 0:
        tr = N.arange(ar.ndim)
        tr[0] = axis
        tr[axis] = 0
        ar = ar.transpose(tr).copy()

    # bring input arrays into required shape
    # ar[nj,ni] and val[nv]
    ar_shape = list(ar.shape)
    nj = ar_shape[0]
    ni = ar.size/nj
    ar.shape = (nj,ni)
    is_scalar = N.isscalar(val)
    val_ndim = N.array(val).ndim
    val = N.atleast_1d(val)
    nv = val.size

    # determine the r-index
    if ar.shape[0] == 1:
        rj = N.zeros((1,ar.shape[1]), dtype=N.int)
    else:
        if ar[0,0] < ar[1,0]: # ascending order
            j1 = N.apply_along_axis(N.searchsorted, 0, ar, val).reshape(nv,ni)
            j1 = N.clip(j1, 1, nj-1)
            j0 = j1-1
            i = N.arange(ni).reshape(1,ni)
            rj = j0 + 1.0*(val.reshape(nv,1) - ar[j0,i])/(ar[j1,i] - ar[j0,i])
            if clip:
                rj = N.clip(rj, -ep, (nj-1)+ep)
            else:
                mask = (rj<-ep)|(rj>nj-1+ep)
                if mask.any():
                    rj = N.ma.array(rj, mask=mask)
        else: #descending order
            j1 = N.apply_along_axis(N.searchsorted, 0, ar[::-1], val).reshape(nv,ni)
            j1 = nj - j1 - 1
            j1 = j1.clip(0,nj-2)
            j0 = j1+1
            i = N.arange(ni).reshape(1,ni)
            rj = j0 - 1.0*(val.reshape(nv,1) - ar[j0,i])/(ar[j1,i] - ar[j0,i])
            if clip:
                rj = N.clip(rj, -ep, (nj-1)+ep)
            else:
                mask = (rj<-ep)|(rj>nj-1+ep)
                if mask.any():
                    rj = N.ma.array(rj, mask=mask)



    # back to multdimensional
    ar.shape = ar_shape
    ar_shape[0] = nv
    rj.shape = ar_shape

    # and transpose if necessary
    if axis != 0:
        if isinstance(rj, N.ma.MaskedArray):
            mask = rj.mask.transpose(tr)
            data = rj.filled(0.).transpose(tr).copy()
            rj = N.ma.array(data, mask=mask)
        else:
            rj = rj.transpose(tr).copy()

    if round:
        rj = N.round(rj).astype(N.int)

    if is_scalar and squeeze:
        rj_shape = list(rj.shape)
        del(rj_shape[axis])
        rj.shape = tuple(rj_shape)
        #if len(ar_shape) == 1:
        #    rj = rj[0]
    return rj

#@-node:schmidli.20080321230916:rindex
#@+node:schmidli.20080322212349:rindex2
def rindex2(ar, val, axis=0, ep=0.5, clip=True, round=False):
    """
        As function rindex, but returns a complete selection tuple.
    """

    val_ndim = N.array(val).ndim
    rj = rindex(ar, val, axis=axis, ep=ep, clip=clip, round=round, 
                squeeze=False)

    # determine selection tuples for the remaining dimensions
    rj_shape = list(rj.shape)
    rtup = []
    for i in xrange(rj.ndim):
        idx = N.arange(rj.shape[i])
        idx_shape = N.ones(rj.ndim)
        idx_shape[i] = rj.shape[i]
        idx.shape = idx_shape
        rtup.append(idx)
    rtup[axis] = rj

    if val_ndim == 0:
        for i in xrange(rj.ndim):
            r_shape = list(rtup[i].shape)
            if i != axis: del(r_shape[axis])
            rtup[i].shape = r_shape
    return tuple(rtup)
#@-node:schmidli.20080322212349:rindex2
#@+node:schmidli.20080325135032:rline
def rline(edges, delta=None, ns=None):
    """
        Returns a discretized polyline
        -> 

        edges[npts,ndim]    coordinates of the edge points of the 
                            line segments
        delta           length of a discretized segment
        ns              number of discretized segments
    """

    edges = N.array(edges)
    if edges.ndim != 2:
        raise ValueError('edges must be 2-dimensional')
    b = edges[1:,:] - edges[0:-1,:]
    nlines = b.shape[0]                     # number of line segments
    d = N.zeros(nlines, dtype=N.double)
    for i in range(nlines):
        d[i] = N.sqrt((b[i]**2).sum())
    length = d.sum()
    if delta is None: delta = 1
    if ns is None: ns = int(length/delta)   # number of discret segments

    # number of discretized segments per line segment
    n = N.zeros(nlines, dtype=N.int_)
    for i in range(nlines):
        n[i] = N.round(d[i]/length*ns)

    rj = N.zeros((n.sum()+1,edges.shape[1]), dtype=N.double)
    idx = N.zeros(nlines+1, dtype=N.int_)
    idx[1:] = n.cumsum()
    for i in xrange(nlines):
        t = N.linspace(0,1,n[i]+1)
        t.shape = (t.shape[0],1)
        rj[idx[i]:idx[i+1]+1,:] = edges[i,:] + b[i,:]*t
    return rj
#@-node:schmidli.20080325135032:rline
#@+node:schmidli.20080321230001.13:ismasked_tuple
def ismasked_tuple(sltup):
    """ """
    for n in range(len(sltup)):
        if isinstance(sltup[n], N.ma.MaskedArray):
            return True
    return False
#@-node:schmidli.20080321230001.13:ismasked_tuple
#@-others
#@-node:schmidli.20080321230001.1:@thin xarray.py
#@-leo

