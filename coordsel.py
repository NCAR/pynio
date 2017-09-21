#@+leo-ver=4-thin
#@+node:schmidli.20080322120238:@thin pymfio/coordsel.py
#@@language python
#@<< coordsel declarations >>
#@+node:schmidli.20080322120238.1:<< coordsel declarations >>
""" Coordinate space selection for NioVariable

    data = f.variables[varname][selobj]

    where selobj is one of the following:
        input string (inp)
        coordinate space selection object (csel)
        index space selection object (isel)
        index space selection object (correct dimensionality; xsel)
        a basic numpy slice object
"""
from __future__ import print_function
import copy
import datetime
import time
import numpy as N
import Nio
import collections
from _xarray import _intp, _rindex, _xArray

__version__ = '0.1.0'
__all__ = ['get_variable', 'inp2csel', 'inp2isel', 'inp2xsel', 'idxsel2xsel', \
        'xSelect', 'crdSelect', 'idxSelect', '__version__']

fact = {'k': 10**3, 'M': 10**6, 'h': 3600, 'm': 60, 'H': 100}
fact_keys = fact.keys()
#@-node:schmidli.20080322120238.1:<< coordsel declarations >>
#@nl
#@+others
#@+node:schmidli.20080322120238.2:get_variable
def get_variable(file, varname, xsel):
    """ get variable from file using extended selection """

    dims = file.variables[varname].cf_dimensions
    order = []
    do_transpose = False
    if isinstance(xsel, str):
        xsel = crdSelect(xsel, dims)
        if hasattr(xsel,'names'):
            tdims = list(dims) 
            for name in xsel.names:
                order.append(tdims.index(name))
            for i in range(len(order)-1):
                if order[i] > order[i+1]:
                    do_transpose = True
            if not do_transpose:
                order = []
    if isinstance(xsel, crdSelect):
        xsel = crdsel2idxsel(file, xsel)
    if isinstance(xsel, idxSelect):
        xsel = idxsel2xsel(file, xsel, dims, order)

    if hasattr(xsel,'order'):
        order = xsel.order
    if not isinstance(xsel, xSelect) or xsel.isbasic:
        ret = file.file.variables[varname][xsel]
    else:
        if xsel.masked:
            ret = file.file.variables[varname][:]
            ret = _xArray(ret)[xsel]
        else:
            bb = xsel.bndbox()
            ret = file.file.variables[varname][bb]
            rsel = xsel - bb
            ret = _intp(ret, rsel)

    if do_transpose and len(order) > 1:
        ret = ret.transpose(order)

    return ret
#@-node:schmidli.20080322120238.2:get_variable
#@+node:schmidli.20080322120238.3:inp2csel
def inp2csel(file, varname, csel):
    """ Convert the input string to a coordinate space selection object. """

    dims = file.variables[varname].cf_dimensions
    if isinstance(csel, str):
        csel = crdSelect(csel, dims)

    return csel
#@-node:schmidli.20080322120238.3:inp2csel
#@+node:schmidli.20080322135237:inp2isel
def inp2isel(file, varname, isel):
    """ Convert the input isel to a index space selection object. """

    dims = file.variables[varname].cf_dimensions
    if isinstance(isel, str):
        isel = crdSelect(isel, dims)
    if isinstance(isel, crdSelect):
        isel = crdsel2idxsel(file, isel)

    return isel
#@-node:schmidli.20080322135237:inp2isel
#@+node:schmidli.20080322135142:inp2xsel
def inp2xsel(file, varname, xsel):
    """ Convert the input xsel to a index space selection object. """

    dims = file.variables[varname].cf_dimensions
    if isinstance(xsel, str):
        xsel = crdSelect(xsel, dims)
    if isinstance(xsel, crdSelect):
        xsel = crdsel2idxsel(file, xsel)
    if isinstance(xsel, idxSelect):
        xsel = idxsel2xsel(file, xsel, dims,[])

    return xsel
#@-node:schmidli.20080322135142:inp2xsel
#@+node:schmidli.20080322120238.4:class crdSelect
class crdSelect(dict):
    """ crdSelect(inp)

    Create a coordinate space selection object from inp

    Examples:
        crdSelect('time|d20070321-6:18:3h xc|10k:20k:2k')
    """
    #@  @+others
    #@+node:schmidli.20080322120238.5:__init__
    def __init__(self, inp, dimensions):
        """ crdSelect(inp, dimensions) """
        self.names = []
        if isinstance(inp, str):
            inp = inp.strip()
            inpv = inp.split()
            data = {}
            if '|' in inp:      # named axes input
                for item in inpv:
                    itemv = item.split('|')
                    if len(itemv) < 2:
                        raise ValueError("Invalid input format")
                    key = itemv[0]
                    value = '|'.join(itemv[1:])
                    if dimensions is not None:
                        if key in dimensions:
                            data[key] = axisSelect(value)
                            self.names.append(key)
                        else:
                            pass
                            #raise ValueError, '"' + key + '" is not a valid axis name (UseAxisAttribute option setting?)'
                    else:
                        data[key] = axisSelect(value)
                        self.names.append(key)
            else:               # positional input
                if dimensions is None:
                    raise TypeError("Invalid type for dimensions")
                if len(inpv) > len(dimensions):
                    inpv = inpv[:len(dimensions)]
                for i in range(len(inpv)):
                    key = dimensions[i]
                    data[key] = axisSelect(inpv[i])
                    self.names.append(key)
                if len(inpv) < len(dimensions):
                    for i in range(len(inpv), len(dimensions)):
                        key = dimensions[i]
                        data[key] = axisSelect('i:')
                        self.names.append(key)

            dict.__init__(self, data)
        else:
            raise TypeError("Invalid input type")
    #@-node:schmidli.20080322120238.5:__init__
    #@+node:schmidli.20080322120238.6:__str__
    def __str__(self):
        """ string representation """
        _str = 'crdSelect(( '
        for key in self.keys():
            _str += str(key) +': ' + str(self[key]) + ', '
        _str = _str[:-2] + ' ))'

        return(_str)
    #@-node:schmidli.20080322120238.6:__str__
    #@-others
#@-node:schmidli.20080322120238.4:class crdSelect
#@+node:schmidli.20080322120238.7:class idxSelect
class idxSelect(dict):
    """ idxSelect(inp)

    Create a index space selection object from inp

    Examples:
        idxSelect(dimensions)
    """

    #@  @+others
    #@+node:schmidli.20080322120238.8:__init__
    def __init__(self, dimensions):
        """ idxSelect(dimensions) """

        if not isinstance(dimensions, collections.Iterable):
            raise TypeError("Invalid argument type")
        data = {}
        for key in dimensions:
            data[key] = axisSelect('i:')

        self = dict.__init__(self, data)
    #@-node:schmidli.20080322120238.8:__init__
    #@+node:schmidli.20080322120238.9:__str__
    def __str__(self):
        """ string representation """

        _str = 'idxSelect(( '
        for key in self.keys():
            _str += str(key) +': ' + str(self[key]) + ', '
        _str = _str[:-2] + ' ))'

        return(_str)
    #@-node:schmidli.20080322120238.9:__str__
    #@-others
#@-node:schmidli.20080322120238.7:class idxSelect
#@+node:schmidli.20080322120238.10:class xSelect
class xSelect(tuple):
    """ xSelect(inp)

    Create an extended selection object to be used with _xArray.

    """

    #@  @+others
    #@+node:schmidli.20080322120238.11:bndbox
    def bndbox(self):
        """ Return the bounding box of the current selection object. """
        ret = []
        for idx in self:
            if isinstance(idx, slice):
                start = idx.start
                stop = idx.stop
            elif isinstance(idx, N.ndarray):
                start = idx.min()
                stop = idx.max()
            else:
                idx = N.atleast_1d(idx)
                start = idx.min()
                stop = idx.max()
            if isinstance(start, float):
                start = N.floor(start).astype(N.int)
            if isinstance(stop, float):
                stop = N.ceil(stop).astype(N.int)
            if not isinstance(idx, slice):
                stop += 1
            # should be true only for interpolation cases, therefore stop>= 2
            if start < 0: 
                start = 0
                stop = max(2, stop)

            ret.append(slice(start, stop))
        return tuple(ret)

        # does this do anything?
        bb = property(bndbox)   
    #@-node:schmidli.20080322120238.11:bndbox
    #@+node:schmidli.20080322120238.12:__sub__
    def __sub__(self, bb):
        """ Subtract the bounding box (bb) from the current selection object. """
        rsel = list(self)
        if len(rsel) != len(bb):
            raise ValueError("Incompatible operands")
        for i in range(len(rsel)):
            if isinstance(self[i], slice):
                rsel[i] = slice(self[i].start-bb[i].start, \
                            self[i].stop-bb[i].start, self[i].step)
            else:
                rsel[i] = self[i] - bb[i].start
        return tuple(rsel)
    #@-node:schmidli.20080322120238.12:__sub__
    #@+node:schmidli.20080322120238.13:__str__
    def __str__(self):
        """ string representation """
        _str = 'xSelect('
        for item in self:
            _str += str(item) + ', '
        _str = _str[:-2] + '; isbasic: ' +str(self.isbasic)+ ')'

        return(_str)
    #@-node:schmidli.20080322120238.13:__str__
    #@-others
#@-node:schmidli.20080322120238.10:class xSelect
#@+node:schmidli.20080322120238.14:crdsel2idxsel
def crdsel2idxsel(file, csel):
    """ convert a coordinate space selection object to an index space
    selection object
    """

    isel = idxSelect(csel.keys())
    mdcrdname = {}
    for axis in isel.keys():
        if csel[axis].mdcrd is None:
            isel[axis] = csel[axis].toindex(file, axis)
        else:
            isel[axis] = axisIdxSelect(slice(0, file.cf_dimensions[axis]))
            mdcrdname[axis] = csel[axis].mdcrd

    if len(mdcrdname) > 1:
        raise NotImplementedError("More than one multi-dimensional coordinate are not yet supported")

    for axis in mdcrdname.keys():
        isel[axis] = csel[axis].toindex(file, axis, mdcrd=mdcrdname[axis], 
                                        isel=isel)
    return isel
#@-node:schmidli.20080322120238.14:crdsel2idxsel
#@+node:schmidli.20080322120238.15:idxsel2xsel
def idxsel2xsel(file, isel, dimensions, order):
    """ convert a index space selection object to an xSelect object
    """

    if not isinstance(isel, idxSelect):
        raise TypeError('wrong argument type')

    xsel = {}
    xsel_size = {}
    xsel_dims = {}
    isarray = False
    interp = False
    masked = False
    multidim = False

    i = 0
    for axis in dimensions:
        inc_i = True
        try:
            idx = isel[axis]
            if idx.interp: interp = True
            if idx.isarray: 
                isarray = True
                if idx.dims is not None: multidim = True
                if isinstance(idx.v, N.ma.MaskedArray): masked = True
            xsel_dims[axis] = idx.dims
            idx = idx.v
            if isinstance(idx, slice):
                dimsize = file.cf_dimensions[axis]
                res = [idx.start, idx.stop, idx.step]
                if (idx.step is not None and idx.step < 0):
                    if idx.start is None: res[0] = dimsize - 1
                    if idx.stop is None: res[1] = None
                else:
                    if idx.start is None: res[0] = 0 
                    if idx.stop is None: res[1] = dimsize
                    if idx.step is None: res[2] = 1
                xsel[axis] = slice(res[0], res[1], res[2])
            elif N.isscalar(idx):
                xsel[axis] = idx
                if len(order) > 0:
                    order.remove(i)
                    for val in order:
                        if val > i:
                            order[order.index(val)] = val - 1
                            inc_i = False
            else:
                #xsel[axis] = idx.copy()
                xsel[axis] = copy.copy(idx)
                if len(idx.shape) == 0 or idx.shape == 1:
                    if len(order) > 0:
                        order.remove(i)
                        for val in order:
                            if val > i:
                                order[order.index(val)] = val - 1
                                inc_i = False
        except KeyError:
            dimsize = file.cf_dimensions[axis]
            xsel[axis] = (slice(0, dimsize, 1))
            xsel_dims[axis] = None
        if inc_i:
            i += 1

    if isarray:
        # convert slices to 1d-arrays and determine result size
        for axis in dimensions:
            idx = xsel[axis]
            if isinstance(idx, slice):
                xsel[axis] = N.arange(idx.start, idx.stop, idx.step)
            if xsel_dims[axis] is None:
                if is_scalar(xsel[axis]):
                    xsel_size[axis] = 0
                else:
                    xsel_size[axis] = len(xsel[axis])
            else:
                xsel_size[axis] = isel[axis].axlen

        # determine shape of xsel
        dim_ret = []
        for axis in dimensions:
            if xsel_size[axis] != 0:
                dim_ret.append(xsel_size[axis])
        ndim_ret = len(dim_ret)

        # all 1d arrays
        if not multidim:
            i = 0
            for axis in dimensions:
                if xsel_size[axis] != 0:
                    idx_shape = N.ones(ndim_ret,dtype="int32")
                    idx_shape[i] = dim_ret[i]
                    xsel[axis].shape = idx_shape
                    i += 1
        # at least one multidimensional coordinate
        else:
            i = 0
            for axis in dimensions:
                if xsel_dims[axis] is None:
                    if xsel_size[axis] != 0:
                        idx_shape = N.ones(ndim_ret,dtype="int32")
                        idx_shape[i] = dim_ret[i]
                        xsel[axis].shape = idx_shape
                        i += 1
                else:
                    idx_shape2 = {}
                    for axis2 in dimensions:
                        if xsel_size[axis2] != 0:
                            if axis2 in xsel_dims[axis]:
                                idx_shape2[axis2] = isel[axis].dimsize(axis2)
                            else:
                                idx_shape2[axis2] = 1
                    idx_shape = []
                    for axis2 in dimensions:
                        if axis2 in idx_shape2:
                            idx_shape.append(idx_shape2[axis2])
                    if isel[axis].type != 'scalar': 
                        i += 1

    # check if we only need basic slicing
    if not isarray and not interp:
        isbasic = True
    else:
        isbasic = False

    ret = []
    for axis in dimensions:
        ret.append(xsel[axis])
    ret = xSelect(ret)
    ret.isbasic = isbasic
    ret.interp = interp
    ret.masked = masked
    ret.order = order

    return ret            
#@-node:schmidli.20080322120238.15:idxsel2xsel
#@+node:schmidli.20080322120238.16:numpy2xsel
def numpy2xsel(isel):
    """ convert a numpy selection object to an xselection object

        extended numpy selection object:
            if multidim: the dimensionality of idx is NOT changed
            else: convert 1d-idx to ndim arrays
    """

    if N.isscalar(isel): isel = tuple((isel,))
    if isinstance(isel, slice): isel = tuple((isel,))
    xsel = []
    isarray = False
    interp = False
    multidim = False
    do_convert = True

    if not isinstance(isel, tuple):
        raise TypeError("wrong argument type")

    for idx in isel:
        if isinstance(idx, slice):
            xsel.append(idx)
        elif N.isscalar(idx):
            xsel.append(idx)
            if idx.dtype in (N.float32, N.float64):
                interp = True
        else:
            isarray = True
            idx = N.atleast_1d(idx)
            xsel.append(idx)
            isarray = True
            if idx.dtype in (N.float32, N.float64):
                interp = True
            if idx.ndim > 1: 
                multidim = True      # conversion not supported for multidim

    # convert selection objects to compatible _intp arrays if necessary
    if isarray and not multidim:
        # convert slices to 1d-arrays
        for i in range(len(xsel)):
            if isinstance(xsel[i], slice):
                xsel[i] = N.arange(xsel[i].start, xsel[i].stop, xsel[i].step)
                xsel_size[i] = len(xsel[i])
        dim_ret = []
        for i in range(len(xsel)):
            if not N.isscalar(xsel[i]):
                if xsel[i].ndim > 0:
                    dim_ret.append(len(xsel[i]))
        ndim_ret = len(dim_ret)
        j = 0
        for i in range(len(xsel)):
            if not N.isscalar(xsel[i]):
                idx_shape = N.ones(ndim_ret)
                idx_shape[j] = dim_ret[i]
                xsel[i].shape = idx_shape
                j += 1

    # check if we only need basic slicing
    if not isarray and not interp:
        isbasic = True
    else:
        isbasic = False

    ret = xSelect(xsel)
    ret.isbasic = isbasic
    ret.interp = interp

    return ret            
#@-node:schmidli.20080322120238.16:numpy2xsel
#@+node:schmidli.20080322120238.17:class axisSelect
class axisSelect(object):
    """ axisSelect(inp)

    Create an axis selection object.

    Parameters:
        inp     a scalar, slice, or vector selection object

    The syntax for inp is as follows:
        [crdname|]<pre><selection><post>

    <pre> is one of:
        None    <selection> is in native coordinate space
        d       <selection> is in ISO-8601 date format
        i       <selection> is in index space

    <selection> is one of:
        #       for a scalar
        #:#:#   for a slice
        #,#,... for a vector
    where # is a number with an optional multiplier (e.g. 10k), 
    or a ISO-date. Valid multipliers include:
        k (10**3), M (10**6), h (3600), m (60), H (100)

    <post> is one of:
        i       interpolate data to exact location
        n       round to nearest index

    Examples:
        cidx = axisSelect('d20070321-09')
        cidx = axisSelect('d20070321-09:18:3h')
        cidx = axisSelect('i4')
        cidx = axisSelect('i10,20,24i')
        cidx = axisSelect('ZP|1500')
        cidx = axisSelect('20k:100k:5k')
        cidx = axisSelect('ZP|1500n')
    """
    #@  @+others
    #@+node:schmidli.20080322120238.18:__init__
    def __init__(self, inp):
        """ Overview of attributes:
            .type   one of 'scalar', 'slice', 'vector'
            .fmt    one of 'number', 'datetime'
            .iscrd  True/False
            .interp False/True
            .clip   True/False
        """

        if not isinstance(inp, str):
            raise TypeError("Invalid argument type")
        if len(inp) == 0:
            raise ValueError("Empty string is not a valid input")

        # default settings
        self.type = 'slice'
        self.fmt = 'number'
        self.iscrd = True
        self.clip = True

        # check prefix

        if inp[0] == 'd':
            #raise NotImplementedError, "Date/time selection is not yet implemented"
            self.fmt = 'datetime'
            inp = inp[1:]
        elif inp[0] == 'i':
            self.iscrd = False
            inp = inp[1:]

        # check postfix
        if inp[-1] in 'in':
            postfix = inp[-1]
            inp = inp[:-1]
        else:
            postfix = None

        # check if to clip field
        if inp[-1] == 'm':
            self.clip = False
            inp = inp[:-1]

        if len(inp) == 0:
            raise ValueError("Invalid input string")   

        # check for multi-dimensional coordinate name
        inpv = inp.split('|')
        if len(inpv) > 1:
            self.mdcrd = inpv[0]
            inp = inpv[1]
            if postfix == 'n':          # default is to interpolate
                self.interp = False
            else:
                self.interp = True
        else:
            self.mdcrd = None
            inp = inpv[0]
            if postfix == 'i':          # default is to not interpolate
                self.interp = True
            else:
                self.interp = False

        # determine selection type
        inpv = inp.split(':')
        if len(inpv) > 1 and len(inpv) <= 3:
            self.type = 'slice'
        elif len(inpv) > 3:
            raise NotImplementedError("eslice is not yet implemented")
        else:
            inpv = inp.split(',')
            if len(inpv) > 1:
                self.type = 'vector'
                if inpv[-1] == '': inpv = inpv[:-1]
            else:
                self.type = 'scalar'

        # parse the selection string
        data = []
        if self.type == 'slice' and len(inpv) == 3:
            step = inpv[2]
            if len(step) == 0:
                self.__step = None
            else:
                if step[0] == 'i':
                    self._index_step = 1
                    step = step[1:]
                self.__step = str2float(step)
                inpv = inpv[:2]
        else:
            self.__step = None
        is_first = True
        for item in inpv:
            if self.fmt == 'datetime':
                if is_first:
                    data.append(str2datetime(item))
                    is_first = False
                else:
                    data.append(str2datetime(item, templ=data[-1]))
            else:
                data.append(str2float(item))
        if self.type == 'slice':
            if len(inpv) == 1:
                data = [None, data[0]]
            elif len(inpv) == 2:
                data = [data[0], data[1]]
            if self.mdcrd is not None:
                if data[0] is None or data[1] is None or self.__step is None:
                    raise ValueError("must specify a complete slice for multidimensional coordinates")
            if self.interp:
                if data[0] is None or data[1] is None or self.__step is None:
                    raise ValueError("must specify a complete slice in interpolation mode")
        self.__data = data
    #@-node:schmidli.20080322120238.18:__init__
    #@+node:schmidli.20080322120238.19:__str__
    def __str__(self):
        if self.type == 'scalar':
            _str = 'scalar(' + str(self.__data[0]) + ')'
        elif self.type == 'slice':
            _str = 'slice(' + str(self.start) + ', ' + str(self.stop) + ', ' \
                            + str(self.step) + ')'
        elif self.type == 'vector':
            _str = 'vector('
            for item in self.__data:
                _str += str(item) + ' '
            _str = _str[:-1] + ')'
        _str = 'axisSelect(' + _str + ', iscrd: ' + str(self.iscrd) \
            + ', interp: ' + str(self.interp) + ')'
        return _str
    #@-node:schmidli.20080322120238.19:__str__
    #@+node:schmidli.20080322120238.20:__getitem__
    def __getitem__(self, index):
        if self.type == 'vector':
            return self.__data[index]
        else:
            raise LookupError("Not valid method for "+self.type+".")
    #@-node:schmidli.20080322120238.20:__getitem__
    #@+node:schmidli.20080322120238.21:__len__
    def __len__(self):
        return len(self.__data)
    #@-node:schmidli.20080322120238.21:__len__
    #@+node:schmidli.20080322163604:__iter__
    def __iter__(self):
        if self.type == 'slice':
            data = N.arange(self.start, self.stop, self.step)
            if data[-1]+self.step == self.stop:
                data = N.concatenate((data, [self.stop]))
            data = data.tolist()
        elif self.type == 'scalar':
            data = (self.v)
        else:
            data = tuple(self.v)
        return iter(data)
    #@-node:schmidli.20080322163604:__iter__
    #@+node:schmidli.20080322172813:tolist
    def tolist(self):
        if self.type == 'slice':
            data = N.arange(self.start, self.stop, self.step)
            if data[-1]+self.step == self.stop:
                data = N.concatenate((data, [self.stop]))
            data = data.tolist()
        elif self.type == 'scalar':
            data = [self.v]
        else:
            data = list(self.v)
        return data
    #@-node:schmidli.20080322172813:tolist
    #@+node:schmidli.20080322120238.22:toindex
    def toindex(self, file, axis, mdcrd=None, isel=None, clip=True, ep=0.0):
        """ Convert a axisSelect object from coordinate space to index space
        """

        dimsize = None; refdate = None
        dims = None; axis_no = 0
        if self.iscrd:
            if file.cf2dims is not None:
                axfile = file.cf2dims[axis]
            else:
                axfile = axis
            if not axfile in file.file.variables:
                self.iscrd = False
        if self.iscrd:
            if mdcrd is None:
                if file.cf2dims is not None:
                    axfile = file.cf2dims[axis]
                else:
                    axfile = axis
                crd = file.file.variables[axfile]
                if self.fmt == 'datetime':
                    refdate = get_refdate(crd)
                crd = crd[:]
            else:
                crd = get_variable(file, mdcrd, isel)
                var = file.variables[mdcrd]
                dims = list(var.cf_dimensions)
                for axis2 in isel.keys():
                    if isel[axis2].type == 'scalar' and axis2 != axis: 
                        try:
                            dims.remove(axis2)
                        except ValueError:
                            pass
                axis_no = dims.index(axis)
                if self.type == 'scalar': dims.remove(axis)
                if var.rank < 2:
                    raise ValueError("Coordinate variable "+self.mdcrd+ " is not multidimensional")
        else:
            dimsize = file.cf_dimensions[axis]
            crd = None
        ret = self.toindex_crd(crd, axis=axis, axis_no=axis_no, dimsize=dimsize, refdate=refdate, 
                               clip=clip, ep=0.0)
        ret.dims = dims
        ret.axis = axis
        return ret
    #@-node:schmidli.20080322120238.22:toindex
    #@+node:schmidli.20080322120238.23:toindex_crd
    def toindex_crd(self, crd, axis= None, axis_no=0, dimsize=None, refdate=None, clip=True, ep=0.0):
        """ Convert a axisSelect object from coordinate space to index space
        """

        interp = self.interp
        round_ = not interp
        clip = self.clip
        ep = 0.5

        cidx = self
        data = copy.copy(self.__data)
        idx = axisIdxSelect(self)

        # convert datetime to seconds since a reference date
        if cidx.fmt == 'datetime':
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = data[i] - refdate
                    data[i] = data[i].days*86400. + data[i].seconds

        # if interp=True: convert slice to vector object
        if cidx.type == 'slice' and interp:
            if data[0] is None:
                if cidx.iscrd:
                    start = crd.min()
                else:
                    if cidx.step < 0:
                        start = dimsize
                    else:
                        start = 0.0
            else:
                start = data[0]
                if start < 0: start += dimsize
            if data[1] is None: 
                if cidx.iscrd:
                    stop = crd.max()
                else:
                    if cidx.step < 0:
                        stop = 0
                    else:
                        stop = dimsize
            else:
                stop = data[1]
                if stop < 0: stop += dimsize
            if cidx.iscrd and hasattr(self,'_index_step'):
                start = _rindex(crd, start, axis=axis_no, round=round_, clip=clip, ep=ep)
                stop = _rindex(crd, stop, axis=axis_no, round=round_, clip=clip, ep=ep)
                cidx.iscrd = False

            data = N.arange(start, stop, cidx.step)
            if len(data) > 0 and data[-1]+cidx.step == stop:
                data = N.concatenate((data, [stop]))        
            idx.type = 'vector'

        # convert from coordinate to index space
        if cidx.iscrd:
            if idx.type == 'slice':
                dir = 1
                if cidx.step is not None:
                    if len(crd > 1):
                        if hasattr(self,'_index_step'):
                            if idx.step < 0: dir = -1
                            idx.step = N.round(idx.step).astype(N.int)
                        else:
                            # Note this assumes equally spaced coordinate values.
                            idx.step = cidx.step/(crd[1]-crd[0])
                            if idx.step < 0: dir = -1
                            idx.step = N.round(idx.step).astype(N.int)
                        # step size less than spacing is treated as default step (pos or neg)
                        if idx.step == 0: 
                            if dir == 1: 
                                idx.step = 1
                            else: 
                                idx.step = -1 
                    else:
                        idx.step = None
                for i in range(len(data)):
                    if data[i] is not None: 
                        data[i] = _rindex(crd, data[i], axis=axis_no, round=False, clip=clip, ep=ep)
                        if dir == 1:
                            if i == 0: data[i] = N.ceil(data[i]).astype(N.int)
                            if i == 1: data[i] = N.floor(data[i]).astype(N.int) + 1
                        else:
                            if i == 0: data[i] = N.floor(data[i]).astype(N.int)
                            if i == 1: 
                                data[i] = N.ceil(data[i]).astype(N.int) - 1
                                if data[i] == -1: data[i] = None
            else:
                if crd is None: raise ValueError("Missing coordinate variable")
                if cidx.type == 'scalar':
                    data[0] = _rindex(crd, data[0], axis=axis_no, round=round_, clip=clip, ep=ep)
                else:
                    data = _rindex(crd, data, axis=axis_no, round=round_, clip=clip, ep=ep)
        else:
            if not interp:
                if idx.type == 'slice':
                    if idx.step is not None: 
                        if idx.step < 0: 
                            dir = -1
                        else:
                            dir = 1
                        idx.step = N.round(idx.step).astype(N.int)
                        if idx.step == 0: idx.step = dir
                        if idx.step < 0:
                            if data[0] is not None: 
                                data[0] = N.floor(data[0]).astype(N.int)
                                if data[0] < 0:
                                    data[0] += dimsize
                            if data[1] is not None: 
                                data[1] = N.ceil(data[1]).astype(N.int)-1
                                if data[1] < -1:
                                    data[1] += dimsize
                                elif data[1] == -1:
                                   data[1] = None
                    if idx.step is None or idx.step > 0:
                        if data[0] is not None: 
                            data[0] = N.ceil(data[0]).astype(N.int)
                            if data[0] < 0:
                                data[0] += dimsize
                        if data[1] is not None: 
                            data[1] = N.floor(data[1]).astype(N.int)+1
                            if data[1] < 0:
                                data[1] += dimsize
                else: 
                # not a slice
                    for i in range(len(data)):
                        data[i] = N.round(data[i]).astype(N.int)
                        if data[i] < 0: data[i] += dimsize

        if cidx.type == 'scalar' and len(data) == 1: 
            data = data[0]
            if dimsize is not None:
                # dimsize is only defined if operating in index space.
                # since later processing (xSelect.bndbox) does not handle index space values less than 0
                # convert them here.
                if data < -dimsize or data >= dimsize:    
                    raise IndexError(axis + " axis index out of range")
                elif data < 0:
                    data += dimsize
        if idx.type != 'slice': 
            if not isinstance(data, N.ma.MaskedArray):
                data = N.asarray(data)
        #if cidx.type == 'scalar' and len(data) == 1: data = data[0]
        idx.setdata(data)
        return idx
    #@-node:schmidli.20080322120238.23:toindex_crd
    #@+node:schmidli.20080322120238.24:getstart
    def getstart(self):
        """ return start value of slice object """

        if self.type == 'slice':
            return self.__data[0]
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.24:getstart
    #@+node:schmidli.20080322120238.25:setstart
    def setstart(self, value):
        """ return start value of slice object """

        if self.type == 'slice':
            self.__data[0] = value
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.25:setstart
    #@+node:schmidli.20080322120238.26:getstop
    start = property(getstart, setstart)
    def getstop(self):
        """ return stop value of slice object """

        if self.type == 'slice':
            return self.__data[1]
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.26:getstop
    #@+node:schmidli.20080322120238.27:setstop
    def setstop(self, value):
        """ return stop value of slice object """

        if self.type == 'slice':
            self.__data[1] = value
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.27:setstop
    #@+node:schmidli.20080322120238.28:getstep
    stop = property(getstop, setstop)
    def getstep(self):
        """ return step value of slice object """

        if self.type == 'slice':
            return self.__step
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.28:getstep
    #@+node:schmidli.20080322120238.29:setstep
    def setstep(self, value):
        """ return step value of slice object """

        if self.type == 'slice':
            self.__step = value
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.29:setstep
    #@+node:schmidli.20080322120238.30:getvalue
    step = property(getstep, setstep)
    def getvalue(self):
        """ return the scalar object """
        if self.type == 'scalar':
            return self.__data[0]
        elif self.type == 'slice':
            return slice(self.start, self.stop, self.step)
        elif self.type == 'vector':
            return self.__data
    #@-node:schmidli.20080322120238.30:getvalue
    #@+node:schmidli.20080322120238.31:setvalue
    def setvalue(self, value):
        """ set the scalar object """
        if isinstance(value, slice):
            self.type = 'slice'
            self.__data = [value.start, value.stop]
            self.__step = value.step
        elif N.isscalar(value):
            self.type = 'scalar'
            self.__data = [value]
        else:
            self.type = 'vector'
            self.__data = value
        #if self.type == 'scalar':
        #    self.__data[0] = value
        #elif self.type == 'vector':
        #    self.__data = value
        #else:
        #    raise AttributeError, "Not valid operation for type=slice"
    #@-node:schmidli.20080322120238.31:setvalue
    #@+node:schmidli.20080322120238.32:is_vector
    v = property(getvalue, setvalue)
    def is_vector(self):
        if self.type == 'vector':
            return True
        elif self.type == 'scalar':
            if not N.isscalar(self.__data[0]):
                return True
        return False
    #@-node:schmidli.20080322120238.32:is_vector
    #@+node:schmidli.20080322120238.33:_getdata
    isvect = property(is_vector)
    def _getdata(self):
        return copy.copy(self.__data)
    #@-node:schmidli.20080322120238.33:_getdata
    #@+node:schmidli.20080322120238.34:_getstep
    def _getstep(self):
        return self.__step
    #@-node:schmidli.20080322120238.34:_getstep
    #@-others
#@-node:schmidli.20080322120238.17:class axisSelect
#@+node:schmidli.20080322120238.35:class axisIdxSelect
class axisIdxSelect(object):
    """ axisIdxSelect(cidx)

    Create an axis selection object (index space only)

    Parameters:
        cidx    an coordinate space axis selection object (axisSelect)

    Examples:
        cidx = axisSelect('d20070321-09:18:3h')
        idx = axisIdxSelect(cidx)
    """
    #@  @+others
    #@+node:schmidli.20080322120238.36:__init__
    def __init__(self, cidx):
        """ Overview of attributes:
            .type   one of 'scalar', 'slice', 'vector'
            .interp False/True
        """

        if isinstance(cidx, axisSelect):
            # copy data members
            self.type = cidx.type
            self.interp = cidx.interp
            self.axis = None
            self.dims = None
            #self.__data = cidx._getdata()
            self.__step = cidx._getstep()
            self.__data = None
        elif isinstance(cidx, slice):
            self.type = 'slice'
            self.interp = False
            self.axis = None
            self.dims = None
            self.__data = [cidx.start, cidx.stop]
            self.__step = cidx.step
        else:
            raise TypeError("Invalid argument type")
    #@-node:schmidli.20080322120238.36:__init__
    #@+node:schmidli.20080322120238.37:__str__
    def __str__(self):
        if self.type == 'scalar':
            _str = 'scalar(' + str(self.__data) + ')'
        elif self.type == 'slice':
            _str = 'slice(' + str(self.start) + ', ' + str(self.stop) + ', ' + str(self.step) + ')'
        elif self.type == 'vector':
            _str = 'vector('
            for item in self.__data:
                _str += str(item) + ' '
            _str = _str[:-1] + ')'
        _str = 'axisIdxSelect(' + _str + ', interp: ' + str(self.interp) \
                + ', dims: ' + str(self.dims) + ')'
        return _str
    #@-node:schmidli.20080322120238.37:__str__
    #@+node:schmidli.20080322120238.38:__getitem__
    def __getitem__(self, index):
        if self.type == 'vector':
            return self.__data[index]
        else:
            raise LookupError("Not valid method for "+self.type+".")
    #@-node:schmidli.20080322120238.38:__getitem__
    #@+node:schmidli.20080322120238.39:getstart
    def getstart(self):
        """ return start value of slice object """

        if self.type == 'slice':
            return self.__data[0]
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.39:getstart
    #@+node:schmidli.20080322120238.40:setstart
    def setstart(self, value):
        """ return start value of slice object """

        if self.type == 'slice':
            self.__data[0] = value
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.40:setstart
    #@+node:schmidli.20080322120238.41:getstop
    start = property(getstart, setstart)
    def getstop(self):
        """ return stop value of slice object """

        if self.type == 'slice':
            return self.__data[1]
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.41:getstop
    #@+node:schmidli.20080322120238.42:setstop
    def setstop(self, value):
        """ return stop value of slice object """

        if self.type == 'slice':
            self.__data[1] = value
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.42:setstop
    #@+node:schmidli.20080322120238.43:getstep
    stop = property(getstop, setstop)
    def getstep(self):
        """ return step value of slice object """

        if self.type == 'slice':
            return self.__step
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.43:getstep
    #@+node:schmidli.20080322120238.44:setstep
    def setstep(self, value):
        """ return step value of slice object """

        if self.type == 'slice':
            self.__step = value
        else:
            raise AttributeError("Not valid attribute for "+self.type+".")
    #@-node:schmidli.20080322120238.44:setstep
    #@+node:schmidli.20080322120238.45:getvalue
    step = property(getstep, setstep)
    def getvalue(self):
        """ return the scalar object """
        if self.type == 'scalar':
            return self.__data
        elif self.type == 'slice':
            return slice(self.start, self.stop, self.step)
        elif self.type == 'vector':
            return self.__data
    #@-node:schmidli.20080322120238.45:getvalue
    #@+node:schmidli.20080322120238.46:setvalue
    def setvalue(self, value):
        """ set the scalar object """
        if self.type == 'scalar':
            self.__data = value
        elif self.type == 'vector':
            self.__data = value
        else:
            raise AttributeError("Not valid operation for type=slice")
    #@-node:schmidli.20080322120238.46:setvalue
    #@+node:schmidli.20080322120238.47:setdata
    v = property(getvalue, setvalue)
    def setdata(self, value):
        self.__data = value
    #@-node:schmidli.20080322120238.47:setdata
    #@+node:schmidli.20080322120238.48:is_array
    def is_array(self):
        if self.type == 'vector':
            return True
        elif self.type == 'scalar':
            if not N.isscalar(self.__data):
                return True
        return False
    #@-node:schmidli.20080322120238.48:is_array
    #@+node:schmidli.20080322120238.49:ndim_
    isarray = property(is_array)
    def ndim_(self):
        if not self.isarray:
            raise AttributeError("Not valid attribute for "+self.type+".")
        return self.__data.ndim
    #@-node:schmidli.20080322120238.49:ndim_
    #@+node:schmidli.20080322120238.50:axlen_
    ndim = property(ndim_)
    def axlen_(self):
        if self.dims is not None:
            return self.dimsize(self.axis)
        else:
            if self.type == 'scalar':
                return 0
            elif self.type == 'slice':
                return -1
            else:
                return self.__data.size
    #@-node:schmidli.20080322120238.50:axlen_
    #@+node:schmidli.20080322120238.51:dimsize
    axlen = property(axlen_)
    def dimsize(self, axis):
        if self.dims is None: return 0
        if axis in self.dims:
            i = self.dims.index(axis)
            return self.__data.shape[i]
        else:
            return 0
    #@-node:schmidli.20080322120238.51:dimsize
    #@-others
#@-node:schmidli.20080322120238.35:class axisIdxSelect
#@+node:schmidli.20080322120238.52:class realSelect
class realSelect(object):
    """ realSelect(sel)

    Create an extended selection object, containing floats and
    masked arrays

    Examples:
        rsel = realSelect()[5.5, 2.2:4.4:1.1, (2.2,3,3.5),:]]
    """
    #@  @+others
    #@+node:schmidli.20080322120238.53:__init__
    def __init__(self):
        pass
    #@-node:schmidli.20080322120238.53:__init__
    #@+node:schmidli.20080322120238.54:__str__
    def __str__(self):
        _str = str(self)
        return _str
    #@-node:schmidli.20080322120238.54:__str__
    #@+node:schmidli.20080322120238.55:__getitem__
    def __getitem__(self, index):
        return index
    #@-node:schmidli.20080322120238.55:__getitem__
    #@-others
#@-node:schmidli.20080322120238.52:class realSelect
#@+node:schmidli.20080322120238.56:str2float
def str2float(inp):
    """ Convert a string with an optional postfix into a float
        postfix must be in 'kMhmH'
    """

    if inp == '': return None

    postfix = inp[-1]
    if postfix in fact_keys:
        ret = float(inp[:-1])*fact[postfix]
    else:
        ret = float(inp)
    return ret
#@-node:schmidli.20080322120238.56:str2float
#@+node:schmidli.20080322120238.57:str2datetime
def str2datetime(inp, isend=False, templ=None):
    """ Convert a iso-date string into a datetime object

        Example: '20070321T0630'
    """

    if inp == '': return None

    inpv = inp.split('-')
    date = ''; time = ''
    if len(inpv) < 2:
        time = inpv[0]
    else:
        (date, time) = inpv[:2]

    # set default values
    if templ is not None:
        d = templ
        (year, month, day) = (d.year, d.month, d.day)
        (hour, min, sec) = (0, 0, 0)
    else:
        (year, month, day) = (0, 1, 1)
        if isend:
            (hour, min, sec) = (23, 59, 59)
        else:
            (hour, min, sec) = (0, 0, 0)

    # parse date
    if len(date) == 0:
        pass
    elif len(date) <= 2:
        day = int(date)
    elif len(date) <= 4:
        month = int(date[:-2])
        day = int(date[-2:])
    elif len(date) <= 8:
        year = int(date[:-4])
        month = int(date[-4:-2])
        day = int(date[-2:])
    else:
        raise ValueError('Invalid date format')

    # parse time
    if len(time) == 0:
        pass
    elif len(time) <= 2:
        hour = int(time)
    elif len(time) <= 4:
        hour = int(time[:2])
        min = int(time[2:])
    elif len(time) <= 6:
        hour = int(time[:2])
        min = int(time[2:4])
        sec = int(time[4:])
    else:
        raise ValueError('Invalid time format')

    ret = datetime.datetime(year, month, day, hour, min, sec)
    return ret
#@-node:schmidli.20080322120238.57:str2datetime
#@+node:schmidli.20080322120238.58:get_refdate
def get_refdate(inp):
    if not hasattr(inp, 'units'):
        raise ValueError('Missing units attribute for time coordinate variable: ' + str(inp))
    tunits = inp.units.split(' ')
    tunits = tunits[2]+' '+tunits[3]
    ret = datetime.datetime(*time.strptime(tunits, "%Y-%m-%d %H:%M:%S")[:5])
    return ret
#@-node:schmidli.20080322120238.58:get_refdate
#@+node:schmidli.20080322120238.59:is_scalar
def is_scalar(inp):
    if isinstance(inp, N.ndarray) and inp.ndim == 0:
        return True
    elif N.isscalar(inp):
        return True
    else:
        return False
#@-node:schmidli.20080322120238.59:is_scalar
#@-others
#@-node:schmidli.20080322120238:@thin pymfio/coordsel.py
#@-leo
