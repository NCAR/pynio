'''
PyNIO enables NetCDF-like access for
    NetCDF 3 (rw), HDF 4 (rw), GRIB 1 (r), CCM (r), and optionally
    NetCDF 4 (rw), HDFEOS 2/4 (r), GRIB 2 (r), HDFEOS 5 (r), shapefiles
    and other GDAL OGR-supported (r) data formats.

import Nio

Class NioFile:

f = Nio.open_file(filepath, mode='r', options=None, history='',format='')

To see summary of file contents, including all dimensions, attributes,
and variables:
   print f
attributes:
   dimensions -- dictionary with dimension names as keys and dimension lengths as values
   variables -- dictionary with variable names as keys and variable objects as values
   attributes (or __dict__) --  contains the global attributes associated with the file
methods:
   close(history='')
   create_dimension(name, length)
   create_variable(name, type, dimensions)
For more detailed information:
    print f.__doc__

Class NioOptions

opt = Nio.options()

To set format-specific options assign option names and settings as attributes
and values of 'opt'. Then pass 'opt' as the optional options argument to
Nio.open_file.
To see valid options:
    print opt.__doc__ 

Class NioVariable

v = f.variables['varname']

To see summary of variable contents including all dimensions,
associated coordinate variables, and attributes:
    print v 
Attributes:
    rank -- a scalar value indicating the number of dimensions
    shape -- a tuple containing the number of elements in each dimension
    dimensions -- a tuple containing the dimensions names in order
    attributes (or __dict__) -- a dictionary containing the variable attributes
Methods:
    assign_value(value) -- assign a value to a variable in the file.
    get_value() -- retrieve the value of a variable in the file.
    typecode() -- return a character code representing the variable's type.
    set_option(option,value) -- set certain options.

For more detailed information:
    print v.__doc__

For complete documentation see:

        http://www.pyngl.ucar.edu/Nio.html
'''

from nio import *
from nio import _C_API

#
#  Get version number and flag for NumPy compatibility.
#
#  Also, get the __array_module__  and __array_module_version__
#  attributes.
#
import pynio_version
__version__              = pynio_version.version
__array_module__         = pynio_version.array_module
__array_module_version__ = pynio_version.array_module_version
__formats__              = pynio_version.formats
del pynio_version

__all__ = [ 'open_file', 'option_defaults', 'options' ]

def pyniopath_ncarg():
#
#  Find the root directory that contains the supplemental PyNIO files,
#  in particular, the grib2 codetables. For now the default is to look
#  for a path in sys.path that contains the subdirectory PyNIO/ncarg. 
#   Otherwise, check NCARG_ROOT/lib/ncarg.
#
    import sys
    pynio_ncarg = None
    for path in sys.path:
        trypath = os.path.join(path,"PyNIO","ncarg")
        if os.path.exists(trypath):
            pynio_ncarg = trypath
            break

    if pynio_ncarg == None:
        ncarg_dir = os.environ.get("NCARG_ROOT")
        if ncarg_dir == None or not os.path.exists(ncarg_dir) \
          or not os.path.exists(os.path.join(ncarg_dir,"lib","ncarg")):
            if not __formats__['grib2']:
                return ""
            else:
                print "No path found to PyNIO/ncarg data directory and no usable NCARG installation found"
                sys.exit()
        else:
            pynio_ncarg = os.path.join(ncarg_dir,"lib","ncarg")

    return pynio_ncarg

#
# Set the NCARG_NCARG environment variable.
# This should allow the grib2_codetables directory to be found without
# requiring any environment variables to be set by the user

import os
os.environ["NCARG_NCARG"] = pyniopath_ncarg()
del pyniopath_ncarg
del os

#
# This part of the code creates the NioFile and NioVariable classes as
# proxies for the C API private classes _NioFile and _NioVariable
# It also implements support for Juerg Schmidli's coordinate selection 
# code


from nio import _Nio
import numpy as np
from numpy import ma
from coordsel import get_variable, inp2csel, inp2isel, inp2xsel, idxsel2xsel, \
                     xSelect, crdSelect, idxSelect 
import coordsel as cs

_Nio.option_defaults['UseAxisAttribute'] = False
_Nio.option_defaults['MaskedArrayMode'] = 'MaskedIfFillAtt'
_Nio.option_defaults['ExplicitFillValues'] = None
_Nio.option_defaults['MaskBelowValue'] = None
_Nio.option_defaults['MaskAboveValue'] = None

def get_integer_version(strversion):
    ''' converts string version number into an integer '''
    d = strversion.split('.')
    if len(d) > 2:
       v = int(d[0]) * 10000 + int(d[1]) * 100 + int(d[2][0])
    elif len(d) is 2:
       v = int(d[0]) * 10000 + int(d[1]) * 100
    else:
       v = int(d[0]) * 10000
    return v

_is_new_ma = get_integer_version(np.__version__) > 10004
del get_integer_version

_builtins = ['__class__', '__delattr__', '__doc__', '__getattribute__', '__hash__', \
            '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', \
            '__setattr__', '__str__', '__weakref__', '__getitem__', '__setitem__', '__len__' ]

_localatts = ['attributes','_obj','variables','file','varname', \
              'cf_dimensions', 'cf2dims', 'ma_mode', 'explicit_fill_values', \
              'mask_below_value', 'mask_above_value', 'set_option', 'create_variable', 'close', 'assign_value', 'get_value']    

# The Proxy concept is adapted from Martelli, Ravenscroft, & Ascher, 'Python Cookbook' Second Edition, 6.6

class _Proxy(object):
    ''' base class for all proxies '''


    def __init__(self, obj):
        super(_Proxy, self).__init__()
        super(_Proxy,self).__setattr__('_obj', obj)
        super(_Proxy,self).__setattr__('attributes',{})
        for key in obj.__dict__.keys():
           super(_Proxy,self).__setattr__(key,obj.__dict__[key])
           self.attributes[key] = obj.__dict__[key]

    def __getattribute__(self, attrib):

        if attrib in _localatts or attrib in _builtins:
            return super(_Proxy,self).__getattribute__(attrib)
        else:
            return getattr(self._obj,attrib)

    def __setattr__(self, attrib, value):
        if attrib in _builtins:
            raise AttributeError, "Attempt to modify read only attribute"
        elif attrib in _localatts:
            super(_Proxy,self).__setattr__(attrib,value)
        else:
            setattr(self._obj,attrib,value)
            self.attributes[attrib] = self._obj.__dict__[attrib]

    def __delattr__(self, attrib):
        if attrib in _localatts or attrib in _builtins:
            raise AttributeError, "Attempt to delete read only attribute"
        else:
            delattr(self._obj,attrib)
            del(self.attributes[attrib])

def _make_binder(unbound_method):
    def f(self, *a, **k): return unbound_method(self._obj, *a, **k)
    # in 2.4, only: f.__name__ = unbound_method.__name__
    return f

_known_proxy_classes = {  }

def _proxy(obj, *specials, **regulars):
    ''' factory-function for a proxy able to delegate special methods '''
    # do we already have a suitable customized class around?
    obj_cls = type(obj)
    key = obj_cls, specials
    cls = _known_proxy_classes.get(key)
    if cls is None:
        # we don't have a suitable class around, so let's make it
        cls_dict = {}
        cls_dict['__doc__'] = obj_cls.__doc__
        # this removes the underscore in the name as supplied by the C API module
        # to give the user-visible name
        name = obj_cls.__name__[1:] 
        cls = type(name, (_Proxy,), cls_dict)
        for name in specials:
            name = '__%s__' % name
            unbound_method = getattr(obj_cls, name)
            setattr(cls, name, _make_binder(unbound_method))
        # also cache it for the future
        _known_proxy_classes[key] = cls
        for key in regulars.keys():
            setattr(cls, key, regulars[key])
    # instantiate and return the needed proxy
    instance = cls(obj)
    return instance

def _fill_value_to_masked(self, a):

    #
    # _FillValue is the preferred fill value attribute but if it is not set
    # then look for missing_value
    #
    if self.file.ma_mode == 'maskednever':
        # MaskedNever -- just return a numpy array
         return a
    elif self.file.ma_mode == 'maskedexplicit':
        # handle user-specified masking -- first ranges, then explicit single values
        # note that masked_where does not remove previously applied mask values 
        # so it can be applied in stages

        if self.file.mask_below_value is not None and self.file.mask_above_value is not None:
            if self.file.mask_below_value > self.file.mask_above_value: 
                # mask a band of values
                a = ma.masked_where((a < self.file.mask_below_value) & (a > self.file.mask_above_value),a,copy=0)
            else:
                # mask high and low values                    
                a = ma.masked_where((a < self.file.mask_below_value) | (a > self.file.mask_above_value),a,copy=0)
        elif self.file.mask_below_value is not None:
            # mask low values
            a = ma.masked_where(a < self.file.mask_below_value,a,copy=0)
        elif self.file.mask_above_value is not None:
            # mask high values
            a = ma.masked_where(a > self.file.mask_above_value,a,copy=0)

        # now apply single fill values
        if hasattr(self.file.explicit_fill_values,'__iter__'):
            # multiple explicit fill values
            for fval in self.file.explicit_fill_values:
                a = ma.masked_where(a == fval,a,copy=0)
            a.set_fill_value(self.file.explicit_fill_values[0])
        elif self.file.explicit_fill_values is not None:
            a = ma.masked_where(a == self.file.explicit_fill_values,a,copy=0)
            a.set_fill_value(self.file.explicit_fill_values)

    elif self.file.ma_mode == 'maskediffillattandvalue':
        # MaskedIfFillAttAndValue -- return a masked array only if there are actual fill values
        if self.__dict__.has_key('_FillValue'):
            if a.__contains__(self.__dict__['_FillValue']):
                a = ma.masked_where(a == self.__dict__['_FillValue'],a,copy=0)
                a.set_fill_value(self.__dict__['_FillValue'])
        elif self.__dict__.has_key('missing_value'):
            if a.__contains__(self.__dict__['missing_value']):
                a = ma.masked_where(a == self.__dict__['missing_value'],a,copy=0)
                a.set_fill_value(self.__dict__['missing_value'])
    else: 
        # Handles MaskedIfFillAtt and MaskedAlways
        if self.__dict__.has_key('_FillValue'):
            a = ma.masked_where(a == self.__dict__['_FillValue'],a,copy=0)
            a.set_fill_value(self.__dict__['_FillValue'])
        elif self.__dict__.has_key('missing_value'):
            a = ma.masked_where(a == self.__dict__['missing_value'],a,copy=0)
            a.set_fill_value(self.__dict__['missing_value'])
        elif self.file.ma_mode == 'maskedalways':
            # supply a mask of all False, but just allow the fill_value to default
            mask = np.zeros(a.shape,dtype='?')
            a = ma.array(a,mask=mask,copy=False) 

    return a
   
def __getitem__(self, xsel):
    ''' 
Return data as specified by selection object xsel from an NioVariable. 
Depending on the setting of option MaskedArrayMode and the presence of 
a fill value attribute and/or fill values in the data, the return value will
be a MaskedArray or a normal NumPy array.
    '''

    ret = get_variable(self.file, self.varname, xsel)

    ret = _fill_value_to_masked(self,ret)

    return ret

def get_value(self):
    '''
Retrieve the value of a variable in the file.

v = f.variables['varname']
val = v.get_value()

'val' is returned as a NumPy array or a MaskedArray depending on the value
of option MaskedArrayMode. This method is the only way to retrieve the value
of a scalar variable in a file. For array variables basic or extended selection
syntax is more flexible.
    '''
    
    ret  = self._obj.get_value()
    ret = _fill_value_to_masked(self,ret)

    return ret

def _masked_to_fill_value(self,value):
    fill_value = None
    add_fill_value_att = False

    if ma.isMaskedArray(value):
        #
        # If the file variable already has a _FillValue or missing_value attribute
        # use it for the fill_value when converting the masked array.
        # _FillValue is the preferred fill value attribute but if it is not set
        # then look for missing_value.
        # Note that it is important to generate the fill_value with the correct type.
        # If there is an existing fill value in the file, make it conform to that type (??)
        # Otherwise use the type of the numpy array value
        #
        if self.__dict__.has_key('_FillValue'):
            fval = self.__dict__['_FillValue']
            fill_value = np.array(fval,dtype=fval.dtype)
        elif self.__dict__.has_key('missing_value'):
            fval = self.__dict__['missing_value']
            fill_value = np.array(fval,dtype=fval.dtype)
        elif _is_new_ma:
            fill_value = np.array(value.fill_value,dtype=value.dtype)
            add_fill_value_att = True
        else:
            fill_value = np.array(value.fill_value(),dtype=value.dtype)
            add_fill_value_att = True
        value = value.filled(fill_value)

    if add_fill_value_att:
        setattr(self._obj,'_FillValue',fill_value)

    return value

def __setitem__(self, xsel, value):
    ''' 
Assign elements of value to file variable with subscripts specified by the 
selection object xsel. If the value is a masked array fill it using the file 
variable fill value if it exists; otherwise use the masked array fill value.
    '''

    xsel = inp2xsel(self.file, self.varname, xsel)

    value = _masked_to_fill_value(self,value)

    if not isinstance(xsel, xSelect) or xsel.isbasic:
        self._obj[xsel] = value
    else:
        bb = xsel.bndbox()
        rsel = xsel - bb
        ret = self._obj[bb]
        ret[rsel] = value

    return

def assign_value(self,value):
    '''
Assign a value to a variable in the file.

v = f.variables['varname']
v.assign_value(value)
value - a NumPy array, a MaskedArray, or a Python sequence of values 
that are coercible to the type of variable 'v'.

This method is the only way to assign to a scalar variable in the file. 
For array variables basic or extended selection syntax is more flexible.
    '''

    value = _masked_to_fill_value(self,value)
    self._obj.assign_value(value)

    return

def close(self,history=""):
    '''
Close a file, ensuring all modifications are up-to-date if open for writing.

f.close([history])
history -- optional string appended to the global 'history' attribute
before closing a writable file. The attribute is created if it does not
already exist.
Read or write access attempts on the file object after closing
raise an exception.
    '''

    self._obj.close(history)
    # Delete the proxy variable and file references so that the objects can be freed.
    # Note that the attributes cannot be deleted, but setting them to None accomplishes 
    # the same objective.

    self.variables = None
    self.file = None

    return

def create_variable(self,name,type,dimensions):
    '''
Create a new variable with given name, type, and dimensions in a writable file.

f.create_variable(name,type,dimensions)
name -- a string specifying the variable name.
type -- a type identifier. The following are currently supported:
    'd' -- 64 bit real
    'f' -- 32 bit real
    'l','i' -- 32 bit integer
    'h' -- 16 bit integer
    'b' -- 8 bit integer
    'S1','c' -- character
dimensions -- a tuple of dimension names (strings), previously defined
    '''

    #print 'in create variable'
    v = self._obj.create_variable(name,type,dimensions)
    if not v is None:
        vp  = _proxy(v,'str','len',
                     __setitem__=__setitem__,__getitem__=__getitem__,get_value=get_value,assign_value=assign_value)
        vp.file = self
        vp.varname = name
        vp.cf_dimensions = vp.dimensions
        self.variables[name] = vp
        return vp
    else:
        return None

def _get_masked_array_mode(options,option_defaults):
    ''' 
        get the MaskedArrayMode value considering the default setting and any option value set 
        when the file is opened 
    '''

    # ma_mode specifies when to return masked arrays
    # MaskedIfFillAtt: return a masked array iff file variable has a _FillValue or a missing_value (default)
    # MaskedNever: never return a masked array for any variable
    # MaskedAlways: return a masked array for all variables
    # MaskedIfFillAttAndValue: return a masked array iff file variable has a _FillValue or a missing_value and
    #                          the returned data array actually contains 1 or more fill values.
    # MaskedExplicit: only mask values specified explicitly using options ExplicitFillValues, MaskBelowValue,
    #                 and MaskAboveValue; ignore fill value attributes associated with the variable

    optvals = [ 'maskednever', 'maskediffillatt', 'maskedalways', 'maskediffillattandvalue', 'maskedexplicit' ]

    if options is not None:
        for key in options.__dict__.keys():
            lkey = key.lower()
            if not lkey == 'maskedarraymode':
                continue
            val = options.__dict__[key]
            lval = val.lower()
            if optvals.count(lval) == 0:
                raise ValueError, 'Invalid value for MaskArrayMode option'
            return  lval

    if option_defaults.has_key('MaskedArrayMode'):
        return option_defaults['MaskedArrayMode'].lower()
    else:
        return 'maskediffillatt'

def _get_axis_att(options,option_defaults):
    ''' Get a value for the UseAxisAttribute option '''
    if options == None:
        if option_defaults.has_key('UseAxisAttribute'):
            return option_defaults['UseAxisAttribute']
        else:
            return False
    for key in options.__dict__.keys():
        lkey = key.lower()
        if not lkey == 'useaxisattribute':
            continue
        val = options.__dict__[key]
        if val:
            return True
        return False
    return False

def _get_option_value(options,option_defaults,name):
    ''' Get an option value when the file is opened, considering 
        the default value and and any value set using  options keyword.
        The name must be one of the valid Python-level options and 
        have a default value of None.
    '''

    if options is not None:
        lname = name.lower()
        for key in options.__dict__.keys():
            lkey = key.lower()
            if not lkey == lname:
                continue
            val = options.__dict__[key]
            if val:
                return val
            return None

    if option_defaults.has_key(name):
        return option_defaults[name]
    else:
        return None

def set_option(self,option,value):
    '''
    Set certain options for an open NioFile instance.
    The option name is specified as a string
    Options that can be set include:
    MaskedArrayMode -- Specify MaskedArray bahavior (string):
        'MaskedIfFillAtt' -- Return a masked array iff file variable has a 
            _FillValue or a missing_value attribute (default).
        'MaskedNever' -- Never return a masked array for any variable.
        'MaskedAlways' -- Return a masked array for all variables.
        'MaskedIfFillAttAndValue' -- Return a masked array iff file variable has a 
            _FillValue or a missing_value attribute and the returned data array 
            actually contains 1 or more fill values.
        'MaskedExplicit' -- Only mask values specified explicitly using options 
            'ExplicitFillValues, MaskBelowValue, and/or MaskAboveValue; 
            ignore fill value attributes associated with the variable.
    ExplicitFillValues -- A scalar value or a sequence of values to be masked in the 
        return array. The first value becomes the fill_value attribute of the MaskedArray.
        Setting this option causes MaskedArrayMode to be set to 'MaskedExplicit'.
    MaskBelowValue -- A scalar value all values less than which are masked. However, if
        MaskAboveValue is less than MaskBelowValue, a range of of values become masked.
        Setting this option causes MaskedArrayMode to be set to 'MaskedExplicit'.
    MaskAboveValue -- A scalar value all values greater than which are masked. However, if
        MaskBelowValue is greater than MaskAboveValue, a range of of values become masked.
        Setting this option causes MaskedArrayMode to be set to 'MaskedExplicit'.
    '''
    valid_opts = {'maskedarraymode':'ma_mode', 'explicitfillvalues':'explicit_fill_values',
                  'maskbelowvalue' : 'mask_below_value', 'maskabovevalue' : 'mask_above_value' }
    if hasattr(option,'lower'):
        loption = option.lower()
    else:
        loption = option
    if not valid_opts.has_key(loption):
        raise KeyError, "Option %s invalid or cannot be set on open NioFile" % (option,)
        
    if hasattr(value,'lower'):
        lvalue = value.lower()
    else:
        lvalue = value
    if loption == 'explicitfillvalues' or loption == 'maskbelowvalue' or loption == 'maskabovevalue':
        if lvalue is not None:
            setattr(self,'ma_mode','maskedexplicit')

    return setattr(self,valid_opts[loption],lvalue)

def open_file(filename, mode = 'r', options=None, history='', format=''):
    '''
Open a file containing data in a supported format for reading and/or writing.

f = Nio.open_file(filepath, mode='r',options=None, history='', format='')
filepath -- path of file with data in a supported format. Either the path must
end with an extension indicating the expected format of the file (it need not
be part of the actual file name), or it must be specified using the optional 
'format' argument. Valid extensions include:
    .nc, .cdf, .netcdf, .nc3, .nc4,  -- NetCDF
    .gr, .grb, .grib, .gr1, .grb1, .grib1, .gr2, .grb2, .grib2, -- GRIB
    .hd, .hdf -- HDF
    .he2, .he4, .hdfeos -- HDFEOS
    .he5, .hdfeos5 -- HDFEOS5
    .shp, .mif, .gmt, .rt1 -- shapefiles, other formats supported by GDAL OGR
    .ccm -- CCM history files
Extensions are handled case-insensitvely, i.e.: .grib, .GRIB, and .Grib all
indicate a GRIB file.
mode -- access mode (optional):
     'r' -- open an existing file for reading
     'w','r+','rw','a' -- open an existing file for modification
     'c' -- create a new file open for writing
options -- instance of NioOptions class used to specify generic or 
format-specific options.
history -- a string specifying text to be appended to the file's global
attribute. The attribute is created if it does not exist. Only valid
if the file is opened for writing.
format -- a string specifying the expected format. Valid strings are the
same as the extensions listed above without the initial period (.). 

Returns an NioFile object.
    '''

    ma_mode  = _get_masked_array_mode(options,_Nio.option_defaults)
    use_axis_att = _get_axis_att(options,_Nio.option_defaults)
    explicit_fill_values = _get_option_value(options,_Nio.option_defaults,'ExplicitFillValues')
    mask_below_value = _get_option_value(options,_Nio.option_defaults,'MaskBelowValue')
    mask_above_value = _get_option_value(options,_Nio.option_defaults,'MaskAboveValue')

    file = _Nio.open_file(filename,mode,options,history,format)

    file_proxy = _proxy(file, 'str', create_variable=create_variable,close=close)
    setattr(file_proxy.__class__,'set_option',set_option)
    file_proxy.file = file
    if not (explicit_fill_values is None and mask_below_value is None and mask_above_value is None):
        file_proxy.ma_mode = 'maskedexplicit'
    else:
        file_proxy.ma_mode = ma_mode
    file_proxy.explicit_fill_values = explicit_fill_values
    file_proxy.mask_below_value = mask_below_value
    file_proxy.mask_above_value = mask_above_value

    if use_axis_att:
        cf_dims = _get_cf_dims(file_proxy)
        newdims = {}
        cf2dims = {}
        dimensions = file_proxy.dimensions
        for dim in dimensions:
            try:
                newdim = cf_dims[dim]
            except KeyError:
                newdim = dim
            newdims[newdim] = dimensions[dim]
            cf2dims[newdim] = dim
    else:
        cf2dims = None
        newdims = file_proxy.dimensions
    file_proxy.cf_dimensions = newdims
    file_proxy.cf2dims = cf2dims

    variable_proxies = {}
    for var in file.variables.keys():
        vp  = _proxy(file.variables[var],'str','len',
                     __setitem__=__setitem__,__getitem__=__getitem__,get_value=get_value,assign_value=assign_value)
        vp.file = file_proxy
        vp.varname = var
        variable_proxies[var] = vp
        if use_axis_att:
            newdims = []
            dimensions = vp.dimensions
            for dim in dimensions:
                try:
                    newdim = cf_dims[dim]
                except KeyError:
                    newdim = dim
                newdims.append(newdim)
            vp.cf_dimensions = tuple(newdims)
        else:
            vp.cf_dimensions = vp.dimensions
    file_proxy.variables = variable_proxies

    #print file_proxy, file_proxy.variables
    return file_proxy


def _get_cf_dims(file):
    ret = {}
    for dim in file.dimensions:
        if dim in file._obj.variables:
            try:
                axis = file._obj.variables[dim].axis
                ret[dim] = axis.lower()
            except AttributeError:
                pass
    return ret
            
def options():
    '''
Return an NioOptions object for specifying format-specific options.
opt = Nio.options()
Assign 'opt' as the third (optional) argument to Nio.open_file.
print opt.__doc__ to see valid options.
    '''
    opt = _Nio.options()
    return opt


 
