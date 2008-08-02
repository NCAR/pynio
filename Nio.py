"""
PyNIO enables NetCDF-like access for
NetCDF (rw), HDF (rw), HDFEOS (r)(optional), GRIB (r), and CCM (r) data files.

import Nio

Class NioFile:

f = Nio.open_file(filepath, mode='r', options=None, history='')

To see summary of file contents, including all dimensions, attributes,
and variables:
   print f
attributes:
   dimensions -- dimension names (keys), dimension lengths (values)
   variables -- variable names (keys), variable objects (values)
   __dict__ --  contains the global attributes associated with the file
methods:
   close(history='')
   create_dimension(name, length)
   create_variable(name, type,dimensions)
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
    __dict__ -- a dictionary containing the variable attributes
Methods:
    assign_value(value) -- assign a value to a variable in the file.
    get_value() -- retrieve the value of a variable in the file.
    typecode() -- return a character code representing the variable's type.
For more detailed information:
    print v.__doc__

For complete documentation see:

        http://www.pyngl.ucar.edu/Nio.html
"""

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
del pynio_version

__all__ = [ 'open_file', 'option_defaults', 'options' ]

def pyniopath_ncarg():
#
#  Find the root directory that contains the supplemental PyNIO files,
#  in particular, the grib2 codetables. For now the default is to look
#  in site-packages/PyNGL/ncarg. Otherwise, check the PYNGL_NCARG
#  environment variable. This may change if the grib2 codetables 
#  are moved into the PyNIO tree.
#  
#
  import sys
  pkgs_path = None
  for path in sys.path:
    slen = len('site-packages')
    i = path.rfind('site-packages')
    if i > -1 and i + slen == len(path):
      pkgs_path = path
      break

  pyngl1_dir  = os.path.join(pkgs_path,"PyNGL","ncarg")
  pyngl2_dir  = os.environ.get("PYNGL_NCARG")
  ncarg_dir  = os.environ.get("NCARG_ROOT")

  if pyngl2_dir != None and os.path.exists(pyngl2_dir):
    pyngl_ncarg = pyngl2_dir
  elif os.path.exists(pyngl1_dir):
    pyngl_ncarg = pyngl1_dir
  else:
    if os.path.exists(ncarg_dir):
        pyngl_ncarg = os.path.join(ncarg_dir,"lib","ncarg")
        if not os.path.exists(pyngl_ncarg):
            print "pynglpath: directory " + pyngl1_dir + \
                  "\n           does not exist and " + \
                  "environment variable PYNGL_NCARG is not set and " + \
                  "no usable NCARG installation found"
            sys.exit()
    else:
        sys.exit()

  return pyngl_ncarg

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
import numpy
from numpy import ma
from coordsel import *

_Nio.option_defaults['UseAxisAttribute'] = False
_Nio.option_defaults['MaskedArrayMode'] = 'MaskedIfFillAtt'
_Nio.option_defaults['ExplicitFillValues'] = None
_Nio.option_defaults['MaskBelowValue'] = None
_Nio.option_defaults['MaskAboveValue'] = None

def get_integer_version(strversion):
    ''' converts string version number into an integer '''
    d = strversion.split('.')
    if len(d) > 2:
       v = int(d[0]) * 10000 + int(d[1]) * 100 + int(d[2])
    elif len(d) is 2:
       v = int(d[0]) * 10000 + int(d[1]) * 100
    else:
       v = int(d[0]) * 10000
    return v

_is_new_ma = get_integer_version(numpy.__version__) > 10004
del get_integer_version

class _Proxy(object):
    """ base class for all proxies """
    def __init__(self, obj):
        super(_Proxy, self).__init__(obj)
        super(_Proxy,self).__setattr__('_obj', obj)
        super(_Proxy,self).__setattr__('attributes',{})
        for key in obj.__dict__.keys():
           super(_Proxy,self).__setattr__(key,obj.__dict__[key])
           self.attributes[key] = obj.__dict__[key]

    def __getattribute__(self, attrib):
        localatts = ['__doc__','__setattr__','attributes','_obj','variables','file','varname', \
	             'create_variable','cf_dimensions', 'cf2dims', 'ma_mode', 'explicit_fill_values', \
                     'mask_below_value', 'mask_above_value', 'set_option','__class__']

        if attrib in localatts:
            return super(_Proxy,self).__getattribute__(attrib)
        else:
            return getattr(self._obj,attrib)

    def __setattr__(self, attrib, value):
        localatts = ['__doc__','__setattr__','attributes','_obj','variables','file','varname', \
                     'cf_dimensions', 'cf2dims', 'ma_mode', 'explicit_fill_values', \
                     'mask_below_value', 'mask_above_value', 'set_option','__class__']    
        if attrib in localatts:
            super(_Proxy,self).__setattr__(attrib,value)
        else:
            setattr(self._obj,attrib,value)

    def __delattr__(self, attrib):
        localatts = ['__doc__','__setattr__cd ','attributes','_obj','variables','file','varname', \
                     'cf_dimensions', 'cf2dims', 'ma_mode','explicit_fill_values', \
                     'mask_below_value', 'mask_above_value', 'set_option','__class__' ] 
        if attrib in localatts:
            raise AttributeError, "Attempt to modify read only attribute"
        else:
            delattr(self._obj,attrib)

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
        for key in regulars.keys():
            setattr(cls, key, regulars[key])
            
        # also cache it for the future
        _known_proxy_classes[key] = cls
    # instantiate and return the needed proxy
    instance = cls(obj)
    return instance

    
def __getitem__(self, xsel):
    """ Return data specified by the extended selection object xsel.
        If there is a fill value return a masked array; otherwise an ndarray
    """
    ret = get_variable(self.file, self.varname, xsel)

    #
    # _FillValue is the preferred fill value attribute but if it is not set
    # then look for missing_value
    #
    if self.file.ma_mode == 'maskednever':
        # MaskedNever -- just return a numpy array
        return ret
    elif self.file.ma_mode == 'maskedexplicit':
	# handle user-specified masking -- first ranges, then explicit single values
        # note that masked_where does not remove previously applied mask values 
        # so it can be applied in stages

	if self.file.mask_below_value is not None and self.file.mask_above_value is not None:
            if self.file.mask_below_value > self.file.mask_above_value: 
                # mask a band of values
                ret = ma.masked_where((ret < self.file.mask_below_value) & (ret > self.file.mask_above_value),ret,copy=0)
            else:
                # mask high and low values                    
                ret = ma.masked_where((ret < self.file.mask_below_value) | (ret > self.file.mask_above_value),ret,copy=0)
        elif self.file.mask_below_value is not None:
            # mask low values
            ret = ma.masked_where(ret < self.file.mask_below_value,ret,copy=0)
        elif self.file.mask_above_value is not None:
            # mask high values
            ret = ma.masked_where(ret > self.file.mask_above_value,ret,copy=0)

	# now apply single fill values
        if hasattr(self.file.explicit_fill_values,'__iter__'):
            # multiple explicit fill values
            for fval in self.file.explicit_fill_values:
                ret = ma.masked_where(ret == fval,ret,copy=0)
            ret.set_fill_value(self.file.explicit_fill_values[0])
        elif self.file.explicit_fill_values is not None:
            ret = ma.masked_where(ret == self.file.explicit_fill_values,ret,copy=0)
            ret.set_fill_value(self.file.explicit_fill_values)

    elif self.file.ma_mode == 'maskediffillattandvalue':
        # MaskedIfFillAttAndValue -- return a masked array only if there are actual fill values
        if self.__dict__.has_key('_FillValue'):
            if ret.__contains__(self.__dict__['_FillValue'][0]):
                ret = ma.masked_where(ret == self.__dict__['_FillValue'][0],ret,copy=0)
                ret.set_fill_value(self.__dict__['_FillValue'][0])
        elif self.__dict__.has_key('missing_value'):
            if ret.__contains__(self.__dict__['missing_value'][0]):
                ret = ma.masked_where(ret == self.__dict__['missing_value'][0],ret,copy=0)
                ret.set_fill_value(self.__dict__['missing_value'][0])
    else: 
        # Handles MaskedIfFillAtt and MaskedAlways
        if self.__dict__.has_key('_FillValue'):
            ret = ma.masked_where(ret == self.__dict__['_FillValue'][0],ret,copy=0)
            ret.set_fill_value(self.__dict__['_FillValue'][0])
        elif self.__dict__.has_key('missing_value'):
            ret = ma.masked_where(ret == self.__dict__['missing_value'][0],ret,copy=0)
            ret.set_fill_value(self.__dict__['missing_value'][0])
        elif self.file.ma_mode == 'maskedalways':
            # supply a mask of all False, but just allow the fill_value to default
            mask = numpy.zeros(ret.shape,dtype='?')
            ret = ma.array(ret,mask=mask)

    return ret


def __setitem__(self, xsel,value):
    """ Set data into file variable with subscripts specified by the extended selection object xsel.
       If the value is a masked array fill it using the file variable fill value if it exists; 
       otherwise use the masked array fill value.
    """

    fill_value = None
    add_fill_value_att = False
    xsel = inp2xsel(self.file, self.varname, xsel)

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
            fval = self.__dict__['_FillValue'][0]
            fill_value = numpy.array(fval,dtype=fval.dtype)
        elif self.__dict__.has_key('missing_value'):
            fval = self.__dict__['missing_value'][0]
            fill_value = numpy.array(fval,dtype=fval.dtype)
        elif _is_new_ma:
            fill_value = numpy.array(value.fill_value,dtype=value.dtype)
            add_fill_value_att = True
        else:
            fill_value = numpy.array(value.fill_value(),dtype=value.dtype)
            add_fill_value_att = True
        value = value.filled(fill_value)

    if not isinstance(xsel, xSelect) or xsel.isbasic:
        self._obj[xsel] = value
    else:
        bb = xsel.bndbox()
        rsel = xsel - bb
        ret = self._obj[bb]
        ret[rsel] = value

    if add_fill_value_att:
        setattr(self._obj,'_FillValue',fill_value)

def _create_variable(self,name,type,dimensions):
    """create variable and store a reference"""
    #print 'in create variable'
    v = self._obj.create_variable(name,type,dimensions)
    if not v is None:
        vp  = _proxy(v,'str','len',__setitem__=__setitem__,__getitem__=__getitem__)
        vp.file = self
        vp.varname = name
        vp.cf_dimensions = vp.dimensions
        self.variables[name] = vp
    return vp

def _get_masked_array_mode(options,option_defaults):
    ''' 
        get the MaskedArrayMode value considering the default setting and any option value set 
        when the file is opened 
    '''

    # ma_mode specifies when to return masked arrays
    # MaskedNever: never return a masked array for any variable: default for backwards compatibility
    # MaskedIfFillAtt: return a masked array iff file variable has a _FillValue or a missing_value
    # MaskedAlways: return a masked array for all variables
    # MaskedIfFillAttAndValue: return a masked array iff file variable has a _FillValue or a missing_value and
    #                          the returned data array actually contains 1 or more fill values.

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
    return setattr(self,valid_opts[loption],lvalue)

def open_file(filename, mode = 'r', options=None, history='',format=''):

    ma_mode  = _get_masked_array_mode(options,_Nio.option_defaults)
    use_axis_att = _get_axis_att(options,_Nio.option_defaults)
    explicit_fill_values = _get_option_value(options,_Nio.option_defaults,'ExplicitFillValues')
    mask_below_value = _get_option_value(options,_Nio.option_defaults,'MaskBelowValue')
    mask_above_value = _get_option_value(options,_Nio.option_defaults,'MaskAboveValue')

    file = _Nio.open_file(filename,mode,options,history,format)

    file_proxy = _proxy(file, 'str', create_variable=_create_variable)
    setattr(file_proxy.__class__,'set_option',set_option)
    file_proxy.file = file
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
        vp  = _proxy(file.variables[var],'str','len',__setitem__=__setitem__,__getitem__=__getitem__)
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
        opt = _Nio.options()
        return opt


 