/*******************************************************
 * $Id: niomodule.c 16535 2016-06-17 23:24:19Z dbrown $
 *******************************************************/

/*
 * Objects representing NIO files and variables.
 *
 * David I. Brown
 * Adapted from netcdfmodule.c which was
 * Written by Konrad Hinsen
 * last revision: 1998-3-14
*/

#ifndef PYNIO
#define PYNIO
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "netcdf.h"
#include "Python.h"
#include "structmember.h"
#include "nio.h"
#include <numpy/arrayobject.h>

#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

/* Py_ssize_t for old Pythons */
/* This code is as recommended by: */
/* http://www.python.org/dev/peps/pep-0353/#conversion-guidelines */
#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#endif

#define _NIO_MODULE
#include "niomodule.h"


/* Python 3.7 changes the unicode char* to const char* */
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 7
typedef const char py3_char;
#else
typedef char py3_char;
#endif

static PyObject* get_NioError(void);
static PyObject* get_NioFileType(void);
static PyObject* get_NioVariableType(void);
static PyObject* get_Niomodule(void);
static char* get_errbuf(void);


/* Converts unicode, bytes, or str to UTF8 unicode object.
 * The ob argument can be unicode, bytes, or str;
 *
 * A new reference will always be returned.
 * */
static PyObject* pystr_to_utf8(PyObject *ob) {
	PyObject *bytes, *result;
	int del_bytes = 0;

#if PY_MAJOR_VERSION >= 3
	Py_ssize_t bytelen;
	char *bytedata = NULL;

	if (PyUnicode_Check(ob)){
		bytes = PyUnicode_AsUTF8String(ob);
		del_bytes = 1;
	}
	else if (PyBytes_Check(ob)) {
		bytes = ob;
		del_bytes = 0;
	} else {
		bytes = NULL;
		del_bytes = 0;
		PyErr_SetString(PyExc_ValueError, "argument is not a string type");
	}

	if (bytes) {
		bytelen = PyBytes_Size(bytes);
		bytedata = PyBytes_AsString(bytes);

		result = PyUnicode_DecodeUTF8(bytedata, bytelen, "strict");
	} else {
		result = NULL;
		PyErr_SetString(PyExc_ValueError, "creation of bytes failed");
	}
#else
	del_bytes = 0;

	if (PyUnicode_Check(ob)) {
		bytes = PyUnicode_AsUTF8String(ob);

	}
	else if (PyString_Check(ob)) {
		/* Make a copy of the string so that a new reference is returned */
		bytes = PyString_FromString(PyString_AsString(ob));
	} else {
		PyErr_SetString(PyExc_ValueError, "argument is not a string type");
		return NULL;
	}

	if (bytes) {
		result = bytes;
	} else {
		result = NULL;
		PyErr_SetString(PyExc_ValueError, "creation of bytes failed");
	}
#endif

	if (del_bytes) {
		Py_XDECREF(bytes);
	}

	return result;
}

/* This will convert a Unicode/PyString to a UTF8 char* string.
 *
 * The resulting string must be free'd by the user.
 */
static py3_char* as_utf8_char_and_size(PyObject *ob, Py_ssize_t *size) {
	char *result;
	char *tempstr;
	Py_ssize_t local_size;
	PyObject *pystr = NULL;

#if PY_MAJOR_VERSION >= 3
	tempstr = (char*) PyUnicode_AsUTF8AndSize(ob, &local_size);
#else
	/* First, need to convert the object to UTF8 unicode/string object */
	if (PyUnicode_Check(ob) || PyString_Check(ob)) {
		pystr = pystr_to_utf8(ob);
	} else {
		pystr = NULL;
	}

	if (pystr) {
		tempstr = PyString_AsString(pystr);
		local_size = strlen(tempstr);
	} else {
		result = NULL;
		if (size) {
			*size = 0;
		}

		return result;
	}
#endif

	/* create a copy of tempstr */
	result = calloc(local_size+1, sizeof(char));
	if (result) {
		strcpy(result, tempstr);
	} else {
		result = (char*) PyErr_NoMemory();
	}

	if (size) {
		*size = 0;
	}

	if (pystr) {
		Py_XDECREF(pystr);
	}

	return (py3_char*) result;
}

static py3_char* as_utf8_char(PyObject *ob) {
	return as_utf8_char_and_size(ob, NULL);
}

static PyObject* char_to_pystr(const char *s, Py_ssize_t len) {
	PyObject *result = NULL;

#if PY_MAJOR_VERSION >= 3
	result = PyUnicode_DecodeUTF8(s, len, "strict");
#else
	result = PyString_FromStringAndSize(s, len);
#endif
	return result;
}

static int is_string_type(PyObject *ob) {
	int result;

#if PY_MAJOR_VERSION >= 3
	result = PyUnicode_Check(ob) || PyBytes_Check(ob);
#else
	result = PyUnicode_Check(ob) || PyString_Check(ob);
#endif

	return result;

}

#define DICT_SETITEMSTRING(d, key, val) PyDict_SetItem(d, char_to_pystr(key, strlen(key)), val)


/* all doc strings defined within the C interface */

/* Nio.open_file.__doc__ */

static char *open_file_doc =
		"\n\
Open a file containing data in a supported format for reading and/or writing.\
\n\n\
f = Nio.open_file(filepath, mode='r',options=None, history='')\n\n\
filepath -- path of file with data in a supported format. The path  must end\n\
with an extension indicating the expected format of the file, whether or not\n\
it is part of the actual file name. Valid extensions include:\n\
    .nc, .cdf, .netcdf, .nc3, .nc4,  -- NetCDF\n\
    .gr, .grb, .grib, .gr1, .grb1, .grib1, .gr2, .grb2, .grib2, -- GRIB\n\
    .hd, .hdf -- HDF\n\
    .he2, .he4, .hdfeos -- HDFEOS\n\
    .he5, .hdfeos5 -- HDFEOS5\n\
    .shp, .mif, .gmt, .rt1 -- shapefiles, other formats supported by GDAL OGR\n\
    .ccm -- CCM history files\n\
Extensions are handled case-insensitvely, i.e.: .grib, .GRIB, and .Grib all\n\
indicate a GRIB file.\n\
mode -- access mode (optional):\n\
     'r' -- open an existing file for reading\n\
     'w','r+','rw','a' -- open an existing file for modification\n\
     'c' -- create a new file open for writing\n\
options -- instance of NioOptions class used to specify format-specific\n\
    options\n\
history -- a string specifying text to be appended to the file\'s global\n\
    attribute. The attribute is created if it does not exist. Only valid\n\
    if the file is open for writing\n\n\
Returns an NioFile object.\n\
";

/* Nio.options.__doc__ */

static char *options_doc =
		"\n\
Return an NioOptions object for specifying format-specific options.\n\n\
opt = Nio.options()\n\
Assign 'opt' as the third (optional) argument to Nio.open_file.\n\
print opt.__doc__ to see valid options.\n\
";

/*
 * opt = Nio.options()
 * opt.__doc__
 */
static char *option_class_doc =
		"\n\
NioOptions object\n\n\
Set options by assigning attributes to this object and then passing the\n\
object as an optional argument to Nio.open_file:\n\
opt.OptionName = value\n\
All option names and string option values are handled case-insensitively.\n\
\n\
Generic options:\n\
    MaskedArrayMode -- Specify MaskedArray bahavior (string):\n\
        'MaskedIfFillAtt' -- Return a masked array iff file variable has a\n\
            _FillValue or a missing_value attribute (default).\n\
        'MaskedNever' -- Never return a masked array for any variable.\n\
        'MaskedAlways' -- Return a masked array for all variables.\n\
        'MaskedIfFillAttAndValue' -- Return a masked array iff file variable has a\n\
            _FillValue or a missing_value attribute and the returned data array\n\
            actually contains 1 or more fill values.\n\
        'MaskedExplicit' -- Only mask values specified explicitly using options\n\
            'ExplicitFillValues, MaskBelowValue, and/or MaskAboveValue;\n\
            ignore fill value attributes associated with the variable.\n\
    ExplicitFillValues -- A scalar value or a sequence of values to be masked in the\n\
        return array. The first value becomes the fill_value attribute of the MaskedArray.\n\
        Setting this option causes MaskedArrayMode to be set to 'MaskedExplicit'.\n\
    MaskBelowValue -- A scalar value all values less than which are masked. However, if\n\
        MaskAboveValue is less than MaskBelowValue, a range of of values become masked.\n\
        Setting this option causes MaskedArrayMode to be set to 'MaskedExplicit'.\n\
    MaskAboveValue -- A scalar value all values greater than which are masked. However, if\n\
        MaskBelowValue is greater than MaskAboveValue, a range of of values become masked.\n\
        Setting this option causes MaskedArrayMode to be set to 'MaskedExplicit'.\n\
    UseAxisAttribute -- A boolean option that if set True when using extended subscripting,\n\
        and if coordinate variables have the CF-compliant 'axis' attribute, expects the\n\
        short names ('T','Z','Y' or 'X') instead of the actual coordinate names in the\n\
         subscript specification.\n\
\n\
NetCDF file options:\n\
    Format -- Specify the format of newly created files (string):\n\
        'Classic' -- (default) standard file (generally file size < 2GB)\n\
        'LargeFile' or '64BitOffset' -- (fixed-size variables or record\n\
            elements of unlimited dimension variables each up to 4GB)\n\
        'NetCDF4Classic' -- Classic mode NetCDF 4 file (uses HDF 5 as the\n\
            underlying format but is restricted to features of the classic\n\
            NetCDF data model).\n\
        'NetCDF4' -- NetCDF 4 file (uses HDF 5 as the underlying format).\n\
    CompressionLevel -- Specify the level of data compression on a scale\n\
            of 0 - 9 (ignored unless Format is set to 'NetCDF4Classic' or 'NetCDF4').\n\
    HeaderReserveSpace -- Reserve <int-value> extra bytes in the header\n\
        of a file open for writing. Used to subsequently add dimensions,\n\
        attributes, and variables to existing files efficiently.\n\
    MissingToFillValue -- If set True (the default), create a virtual\n\
        '_FillValue' attribute only for variables that have a\n\
        'missing_value' but no '_FillValue'.\n\
    PreFill -- If set True (the default), fill all elements of newly\n\
        defined variables with a fill value. If set False, elements are\n\
        undefined until data is written.\n\
    SafeMode -- Close the file after each individual operation on the file.\n\
\n\
GRIB files options\n\
    DefaultNCEPTable -- (GRIB 1 only) Specify the table to use in certain\n\
        ambiguous cases:\n\
        'Operational' -- (default) Use the NCEP operational parameter table\n\
        'Reanalysis' -- Use the NCEP reanalysis parameter table\n\
    InitialTimeCoordinateType -- Specify the type of the coordinate\n\
        associated with initial_time (as opposed to forecast_time)\n\
        dimensions:\n\
        'Numeric' -- (default) use CF-compliant numeric coordinates\n\
        'String' -- use date strings as the coordinates\n\
    SingleElementDimensions -- Specify that dimensional types with only a\n\
        single representative element be treated as full-fledged dimensions.\n\
        'None' -- (default) no single element dimensions are created\n\
        'All' -- all possible single element dimensions are created.\n\
        The names of individual dimension types may be specified individually:\n\
        'Initial_time', 'Forecast_time', 'Level', 'Ensemble', or 'Probability'.\n\
    ThinnedGridInterpolation -- Specify the type of interpolation for\n\
        thinned (GRIB 'quasi-regular') grids:\n\
        'Cubic' -- (cubic) use cubic interpolation\n\
        'Linear' -- use linear interpolation\n\n\
";

static char *niofile_type_doc =
		"\n\
NioFile object\n\n\
Given:\n\
f = Nio.open_file(filepath, mode='r', options=None, history='',format='')\n\
\n\
To see summary of file contents, including all attributes:\n\
   print f\n\
Assign global attributes to writable files or groups using:\n\
    f.global_att = global_att_value\n\
Attributes:\n\
   name -- the name of the file or group\n\
   dimensions -- dictionary of dimension lengths with dimension name keys\n\
   variables -- dictionary of variable objects with variable name keys\n\
   attributes --  dictionary of global file or group attributes with attribute name keys\n\
       (the following are applicable for advanced formats NetCDF4 and HDF5 only)\n\
   groups -- dictionary of groups with group name keys\n\
   ud_types -- dictionary of user-defined data type definitions with data type name keys\n\
   chunk_dimensions -- dictionary of chunking dimension sizes with dimension name keys\n\
   parent -- reference to the parent group, parent file for the root group, or None for a file\n\
   path -- the path of a group relative to the root group ('/'), or the file path for a file\n\
Methods:\n\
   close(history='') -- close the file\n\
   create_dimension(name, length) -- create a dimension in the file\n\
   create_variable(name, type, dimensions) -- create a variable in the file\n\
   unlimited(dimension_name) -- returns True if dimension_name refers to an unlimited dimension; False otherwise\n\
       (the following are applicable for advanced formats NetCDF4 and HDF5 only)\n\
   create_group(name) -- create a group in the file or group.\n\
   create_vlen(name,type,dimensions) -- create a variable length array variable in the file or group.\n\
   create_compound(name,type,dimensions) -- create a compound variable with the given type and dimensions.\n\
   create_compound_type(name, type)  -- create a user-defined compound type; with member names, sizes\n\
        and types as defined in the type sequence argument.\n\
";

/* NioFile object method doc strings */

/*
 * f = Nio.open_file(..)
 * f.close.__doc__
 * f.create_dimension.__doc__
 * f.create_chunk_dimension.__doc__
 * f.create_variable.__doc__
 * f.create_group.__doc__
 */

static char *close_doc =

		"\n\
Close a file, ensuring all modifications are up-to-date if open for writing.\
\n\n\
f.close([history])\n\
history -- optional string appended to the global 'history' attribute\n\
before closing a writable file. The attribute is created if it does not\n\
already exist.\n\
Read or write access attempts on the file object after closing\n\
raise an exception.\n\
";

static char *create_dimension_doc =
		"\nCreate a new dimension with the given name and length in a writable file.\n\n\
f.create_dimension(name,length)\n\
name -- a string specifying the dimension name.\n\
length -- a positive integer specifying the dimension length. If set to\n\
None or 0, specifies the unlimited dimension.\n";

static char *create_chunk_dimension_doc =
		"\nCreate a chunking size for a dimension that has been defined but not yet used in a writable file.\n\
The size must be no larger than the dimension size; once set it cannot be changed.\n\
f.create_chunk_dimension(name,length)\n\
name -- a string specifying the dimension name.\n\
length -- a positive integer specifying the dimension length. If set to\n\
None or 0, it will be reset to 1.\n";

static char *create_variable_doc =
		"\n\
Create a new variable with given name, type, and dimensions in a writable file.\
\n\n\
f.create_variable(name,type,dimensions)\n\
name -- a string specifying the variable name.\n\
type -- a type identifier. The following are currently supported:\n\
    'd' -- 64 bit real\n\
    'f' -- 32 bit real\n\
    'l' -- 32 bit integer\n\
    'L' -- 32 bit unsigned integer\n\
    'q' -- 64 bit integer\n\
    'Q' -- 64 bit unsigned integer\n\
    'h' -- 16 bit integer\n\
    'H' -- 16 bit unsigned integer\n\
    'b' -- 8 bit integer\n\
    'B' -- 8 bit unsigned integer\n\
    'S1','c' -- character\n\
dimensions -- a tuple of dimension names (strings), previously defined\n\
";

static char *create_group_doc =
		"\n\
Create a new group with given name in a writable file.\
\n\n\
f.create_group(name)\n\
name -- a string specifying the group name.\n\
";

static char *create_vlen_doc =
		"\n\
Create a new variable length array variable with given name in a writable file.\
\n\n\
f.create_vlen(name,type,dimensions)\n\
name -- a string specifying the vlen variable name.\n\
type -- the variable type.\n\
dimensions -- the dimensions of the vlen variable.\n\
";

static char *create_compound_type_doc =
		"\n\
Create a new compound variable type with given name in a writable file.\
\n\n\
f.create_compound_type(name,type)\n\
name -- a string specifying the compound type name.\n\
type -- a sequence containing the name, type, and number of elements for each member of the compound type\n\
";

static char *create_compound_doc =
		"\n\
Create a new compound variable with given name in a writable file.\
\n\n\
f.create_compound(name,type,dimensions)\n\
name -- a string specifying the compound variable name.\n\
type -- the variable type.\n\
dimensions -- the dimensions of the compound variable.\n\
";

static char *unlimited_doc =
		"\n\
Returns True or False depending on whether the named dimension is unlimited.\n\n\
f.unlimited(name)\n\
name -- a string specifying the dimension name to be queried for the unlimited property.\n\
";

static char *niovariable_type_doc =
		"\n\
NioVariable object\n\n\
Given \n\
    v = f.variables['varname']\n\
Get summary of variable contents using:\n\
    print v\n\
Assign variable attributes for writable files using, e.g:\n\
    v.units = 'meters'\n\
Get or assign variable values using slicing syntax:\n\
    val = v[:]\n\
assigns all elements of variable to 'val', retaining dimensionality;\n\
    val = v[0,:]\n\
assigns one element of the first dimension and all elements of the remaining dimensions.\n\
Attributes:\n\
    rank -- a scalar value indicating the number of dimensions\n\
    shape -- a tuple containing the number of elements in each dimension\n\
    dimensions -- a tuple containing the dimensions names in order\n\
    attributes -- a dictionary of variable attributes with attribute name keys\n\
    size -- a scalar value indicating the size in bytes of the variable\n\
    name -- the name of the variable\n\
    parent -- reference to the group or file to which the variable belongs\n\
    path -- the path of the variable relative to the root group ('/')\n\
Methods:\n\
    assign_value(value) -- assign a value to a variable in the file.\n\
    get_value() -- retrieve the value of a variable in the file.\n\
    typecode() -- return a character code representing the variable's type.\n\
    set_option(option,value) -- set certain options.\n\
";

/* NioVariable object method doc strings */

/*
 * v = f.variables['varname']
 * v.assign_value.__doc__
 * v.get_value.__doc__
 * v.typecode.__doc__
 */

static char *assign_value_doc =
		"\n\
Assign a value to a variable in the file.\n\n\
v = f.variables['varname']\n\
v.assign_value(value)\n\
value - a NumPy array or a Python sequence of values that are coercible\n\
to the type of variable 'v'.\n\
This method is the only way to assign a scalar value. There is no way to\n\
indicate a slice. For array variables direct assignment using slicing\n\
syntax is more flexible.\n\
";

static char *get_value_doc =
		"\n\
Retrieve the value of a variable in the file.\n\n\
v = f.variables['varname']\n\
val = v.get_value()\n\
'val' is returned as a NumPy array.\n\
This method is the only way to retrieve the scalar value from a file.\n\
There is no way to indicate a slice. For array variables direct assignment\n\
using slicing syntax is more flexible.\n\
";

static char *typecode_doc =
		"\n\
Return a character code representing the variable's type.\n\n\
v = f.variables['varname']\n\
t = v.typecode()\n\
Return variable 't' will be one of the following:\n\
    'd' -- 64 bit real\n\
    'f' -- 32 bit real\n\
    'i' -- 32 bit integer\n\
    'I' -- 32 bit unsigned integer\n\
    'l' -- 32 or 64 bit integer (platform dependent)\n\
    'L' -- 32 or 64 bit unsigned integer (platform dependent)\n\
    'q' -- 64 bit integer\n\
    'Q' -- 64 bit unsigned integer\n\
    'h' -- 16 bit integer\n\
    'H' -- 16 bit unsigned integer\n\
    'b' -- 8 bit integer\n\
    'B' -- 8 bit unsigned integer\n\
    'S1', 'c' -- character\n\
    'S' -- string\n\
    'v' -- Vlen\n\
    'x' -- Cmpound\n\
";

static void
NioVariableObject_dealloc(NioVariableObject *self);
static void
NioFileObject_dealloc(NioFileObject *self);

/*
 * global used in NclMultiDValData
 */
short NCLnoPrintElem = 0;

size_t NCLtotalVariables = 0;
size_t NCLtotalGroups = 0;

static int nio_file_init(NioFileObject *self);
static NioFileObject* nio_read_group(NioFileObject* file,
		NclFileGrpNode *grpnode);
static NioFileObject* nio_create_group(NioFileObject* file,
		NrmQuark gname);
static NioVariableObject* nio_read_advanced_variable(NioFileObject* file,
		NclFileVarNode* varnode, int id);
static NioVariableObject *nio_variable_new(NioFileObject *file,
		char *name, int id, int type, int ndims, NrmQuark *qdims, int nattrs);
void _convertObj2COMPOUND(PyObject* pyobj, obj* listids,
		NclFileCompoundRecord *comprec, ng_size_t n_dims, ng_size_t curdim,
		ng_usize_t* counter);

#if PY_MAJOR_VERSION < 3
static char err_buf[256];
static PyObject *Niomodule; /* the Nio Module object */

/* Error object and error messages for nio-specific errors */

static PyObject *NIOError;
#endif

static char *nio_errors[] = { "No Error", /* 0 */

"Not a NIO id", "Too many NIO files open", "NIO file exists && NC_NOCLOBBER",
		"Invalid Argument", "Write to read only",
		"Operation not allowed in data mode",
		"Operation not allowed in define mode", "Coordinates out of Domain",
		"MAX_NC_DIMS exceeded", "String match to name in use",
		"Attribute not found", "MAX_NC_ATTRS exceeded", "Not a NIO data type",
		"Invalid dimension id", "NC_UNLIMITED in the wrong index",
		"MAX_NC_VARS exceeded", "Variable not found",
		"Action prohibited on NC_GLOBAL varid", "Not an NIO supported file",
		"In Fortran, string too short", "MAX_NC_NAME exceeded",
		"NC_UNLIMITED size already in use", /* 22 */
		"Memory allocation error", "attempt to set read-only attributes",
		"invalid mode specification", "", "", "", "", "", "", "XDR error" /* 32 */
};

static void nio_seterror(int nio_ncerr) {
	if (nio_ncerr != 0) {
		char *error = "Unknown error";

		if (nio_ncerr > 0 && nio_ncerr <= 32)
			error = nio_errors[nio_ncerr];

		PyErr_SetString(get_NioError(), error);
	}
}

/*
 * Python equivalents to NIO data types
 *
 * Attention: the following specification may not be fully portable.
 * The comments indicate the correct NIO specification. The assignment
 * of Python types assumes that 'short' is 16-bit and 'int' is 32-bit.
 */

#if 0
int data_types[] =
{	-1, /* not used */

	NPY_BYTE, /* signed 8-bit int */
	NPY_CHAR, /* 8-bit character */
	NPY_SHORT, /* 16-bit signed int */
	NPY_INT, /* 32-bit signed int */
	NPY_FLOAT, /* 32-bit IEEE float */
	NPY_DOUBLE /* 64-bit IEEE float */
};
#endif

int data_type(NclBasicDataTypes ntype) {
	switch (ntype) {
	case NCL_short:
		return NPY_SHORT;
	case NCL_ushort:
		return NPY_USHORT;
	case NCL_int:
		return NPY_INT; /* netcdf 3.x has only a long type */
	case NCL_uint:
		return NPY_UINT; /* netcdf 3.x has only a long type */
	case NCL_long:
		return NPY_LONG;
	case NCL_ulong:
		return NPY_ULONG;
	case NCL_int64:
		return NPY_LONGLONG;
	case NCL_uint64:
		return NPY_ULONGLONG;
	case NCL_float:
		return NPY_FLOAT;
	case NCL_double:
		return NPY_DOUBLE;
	case NCL_byte:
		return NPY_BYTE;
	case NCL_ubyte:
		return NPY_UBYTE;
	case NCL_char:
		return NPY_CHAR;
	case NCL_logical:
		return NPY_BOOL;
	case NCL_string:
		return NPY_STRING;
	case NCL_vlen:
		return NPY_VLEN;
	case NCL_compound:
		return NPY_COMPOUND;
	case NCL_enum:
		return NPY_ENUM;
	case NCL_opaque:
		return NPY_OPAQUE;
	case NCL_reference:
		return NPY_REFERENCE;
	case NCL_list:
		return NPY_LIST;
	case NCL_listarray:
		return NPY_LISTARRAY;
	default:
		return NPY_NOTYPE;
	}
	return NPY_NOTYPE;
}

/* Utility functions */

static void define_mode(NioFileObject *file, int define_flag) {
	if (file->define != define_flag) {
#if 0
		if (file->define)
		ncendef(file->id);
		else
		ncredef(file->id);
#endif
		file->define = define_flag;
	}
}

static char *
typecode(int type) {
	static char buf[3];

	memset(buf, 0, 3);

	switch (type) {
	case NPY_BOOL:
		buf[0] = NPY_BOOLLTR;
		break;
	case NPY_BYTE:
		buf[0] = NPY_BYTELTR;
		break;
	case NPY_UBYTE:
		buf[0] = NPY_UBYTELTR;
		break;
	case NPY_SHORT:
		buf[0] = NPY_SHORTLTR;
		break;
	case NPY_USHORT:
		buf[0] = NPY_USHORTLTR;
		break;
	case NPY_INT:
		buf[0] = NPY_INTLTR;
		break;
	case NPY_UINT:
		buf[0] = NPY_UINTLTR;
		break;
	case NPY_LONG:
		buf[0] = NPY_LONGLTR;
		break;
	case NPY_ULONG:
		buf[0] = NPY_ULONGLTR;
		break;
	case NPY_LONGLONG:
		buf[0] = NPY_LONGLONGLTR;
		break;
	case NPY_ULONGLONG:
		buf[0] = NPY_ULONGLONGLTR;
		break;
	case NPY_FLOAT:
		buf[0] = NPY_FLOATLTR;
		break;
	case NPY_DOUBLE:
		buf[0] = NPY_DOUBLELTR;
		break;
	case NPY_STRING:
		buf[0] = NPY_STRINGLTR;
		break;
#if 0
		case NPY_CHAR:
		buf[0] = NPY_STRINGLTR;
		break;
#else
	case NPY_CHAR:
		buf[0] = NPY_CHARLTR;
		break;
#endif
	case NPY_VLEN:
		buf[0] = NPY_VLENLTR;
		break;
	case NPY_COMPOUND:
		buf[0] = NPY_COMPOUNDLTR;
		break;
	case NPY_ENUM:
		buf[0] = NPY_ENUMLTR;
		break;
	case NPY_OPAQUE:
		buf[0] = NPY_OPAQUELTR;
		break;
	case NPY_LIST:
		buf[0] = NPY_LISTLTR;
		break;
	case NPY_LISTARRAY:
		buf[0] = NPY_LISTARRAYLTR;
		break;
	case NPY_REFERENCE:
		buf[0] = NPY_REFERENCELTR;
		break;
	default:
		buf[0] = ' ';
		break;
	}
	return &buf[0];
}

static NrmQuark nio_type_from_code(int code) {
	NrmQuark type;
	switch ((char) code) {
	case 'c':
		type = NrmStringToQuark("character");
		break;
	case 'S':
		type = NrmStringToQuark("string");
		break;
	case '1':
	case 'b':
		type = NrmStringToQuark("byte");
		break;
	case 'B':
		type = NrmStringToQuark("ubyte");
		break;
	case 'h':
		type = NrmStringToQuark("short");
		break;
	case 'H':
		type = NrmStringToQuark("ushort");
		break;
	case 'i':
		type = NrmStringToQuark("integer");
		break;
	case 'I':
		type = NrmStringToQuark("uint");
		break;
	case 'l':
		type = NrmStringToQuark("long");
		break;
	case 'L':
		type = NrmStringToQuark("ulong");
		break;
	case 'q':
		type = NrmStringToQuark("int64");
		break;
	case 'Q':
		type = NrmStringToQuark("uint64");
		break;
	case 'f':
		type = NrmStringToQuark("float");
		break;
	case 'd':
		type = NrmStringToQuark("double");
		break;
	case '?':
		type = NrmStringToQuark("logical");
		break;
	case 'v':
		type = NrmStringToQuark("list");
		break;
	case 'x':
		type = NrmStringToQuark("compound");
		break;
	case 'O':
		type = NrmStringToQuark("object");
		break;
	default:
		type = NrmNULLQUARK;
	}
	return type;
}

static void collect_attributes(void *fileid, int varid, PyObject *attributes,
		int nattrs) {
	NclFile file = (NclFile) fileid;
	NclFileAttInfoList *att_list = NULL;
	NclFAttRec *att;
	NclFVarRec *fvar = NULL;
	char *name;
	npy_intp length;
	int py_type;
	int i;
	if (varid > -1) {
		att_list = file->file.var_att_info[varid];
		fvar = file->file.var_info[varid];
	}
	for (i = 0; i < nattrs; i++) {
		NclMultiDValData md;
		if (varid < 0) {
			att = file->file.file_atts[i];
			name = NrmQuarkToString(att->att_name_quark);
			md = _NclFileReadAtt(file, att->att_name_quark, NULL);
		} else if (!(att_list && fvar)) {
			PyErr_SetString(get_NioError(),
					"internal attribute or file variable error");
			return;
		} else {
			att = att_list->the_att;
			name = NrmQuarkToString(att->att_name_quark);
			md = _NclFileReadVarAtt(file, fvar->var_name_quark,
					att->att_name_quark, NULL);
			att_list = att_list->next;
		}
		if (att->data_type == NCL_string) {
			PyObject *string = NULL;
			char *satt = NrmQuarkToString(*((NrmQuark *) md->multidval.val));
			if (satt != NULL) {
				string = char_to_pystr(satt, strlen(satt));
			} else {
				string = char_to_pystr("", strlen(""));
			}
			if (string != NULL) {
				DICT_SETITEMSTRING(attributes, name, string);
				Py_DECREF(string);
			}
		} else {
			PyObject *array;
			length = (npy_intp) md->multidval.totalelements;
			py_type = data_type(att->data_type);
			/* following three if clauses are temporary until these types are actually supported */
			if (py_type == NPY_REFERENCE) {
				py_type = NPY_LONG;
			}
			if (py_type == NPY_COMPOUND) {
				py_type = NPY_LONG;
			}
			if (py_type == NPY_ENUM) {
				py_type = NPY_LONG;
			}
			array = PyArray_SimpleNew(1, &length, py_type);
			if (array != NULL) {
				memcpy(PyArray_BYTES((PyArrayObject *) array), md->multidval.val,
						(size_t) length * md->multidval.type->type_class.size);
				array = PyArray_Return((PyArrayObject *) array);
				if (array != NULL) {
					DICT_SETITEMSTRING(attributes, name, array);
					Py_DECREF(array);
				}
			}
		}
	}
}

static int set_attribute(NioFileObject *file, int varid, PyObject *attributes,
		char *name, PyObject *value) {
	NclFile nfile = (NclFile) file->id;
	NhlErrorTypes ret;
	NclMultiDValData md = NULL;
	PyArrayObject *array = NULL;

	if (!value || value == Py_None) {
		/* delete attribute */
		if (varid == NC_GLOBAL) {
			ret = _NclFileDeleteAtt(nfile, NrmStringToQuark(name));
		} else {
			ret = _NclFileDeleteVarAtt(nfile,
					nfile->file.var_info[varid]->var_name_quark,
					NrmStringToQuark(name));
		}
		PyObject_DelItemString(attributes, name);
		return 0;
	}

	if (is_string_type(value)) {
		ng_size_t len_dims = 1;
		NrmQuark *qval = malloc(sizeof(NrmQuark));

		py3_char* utf8 = as_utf8_char(value);

		qval[0] = NrmStringToQuark(utf8);

		free(utf8);

		md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0, (void*) qval,
				NULL, 1, &len_dims, TEMPORARY, NULL,
				(NclTypeClass) nclTypestringClass);
	} else {
		ng_size_t dim_sizes = 1;
		int n_dims;
		NrmQuark qtype;
		int pyarray_type = NPY_NOTYPE;
		PyArrayObject *tmparray = (PyArrayObject *) PyDict_GetItemString(
				attributes, name);

		if (tmparray != NULL) {
			pyarray_type = PyArray_TYPE(tmparray);
		}

		array = (PyArrayObject *) PyArray_ContiguousFromAny(value, pyarray_type, 0, 1);
		if (array) {
			n_dims = (PyArray_NDIM(array) == 0) ? 1 : PyArray_NDIM(array);
			qtype = nio_type_from_code(PyArray_DTYPE(array)->type);
			if (PyArray_ITEMSIZE(array) == 8
					&& qtype == NrmStringToQuark("long")) {
				PyArrayObject *array2 = (PyArrayObject *) PyArray_Cast(array,
						NPY_INT);
				Py_DECREF(array);
				array = array2;
				qtype = NrmStringToQuark("integer");
				sprintf(get_errbuf(),
						"output format does not support 8-byte integers; converting to 4-byte integer variable (%s): possible data loss due to overflow",
						name);
				PyErr_SetString(get_NioError(), get_errbuf());
				PyErr_Print();

			}

			if (array) {
				ng_size_t *dims;
				void *data;
				if (PyArray_NDIM(array) == 0) {
					dims = &dim_sizes;
				} else {
					/* TODO:  Make sure these pointers are the same in new api */
					dims = (ng_size_t *) PyArray_DIMS(array);
				}
				data = malloc(PyArray_NBYTES(array));
				memcpy(data, PyArray_DATA(array), PyArray_NBYTES(array));

				md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
						(void*) data, NULL, n_dims, dims, TEMPORARY, NULL,
						_NclNameToTypeClass(qtype));

			}
		}
	}
	if (!md) {
		nio_seterror(23);
		return -1;
	}

	if (varid == NC_GLOBAL) {
		ret = _NclFileWriteAtt(nfile, NrmStringToQuark(name), md, NULL);
	} else {
		ret = _NclFileWriteVarAtt(nfile,
				nfile->file.var_info[varid]->var_name_quark,
				NrmStringToQuark(name), md, NULL);
	}
	if (ret > NhlFATAL) {
		if (is_string_type(value)) {
			DICT_SETITEMSTRING(attributes, name, value);
		} else if (array) {
			DICT_SETITEMSTRING(attributes, name, (PyObject * )array);
		}
	}
	return 0;
}

static int check_if_open(NioFileObject *file, int mode) {
	/* mode: -1 read, 1 write, 0 other */
	if (file->open) {
		if (mode != 1 || file->write) {
			return 1;
		} else {
			PyErr_SetString(get_NioError(), "write access to read-only file");
			return 0;
		}
	} else {
		PyErr_SetString(get_NioError(), "file has been closed");
		return 0;
	}
}

/*
 * NioFile object
 * (type declaration in niomodule.h)
 */

/* Destroy file object */

static void NioFileObject_dealloc(NioFileObject *self) {
	PyObject *keys, *key;
	Py_ssize_t i;

	if (self->parent) {/* group objects should just delete themselves */
		PyObject_DEL(self);
		return;
	}
	if (self->being_destroyed) {
		/* already closed and mostly deallocated -- just del object and return */
		Py_XDECREF(self->name);
		Py_XDECREF(self->mode);
		PyObject_DEL(self);
		return;
	} else if (self->open) {
		NioFile_Close(self); /* calls this function back so we do not want to run it again */
		return;
	}

	/* destroy the contents of all groups except the root ('/') group */
	keys = PyDict_Keys(self->groups);
	if (NULL != keys) {
		for (i = 0; i < PyList_Size(keys); i++) {

			py3_char *utf8 = NULL;

			key = PyList_GetItem(keys, i);
			NioFileObject *g = (NioFileObject *) PyDict_GetItem(self->groups,
					key);
			utf8 = as_utf8_char(g->name);
			if (g != NULL && strcmp(utf8, "/")) {
				_NclDestroyObj((NclObj) g->gnode);
			}

			free(utf8);

			Py_CLEAR(g->variables);
			Py_CLEAR(g->dimensions);
			Py_CLEAR(g->chunk_dimensions);
			Py_CLEAR(g->ud_types);
			Py_CLEAR(g->groups);
			Py_CLEAR(g->attributes);
			Py_XDECREF(g->name);
			Py_XDECREF(g->full_path);
			Py_XDECREF(g->mode);
			Py_XDECREF(g->type);
			PyDict_DelItem(self->groups, key);
		}
		Py_DECREF(keys);
	}
	if (NULL != self->id)
		_NclDestroyObj((NclObj) self->id);

	Py_CLEAR(self->variables);
	Py_CLEAR(self->dimensions);
	Py_CLEAR(self->chunk_dimensions);
	Py_CLEAR(self->ud_types);
	Py_CLEAR(self->groups);
	Py_CLEAR(self->attributes);
	Py_XDECREF(self->full_path);
	Py_XDECREF(self->type);

	/* The name and mode components are kept around for the benefit of the repr method */
	self->being_destroyed = 1; /* indicates file is closed and data structures have been mostly torn down */
	return;
}

static int GetNioMode(char* filename, char *mode) {
	struct stat buf;
	int crw;
	char *cp = NULL;
	char *fbuf;

	fbuf = malloc(strlen(filename) + 1);

	strcpy(fbuf, filename);
	cp = strrchr(fbuf, '.');

	if (mode == NULL)
		mode = "r";

	switch (mode[0]) {
	case 'a':
		if (stat(_NGResolvePath(fbuf), &buf) < 0) {
			if (cp)
				*cp = '\0';
		}
		if (stat(_NGResolvePath(fbuf), &buf) < 0)
			crw = -1;
		else
			crw = 0;
		break;
	case 'c':
		crw = -1;
		break;
	case 'r':
		if (strlen(mode) > 1 && (mode[1] == '+' || mode[1] == 'w')) {
			if (stat(_NGResolvePath(fbuf), &buf) < 0) {
				if (cp)
					*cp = '\0';
			}
			if (stat(_NGResolvePath(fbuf), &buf) < 0)
				crw = -1;
			else
				crw = 0;
		} else
			crw = 1;
		break;
	case 'w':
		if (stat(_NGResolvePath(fbuf), &buf) < 0) {
			if (cp)
				*cp = '\0';
		}
		if (stat(_NGResolvePath(fbuf), &buf) < 0)
			crw = -1;
		else
			crw = 0;
		break;
	default:
		crw = -2;
	}
	free(fbuf);
	return crw;
}

/* Create file object */

NioFileObject *
NioFile_Open(py3_char *filename, py3_char *mode) {
	NioFileObject *self = PyObject_New(NioFileObject, (PyTypeObject*) get_NioFileType());
	NclFile file = NULL;
	int crw;
	char *name;

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *	           __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */

	/*nio_ncerr = 0;*/

	if (self == NULL)
		return NULL;
	self->groups = NULL;
	self->dimensions = NULL;
	self->chunk_dimensions = NULL;
	self->variables = NULL;
	self->attributes = NULL;
	self->ud_types = NULL;
	self->name = NULL;
	self->mode = NULL;
	self->type = char_to_pystr("file", strlen("file"));
	self->full_path = NULL;
	self->parent = NULL;
	self->top = NULL;
	self->open = 0;
	self->being_destroyed = 0;
	self->id = NULL;
	self->gnode = NULL;
	self->weakreflist = NULL;

	crw = GetNioMode(filename, mode);
	file = _NclOpenFile(NULL, NULL, Ncl_File, 0, TEMPORARY,
			NrmStringToQuark(filename), crw);
	if (file) {
		self->id = (void *) file;
		self->define = 1;
		self->open = 1;
		self->write = (crw != 1);
		name = strrchr(filename, '/');
		if (name && strlen(name) > 1) {
			self->name = char_to_pystr(name + 1, strlen(name)-1);
			self->full_path = char_to_pystr(filename, strlen(filename));
		} else {
			self->name = char_to_pystr(filename, strlen(filename));
			self->full_path = char_to_pystr(filename, strlen(filename));
		}
		self->mode = char_to_pystr(mode, strlen(mode));
		nio_file_init(self);
	} else {
		NioFileObject_dealloc(self);
		PyErr_SetString(get_NioError(), "Unable to open file");
		return NULL;
	}

	return self;
}

/* Create variables from file */

static NclFileVarNode* getVarFromGroup(NclFileGrpNode* grpnode, NrmQuark vname) {
	NclFileGrpRecord* grprec;
	NclFileVarRecord* varrec;
	NclFileVarNode* varnode = NULL;
	int i;

	varrec = grpnode->var_rec;
	if (varrec != NULL) {
		for (i = 0; i < varrec->n_vars; ++i) {
			varnode = &(varrec->var_node[i]);
			if ((vname == varnode->name) || (vname == varnode->real_name))
				return varnode;
		}
	}

	grprec = grpnode->grp_rec;
	if (grprec != NULL) {
		for (i = 0; i < grprec->n_grps; ++i) {
			varnode = getVarFromGroup(grprec->grp_node[i], vname);
			if (varnode != NULL)
				return varnode;
		}
	}

	return NULL;
}

/* Get user-define node from file */

static NclFileUDTNode* getUDTfromGroup(NclFileGrpNode* grpnode, NrmQuark name) {
	NclFileGrpRecord* grprec;
	NclFileUDTRecord* udtrec;
	NclFileUDTNode* udtnode = NULL;
	int i;

	udtrec = grpnode->udt_rec;
	if (NULL != udtrec) {
		for (i = 0; i < udtrec->n_udts; ++i) {
			udtnode = &(udtrec->udt_node[i]);
			if (name == udtnode->name)
				return udtnode;
		}
	}

	grprec = grpnode->grp_rec;
	if (NULL != grprec) {
		for (i = 0; i < grprec->n_grps; ++i) {
			udtnode = getUDTfromGroup(grprec->grp_node[i], name);
			if (NULL != udtnode)
				return udtnode;
		}
	}

	return NULL;
}

static int dimNvarInfofromGroup(NioFileObject *self, NclFileGrpNode* grpnode,
		int* ndims, int* nvars, int* ngrps, int* ngattrs) {
	NclFileDimRecord* dimrec;
	NclFileDimNode* dimnode;

	NclFileVarRecord* varrec;
	NclFileVarNode* varnode;

	NclFileGrpRecord* grprec;

	char* name;
	ng_size_t size;
	PyObject* size_ob;
	PyObject* str;

	NioVariableObject *variable;
	NioFileObject *group;

	int i;
	py3_char *path;

	if (NULL == grpnode)
		return 0;

	if (NULL != grpnode->att_rec)
		*ngattrs += grpnode->att_rec->n_atts;

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *                  __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */

	path = as_utf8_char(self->full_path);
	dimrec = grpnode->dim_rec;

	if (NULL != dimrec) {
		*ndims += dimrec->n_dims;
		for (i = 0; i < dimrec->n_dims; ++i) {
			dimnode = &(dimrec->dim_node[i]);
			/*if(dimnode->is_unlimited)*/
			if ((dimnode->is_unlimited) && (0 > self->recdim))
				if (dimnode->is_unlimited)
					self->recdim = i;

			name = NrmQuarkToString(dimnode->name);
			size = dimnode->size;
#if PY_MAJOR_VERSION < 3
			size_ob = PyInt_FromSize_t(size);
#else
			size_ob = PyLong_FromSize_t(size);
#endif
			DICT_SETITEMSTRING(self->dimensions, name, size_ob);
			if (!strcmp(path, "/")) {
				/*str = PyString_FromFormat("%s", name);*/
				str = pystr_to_utf8(PyUnicode_FromFormat("%s", name));
				PyDict_SetItem(self->top->dimensions, str,
						(PyObject *) size_ob);
			} else {
				/*str = PyString_FromFormat("%s/%s", path, name);*/
				str = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name));
				PyDict_SetItem(self->top->dimensions, str,
						(PyObject *) size_ob);
			}
			Py_DECREF(str);
			Py_DECREF(size_ob);

			/*
			 *fprintf(stderr, "\tin file: %s, line: %d\n", __FILE__, __LINE__);
			 *fprintf(stderr, "\tDim %d: name: <%s>, size: %ld\n", i, name, size);
			 */
		}
	}

	dimrec = grpnode->chunk_dim_rec;

	if (NULL != dimrec) {
		for (i = 0; i < dimrec->n_dims; ++i) {
			dimnode = &(dimrec->dim_node[i]);

			name = NrmQuarkToString(dimnode->name);
			size = dimnode->size;
#if PY_MAJOR_VERSION < 3
			size_ob = PyInt_FromSize_t(size);
#else
			size_ob = PyLong_FromSize_t(size);
#endif
			DICT_SETITEMSTRING(self->chunk_dimensions, name, size_ob);
			Py_DECREF(size_ob);

			/*
			 *fprintf(stderr, "\tin file: %s, line: %d\n", __FILE__, __LINE__);
			 *fprintf(stderr, "\tDim %d: name: <%s>, size: %ld\n", i, name, size);
			 */
		}
	}

	varrec = grpnode->var_rec;
	if (NULL != varrec) {
		*nvars += varrec->n_vars;
		for (i = 0; i < varrec->n_vars; ++i) {
			varnode = &(varrec->var_node[i]);
			name = NrmQuarkToString(varnode->name);

			/*
			 *fprintf(stderr, "\tin file: %s, line: %d\n", __FILE__, __LINE__);
			 *fprintf(stderr, "\tVar %d: name: <%s>\n", i, name);
			 */

			variable = nio_read_advanced_variable(self, varnode, i);
			DICT_SETITEMSTRING(self->variables, name, (PyObject * )variable);
			if (!strcmp(path, "/") || strlen(path) == 0) {
				/*str = PyString_FromFormat("%s", name);*/
				str = pystr_to_utf8(PyUnicode_FromFormat("%s", name));

				PyDict_SetItem(self->top->variables, str,
						(PyObject *) variable);
			} else {
				/*str = PyString_FromFormat("%s/%s", path, name);*/
				str = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name));
				PyDict_SetItem(self->top->variables, str,
						(PyObject *) variable);
			}
			Py_DECREF(str);
			Py_DECREF(variable);
		}
	}

	grprec = grpnode->grp_rec;
	if (NULL != grprec) {
		*ngrps += grprec->n_grps;
		for (i = 0; i < grprec->n_grps; ++i) {
			name = NrmQuarkToString(grprec->grp_node[i]->name);

			/*
			 *fprintf(stderr, "\tin file: %s, line: %d\n", __FILE__, __LINE__);
			 *fprintf(stderr, "\tGrp %d: name: <%s>\n", i, name);
			 */

			group = nio_read_group(self, grprec->grp_node[i]);
			DICT_SETITEMSTRING((PyObject * )self->groups, name,
					(PyObject * )group);
			if (!strcmp(path, "/") || strlen(path) == 0) {
				/*str = PyString_FromFormat("%s", name);*/
				str = pystr_to_utf8(PyUnicode_FromFormat("%s",name));
				PyDict_SetItem(self->top->groups, str, (PyObject *) group);
			} else {
				/*str = PyString_FromFormat("%s/%s", path, name);*/
				str = pystr_to_utf8(PyUnicode_FromFormat("%s/%s",path,name));
				PyDict_SetItem(self->top->groups, str, (PyObject *) group);
			}
			Py_DECREF(str);
			Py_DECREF(group);

			/*dimNvarInfofromGroup(self, grprec->grp_node[i],
			 ndims, nvars, ngrps, ngattrs);*/
		}
	}

	free (path);

	return 0;
}

static void collect_advancedfile_attributes(NioFileObject *self,
		NclFileAttRecord* attrec, PyObject *attributes) {
	NclFileAttNode* attnode;
	char *name;
	npy_intp length;
	int py_type;
	int i;
	py3_char *path = NULL;
	PyObject *attname;

	if (NULL == attrec)
		return;

	if (self) {
		path = as_utf8_char(self->full_path);
	}

	for (i = 0; i < attrec->n_atts; ++i) {
		attnode = &(attrec->att_node[i]);
		name = NrmQuarkToString(attnode->name);
		length = (npy_intp) attnode->n_elem;
		py_type = data_type(attnode->type);

		/* the following is temporary until these types are actually supported */
		if (py_type == NPY_REFERENCE || py_type == NPY_COMPOUND
				|| py_type == NPY_ENUM || py_type == NPY_VLEN) {
			PyObject *astring = NULL;

			switch (py_type) {
			case NPY_REFERENCE:
				/*astring = PyString_FromFormat("ref_type [%d]", length);*/
				astring = pystr_to_utf8(PyUnicode_FromFormat("ref_type [%d]", length));
				break;
			case NPY_COMPOUND:
				/*astring = PyString_FromFormat("compound_type [%d]", length);*/
				astring = pystr_to_utf8(PyUnicode_FromFormat("compound_type [%d]", length));
				break;
			case NPY_ENUM:
				/*astring = PyString_FromFormat("enum_type [%d]", length);*/
				astring = pystr_to_utf8(PyUnicode_FromFormat("enum_type [%d]", length));
				break;
			case NPY_VLEN:
				/*astring = PyString_FromFormat("vlen_type [%d]", length);*/
				astring = pystr_to_utf8(PyUnicode_FromFormat("vlen_type [%d]", length));
				break;
			}
			if (astring != NULL) {
				DICT_SETITEMSTRING(attributes, name, astring);
				if (path) {
					if (!strcmp(path, "/")) {
						/*attname = PyString_FromFormat("%s", name);*/
						attname = pystr_to_utf8(PyUnicode_FromFormat("%s", name));
						PyDict_SetItem(self->top->attributes, attname, astring);
					} else {
						/*attname = PyString_FromFormat("%s/%s", path, name);*/
						attname = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", length));
						PyDict_SetItem(self->top->attributes, attname, astring);
					}
					Py_DECREF(attname);
				}
				Py_DECREF(astring);
			}
		} else if (attnode->type == NCL_string) {
			if (attnode->n_elem > 1) {
				PyObject *array = NULL;
				int j;
				int maxlen = 0;
				npy_intp n_elem = attnode->n_elem;
				for (j = 0; j < n_elem; j++) {
					int tlen = strlen(
							NrmQuarkToString(((NrmQuark *) attnode->value)[j]));
					maxlen = maxlen < tlen ? tlen : maxlen;
				}
				array = (PyObject *) PyArray_New(&PyArray_Type, 1, &n_elem,
						data_type(attnode->type), NULL, NULL, maxlen, 0, NULL);
				if (array) {
					PyObject *pystr;
					PyArrayObject *pyarray = (PyArrayObject *) array;
					for (j = 0; j < n_elem; j++) {
						py3_char *valstr = NrmQuarkToString(((NrmQuark*) attnode->value)[j]);
						pystr = char_to_pystr(valstr, strlen(valstr));
						PyArray_SETITEM(pyarray,
								PyArray_BYTES(pyarray) + j * PyArray_ITEMSIZE(pyarray),
								pystr);
						Py_DECREF(pystr);
					}
					DICT_SETITEMSTRING(attributes, name, array);
					if (path) {
						if (!strcmp(path, "/")) {
							/*attname = PyString_FromFormat("%s", name);*/
							attname = pystr_to_utf8(PyUnicode_FromFormat("%s", name));
							PyDict_SetItem(self->top->attributes, attname,
									array);
						} else {
							/*attname = PyString_FromFormat("%s/%s", path, name);*/
							attname = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name));
							PyDict_SetItem(self->top->attributes, attname,
									array);
						}
						Py_DECREF(attname);
					}
				}
			} else {
				PyObject *astring;
				char *satt = NrmQuarkToString(*((NrmQuark *) attnode->value));
				if (satt != NULL)
					astring = char_to_pystr(satt, strlen(satt));
				else
					astring = char_to_pystr("", strlen(""));
				if (astring != NULL) {
					DICT_SETITEMSTRING(attributes, name, astring);
					if (path) {
						if (!strcmp(path, "/")) {
							/*attname = PyString_FromFormat("%s", name);*/
							attname = pystr_to_utf8(PyUnicode_FromFormat("%s", name));
							PyDict_SetItem(self->top->attributes, attname,
									astring);
						} else {
							/*attname = PyString_FromFormat("%s/%s", path, name);*/
							attname = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name));
							PyDict_SetItem(self->top->attributes, attname,
									astring);
						}
						Py_DECREF(attname);
					}
				}
				Py_DECREF(astring);
			}
		} else {
			PyObject *array = NULL;
			if (attnode->value == NULL) {
				array = PyArray_SimpleNew(1, &length, NPY_LONG);
				if (array != NULL) {
					memset(PyArray_BYTES((PyArrayObject *) array), 0,
							(size_t) sizeof(long) * length);
					array = PyArray_Return((PyArrayObject *) array);
				}
			} else {
				array = PyArray_SimpleNew(1, &length, py_type);
				if (array != NULL) {
					memcpy(PyArray_BYTES((PyArrayObject *) array), attnode->value,
							(size_t) length * _NclSizeOf(attnode->type));
				}
				array = PyArray_Return((PyArrayObject *) array);
			}
			if (array != NULL) {
				DICT_SETITEMSTRING(attributes, name, array);
				if (path) {
					if (!strcmp(path, "/")) {
						/*attname = PyString_FromFormat("%s", name);*/
						attname = pystr_to_utf8(PyUnicode_FromFormat("%s", name));
						PyDict_SetItem(self->top->attributes, attname, array);
					} else {
						/*attname = PyString_FromFormat("%s/%s", path, name);*/
						attname = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name));
						PyDict_SetItem(self->top->attributes, attname, array);
					}
					Py_DECREF(attname);
				}
				Py_DECREF(array);
			}
		}

	}
	free(path);
}

static int nio_file_init(NioFileObject *self) {
	NclFile file = (NclFile) self->id;
	int ndims, nvars, ngattrs;
	int i, j;
	int scalar_dim_ix = -1;
	NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
	self->dimensions = PyDict_New();
	self->chunk_dimensions = PyDict_New();
	self->variables = PyDict_New();
	self->ud_types = PyDict_New();
	self->groups = PyDict_New();
	self->attributes = PyDict_New();

	self->recdim = -1; /* for now */
	if (file->file.advanced_file_structure) {
		NclAdvancedFile advfile = (NclAdvancedFile) self->id;
		char *name = NrmQuarkToString(advfile->advancedfile.grpnode->name);
		NioFileObject *group;

		ndims = 0;
		nvars = 0;
		ngattrs = 0;

		self->gnode = advfile;
		self->parent = NULL;
		self->top = self;
		Py_INCREF(self->top);
		group = nio_read_group(self,
				((NclAdvancedFile) self->gnode)->advancedfile.grpnode);
		group->id = self->id;
		group->gnode = self->gnode;
		DICT_SETITEMSTRING((PyObject * )self->groups, name, (PyObject * )group);
		Py_DECREF(group);
		/*dimNvarInfofromGroup(self, advfile->advancedfile.grpnode,
		 &ndims, &nvars, &ngrps, &ngattrs);*/
		/*collect_advancedfile_attributes(self,advfile->advancedfile.grpnode->att_rec,
		 self->attributes);*/
	} else {
		ndims = file->file.n_file_dims;
		nvars = file->file.n_vars;
		ngattrs = file->file.n_file_atts;

		for (i = 0; i < ndims; i++) {
			char *name;
			ng_size_t size;
			PyObject *size_ob;
			NclFDimRec *fdim = file->file.file_dim_info[i];
			if (fdim->is_unlimited) {
				self->recdim = i;
			}
			if (fdim->dim_name_quark != scalar_dim) {
				name = NrmQuarkToString(fdim->dim_name_quark);
				size = fdim->dim_size;
#if PY_MAJOR_VERSION < 3
				size_ob = PyInt_FromSize_t(size);
#else
				size_ob = PyLong_FromSize_t(size);
#endif
				DICT_SETITEMSTRING(self->dimensions, name, size_ob);
				Py_DECREF(size_ob);
			} else {
				scalar_dim_ix = i;
			}
		}
		for (i = 0; i < nvars; i++) {
			char *name;
			NclBasicDataTypes datatype;
			NclFVarRec *fvar;
			NclFileAttInfoList *att_list;
			int ndimensions, nattrs;
			NioVariableObject *variable;
			NrmQuark *qdims;

			fvar = file->file.var_info[i];
			ndimensions = fvar->num_dimensions;
			datatype = fvar->data_type;
			nattrs = 0;
			att_list = file->file.var_att_info[i];
			while (att_list != NULL) {
				nattrs++;
				att_list = att_list->next;
			}
			name = NrmQuarkToString(fvar->var_name_quark);
			if (ndimensions == 1 && fvar->file_dim_num[0] == scalar_dim_ix) {
				ndimensions = 0;
				qdims = NULL;
			} else if (ndimensions > 0) {
				qdims = (NrmQuark *) malloc(ndimensions * sizeof(NrmQuark));

				if (qdims == NULL) {
					PyErr_NoMemory();
					return 0;
				}
				for (j = 0; j < ndimensions; j++) {
					qdims[j] =
							file->file.file_dim_info[fvar->file_dim_num[j]]->dim_name_quark;
				}
			} else
				qdims = NULL;
			variable = nio_variable_new(self, name, i, data_type(datatype),
					ndimensions, qdims, nattrs);
			DICT_SETITEMSTRING(self->variables, name, (PyObject * )variable);
			Py_DECREF(variable);
		}

		collect_attributes(self->id, NC_GLOBAL, self->attributes, ngattrs);
	}

	return 1;
}

/* Create dimension */

int NioFile_CreateDimension(NioFileObject *file, char *name, Py_ssize_t size) {
	PyObject *size_ob;
	NrmQuark qname;
	if (check_if_open(file, 1)) {
		NclFile nfile = (NclFile) file->id;
		NhlErrorTypes ret;

		if (PyDict_GetItemString(file->dimensions, name)) {
			printf("dimension (%s) exists: cannot create\n", name);
			return 0;
		}
		if (size == 0 && file->recdim != -1) {
			if (!nfile->file.advanced_file_structure) {
				PyErr_SetString(get_NioError(),
						"there is already an unlimited dimension");
				return -1;
			}
		}
		define_mode(file, 1);
		qname = NrmStringToQuark(name);
		if (nfile->file.advanced_file_structure) {
			ret = _NclFileAddDim(file->gnode, qname, (ng_size_t) size,
					(size == 0 ? 1 : 0));
			if (ret > NhlWARNING) {
				NioFileObject *pgroup =
						file->parent ?
								file :
								(NioFileObject *) PyDict_GetItemString(
										file->groups, "/");

				py3_char *path = as_utf8_char(pgroup->full_path);
				PyObject *dimpath =
						(!strcmp(path, "/") || strlen(path) == 0) ?
								/*PyString_FromFormat("%s", name) :*/
								pystr_to_utf8(PyUnicode_FromFormat("%s", name)) :
								/*PyString_FromFormat("%s/%s", path, name);*/
								pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name));

				if (size == 0) {
					DICT_SETITEMSTRING(pgroup->dimensions, name, Py_None);
					PyDict_SetItem(pgroup->top->dimensions, dimpath, Py_None);
					file->recdim = _NclFileIsDim(nfile, qname);
				} else {
#if PY_MAJOR_VERSION < 3
					size_ob = PyInt_FromSsize_t(size);
#else
					size_ob = PyLong_FromSsize_t(size);
#endif
					DICT_SETITEMSTRING(pgroup->dimensions, name, size_ob);
					PyDict_SetItem(pgroup->top->dimensions, dimpath, size_ob);
					Py_DECREF(size_ob);
				}
				free(path);
			}
		} else {
			ret = _NclFileAddDim(nfile, qname, (ng_size_t) size,
					(size == 0 ? 1 : 0));
			if (ret > NhlWARNING) {
				if (size == 0) {
					DICT_SETITEMSTRING(file->dimensions, name, Py_None);
					file->recdim = _NclFileIsDim(nfile, qname);
				} else {
#if PY_MAJOR_VERSION < 3
					size_ob = PyInt_FromSsize_t(size);
#else
					size_ob = PyLong_FromSsize_t(size);
#endif
					DICT_SETITEMSTRING(file->dimensions, name, size_ob);
					Py_DECREF(size_ob);
				}
			}
		}
		return 0;
	} else
		return -1;
}

static PyObject *
NioFileObject_new_dimension(NioFileObject *self, PyObject *args) {
	char *name;
	PyObject *size_ob;
	Py_ssize_t size;
	if (!PyArg_ParseTuple(args, "sO", &name, &size_ob))
		return NULL;
	if (size_ob == Py_None)
		size = 0;
#if PY_MAJOR_VERSION < 3
	else if (PyInt_Check(size_ob))
		size = (Py_ssize_t) PyInt_AsSsize_t(size_ob);
#endif
	else if (PyLong_Check(size_ob))
		size = (Py_ssize_t) PyLong_AsSsize_t(size_ob);
	else {
		PyErr_SetString(PyExc_TypeError, "size must be None or integer");
		return NULL;
	}
	if (NioFile_CreateDimension(self, name, (ng_size_t) size) == 0) {
		Py_INCREF(Py_None);
		return Py_None;
	} else
		return NULL;
}

/* Create chunk dimension */

int NioFile_CreateChunkDimension(NioFileObject *file, char *name,
		Py_ssize_t size) {
	PyObject *size_ob;
	NrmQuark qname;

	NclFile gnode = (NclFile) file->gnode;
	NhlErrorTypes ret;

	if (!check_if_open(file, 1))
		return -1;

	if (PyDict_GetItemString(file->chunk_dimensions, name)) {
		printf("chunk_dimension (%s) exists: cannot create\n", name);
		return 0;
	}

	define_mode(file, 1);
	qname = NrmStringToQuark(name);
	ret = _NclFileAddChunkDim(gnode, qname, (ng_size_t) size,
			(size == 0 ? 1 : 0));
	if (ret > NhlWARNING) {
		NioFileObject *pgroup =
				file->parent ?
						file :
						(NioFileObject *) PyDict_GetItemString(file->groups,
								"/");

		py3_char *path = as_utf8_char(pgroup->full_path);
		PyObject *dimpath =
				(!strcmp(path, "/") || strlen(path) == 0) ?
						/*PyString_FromFormat("%s", name) : */
						pystr_to_utf8(PyUnicode_FromFormat("%s", name)) :
						/*PyString_FromFormat("%s/%s", path, name);*/
						pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name));

#if PY_MAJOR_VERSION < 3
		size_ob = PyInt_FromSsize_t(size);
#else
		size_ob = PyLong_FromSsize_t(size);
#endif
		DICT_SETITEMSTRING(pgroup->chunk_dimensions, name, size_ob);
		PyDict_SetItem(pgroup->top->chunk_dimensions, dimpath, size_ob);
		Py_DECREF(size_ob);
		/*Py_DECREF(path); Was this was a bug? */
		Py_DECREF(dimpath);
		free(path);
	}

	return 0;
}

static PyObject *NioFileObject_new_chunk_dimension(NioFileObject *self,
		PyObject *args) {
	char *name;
	PyObject *size_ob;
	Py_ssize_t size;

	if (!PyArg_ParseTuple(args, "sO", &name, &size_ob))
		return NULL;

	if (size_ob == Py_None)
		size = 1;
#if PY_MAJOR_VERSION < 3
	else if (PyInt_Check(size_ob))
		size = (Py_ssize_t) PyInt_AsSsize_t(size_ob);
#endif
	else if (PyLong_Check(size_ob))
		size = (Py_ssize_t) PyLong_AsSsize_t(size_ob);
	else {
		PyErr_SetString(PyExc_TypeError, "size must be None or integer");
		return NULL;
	}

	if (1 > size)
		size = 1;

	if (NioFile_CreateChunkDimension(self, name, (ng_size_t) size) == 0) {
		Py_INCREF(Py_None);
		return Py_None;
	} else
		return NULL;
}

/* Create group */

static NioFileObject* nio_create_group(NioFileObject* niofileobj,
		NrmQuark qname) {
	NioFileObject *self;
	NclFile nclfile = (NclFile) niofileobj->id;
	NclAdvancedFile advfilegroup = NULL;
	NhlErrorTypes ret = -1;
	py3_char *path_buf;
	char* name;
	char* buf;
	Py_ssize_t len;


	name = NrmQuarkToString(qname);

	if (!check_if_open(niofileobj, -1))
		return NULL;

	self = PyObject_New(NioFileObject, (PyTypeObject*)get_NioFileType());
	if (self == NULL)
		return NULL;

	self->dimensions = PyDict_New();
	self->chunk_dimensions = PyDict_New();
	self->ud_types = PyDict_New();
	self->variables = PyDict_New();
	self->groups = PyDict_New();
	self->attributes = PyDict_New();
	self->recdim = -1; /* for now */
	self->being_destroyed = 0;

	self->open = niofileobj->open;
	self->write = niofileobj->write;
	self->define = niofileobj->define;
	self->name = char_to_pystr(name, strlen(name));
	self->mode = niofileobj->mode;
	self->type = char_to_pystr("group", strlen("group"));
	self->parent = niofileobj;
	Py_INCREF(self->parent);

	path_buf = as_utf8_char_and_size(niofileobj->full_path, &len);
	/*len = PyString_Size(niofileobj->full_path);*/
	buf = malloc(len + 1);
	strcpy(buf, path_buf);
	/*strcpy(buf,PyUnicode_AsUTF8(niofileobj->full_path));*/
	if (!strcmp(buf, "/")) {
		self->full_path = char_to_pystr(name, strlen(name));
	} else {
		self->full_path = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path_buf, name));
	}
	free(path_buf);
	free(buf);
	/*printf("path is %s\n",PyUnicode_AsUTF8(self->full_path));*/

	/*
	 *fprintf(stderr, "\nfunc: %s, in file: %s, line: %d\n",
	 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */

	ret = _NclFileAddGrp(niofileobj->gnode, qname);
	if (NhlNOERROR != ret) {
		sprintf(get_errbuf(), "Can not add group (%s) to file",
				NrmQuarkToString(qname));
		PyErr_SetString(get_NioError(), get_errbuf());
	}

	advfilegroup = _NclAdvancedGroupCreate(NULL, NULL, Ncl_File, 0, TEMPORARY,
			niofileobj->gnode, qname);

	self->id = (void *) nclfile;
	self->gnode = (void *) advfilegroup;
	self->top = niofileobj->top;
	Py_INCREF(self->top);

	return self;
}

static NioFileObject* NioFileObject_new_group(NioFileObject *self,
		PyObject *args) {
	NioFileObject *group, *pgroup;
	NrmQuark qname;
	char *name;
	py3_char *path;
	NclFile nfile = (NclFile) self->id;

	if (!check_if_open(self, 1))
		return NULL;

	/*printf("in NioFileObject_new_group\n");*/
	if (!PyArg_ParseTuple(args, "s", &name)) {
		PyErr_SetString(get_NioError(), "invalid argument to create_group method");
		return NULL;
	}

	if (!nfile->file.advanced_file_structure) {
		PyErr_SetString(get_NioError(),
				"invalid operation: file format does not support groups");
		return NULL;
	}

	define_mode(self, 1);

	group = (NioFileObject *) PyDict_GetItemString((PyObject *) self->groups,
			name);
	if (group) {
		printf("group (%s) exists: cannot create\n", name);
		return group;
	}

	qname = NrmStringToQuark(name);
	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *         __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 *fprintf(stderr, "\tnio_create_group(self, %s)\n", name);
	 */

	if (self->parent) {
		group = nio_create_group(self, qname);
		pgroup = self;
	} else {/* top file level object */
		pgroup = (NioFileObject *) PyDict_GetItemString(self->groups, "/");
		if (pgroup)
			group = nio_create_group(pgroup, qname);
	}

	if (group) {

		path = as_utf8_char(pgroup->full_path);
		DICT_SETITEMSTRING((PyObject * )pgroup->groups, name,
				(PyObject * )group);
		if (!strcmp(path, "/") || strlen(path) == 0) {
			PyDict_SetItem(pgroup->top->groups,
					/*PyString_FromFormat("%s", name),*/
					pystr_to_utf8(PyUnicode_FromFormat("%s", name)),
					(PyObject *) group);
		} else {
			PyDict_SetItem(pgroup->top->groups,
					/*PyString_FromFormat("%s/%s", path, name),*/
					pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name)),
					(PyObject *) group);
		}

		free(path);

		return group;
	} else
		return NULL;
}

/* Return a variable object referring to an existing variable */

/* Create ud_type */

static NioVariableObject*
nio_create_advancedfile_ud_type(NioFileObject *file, char *name) {
	NioVariableObject *self;
	NclAdvancedFile advfile = (NclAdvancedFile) file->id;
	NclFileGrpNode* grpnode = advfile->advancedfile.grpnode;
	NclFileUDTNode* udtnode;
	char cname[1024];
	NrmQuark qvar;

	if (!check_if_open(file, -1))
		return NULL;

	self = PyObject_New(NioVariableObject, (PyTypeObject*) get_NioVariableType());
	if (self == NULL)
		return NULL;

	self->file = file;
	Py_INCREF(file);

	strcpy(cname, name);
	strcat(cname, "_compound_type");
	qvar = NrmStringToQuark(cname);
	udtnode = getUDTfromGroup(grpnode, qvar);
	self->id = udtnode->id;
	self->nd = 1;
	self->type = data_type(udtnode->type);
	self->qdims = NULL;
	self->unlimited = 0;
	self->dimensions = (Py_ssize_t *) malloc(sizeof(Py_ssize_t));
	self->dimensions[0] = 1;

	self->name = (char *) malloc(strlen(name) + 1);
	if (self->name != NULL)
		strcpy(self->name, name);

	return self;
}

/* Create variable */

static NioVariableObject*
nio_create_advancedfile_variable(NioFileObject *file, char *name, int id,
		int ndims, NrmQuark *qdims) {
	NioVariableObject *self;
	NclFileGrpNode* grpnode =
			((NclAdvancedFile) file->gnode)->advancedfile.grpnode;
	NclFileVarNode* varnode;
	NrmQuark qvar = NrmStringToQuark(name);
	int i, j;

	if (!check_if_open(file, -1))
		return NULL;

	self = PyObject_New(NioVariableObject, (PyTypeObject*) get_NioVariableType());
	if (self == NULL)
		return NULL;

	self->file = file;
	Py_INCREF(file);

	varnode = getVarFromGroup(grpnode, qvar);
	self->id = id;
	self->nd = ndims;
	self->type = data_type(varnode->type);
	self->qdims = qdims;
	self->unlimited = 0;
	self->dimensions = NULL;
	/*
	 *self->qchunkdims = NULL;
	 *self->chunk_dimensions = NULL;
	 */

	if (ndims > 0) {
		NclFileDimRecord* dimrec = grpnode->dim_rec;
		NclFileDimNode* dimnode;

		self->dimensions = (Py_ssize_t *) malloc(ndims * sizeof(Py_ssize_t));
		if ((self->dimensions != NULL) && (NULL != dimrec)) {
			int dimid = -1;
			for (i = 0; i < ndims; ++i) {
				dimid = -1;
				for (j = 0; j < dimrec->n_dims; ++j) {
					dimnode = &(dimrec->dim_node[j]);
					if (dimnode->name == qdims[i]) {
						dimid = j;
						self->dimensions[i] = (Py_ssize_t) dimnode->size;
						if (dimnode->is_unlimited)
							self->unlimited = 1;
						break;
					}
				}

				if (0 > dimid) {
					sprintf(get_errbuf(), "Dimension (%s) not found",
							NrmQuarkToString(qdims[i]));
					PyErr_SetString(get_NioError(), get_errbuf());
					return NULL;
				}
			}
		}
	}
	self->name = (char *) malloc(strlen(name) + 1);
	if (self->name != NULL)
		strcpy(self->name, name);
	self->attributes = PyDict_New();
	collect_advancedfile_attributes(NULL, varnode->att_rec, self->attributes);
	return self;
}

NioVariableObject *
NioFile_CreateVariable(NioFileObject *file, char *name, int typecode,
		char **dimension_names, int ndim) {

	if (check_if_open(file, 1)) {
		NioVariableObject *variable;
		int i, id;
		NclFile nfile = (NclFile) file->id;
		NrmQuark *qdims = NULL;
		int dimid;
		NhlErrorTypes ret;
		NrmQuark qvar;
		NrmQuark qtype;
		int ncl_ndims = ndim;
		char *path;
		NioFileObject *pgroup=NULL;

		define_mode(file, 1);

		variable = (NioVariableObject *) PyDict_GetItemString(file->variables,
				name);
		if (variable) {
			printf("variable (%s) exists: cannot create\n", name);
			return variable;
		}

		if (ndim > 0) {
			qdims = (NrmQuark *) malloc(ndim * sizeof(NrmQuark));
			if (!qdims) {
				return (NioVariableObject *) PyErr_NoMemory();
			}
		} else if (ndim == 0) {
			qdims = (NrmQuark *) malloc(sizeof(NrmQuark));
			dimid = -1;
			if (!qdims) {
				return (NioVariableObject *) PyErr_NoMemory();
			}
			*qdims = NrmStringToQuark("ncl_scalar");
			ncl_ndims = 1;
		}
		qtype = nio_type_from_code(typecode);
		qvar = NrmStringToQuark(name);
		if (nfile->file.advanced_file_structure) {
			for (i = 0; i < ndim; i++) {
				qdims[i] = NrmStringToQuark(dimension_names[i]);
				dimid = _NclFileIsDim(file->gnode, qdims[i]);
				if (dimid == -1) {
					sprintf(get_errbuf(), "Dimension (%s) not found",
							dimension_names[i]);
					PyErr_SetString(get_NioError(), get_errbuf());
					if (qdims != NULL)
						free(qdims);
					return NULL;
				}
			}
			if (file->parent) {
				pgroup = file;
			} else {/* top file level object */
				pgroup = (NioFileObject *) PyDict_GetItemString(file->groups,
						"/");
			}
			variable = (NioVariableObject *) PyDict_GetItemString(
					pgroup->variables, name);
			if (variable) {
				printf("variable (%s) exists: cannot create\n", name);
				return variable;
			}
			ret = _NclFileAddVar(pgroup->gnode, qvar, qtype, ncl_ndims, qdims);
			if (ret > NhlWARNING) {
				id = _NclFileIsVar(pgroup->gnode, qvar);
				variable = nio_create_advancedfile_variable(pgroup, name, id,
						ndim, qdims);
			} else {
				sprintf(get_errbuf(), "Error creating variable (%s)", name);
				PyErr_SetString(get_NioError(), get_errbuf());
				if (qdims != NULL)
					free(qdims);
				return NULL;
			}
		} else {
#if 1
			/*
			 *fprintf(stderr, "\nfunc: %s, in file: %s, line: %d\n",
			 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
			 *fprintf(stderr, "\tsizeof(long) = %d\n", sizeof(long));
			 */

			if (sizeof(long) > 4 && qtype == NrmStringToQuark("long")) {
				qtype = NrmStringToQuark("integer");
			}
#endif
			for (i = 0; i < ndim; i++) {
				qdims[i] = NrmStringToQuark(dimension_names[i]);
				dimid = _NclFileIsDim(nfile, qdims[i]);
				if (dimid == -1) {
					sprintf(get_errbuf(), "Dimension (%s) not found",
							dimension_names[i]);
					PyErr_SetString(get_NioError(), get_errbuf());
					if (qdims != NULL)
						free(qdims);
					return NULL;
				}
			}
			ret = _NclFileAddVar(nfile, qvar, qtype, ncl_ndims, qdims);
			if (ret > NhlWARNING) {
				id = _NclFileIsVar(nfile, qvar);
				variable = nio_variable_new(file, name, id,
						data_type(nfile->file.var_info[id]->data_type), ndim,
						qdims, 0);
			} else {
				sprintf(get_errbuf(), "Error creating variable (%s)", name);
				PyErr_SetString(get_NioError(), get_errbuf());
				if (qdims != NULL)
					free(qdims);
				return NULL;
			}
		}

		DICT_SETITEMSTRING(file->variables, name, (PyObject * )variable);

		if (nfile->file.advanced_file_structure) {

			path = as_utf8_char(pgroup->full_path);
			if (!strcmp(path, "/") || strlen(path) == 0) {
				PyDict_SetItem(pgroup->top->variables,
						/*PyString_FromFormat("%s", name), (PyObject *) variable);*/
						pystr_to_utf8(PyUnicode_FromFormat("%s", name)),
						(PyObject *) variable);
			} else {
				PyDict_SetItem(pgroup->top->variables,
						/*PyString_FromFormat("%s/%s", path, name),*/
						pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name)),
						(PyObject *) variable);
			}
			free(path);
		}
		return variable;
	} else {
		return NULL;
	}
}

static PyObject *
NioFileObject_new_variable(NioFileObject *self, PyObject *args) {
	NioVariableObject *var;
	py3_char **dimension_names;
	PyObject *item, *dim;
	NclFile nfile = (NclFile) self->id;
	char *name;
	int ndim;
	char* type;
	int i;
	char errbuf[256];
	char ltype;


	if (!PyArg_ParseTuple(args, "ssO!", &name, &type, &PyTuple_Type, &dim))
		return NULL;

	ltype = type[0];
	if (strlen(type) > 1) {
		if (nfile->file.advanced_file_structure) {
			if (type[0] == 'S' && type[1] == '1') {
				ltype = 'S';
			} else {
				sprintf(errbuf,
						"Cannot create variable (%s): string arrays not yet supported on write",
						name);
				PyErr_SetString(PyExc_TypeError, errbuf);
				return NULL;
			}
		} else {
			if (type[0] == 'S' && type[1] == '1') {
				ltype = 'c';
			} else {
				sprintf(errbuf,
						"Cannot create variable (%s): string arrays not yet supported on write",
						name);
				PyErr_SetString(PyExc_TypeError, errbuf);
				return NULL;
			}
		}
	}
	ndim = PyTuple_Size(dim);
	if (ndim == 0)
		dimension_names = NULL;
	else {
		dimension_names = (py3_char **) malloc(ndim * sizeof(py3_char *));
		if (dimension_names == NULL) {
			PyErr_SetString(PyExc_MemoryError, "out of memory");
			return NULL;
		}
	}
	for (i = 0; i < ndim; i++) {
		item = PyTuple_GetItem(dim, i);
		if (is_string_type(item)) {
			char *item_char = as_utf8_char(item);
			dimension_names[i] = item_char;
		} else {
			PyErr_SetString(PyExc_TypeError, "dimension name must be a string");
			free(dimension_names);
			return NULL;
		}
	}
	var = NioFile_CreateVariable(self, name, (int) ltype, dimension_names,
			ndim);
	if (!var) {
		sprintf(get_errbuf(), "Failed to create variable (%s)", name);
		PyErr_SetString(get_NioError(), get_errbuf());
	}

	for (i=0; i < ndim; i++) {
		free(dimension_names[i]);
	}

	if (dimension_names)
		free(dimension_names);
	return (PyObject *) var;
}

NioVariableObject *NioFile_CreateVLEN(NioFileObject *file, char *name,
		int typecode, char **dimension_names, int ndim) {
	NioVariableObject *variable;
	int i;
	NclFile nfile = (NclFile) file->gnode;
	NrmQuark *qdims = NULL;
	NhlErrorTypes ret;
	NrmQuark qvar;
	NrmQuark qtype;
	NioFileObject *pgroup;

	if (!check_if_open(file, 1))
		return NULL;

	define_mode(file, 1);

	if (file->parent) {
		pgroup = file;
	} else {/* top file level object */
		pgroup = (NioFileObject *) PyDict_GetItemString(file->groups, "/");
	}

	variable = (NioVariableObject *) PyDict_GetItemString(pgroup->variables,
			name);
	if (variable) {
		printf("variable (%s) exists: cannot create\n", name);
		return variable;
	}

	if (ndim > 0) {
		qdims = (NrmQuark *) malloc(ndim * sizeof(NrmQuark));
		if (!qdims)
			return (NioVariableObject *) PyErr_NoMemory();
	} else if (ndim == 0) {
		qdims = (NrmQuark *) malloc(sizeof(NrmQuark));
		if (!qdims)
			return (NioVariableObject *) PyErr_NoMemory();

		*qdims = NrmStringToQuark("ncl_scalar");
	}

	for (i = 0; i < ndim; ++i)
		qdims[i] = NrmStringToQuark(dimension_names[i]);

	qtype = nio_type_from_code(typecode);
	qvar = NrmStringToQuark(name);
	ret = _NclFileAddVlen(nfile, NrmStringToQuark("vlen"), qvar, qtype, qdims,
			ndim);
	if (ret > NhlWARNING) {

		py3_char *path;

		variable = nio_create_advancedfile_variable(file, name, 0, ndim, qdims);
		path = as_utf8_char(pgroup->full_path);
		if (!strcmp(path, "/") || strlen(path) == 0) {
			PyDict_SetItem(pgroup->top->variables,
					/*PyString_FromFormat("%s", name), */
					pystr_to_utf8(PyUnicode_FromFormat("%s", name)),
					(PyObject *) variable);
		} else {
			PyDict_SetItem(pgroup->top->variables,
					/*PyString_FromFormat("%s/%s", path, name),*/
					pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name)),
					(PyObject *) variable);
		}

		DICT_SETITEMSTRING(pgroup->variables, name, (PyObject * )variable);

		free(path);

		return variable;
	} else {
		sprintf(get_errbuf(), "Error creating variable (%s)", name);
		PyErr_SetString(get_NioError(), get_errbuf());
		if (qdims != NULL)
			free(qdims);
		return NULL;
	}
}

static PyObject *NioFileObject_new_vlen(NioFileObject *self, PyObject *args) {
	NioVariableObject *var;
	py3_char **dimension_names;
	PyObject *item, *dim;
	char *name;
	int ndim;
	char* type;
	int i;
	char errbuf[256];
	char ltype;

	NclFile nfile = (NclFile) self->id;

	if (!check_if_open(self, 1))
		return NULL;

	if (!PyArg_ParseTuple(args, "ssO!", &name, &type, &PyTuple_Type, &dim)) {
		PyErr_SetString(get_NioError(), "invalid argument to create_vlen method");
		return NULL;
	}
	if (!nfile->file.advanced_file_structure) {
		PyErr_SetString(get_NioError(),
				"invalid operation: file format does not support variable length arrays");
		return NULL;
	}

	ltype = type[0];
	if (strlen(type) > 1) {
		if (type[0] == 'S' && type[1] == '1')
			ltype = 'c';
		else {
			sprintf(errbuf,
					"Cannot create vlen (%s): string arrays not yet supported on write",
					name);
			PyErr_SetString(PyExc_TypeError, errbuf);
			return NULL;
		}
	}
	ndim = PyTuple_Size(dim);
	if (ndim == 0)
		dimension_names = NULL;
	else {
		dimension_names = (py3_char **) malloc(ndim * sizeof(py3_char *));
		if (dimension_names == NULL) {
			PyErr_SetString(PyExc_MemoryError, "out of memory");
			return NULL;
		}
	}

	for (i = 0; i < ndim; ++i) {
		item = PyTuple_GetItem(dim, i);
		if (is_string_type(item)) {
			py3_char *item_char = as_utf8_char(item);
			dimension_names[i] = item_char;
		}
		else {
			PyErr_SetString(PyExc_TypeError, "dimension name must be a string");
			free(dimension_names);
			return NULL;
		}
	}

	var = NioFile_CreateVLEN(self, name, (int) ltype, dimension_names, ndim);
	if (!var) {
		sprintf(get_errbuf(), "Failed to create vlen (%s)", name);
		PyErr_SetString(get_NioError(), get_errbuf());
	}

	for (i=0; i < ndim; i++) {
		free (dimension_names[i]);
	}

	if (dimension_names)
		free(dimension_names);
	return (PyObject *) var;
}

NioVariableObject *NioFile_CreateCOMPOUNDtype(NioFileObject *file, char *name,
		char **memb_names, int *memb_types, int *memb_sizes, int nmemb) {
	NioVariableObject *variable;
	int i;
	NclFile nfile = (NclFile) file->id;
	NhlErrorTypes ret;
	char cname[1024];
	NrmQuark qvar;
	NrmQuark* memqname = NULL;
	NrmQuark* memqtype = NULL;

	if (!check_if_open(file, 1))
		return NULL;

	define_mode(file, 1);

	variable = (NioVariableObject *) PyDict_GetItemString(file->ud_types, name);
	if (variable) {
		printf("ud_types (%s) exists: cannot create\n", name);
		return variable;
	}

	if (nmemb > 0) {
		memqname = (NrmQuark *) malloc(nmemb * sizeof(NrmQuark));
		if (!memqname)
			return (NioVariableObject *) PyErr_NoMemory();
		memqtype = (NrmQuark *) malloc(nmemb * sizeof(NrmQuark));
		if (!memqtype)
			return (NioVariableObject *) PyErr_NoMemory();
	} else {
		fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
		fprintf(stderr, "\tnmemb = %d\n", nmemb);
		fprintf(stderr, "\tnumber of compound components must great than 0.\n");
		return NULL;
	}

	for (i = 0; i < nmemb; ++i) {
		memqname[i] = NrmStringToQuark(memb_names[i]);
		memqtype[i] = nio_type_from_code(memb_types[i]);
	}

	qvar = NrmStringToQuark(name);
	strcpy(cname, name);
	strcat(cname, "_compound_type");
	ret = _NclFileAddCompound(nfile, NrmStringToQuark(cname), qvar, 0, NULL,
			nmemb, memqname, memqtype, memb_sizes);

	free(memqname);
	free(memqtype);

	if (ret > NhlWARNING) {
		variable = nio_create_advancedfile_ud_type(file, name);

		DICT_SETITEMSTRING(file->ud_types, name, (PyObject * )variable);
		return variable;
	} else {
		sprintf(get_errbuf(), "Error creating ud_type (%s)", name);
		PyErr_SetString(get_NioError(), get_errbuf());
		return NULL;
	}
}

NioVariableObject *NioFile_CreateCOMPOUND(NioFileObject *file, char *name,
		char **dimension_names, int ndim, char **memb_names, int *memb_types,
		int *memb_sizes, int nmemb) {
	NioVariableObject *variable;
	int i;
	NrmQuark *qdims = NULL;
	NhlErrorTypes ret;
	char cname[1024];
	NrmQuark qvar;
	NrmQuark* memqname = NULL;
	NrmQuark* memqtype = NULL;
	NioFileObject *pgroup;

	if (!check_if_open(file, 1))
		return NULL;

	define_mode(file, 1);

	if (file->parent) {
		pgroup = file;
	} else {/* top file level object */
		pgroup = (NioFileObject *) PyDict_GetItemString(file->groups, "/");
	}

	variable = (NioVariableObject *) PyDict_GetItemString(pgroup->variables,
			name);
	if (variable) {
		printf("variable (%s) exists: cannot create\n", name);
		return variable;
	}

	if (ndim > 0) {
		qdims = (NrmQuark *) malloc(ndim * sizeof(NrmQuark));
		if (!qdims)
			return (NioVariableObject *) PyErr_NoMemory();
	} else if (ndim == 0) {
		qdims = (NrmQuark *) malloc(sizeof(NrmQuark));
		if (!qdims)
			return (NioVariableObject *) PyErr_NoMemory();

		*qdims = NrmStringToQuark("ncl_scalar");
	}

	for (i = 0; i < ndim; ++i)
		qdims[i] = NrmStringToQuark(dimension_names[i]);

	if (nmemb > 0) {
		memqname = (NrmQuark *) malloc(nmemb * sizeof(NrmQuark));
		if (!memqname)
			return (NioVariableObject *) PyErr_NoMemory();
		memqtype = (NrmQuark *) malloc(nmemb * sizeof(NrmQuark));
		if (!memqtype)
			return (NioVariableObject *) PyErr_NoMemory();
	} else {
		fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
		fprintf(stderr, "\tnmemb = %d\n", nmemb);
		fprintf(stderr, "\tnumber of compound components must great than 0.\n");
		return NULL;
	}

	for (i = 0; i < nmemb; ++i) {
		memqname[i] = NrmStringToQuark(memb_names[i]);
		memqtype[i] = nio_type_from_code(memb_types[i]);
	}

	qvar = NrmStringToQuark(name);
	strcpy(cname, name);
	strcat(cname, "_compound_type");
	ret = _NclFileAddCompound(pgroup->gnode, NrmStringToQuark(cname), qvar,
			ndim, qdims, nmemb, memqname, memqtype, memb_sizes);

	free(memqname);
	free(memqtype);

	if (ret > NhlWARNING) {

		py3_char *path;
		variable = nio_create_advancedfile_variable(pgroup, name, 0, ndim,
				qdims);
		path = as_utf8_char(pgroup->full_path);
		if (!strcmp(path, "/") || strlen(path) == 0) {
			PyDict_SetItem(pgroup->top->variables,
					/*PyString_FromFormat("%s", name), */
					pystr_to_utf8(PyUnicode_FromFormat("%s", name)),
					(PyObject *) variable);
		} else {
			PyDict_SetItem(pgroup->top->variables,
					/*PyString_FromFormat("%s/%s", path, name),*/
					pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path, name)),
					(PyObject *) variable);
		}

		DICT_SETITEMSTRING(pgroup->variables, name, (PyObject * )variable);

		free(path);

		return variable;
	} else {
		sprintf(get_errbuf(), "Error creating variable (%s)", name);
		PyErr_SetString(get_NioError(), get_errbuf());
		if (qdims != NULL)
			free(qdims);
		return NULL;
	}
}

static PyObject *NioFileObject_new_compound_type(NioFileObject *self,
		PyObject *args) {
	NioVariableObject *var;
	py3_char **memb_names;
	int *memb_types;
	int *memb_sizes;
	PyObject* seq;
	PyObject* item;
	PyObject* type;
	PyObject* seq2;
	PyObject* item2;
	char *name;
	int nmemb, nitem;
	int i;
	char errbuf[256];
	char ltype;
	py3_char *typestr;

	NclFile nfile = (NclFile) self->id;

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */
	if (!check_if_open(self, 1))
		return NULL;

	if (!PyArg_ParseTuple(args, "sO", &name, &type)) {
		PyErr_SetString(get_NioError(),
				"invalid argument to create_compound_type method");
		return NULL;
	}

	if (!nfile->file.advanced_file_structure) {
		PyErr_SetString(get_NioError(),
				"invalid operation: file format does not support user-defined data types");
		return NULL;
	}

	nmemb = PySequence_Size(type);

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 *fprintf(stderr, "\tnmemb = %d\n", nmemb);
	 */

	if (nmemb == 0) {
		memb_names = NULL;
		memb_types = NULL;
		memb_sizes = NULL;
	} else {
		memb_names = (py3_char **) malloc(nmemb * sizeof(py3_char *));
		if (memb_names == NULL) {
			PyErr_SetString(PyExc_MemoryError,
					"out of memory to define compound member name");
			return NULL;
		}
		memb_types = (int *) malloc(nmemb * sizeof(int));
		if (memb_types == NULL) {
			PyErr_SetString(PyExc_MemoryError,
					"out of memory to define compound member type");
			return NULL;
		}
		memb_sizes = (int *) malloc(nmemb * sizeof(int));
		if (memb_sizes == NULL) {
			PyErr_SetString(PyExc_MemoryError,
					"out of memory to define compound member size");
			return NULL;
		}
	}

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */

	seq = PySequence_Fast(type, "expected a sequence");
	for (i = 0; i < nmemb; ++i) {
		item = PySequence_Fast_GET_ITEM(seq, i);
		nitem = PySequence_Size(item);
		/*
		 *fprintf(stderr, "\tItem %d has %d elements\n", i, nitem);
		 */

		seq2 = PySequence_Fast(item, "expected a sequence");

		item2 = PySequence_Fast_GET_ITEM(seq2, 0);
		if (is_string_type(item2)) {
			py3_char* item2_char = as_utf8_char(item2);
			memb_names[i] = item2_char;
			/*
			 *fprintf(stderr, "\tmemb_names[%d] = <%s>\n", i, memb_names[i]);
			 */
		} else {
			PyErr_SetString(PyExc_TypeError, "memb name must be a string");
			free(memb_names);
			return NULL;
		}

		item2 = PySequence_Fast_GET_ITEM(seq2, 1);
		if (is_string_type(item2)) {
			typestr = as_utf8_char(item2);
			ltype = typestr[0];
			if (strlen(typestr) > 1) {
				if (typestr[0] == 'S' && typestr[1] == '1')
					ltype = 'c';
				else {
					sprintf(errbuf,
							"Cannot create compound (%s): string arrays not yet supported on write",
							name);
					PyErr_SetString(PyExc_TypeError, errbuf);
					free(typestr);
					return NULL;
				}
			}
			memb_types[i] = ltype;

			free(typestr);

			/*
			 *fprintf(stderr, "\tmemb_types[%d] = <%c>\n", i, ltype);
			 */
		} else {
			PyErr_SetString(PyExc_TypeError, "memb type must be a string");
			free(memb_names);
			return NULL;
		}

		memb_sizes[i] = 1;
		if (3 <= nitem) {
			item2 = PySequence_Fast_GET_ITEM(seq2, 2);
			if (PyLong_Check(item2)) {
				memb_sizes[i] = (int) PyLong_AsLong(item2);
			}
#if PY_MAJOR_VERSION < 3
			else if (PyInt_Check(item2)) {
				memb_sizes[i] = (int) PyInt_AsLong(item2);
			}
#endif
			else if (is_string_type(item2)) {
				typestr = as_utf8_char(item2);
				sscanf(typestr, "%d", &(memb_sizes[i]));
				free(typestr);

				/*
				 *fprintf(stderr, "\tmemb_size[%d] = <%s>\n", i, typestr);
				 *fprintf(stderr, "\tmemb_size[%d] = %d\n", i, memb_sizes[i]);
				 */
			} else {
				PyErr_SetString(PyExc_TypeError, "memb size must be an integer");
				free(memb_names);
				return NULL;
			}
		}
		/*
		 *fprintf(stderr, "\tmemb_sizes[%d] = %d\n", i, memb_sizes[i]);
		 */
	}

	var = NioFile_CreateCOMPOUNDtype(self, name, memb_names, memb_types,
			memb_sizes, nmemb);
	if (!var) {
		sprintf(errbuf, "Failed to create compound (%s)", name);
		PyErr_SetString(get_NioError(), errbuf);
	}

	for (i = 0; i < nmemb; ++i) {
		free (memb_names[i]);
	}

	if (memb_names)
		free(memb_names);
	if (memb_types)
		free(memb_types);
	if (memb_sizes)
		free(memb_sizes);
	return (PyObject *) var;
}

static PyObject *NioFileObject_new_compound(NioFileObject *self, PyObject *args) {
	NioVariableObject *var;
	py3_char **dimension_names;
	py3_char **memb_names;
	int *memb_types;
	int *memb_sizes;
	PyObject* seq;
	PyObject* item;
	PyObject* dim;
	PyObject* type;
	PyObject* seq2;
	PyObject* item2;
	char *name;
	int ndim, nmemb, nitem;
	int i;
	char errbuf[256];
	char ltype;
	py3_char *typestr;

	NclFile nfile = (NclFile) self->id;

	if (!check_if_open(self, 1))
		return NULL;

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */

	if (!PyArg_ParseTuple(args, "sOO!", &name, &type, &PyTuple_Type, &dim)) {
		PyErr_SetString(get_NioError(), "invalid argument to create_compound method");
		return NULL;
	}
	if (!nfile->file.advanced_file_structure) {
		PyErr_SetString(get_NioError(),
				"invalid operation: file format does not support compound data types");
		return NULL;
	}

	nmemb = PySequence_Size(type);

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 *fprintf(stderr, "\tnmemb = %d\n", nmemb);
	 */

	if (nmemb == 0) {
		memb_names = NULL;
		memb_types = NULL;
		memb_sizes = NULL;
	} else {
		memb_names = (py3_char **) malloc(nmemb * sizeof(py3_char *));
		if (memb_names == NULL) {
			PyErr_SetString(PyExc_MemoryError,
					"out of memory to define compound member name");
			return NULL;
		}
		memb_types = (int *) malloc(nmemb * sizeof(int));
		if (memb_types == NULL) {
			PyErr_SetString(PyExc_MemoryError,
					"out of memory to define compound member type");
			return NULL;
		}
		memb_sizes = (int *) malloc(nmemb * sizeof(int));
		if (memb_sizes == NULL) {
			PyErr_SetString(PyExc_MemoryError,
					"out of memory to define compound member size");
			return NULL;
		}
	}

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */

	seq = PySequence_Fast(type, "expected a sequence");
	for (i = 0; i < nmemb; ++i) {
		item = PySequence_Fast_GET_ITEM(seq, i);
		nitem = PySequence_Size(item);
		/*
		 *fprintf(stderr, "\tItem %d has %d elements\n", i, nitem);
		 */

		seq2 = PySequence_Fast(item, "expected a sequence");

		item2 = PySequence_Fast_GET_ITEM(seq2, 0);
		if (is_string_type(item2)) {
			py3_char *item2_char = as_utf8_char(item2);
			memb_names[i] = item2_char;
			/*
			 *fprintf(stderr, "\tmemb_names[%d] = <%s>\n", i, memb_names[i]);
			 */
		} else {
			PyErr_SetString(PyExc_TypeError, "memb name must be a string");
			free(memb_names);
			return NULL;
		}
		/*FIXME: Does this need to change?? */
		item2 = PySequence_Fast_GET_ITEM(seq2, 1);
		if (is_string_type(item2)) {
			typestr = as_utf8_char(item2);
			ltype = typestr[0];
			if (strlen(typestr) > 1) {
				if (typestr[0] == 'S' && typestr[1] == '1') {
					ltype = 'c';
				}

				else {
					sprintf(errbuf,
							"Cannot create compound (%s): string arrays not yet supported on write",
							name);
					PyErr_SetString(PyExc_TypeError, errbuf);

					free(typestr);

					return NULL;
				}
			}
			memb_types[i] = ltype;

			free(typestr);

			/*
			 *fprintf(stderr, "\tmemb_types[%d] = <%c>\n", i, ltype);
			 */
		} else {
			PyErr_SetString(PyExc_TypeError, "member type must be a string");
			free(memb_names);
			return NULL;
		}

		memb_sizes[i] = 1;
		if (3 <= nitem) {
			item2 = PySequence_Fast_GET_ITEM(seq2, 2);
			if (PyLong_Check(item2)) {
				memb_sizes[i] = (int) PyLong_AsLong(item2);
			}
#if PY_MAJOR_VERSION < 3
			else if (PyInt_Check(item2)) {
				memb_sizes[i] = (int) PyInt_AsLong(item2);
			}
#endif
			else if (is_string_type(item2)) {
				typestr = as_utf8_char(item2);
				sscanf(typestr, "%d", &(memb_sizes[i]));
				free(typestr);
				/*
				 *fprintf(stderr, "\tmemb_size[%d] = <%s>\n", i, typestr);
				 *fprintf(stderr, "\tmemb_size[%d] = %d\n", i, memb_sizes[i]);
				 */
			} else {
				PyErr_SetString(PyExc_TypeError, "memb size must be an integer");
				free(memb_names);
				return NULL;
			}
		}
		/*
		 *fprintf(stderr, "\tmemb_sizes[%d] = %d\n", i, memb_sizes[i]);
		 */
	}

	ndim = PyTuple_Size(dim);
	if (ndim == 0)
		dimension_names = NULL;
	else {
		dimension_names = (py3_char **) malloc(ndim * sizeof(py3_char *));
		if (dimension_names == NULL) {
			PyErr_SetString(PyExc_MemoryError, "out of memory");
			return NULL;
		}
	}

	for (i = 0; i < ndim; ++i) {
		item = PyTuple_GetItem(dim, i);
		if (is_string_type(item)) {
			dimension_names[i] = as_utf8_char(item);
		}
		else {
			PyErr_SetString(PyExc_TypeError, "dimension name must be a string");
			free(dimension_names);
			return NULL;
		}
	}

	var = NioFile_CreateCOMPOUND(self, name, dimension_names, ndim, memb_names,
			memb_types, memb_sizes, nmemb);
	if (!var) {
		sprintf(errbuf, "Failed to create compound (%s)", name);
		PyErr_SetString(get_NioError(), errbuf);
	}

	for (i = 0; i < nmemb; ++i) {
		free(memb_names[i]);
	}

	for (i = 0; i < ndim; ++i) {
		free(dimension_names[i]);
	}

	if (dimension_names)
		free(dimension_names);
	if (memb_names)
		free(memb_names);
	if (memb_types)
		free(memb_types);
	if (memb_sizes)
		free(memb_sizes);
	return (PyObject *) var;
}

static int advfile_is_unlimited(NioFileObject *self, char *name) {
	NclAdvancedFile adv_file;
	NclFileGrpNode* grpnode;
	NclFileDimRecord* dimrec = NULL;
	NioFileObject *parent = self;
	NrmQuark qstr = NrmStringToQuark(name);

	/* 
	 *  Dimensions can exist in the current group or a higher level group, 
	 *  so loop up to the root group as needed
	 * Note we cannot modify name since it could come from PyUnicode_AsUTF8 or from NrmQuarkToString
	 */

	while (parent) {
		if (parent->parent) {
			adv_file = (NclAdvancedFile) parent->gnode;
			grpnode = adv_file->advancedfile.grpnode;
			dimrec = grpnode->dim_rec;
		} else if (!self->parent) {
			char *cp = NULL;
			char *dimname = NULL;
			char *dimpath = NULL;
			char *alloc_dimpath = NULL;
			NioFileObject *grp = NULL;
			int len;
			cp = strrchr(name, '/');
			if (cp >= name) {
				dimname = cp + 1;
				len = cp - name;
				alloc_dimpath = malloc(strlen(name) + 1);
				strcpy(alloc_dimpath, name);
				dimpath = alloc_dimpath;
				dimpath[len] = '\0';
				if (strlen(dimpath) > 0) {
					dimpath = dimpath[0] == '/' ? &(dimpath[1]) : dimpath;
				}
			} else {
				dimname = name;
				dimpath = "/";
			}
			grp = (NioFileObject *) PyDict_GetItemString(parent->groups,
					dimpath);
			adv_file = (NclAdvancedFile) grp->gnode;
			grpnode = adv_file->advancedfile.grpnode;
			dimrec = grpnode->dim_rec;
			qstr = NrmStringToQuark(dimname);
			if (alloc_dimpath)
				free(alloc_dimpath);
		}

		if (dimrec) {
			NclFileDimNode* dimnode;
			int i;
			for (i = 0; i < dimrec->n_dims; i++) {
				dimnode = &(dimrec->dim_node[i]);
				if (qstr == dimnode->name) {
					return (dimnode->is_unlimited);
				}
			}
		}
		parent = parent->parent;
	}
	return -1;
}

static PyObject *
NioFileObject_Unlimited(NioFileObject *self, PyObject *args) {
	PyObject *obj = NULL;
	int i;
	NrmQuark qstr;

	py3_char *str = NULL;
	NclFile nfile = (NclFile) self->id;

	if (!PyArg_ParseTuple(args, "O", &obj))
		return NULL;

	if (is_string_type(obj)) {
		str = as_utf8_char(obj);
		if (!str) {
			PyErr_SetString(PyExc_MemoryError, "out of memory");
			return NULL;
		}
		qstr = NrmStringToQuark(str);
		if (nfile->file.advanced_file_structure) {
			int result = advfile_is_unlimited(self, str);
			free(str);

			if (result < 0) {
				PyErr_SetString(get_NioError(),
						self->parent == NULL ?
								"dimension name not found in file" :
								"dimension name not found in group hierarchy");

				return NULL;
			}
			return result > 0 ? Py_True : Py_False;
		} else {
			free(str);

			for (i = 0; i < nfile->file.n_file_dims; i++) {
				if (nfile->file.file_dim_info[i]->dim_name_quark != qstr)
					continue;
				return (nfile->file.file_dim_info[i]->is_unlimited == 0 ?
						Py_False : Py_True);

			}
			PyErr_SetString(get_NioError(), "dimension name not found in file");
			return NULL;
		}
	}
	PyErr_SetString(PyExc_TypeError, "Invalid type");
	return NULL;
}

/* Return a variable object referring to an existing variable */

/* Not used */
#if 0
static NioVariableObject *
NioFile_GetVariable(NioFileObject *file, char *name) {
	return (NioVariableObject *) PyDict_GetItemString(file->variables, name);
}
#endif

/* Synchronize output */

#if 0
int
NioFile_Sync(NioFileObject *file)
{
	if (check_if_open(file, 0))
	{
#if 0
		define_mode(file, 0);
		if (ncsync(file->id) == -1)
		{
			nio_seterror(23);
			return -1;
		}
		else
#endif
		return 0;
	}
	else
	return -1;
}

static PyObject *
NioFileObject_sync(NioFileObject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ""))
	return NULL;
	if (NioFile_Sync(self) == 0)
	{
		Py_INCREF(Py_None);
		return Py_None;
	}
	else
	return NULL;
}
#endif

/* Close file */

int NioFile_Close(NioFileObject *file) {
	/* The deallocation routine calls this close function before deletion of all the data structures
	 * On the other hand, calling the close method results in this function calling the deallocation routine --
	 * however, in that case, a minimal amount of data structure is retained because the variable has not gone away.
	 * The only access is through the repr function which will state that the file has been closed.
	 */

	/* do not close groups with this function */
	if (file->parent != NULL || file->open == 0) {
		return 0;
	}
	if (check_if_open(file, 0)) {
		PyObject *keys;
		Py_ssize_t i;

		/* it is necessary to delete the variable dictionary here because the
		 variables contain references to the file object */
		file->open = 0;
		/* DecREF references to the file object */

		keys = PyDict_Keys(file->variables);
		for (i = 0; i < PyList_Size(keys); i++) {
			NioVariableObject *v = (NioVariableObject *) PyDict_GetItem(
					file->variables, PyList_GetItem(keys, i));
			Py_DECREF(v->file);
		}
		Py_DECREF(keys);

		keys = PyDict_Keys(file->groups);
		for (i = 0; i < PyList_Size(keys); i++) {
			NioFileObject *g = (NioFileObject *) PyDict_GetItem(file->groups,
					PyList_GetItem(keys, i));
			Py_XDECREF(g->top);
			Py_XDECREF(g->parent);
		}
		Py_DECREF(keys);

		/* deleting the NIO file structures happens in the deallocation routine */
		NioFileObject_dealloc(file);
		return 0;
	} else {
		return -1;
	}
}

static PyObject *
NioFileObject_close(NioFileObject *self, PyObject *args) {
	char *history = NULL;
	if (!PyArg_ParseTuple(args, "|s", &history))
		return NULL;
	if (history && strlen(history) > 0) {
		if (check_if_open(self, 1)) {
			NioFile_AddHistoryLine(self, history);
		}
	}

	if (NioFile_Close(self) == 0) {
		Py_INCREF(Py_None);
		return Py_None;
	} else
		return NULL;
}

/* Method table */

static PyMethodDef NioFileObject_methods[] = {  { "close",
		(PyCFunction) NioFileObject_close, METH_VARARGS },
/* {"sync", (PyCFunction)NioFileObject_sync, 1}, */

{ "create_dimension", (PyCFunction) NioFileObject_new_dimension, METH_VARARGS },
		{ "create_chunk_dimension",
				(PyCFunction) NioFileObject_new_chunk_dimension, METH_VARARGS },
		{ "create_variable", (PyCFunction) NioFileObject_new_variable,
				METH_VARARGS }, { "create_group",
				(PyCFunction) NioFileObject_new_group, METH_VARARGS }, {
				"create_vlen", (PyCFunction) NioFileObject_new_vlen,
				METH_VARARGS }, { "create_compound_type",
				(PyCFunction) NioFileObject_new_compound_type, METH_VARARGS }, {
				"create_compound", (PyCFunction) NioFileObject_new_compound,
				METH_VARARGS }, { "unlimited",
				(PyCFunction) NioFileObject_Unlimited, METH_VARARGS }, { NULL,
				NULL } /* sentinel */
};

static PyMemberDef NioFileObject_members[] = { { "dimensions", T_OBJECT,
		NhlOffset(NioFileObject, dimensions), READONLY, NULL }, { "variables",
		T_OBJECT, NhlOffset(NioFileObject, variables), READONLY, NULL }, {
		"attributes", T_OBJECT, NhlOffset(NioFileObject, attributes), READONLY,
		NULL }, { "groups", T_OBJECT, NhlOffset(NioFileObject, groups),
		READONLY, NULL }, { "name", T_OBJECT, NhlOffset(NioFileObject, name),
		READONLY, NULL }, { "path", T_OBJECT, NhlOffset(NioFileObject,
		full_path), READONLY, NULL }, { NULL, 0, 0, 0, NULL } /* sentinel */
};

/* Attribute access */

PyObject *
NioFile_GetAttribute(NioFileObject *self, char *name) {
	PyObject *value;
	if (check_if_open(self, -1)) {
		if (strcmp(name, "dimensions") == 0) {
			Py_INCREF(self->dimensions);
			return self->dimensions;
		}
		if (strcmp(name, "chunk_dimensions") == 0) {
			Py_INCREF(self->chunk_dimensions);
			return self->chunk_dimensions;
		}
		if (strcmp(name, "ud_types") == 0) {
			Py_INCREF(self->ud_types);
			return self->ud_types;
		}
		if (strcmp(name, "variables") == 0) {
			Py_INCREF(self->variables);
			return self->variables;
		}
		if (strcmp(name, "groups") == 0) {
			Py_INCREF(self->groups);
			return (PyObject *) self->groups;
		}
		if (strcmp(name, "attributes") == 0) {
			Py_INCREF(self->attributes);
			return self->attributes;
		}
		if (strcmp(name, "__dict__") == 0) {
			Py_INCREF(self->attributes);
			return self->attributes;
		}
		if (strcmp(name, "path") == 0) {
			Py_INCREF(self->full_path);
			return (PyObject *) self->full_path;
		}
		if (strcmp(name, "name") == 0) {
			Py_INCREF(self->name);
			return (PyObject *) self->name;
		}
		value = PyDict_GetItemString(self->attributes, name);
		if (value != NULL) {
			Py_INCREF(value);
			return value;
		} else {
			PyErr_Clear();
#if PY_MAJOR_VERSION < 3
			return Py_FindMethod(NioFileObject_methods, (PyObject *) self, name);
#else
			PyObject *name_obj = char_to_pystr(name, strlen(name));
			PyObject *result =  PyObject_GenericGetAttr((PyObject *) self, name_obj);
			Py_DECREF(name_obj);
			return result;
#endif
		}
	} else
		return NULL;
}

static NclMultiDValData createAttMD(NclFile nfile, PyObject *attributes,
		char *name, PyObject *value) {
	NclMultiDValData md = NULL;
	PyArrayObject *array = NULL;

	if (!value || value == Py_None) {
		return md;
	}

	if (is_string_type(value)) {
		ng_size_t len_dims = 1;

		py3_char *value_char = as_utf8_char(value);
		NrmQuark *qval = malloc(sizeof(NrmQuark));
		qval[0] = NrmStringToQuark(value_char);
		md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0, (void*) qval,
				NULL, 1, &len_dims, TEMPORARY, NULL,
				(NclTypeClass) nclTypestringClass);
		free(value_char);
	} else {
		ng_size_t dim_sizes = 1;
		int n_dims;
		NrmQuark qtype;
		int pyarray_type = NPY_NOTYPE;
		PyArrayObject *tmparray = (PyArrayObject *) PyDict_GetItemString(
				attributes, name);
		if (tmparray != NULL)
			pyarray_type = PyArray_TYPE(tmparray);

		array = (PyArrayObject *) PyArray_ContiguousFromAny(value, pyarray_type,
				0, 1);
		if (array) {
			n_dims = (PyArray_NDIM(array) == 0) ? 1 : PyArray_NDIM(array);
			qtype = nio_type_from_code(PyArray_DTYPE(array)->type);
			if (nfile->file.advanced_file_structure && PyArray_ITEMSIZE(array) == 8
					&& qtype == NrmStringToQuark("long")) {
				PyArrayObject *array2 = (PyArrayObject *) PyArray_Cast(array,
						NPY_INT);
				Py_DECREF(array);
				array = array2;
				qtype = NrmStringToQuark("integer");
				sprintf(get_errbuf(),
						"output format does not support 8-byte integers; converting to 4-byte integer variable (%s): possible data loss due to overflow",
						name);
				PyErr_SetString(get_NioError(), get_errbuf());
				PyErr_Print();
			}
			if (NrmStringToQuark("object") == qtype) {
				fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
						__PRETTY_FUNCTION__, __FILE__, __LINE__);
				fprintf(stderr,
						"\tNeed to handle object array, here, probably a compound attribute.\n");
				if (nfile->file.advanced_file_structure) {
					NclAdvancedFile advfile = (NclAdvancedFile) nfile;
					NclFileVarNode* varnode;

					obj *listids = NULL;
					ng_size_t n, n_dims;
					ng_size_t nitems = 1;
					ng_size_t curdim = 0;
					ng_usize_t counter = 0;
					ng_size_t *dims;

					varnode = getVarFromGroup(advfile->advancedfile.grpnode,
							NrmStringToQuark(name));

					if (array) {
						NclObjTypes the_obj_type = Ncl_Typelist;

						n_dims = (ng_size_t) PyArray_NDIM(array);
						dims = (ng_size_t*) malloc(n_dims * sizeof(ng_size_t));
						assert(dims);
						for (n = 0; n < n_dims; ++n) {
							dims[n] = (ng_size_t) PyArray_DIMS(array)[n];
							nitems *= dims[n];
						}

						listids = (obj*) NclMalloc(
								(ng_usize_t) (nitems * sizeof(obj)));
						assert(listids);

						_NclBuildArrayOfList(listids, n_dims, dims);

						_convertObj2COMPOUND((PyObject*) array, listids,
								varnode->comprec, n_dims, curdim, &counter);

						md = _NclCreateVal(NULL, NULL,
								((the_obj_type & NCL_VAL_TYPE_MASK) ?
										Ncl_MultiDValData : the_obj_type), 0,
								listids, NULL, n_dims, dims, TEMPORARY, NULL,
								(NclObjClass) (
										(the_obj_type & NCL_VAL_TYPE_MASK) ?
												_NclTypeEnumToTypeClass(
														the_obj_type) :
												NULL));

						Py_DECREF(array);
					}
				} else {
					fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__,
					__LINE__);
					fprintf(stderr,
							"\tCompound attributes only implemented for advanced file structure.\n");
					return NULL;
				}
			} else if (array) {
				ng_size_t *dims;
				void *data;
				if (PyArray_NDIM(array) == 0) {
					dims = &dim_sizes;
				} else {
					dims = (ng_size_t *) PyArray_DIMS(array);
				}
				data = malloc(PyArray_NBYTES(array));
				memcpy(data, PyArray_DATA(array), PyArray_NBYTES(array));

				md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
						(void*) data, NULL, n_dims, dims, TEMPORARY, NULL,
						_NclNameToTypeClass(qtype));

			}
		}
	}

	return md;
}

static int set_advanced_file_attribute(NioFileObject *file,
		PyObject *attributes, char *name, PyObject *value) {
	NclFile nfile = (NclFile) file->id;
	NhlErrorTypes ret;
	NclMultiDValData md = NULL;

	if (!value || value == Py_None) {
		/* delete attribute */
		ret = _NclFileDeleteAtt(nfile, NrmStringToQuark(name));
		PyObject_DelItemString(attributes, name);
		return 0;
	}

	md = createAttMD(nfile, attributes, name, value);

	if (!md) {
		nio_seterror(23);
		return -1;
	}

	ret = _NclFileWriteAtt(nfile, NrmStringToQuark(name), md, NULL);

	if (ret > NhlFATAL) {
		if (is_string_type(value)) {
			DICT_SETITEMSTRING(attributes, name, value);
		} else {
			PyArrayObject *array = NULL;
			int pyarray_type = NPY_NOTYPE;
			PyArrayObject *tmparray = (PyArrayObject *) PyDict_GetItemString(
					attributes, name);
			if (tmparray != NULL)
				pyarray_type = PyArray_TYPE(tmparray);
			array = (PyArrayObject *) PyArray_ContiguousFromAny(value,
					pyarray_type, 0, 1);
			DICT_SETITEMSTRING(attributes, name, (PyObject * )array);
		}
	}

	return 0;
}

static int set_advanced_variable_attribute(NioFileObject *file,
		NioVariableObject *self, PyObject *attributes, char *name,
		PyObject *value) {
	NclFile nfile = (NclFile) file->id;
	NclFile gnode = file->gnode;
	NhlErrorTypes ret;
	NclMultiDValData md = NULL;

	if (!value || value == Py_None) {
		/* delete attribute */
		ret = _NclFileDeleteVarAtt(gnode, NrmStringToQuark(self->name),
				NrmStringToQuark(name));
		PyObject_DelItemString(attributes, name);
		return 0;
	}

	md = createAttMD(nfile, attributes, name, value);

	if (!md) {
		nio_seterror(23);
		return -1;
	}

	ret = _NclFileWriteVarAtt(gnode, NrmStringToQuark(self->name),
			NrmStringToQuark(name), md, NULL);

	if (ret > NhlFATAL) {
		if (is_string_type(value)) {
			DICT_SETITEMSTRING(attributes, name, value);
		} else {
			PyArrayObject *array = NULL;
			int pyarray_type = NPY_NOTYPE;
			PyArrayObject *tmparray = (PyArrayObject *) PyDict_GetItemString(
					attributes, name);
			if (tmparray != NULL)
				pyarray_type = PyArray_TYPE(tmparray);
			array = (PyArrayObject *) PyArray_ContiguousFromAny(value,
					pyarray_type, 0, 1);
			DICT_SETITEMSTRING(attributes, name, (PyObject * )array);
		}
	}

	return 0;
}

int NioFile_SetAttribute(NioFileObject *self, char *name, PyObject *value) {
	NclFile nfile = (NclFile) self->id;

	/*nio_ncerr = 0;*/
	if (check_if_open(self, 1)) {
		if (strcmp(name, "dimensions") == 0
				|| strcmp(name, "chunk_dimensions") == 0
				|| strcmp(name, "ud_types") == 0
				|| strcmp(name, "variables") == 0 || strcmp(name, "groups") == 0
				|| strcmp(name, "attributes") == 0
				|| strcmp(name, "__dict__") == 0) {
			PyErr_SetString(PyExc_TypeError,
					"attempt to set read-only attribute");
			return -1;
		}
		define_mode(self, 1);
		if (nfile->file.advanced_file_structure) {
			return set_advanced_file_attribute(self, self->attributes, name,
					value);
		} else {
			return set_attribute(self, NC_GLOBAL, self->attributes, name, value);
		}
	} else
		return -1;
}

/* Not actually used */
#if 0
int NioFile_SetAttributeString(NioFileObject *self, char *name, char *value) {
	PyObject *string = char_to_pystr(value, strlen(value));
	if (string != NULL)
		return NioFile_SetAttribute(self, name, string);
	else
		return -1;
}
#endif

int NioFile_AddHistoryLine(NioFileObject *self, char *text) {
	static char *history = "history";
	Py_ssize_t oldlen, newlen;
	py3_char *prev_history;
	char *new_history;
	PyObject *new_string = NULL;
	PyObject *h = NioFile_GetAttribute(self, history);
	int result;

	if (h == NULL) {
		PyErr_Clear();
		oldlen = 0;
		newlen = strlen(text) + 1;
		new_history = (char*) calloc(newlen, sizeof(char));
		strcpy(new_history, text);
	} else {
		/*alloc = PyString_Size(h);
		 old = strlen(PyUnicode_AsUTF8(h));*/

		prev_history = as_utf8_char_and_size(h, &oldlen);

		newlen = oldlen + strlen(text) + 3; /* Need 2 nulls and 1 '\n' */
		new_history = (char*) calloc(newlen, sizeof(char));
		strcpy(new_history, prev_history);
		strcat(new_history, "\n");
		strcat(new_history, text);
		free(prev_history);
	}

	new_string = char_to_pystr(new_history, newlen);
	if (new_string) {
		result = NioFile_SetAttribute(self, history, new_string);
	} else {
		result = -1;
	}

	Py_XDECREF(h);
	Py_XDECREF(new_string);

	return result;


	/*
	new_string = (PyUnicodeObject *) PyUnicode_DecodeUTF8(NULL, new_alloc);
	if (new_string) {
		char *s = new_string->ob_sval;
		int len, ret;
		memset(s, 0, new_alloc + 1);
		if (h == NULL)
			len = -1;
		else {
			strcpy(s, PyUnicode_AsUTF8(h));
			len = strlen(s);
			s[len] = '\n';
		}
		strcpy(s + len + 1, text);
		ret = NioFile_SetAttribute(self, history, (PyObject *) new_string);
		Py_XDECREF(h);
		Py_DECREF(new_string);
		return ret;
	} else
		return -1;
		*/
}

/* Printed representation */
static PyObject *
NioFileObject_repr(NioFileObject *file) {
	char buf[512];

	py3_char *mode_char = as_utf8_char(file->mode);
	py3_char *file_char = as_utf8_char(file->name);

	sprintf(buf, "<%s NioFile object '%.256s', mode '%.10s' at %lx>",
			file->open ? "open" : "closed", file_char,
			mode_char, (long) file);
	free(mode_char);
	free(file_char);
	return char_to_pystr(buf, strlen(buf));
}

#define BUF_INSERT(tbuf) \
	len = strlen(tbuf); \
	while (bufpos + len > buflen - 2) { \
		buf = realloc(buf,buflen + bufinc); \
		buflen += bufinc; \
	} \
	strcpy(&(buf[bufpos]),tbuf); \
	bufpos += len;

void format_object(char *buf, PyObject *obj, int code) {


	PyObject *utf8;
	py3_char *utf8_char;

	switch (code) {
	case 'i':
#if PY_MAJOR_VERSION < 3
		sprintf(buf, "%d", (int) PyInt_AsLong(obj));
#else
		sprintf(buf, "%d", (int) PyLong_AsLong(obj));
#endif
		break;
	case 'l':
		sprintf(buf, "%lld", PyLong_AsLongLong(obj));
		break;
	case 'f':
		sprintf(buf, "%.7g", PyFloat_AsDouble(obj));
		break;
	case 'd':
		sprintf(buf, "%.16g", PyFloat_AsDouble(obj));
		break;
	default:
		utf8 = pystr_to_utf8(PyObject_Str(obj));
		utf8_char = as_utf8_char(utf8);

		sprintf(buf, "%s", utf8_char);
		free(utf8_char);
	}
}

static void insert2buf(char* tbuf, char** buf, int *bufpos, int *buflen,
		int bufinc) {
	int len = strlen(tbuf);

	while ((*bufpos + len) > (*buflen - 2)) {
		*buflen += bufinc;
		*buf = realloc(*buf, *buflen);
	}

	strcpy(&((*buf)[*bufpos]), tbuf);
	*bufpos += len;
}

static void attrec2buf(PyObject *attributes, NclFileAttRecord* attrec,
		char** buf, int *bufpos, int *buflen, int bufinc, char* title,
		char* titlename, char* attname) {
	NclFileAttNode* attnode;
	PyObject *att_val;
	char tbuf[1024];
	char* name;
	int i;

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *    __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */

	if (strlen(title) > 1)
		sprintf(tbuf, "%s:\t%.510s\n", title, titlename);
	else
		sprintf(tbuf, "\t%.510s\n", titlename);
	insert2buf(tbuf, buf, bufpos, buflen, bufinc);

	/*    sprintf(tbuf,"   %s:\n", attname);
	 insert2buf(tbuf, buf, bufpos, buflen, bufinc);*/

	if (NULL == attrec)
		return;

	for (i = 0; i < attrec->n_atts; ++i) {
		attnode = &(attrec->att_node[i]);

		name = NrmQuarkToString(attnode->name);

		sprintf(tbuf, "      %s : ", name);
		insert2buf(tbuf, buf, bufpos, buflen, bufinc);
		att_val = PyDict_GetItemString(attributes, name);

		if (!att_val) {
			continue;
		}
		if (is_string_type(att_val)) {

			py3_char *att_val_char = as_utf8_char(att_val);

			insert2buf(att_val_char, buf, bufpos, buflen, bufinc);
			insert2buf("\n", buf, bufpos, buflen, bufinc);
			free(att_val_char);

		} else {
			int k;
			PyArrayObject *att_arr_val = (PyArrayObject *) att_val;
			if (PyArray_NDIM(att_arr_val) == 0 || PyArray_DIM(att_arr_val, 0) == 1) {
				PyObject *att = PyArray_GETITEM(att_arr_val, PyArray_DATA(att_arr_val));

				format_object(tbuf, att, PyArray_DTYPE(att_arr_val)->type);
				insert2buf(tbuf, buf, bufpos, buflen, bufinc);
				insert2buf("\n", buf, bufpos, buflen, bufinc);
			} else {
				insert2buf("[", buf, bufpos, buflen, bufinc);
				for (k = 0; k < PyArray_DIM(att_arr_val, 0); ++k) {
					PyObject *att = PyArray_GETITEM(att_arr_val, PyArray_BYTES(att_arr_val) + k * PyArray_ITEMSIZE(att_arr_val));
					format_object(tbuf, att, PyArray_DTYPE(att_arr_val)->type);
					insert2buf(tbuf, buf, bufpos, buflen, bufinc);
					if (k < PyArray_DIM(att_arr_val, 0) - 1)
						sprintf(tbuf, ", ");
					else
						sprintf(tbuf, "]\n");
					insert2buf(tbuf, buf, bufpos, buflen, bufinc);
				}
			}
		}
	}
}
char* NioVarInfo2str(NioVariableObject *var, NclFileVarNode* varnode) {
	char *buf[1];
	char *name, *vname;
	char tbuf[1024];
	int bufinc = 8192;
	int buflen = 0;
	int bufpos = 0;
	NioFileObject *file = var->file;
	NclFileAttRecord* attrec;
	NclFileDimRecord* dimrec;
	NclFileDimNode* dimnode;
	NioVariableObject *vobj;
	NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
	int j;

	*buf = malloc(bufinc);
	buflen = bufinc;

	vname = NrmQuarkToString(varnode->name);

	dimrec = varnode->dim_rec;

	if (NULL != dimrec) {
		dimnode = &(dimrec->dim_node[0]);
		if ((1 == dimrec->n_dims) && (dimnode->name == scalar_dim)) {
			sprintf(tbuf, "   %s %s\n", _NclBasicDataTypeToName(varnode->type),
					vname);
			insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
		} else {
			sprintf(tbuf, "   %s %s [ ", _NclBasicDataTypeToName(varnode->type),
					vname);
			insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

			for (j = 0; j < dimrec->n_dims; ++j) {
				dimnode = &(dimrec->dim_node[j]);
				name = NrmQuarkToString(dimnode->name);

				if (j)
					sprintf(tbuf, ", %s|%ld", name, (long) dimnode->size);
				else
					sprintf(tbuf, "%s|%ld", name, (long) dimnode->size);
				insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
			}
			sprintf(tbuf, " ]");
			insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
		}
	}

	vobj = (NioVariableObject *) PyDict_GetItemString(file->variables, vname);

	attrec = varnode->att_rec;
	attrec2buf(vobj->attributes, attrec, buf, &bufpos, &buflen, bufinc, " ",
			" ", vname);
	sprintf(tbuf, "\n");
	insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

	return *buf;
}

char* NioGroupInfo2str(NioFileObject *file, NclFileGrpNode* grpnode,
		char* title, char* attname) {
	char tbuf[1024];
	char titlename[512];
	char *buf[1];
	int bufinc = 8192;
	int buflen = 0;
	int bufpos = 0;
	int i;
	NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");

	char* name;

	NclFileAttRecord* attrec;

	NclFileDimRecord* dimrec;
	NclFileDimNode* dimnode;

	NclFileVarRecord* varrec;
	NclFileVarNode* varnode;

	NclFileGrpRecord* grprec;


	py3_char *name_char = as_utf8_char(file->name);

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *        __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */

	*buf = malloc(bufinc);
	buflen = bufinc;

	sprintf(titlename, "%.510s", name_char);

	free(name_char);

	if (file->parent == NULL) { /* top level file object */
		file = (NioFileObject *) PyDict_GetItemString(file->groups, "/");
	}

	attrec = grpnode->att_rec;
	attrec2buf(file->attributes, attrec, buf, &bufpos, &buflen, bufinc, title,
			titlename, attname);

	sprintf(tbuf, "   dimensions:\n");
	insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

	dimrec = grpnode->dim_rec;

	if (NULL != dimrec) {
		for (i = 0; i < dimrec->n_dims; ++i) {
			dimnode = &(dimrec->dim_node[i]);
			if (dimnode->name == scalar_dim)
				continue;

			name = NrmQuarkToString(dimnode->name);

			if (dimnode->is_unlimited)
				sprintf(tbuf, "      %s = %ld // unlimited\n", name,
						dimnode->size);
			else
				sprintf(tbuf, "      %s = %ld\n", name, dimnode->size);
			insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
		}
	}

	dimrec = grpnode->chunk_dim_rec;

	if (NULL != dimrec) {
		sprintf(tbuf, "   chunk dimensions:\n");
		insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

		for (i = 0; i < dimrec->n_dims; ++i) {
			dimnode = &(dimrec->dim_node[i]);
			if (dimnode->name == scalar_dim)
				continue;

			name = NrmQuarkToString(dimnode->name);

			sprintf(tbuf, "      %s = %ld\n", name, dimnode->size);
			insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
		}
	}

	sprintf(tbuf, "   variables:\n");
	insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

	varrec = grpnode->var_rec;
	if (NULL != varrec) {
		NioVariableObject *vobj;
		char* vname;
		int j;

		for (i = 0; i < varrec->n_vars; ++i) {
			varnode = &(varrec->var_node[i]);
			vname = NrmQuarkToString(varnode->name);

			dimrec = varnode->dim_rec;

			if (NULL != dimrec) {
				dimnode = &(dimrec->dim_node[0]);
				if ((1 == dimrec->n_dims) && (dimnode->name == scalar_dim)) {
					sprintf(tbuf, "   %s %s\n",
							_NclBasicDataTypeToName(varnode->type), vname);
					insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
				} else {
					sprintf(tbuf, "   %s %s [ ",
							_NclBasicDataTypeToName(varnode->type), vname);
					insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

					for (j = 0; j < dimrec->n_dims; ++j) {
						dimnode = &(dimrec->dim_node[j]);

						name = NrmQuarkToString(dimnode->name);

						if (j)
							sprintf(tbuf, ", %s|%ld", name,
									(long) dimnode->size);
						else
							sprintf(tbuf, "%s|%ld", name, (long) dimnode->size);
						insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
					}
					sprintf(tbuf, " ]");
					insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
				}
			}

			vobj = (NioVariableObject *) PyDict_GetItemString(file->variables,
					vname);

			attrec = varnode->att_rec;
#if 1
			attrec2buf(vobj->attributes, attrec, buf, &bufpos, &buflen, bufinc,
					" ", " ", vname);
#else
			sprintf(titlename,"%s", vname);
			attrec2buf(vobj->attributes, attrec, buf, &bufpos, &buflen, bufinc, "Variable", titlename, vname);
#endif
			sprintf(tbuf, "\n");
			insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
		}

	}

	grprec = grpnode->grp_rec;

	if (NULL != grprec) {
		char titlebuf[256];
		char attribuf[256];
		char* grpbuf;

		for (i = 0; i < grprec->n_grps; ++i) {
			NioFileObject *group;
			name = NrmQuarkToString(grprec->grp_node[i]->name);
			sprintf(titlebuf, "Nio group <%s>", name);
			sprintf(attribuf, "group attributes");
			group = (NioFileObject *) PyDict_GetItemString(file->groups, name);

			grpbuf = NioGroupInfo2str(group, grprec->grp_node[i], titlebuf,
					attribuf);

			insert2buf(grpbuf, buf, &bufpos, &buflen, bufinc);
			insert2buf("\n", buf, &bufpos, &buflen, bufinc);
			free(grpbuf);
		}
	}

	return *buf;
}

/* Printed representation */
static PyObject *
NioFileObject_str(NioFileObject *file) {
	char *buf = NULL;
	NclFile nfile = (NclFile) file->id;
	PyObject *pystr;

	if (!check_if_open(file, -1)) {
		PyErr_Clear();
		return NioFileObject_repr(file);
	}

	if (nfile->file.advanced_file_structure) {
		NclFileGrpNode *grpnode =
				((NclAdvancedFile) file->gnode)->advancedfile.grpnode;
		char *name, *attname;
		NioFileObject *group;

		/*
		 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
		 *		__PRETTY_FUNCTION__, __FILE__, __LINE__);
		 */
		if (file->parent == NULL) {
			name = "Nio file";
			attname = "file attributes";
			group = (NioFileObject *) PyDict_GetItemString(file->groups, "/");
			group = file;
		} else {
			name = "Nio group";
			attname = "group attributes";
			group = file;
		}
		buf = NioGroupInfo2str(group, grpnode, name, attname);
	} else {
		char tbuf[1024];
		int len;
		int bufinc = 4096;
		int buflen = 0;
		int bufpos = 0;
		int i;

		NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
		PyObject *att_val;

		py3_char *name_char = as_utf8_char(file->name);

		/*
		 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
		 *		__PRETTY_FUNCTION__, __FILE__, __LINE__);
		 */

		buf = malloc(bufinc);
		buflen = bufinc;
		sprintf(tbuf, "Nio file:\t%.510s\n", name_char);
		BUF_INSERT(tbuf);
		sprintf(tbuf, "   global attributes:\n");
		BUF_INSERT(tbuf);
		free(name_char);

		/* The problem with just printing the Python dictionary is that it
		 * has no consistent ordering. Since we want an order, we will need
		 * to use the nio library's records at least to get the names of atts, dims, and vars.
		 * On the other hand, use Nio python tools to get the attribute values, because this will
		 * format array values in a Pythonic way.
		 */

		for (i = 0; i < nfile->file.n_file_atts; i++) {
			char *attname;
			if (!nfile->file.file_atts[i])
				continue;
			attname = NrmQuarkToString(
					nfile->file.file_atts[i]->att_name_quark);
			sprintf(tbuf, "      %s : ", attname);
			BUF_INSERT(tbuf);
			att_val = PyDict_GetItemString(file->attributes, attname);
			if (is_string_type(att_val)) {

				py3_char *att_val_char = as_utf8_char(att_val);

				BUF_INSERT(att_val_char);
				BUF_INSERT("\n");

				free(att_val_char);

			} else {
				int k;
				PyArrayObject *att_arr_val = (PyArrayObject *) att_val;
				if (PyArray_NDIM(att_arr_val) == 0 || PyArray_DIM(att_arr_val, 0) == 1) {
					PyObject *att = PyArray_GETITEM(att_arr_val, PyArray_DATA(att_arr_val));

					format_object(tbuf, att, PyArray_DTYPE(att_arr_val)->type);
					BUF_INSERT(tbuf);
					BUF_INSERT("\n");
				} else {
					sprintf(tbuf, "[");
					BUF_INSERT(tbuf);
					for (k = 0; k < PyArray_DIM(att_arr_val, 0); k++) {
						PyObject *att = PyArray_GETITEM(att_arr_val, PyArray_BYTES(att_arr_val) + k * PyArray_ITEMSIZE(att_arr_val));
						format_object(tbuf, att, PyArray_DTYPE(att_arr_val)->type);
						BUF_INSERT(tbuf);
						if (k < PyArray_DIM(att_arr_val, 0) - 1) {
							sprintf(tbuf, ", ");
						} else {
							sprintf(tbuf, "]\n");
						}
						BUF_INSERT(tbuf);
					}
				}
			}
		}
		sprintf(tbuf, "   dimensions:\n");
		BUF_INSERT(tbuf);
		for (i = 0; i < nfile->file.n_file_dims; i++) {
			char *dim;
			if (!nfile->file.file_dim_info[i])
				continue;
			if (nfile->file.file_dim_info[i]->dim_name_quark == scalar_dim)
				continue;
			dim = NrmQuarkToString(
					nfile->file.file_dim_info[i]->dim_name_quark);
			if (nfile->file.file_dim_info[i]->is_unlimited) {
				sprintf(tbuf, "      %s = %ld // unlimited\n", dim,
						nfile->file.file_dim_info[i]->dim_size);
			} else {
				sprintf(tbuf, "      %s = %ld\n", dim,
						nfile->file.file_dim_info[i]->dim_size);
			}
			BUF_INSERT(tbuf);
		}

		sprintf(tbuf, "   variables:\n");
		BUF_INSERT(tbuf);

		for (i = 0; i < nfile->file.n_vars; i++) {
			NclFileAttInfoList* step;
			char *vname;
			NioVariableObject *vobj;
			int j, dim_ix;

			if (nfile->file.var_info[i] == NULL) {
				continue;
			}
			vname = NrmQuarkToString(nfile->file.var_info[i]->var_name_quark);
			if (nfile->file.var_info[i]->num_dimensions == 1
					&& nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[0]]->dim_name_quark
							== scalar_dim) {
				sprintf(tbuf, "      %s %s\n",
						_NclBasicDataTypeToName(
								nfile->file.var_info[i]->data_type), vname);
				BUF_INSERT(tbuf);
			} else {
				sprintf(tbuf, "      %s %s [ ",
						_NclBasicDataTypeToName(
								nfile->file.var_info[i]->data_type), vname);
				BUF_INSERT(tbuf);
				for (j = 0; j < nfile->file.var_info[i]->num_dimensions; j++) {
					dim_ix = nfile->file.var_info[i]->file_dim_num[j];
					if (j != nfile->file.var_info[i]->num_dimensions - 1) {
						sprintf(tbuf, "%s, ",
								NrmQuarkToString(
										nfile->file.file_dim_info[dim_ix]->dim_name_quark));
					} else {
						sprintf(tbuf, "%s ]\n",
								NrmQuarkToString(
										nfile->file.file_dim_info[dim_ix]->dim_name_quark));
					}
					BUF_INSERT(tbuf);
				}
			}
			vobj = (NioVariableObject *) PyDict_GetItemString(file->variables,
					vname);
			step = nfile->file.var_att_info[i];
			while (step != NULL) {
				char *aname = NrmQuarkToString(step->the_att->att_name_quark);
				sprintf(tbuf, "         %s :\t", aname);
				BUF_INSERT(tbuf);
				att_val = PyDict_GetItemString(vobj->attributes, aname);
				if (is_string_type(att_val)) {

					py3_char *att_val_char = as_utf8_char(att_val);
					BUF_INSERT(att_val_char);
					BUF_INSERT("\n");
					free(att_val_char);
				} else {
					int k;
					PyArrayObject *att_arr_val = (PyArrayObject *) att_val;
					if (PyArray_NDIM(att_arr_val) == 0
							|| PyArray_DIM(att_arr_val, 0) == 1) {
						PyObject *att = PyArray_GETITEM(att_arr_val, PyArray_DATA(att_arr_val));
						format_object(tbuf, att, PyArray_DTYPE(att_arr_val)->type);
						/*sprintf(tbuf,"%s\n",PyUnicode_AsUTF8(PyObject_Str(att)));*/
						BUF_INSERT(tbuf);
						BUF_INSERT("\n");
					} else {
						sprintf(tbuf, "[");
						BUF_INSERT(tbuf);
						for (k = 0; k < PyArray_DIM(att_arr_val, 0); k++) {
							PyObject *att = PyArray_GETITEM(att_arr_val, PyArray_BYTES(att_arr_val) + k * PyArray_ITEMSIZE(att_arr_val));

							format_object(tbuf, att, PyArray_DTYPE(att_arr_val)->type);
							/*sprintf(tbuf,"%s",PyUnicode_AsUTF8(PyObject_Str(att)));*/
							BUF_INSERT(tbuf);
							if (k < PyArray_DIM(att_arr_val, 0) - 1) {
								sprintf(tbuf, ", ");
							} else {
								sprintf(tbuf, "]\n");
							}
							BUF_INSERT(tbuf);
						}
					}
				}
				step = step->next;
			}
		}
	}

	pystr = char_to_pystr(buf, strlen(buf));
	free(buf);

	return pystr;
}

/* TODO: Implement cyclic garbage collection */
/*
static int
NioFileObject_traverse(NioFileObject *self, visitproc visit, void *arg)
{
	Py_VISIT(self->groups);
	Py_VISIT(self->dimensions);
	Py_VISIT(self->chunk_dimensions);
	Py_VISIT(self->variables);
	Py_VISIT(self->attributes);
	Py_VISIT(self->ud_types);
	Py_VISIT(self->name);
	Py_VISIT(self->mode);
	Py_VISIT(self->type);
	Py_VISIT(self->full_path);
	Py_VISIT((PyObject*)self->parent);
	Py_VISIT((PyObject*)self->top);

    return 0;
}

static int
NioFileObject_clear(Noddy *self)
{
	Py_CLEAR(self->groups);
	Py_CLEAR(self->dimensions);
	Py_CLEAR(self->chunk_dimensions);
	Py_CLEAR(self->variables);
	Py_CLEAR(self->attributes);
	Py_CLEAR(self->ud_types);
	Py_CLEAR(self->name);
	Py_CLEAR(self->mode);
	Py_CLEAR(self->type);
	Py_CLEAR(self->full_path);
	Py_CLEAR((PyObject*)self->parent);
	Py_CLEAR((PyObject*)self->top);

    return 0;
}
*/

/* Type definition */

#if PY_MAJOR_VERSION < 3
static PyTypeObject NioFile_Type =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	"_nio._NioFile", /*tp_name*/
	sizeof(NioFileObject), /*tp_basicsize*/
	0, /*tp_itemsize*/
	/* methods */
	(destructor)NioFileObject_dealloc, /*tp_dealloc*/
	0, /*tp_print*/
	(getattrfunc)NioFile_GetAttribute, /*tp_getattr*/
	(setattrfunc)NioFile_SetAttribute, /*tp_setattr*/
	0, /*tp_compare*/
	(reprfunc)NioFileObject_repr, /*tp_repr*/
	0, /*tp_as_number*/
	0, /*tp_as_sequence*/
	0, /*tp_as_mapping*/
	0, /*tp_hash*/
	0, /*tp_call*/
	(reprfunc)NioFileObject_str, /*tp_str*/
	0, /*tp_getattro*/
	0, /*tp_setattro*/
	0, /*tp_as_buffer*/
	(Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE), /*tp_flags*/
	0, /*tp_doc*/
	(traverseproc)0, /*tp_traverse */
	(inquiry)0, /*tp_clear */
	(richcmpfunc)0, /*tp_richcompare */
	offsetof(NioFileObject,weakreflist), /*tp_weaklistoffset */

	/* Iterator support (use standard) */

	(getiterfunc) 0, /* tp_iter */
	(iternextfunc)0, /* tp_iternext */

	/* Sub-classing (new-style object) support */

	NioFileObject_methods, /* tp_methods */
	NioFileObject_members, /* tp_members */
	0, /* tp_getset */
	0, /* tp_base */
	0, /* tp_dict */
	0, /* tp_descr_get */
	0, /* tp_descr_set */
	0, /* tp_dictoffset */
	(initproc)0, /* tp_init */
	0, /* tp_alloc */
	0, /* tp_new */
	0, /* tp_free */
	0, /* tp_is_gc */
	0, /* tp_bases */
	0, /* tp_mro */
	0, /* tp_cache */
	0, /* tp_subclasses */
	0 /* tp_weaklist */
};
#else
static PyType_Slot niofile_slots[] = {
			{ Py_tp_dealloc, (destructor)NioFileObject_dealloc },
			{ Py_tp_getattr, (getattrfunc)NioFile_GetAttribute },
			{ Py_tp_setattr, (setattrfunc)NioFile_SetAttribute },
			{ Py_tp_repr, (reprfunc)NioFileObject_repr },
			{ Py_tp_str, (reprfunc)NioFileObject_str },
			{ Py_tp_methods, NioFileObject_methods },
			{ Py_tp_members, NioFileObject_members },
			{ 0 },
		};

static PyType_Spec niofile_spec = { "_nio._NioFile", sizeof(NioFileObject), 0,
		  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, niofile_slots };
#endif

/*
 * NIOVariable object
 * (type declaration in niomodule.h)
 */

/* Destroy variable object */

static void NioVariableObject_dealloc(NioVariableObject *self) {

	if (self->qdims != NULL)
		free(self->qdims);
	if (self->dimensions != NULL)
		free(self->dimensions);
	if (self->name != NULL)
		free(self->name);

	Py_XDECREF(self->attributes);
	PyObject_DEL(self);

}

/* Create variable object */

static NioVariableObject* nio_read_advanced_variable(NioFileObject* file,
		NclFileVarNode* varnode, int id) {
	NioVariableObject *self;
	NclFileDimRecord* dimrec = varnode->dim_rec;
	NclFileDimNode* dimnode;
	NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
	NrmQuark* qdims = NULL;
	char* name = NrmQuarkToString(varnode->name);
	int i;

	if (!check_if_open(file, -1))
		return NULL;

	self = PyObject_New(NioVariableObject, (PyTypeObject*) get_NioVariableType());
	if (self == NULL)
		return NULL;

	self->file = file;
	Py_INCREF(file);
	self->id = id;
	self->type = data_type(varnode->type);
	self->nd = 0;
	self->qdims = NULL;
	self->unlimited = 0;
	self->dimensions = NULL;
	self->name = (char *) malloc(strlen(name) + 1);
	if (self->name != NULL)
		strcpy(self->name, name);
	self->attributes = PyDict_New();
	collect_advancedfile_attributes(NULL, varnode->att_rec, self->attributes);

	/* shouldn't this be an error */
	if (NULL == dimrec)
		return self;
	/* --------------- */

	if (dimrec->n_dims == 1 && dimrec->dim_node[0].name == scalar_dim) {
		return self;
	}
	self->dimensions = (Py_ssize_t *) malloc(
			dimrec->n_dims * sizeof(Py_ssize_t));
	if (NULL == self->dimensions) {
		PyErr_NoMemory();
		return self;
	}

	qdims = (NrmQuark *) malloc(dimrec->n_dims * sizeof(NrmQuark));

	if (NULL == qdims) {
		PyErr_NoMemory();
		return self;
	}

	for (i = 0; i < dimrec->n_dims; ++i) {
		dimnode = &(dimrec->dim_node[i]);
		if (dimnode->id < 0) {
			sprintf(get_errbuf(), "Dimension (%s) not found",
					NrmQuarkToString(qdims[i]));
			PyErr_SetString(get_NioError(), get_errbuf());
			return self;
		}
		qdims[i] = dimnode->name;
		self->dimensions[i] = (Py_ssize_t) dimnode->size;
		if (dimnode->is_unlimited)
			self->unlimited = 1;
	}

	self->nd = dimrec->n_dims;
	self->qdims = qdims;

	return self;
}

static NioVariableObject *
nio_variable_new(NioFileObject *file, char *name, int id, int type, int ndims,
		NrmQuark *qdims, int nattrs) {
	NioVariableObject *self;
	NclFile nfile = (NclFile) file->id;
	NclAdvancedFile advfile = (NclAdvancedFile) file->id;
	NclFileVarNode* varnode;
	int i;

	if (check_if_open(file, -1)) {
		self = PyObject_New(NioVariableObject, (PyTypeObject*) get_NioVariableType());
		if (self == NULL)
			return NULL;
		self->file = file;
		Py_INCREF(file);
		self->id = id;
		self->type = type;
		self->nd = ndims;
		self->qdims = qdims;
		self->unlimited = 0;
		self->dimensions = NULL;
		if (ndims > 0) {
			self->dimensions = (Py_ssize_t *) malloc(
					ndims * sizeof(Py_ssize_t));
			if (self->dimensions != NULL) {
				if (nfile->file.advanced_file_structure) {
					NclFileDimRecord* dimrec;
					NclFileDimNode* dimnode;
					varnode = getVarFromGroup(advfile->advancedfile.grpnode,
							NrmStringToQuark(name));
					if (NULL != varnode) {
						dimrec = varnode->dim_rec;
						if (NULL != dimrec) {
							for (i = 0; i < dimrec->n_dims; ++i) {
								dimnode = &(dimrec->dim_node[i]);
								self->dimensions[i] =
										(Py_ssize_t) dimnode->size;
								if (dimnode->is_unlimited)
									self->unlimited = 1;
							}
						}
					}
				} else {
					for (i = 0; i < ndims; i++) {
						int dimid = _NclFileIsDim(nfile, qdims[i]);
						if (dimid < 0) {
							sprintf(get_errbuf(), "Dimension (%s) not found",
									NrmQuarkToString(qdims[i]));
							PyErr_SetString(get_NioError(), get_errbuf());
							return NULL;
						}
						self->dimensions[i] =
								(Py_ssize_t) nfile->file.file_dim_info[dimid]->dim_size;
						if (nfile->file.file_dim_info[dimid]->is_unlimited)
							self->unlimited = 1;
					}
				}
			}
		}

		if (nfile->file.advanced_file_structure) {
			varnode = getVarFromGroup(advfile->advancedfile.grpnode,
					NrmStringToQuark(name));
			if (NULL != varnode) {
				self->attributes = PyDict_New();
				collect_advancedfile_attributes(NULL, varnode->att_rec,
						self->attributes);
			}
		} else {
			self->attributes = PyDict_New();
			collect_attributes(file->id, self->id, self->attributes, nattrs);
		}

		self->name = (char *) malloc(strlen(name) + 1);
		if (self->name != NULL)
			strcpy(self->name, name);
		return self;
	} else
		return NULL;
}

/* Create group object */

static NioFileObject* nio_read_group(NioFileObject* niofileobj,
		NclFileGrpNode *grpnode) {
	NioFileObject *self;
	NclFile nclfile = (NclFile) niofileobj->id;
	NclAdvancedFile advfilegroup = NULL;
	int ndims, nvars, ngrps, ngattrs;
	char* name;
	char* buf;
	Py_ssize_t len;
	py3_char *full_path;
	py3_char *path_buf;


	name = NrmQuarkToString(grpnode->name);

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 *fprintf(stderr, "\tgroup name: <%s>\n", name);
	 */

	if (!check_if_open(niofileobj, -1))
		return NULL;

	if (NULL == grpnode)
		return NULL;

	self = PyObject_New(NioFileObject, (PyTypeObject*) get_NioFileType());
	if (self == NULL)
		return NULL;

	self->dimensions = PyDict_New();
	self->chunk_dimensions = PyDict_New();
	self->ud_types = PyDict_New();
	self->variables = PyDict_New();
	self->groups = PyDict_New();
	self->parent = niofileobj;
	Py_INCREF((PyObject*) niofileobj);
	self->top = niofileobj->top;
	Py_INCREF((PyObject*) self->top);
	self->attributes = PyDict_New();
	self->weakreflist = NULL;
	self->recdim = -1; /* for now */

	self->open = niofileobj->open;
	self->write = niofileobj->write;
	self->define = niofileobj->define;
	self->name = char_to_pystr(name, strlen(name));
	self->mode = niofileobj->mode;
	Py_INCREF(self->mode);
	self->type = char_to_pystr("group", strlen("group"));

	/*len = PyString_Size(niofileobj->full_path);*/
	path_buf = as_utf8_char_and_size(niofileobj->full_path, &len);
	buf = malloc(len + 1);
	strcpy(buf, path_buf);
	len = strlen(buf);
	while (--len >= 0 && buf[len] == '/') {
		buf[len] = '\0';
	}
	if (!strcmp(name, "/") || strlen(buf) == 0) {
		/*self->full_path = PyString_FromFormat("%s", name);*/
		self->full_path = pystr_to_utf8(PyUnicode_FromFormat("%s", name));
	} else {
		/*self->full_path = PyString_FromFormat("%s/%s", buf, name);*/
		self->full_path = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", buf, name));
	}
	free(path_buf);
	free(buf);
	/*printf("path is %s\n",PyUnicode_AsUTF8(self->full_path));*/
	full_path = as_utf8_char(self->full_path);
	advfilegroup = _NclAdvancedGroupCreate(NULL, NULL, Ncl_File, 0, TEMPORARY,
			nclfile, NrmStringToQuark(full_path));

	free(full_path);
	self->gnode = (void *) advfilegroup;
	self->id = niofileobj->id;

	ndims = 0;
	nvars = 0;
	ngrps = 0;
	ngattrs = 0;

	/*
	 */
	dimNvarInfofromGroup(self, grpnode, &ndims, &nvars, &ngrps, &ngattrs);
	collect_advancedfile_attributes(self, grpnode->att_rec, self->attributes);

	return self;
}

/* Return value */

static PyObject *
NioVariableObject_value(NioVariableObject *self, PyObject *args) {
	NioIndex *indices;
	int i;

	if (self->nd == 0)
		indices = NULL;
	else
		indices = NioVariable_Indices(self);
	for (i = 0; i < self->nd; i++) {
		indices[i].no_stop = 1;
		indices[i].no_start = 1;
	}
	return PyArray_Return(NioVariable_ReadAsArray(self, indices));
}

/* Assign value */

static PyObject *
NioVariableObject_assign(NioVariableObject *self, PyObject *args) {
	PyObject *value;
	NioIndex *indices;
	int i;

	if (!PyArg_ParseTuple(args, "O", &value))
		return NULL;
	if (self->nd == 0)
		indices = NULL;
	else
		indices = NioVariable_Indices(self);
	for (i = 0; i < self->nd; i++) {
		indices[i].no_stop = 1;
		indices[i].no_start = 1;
	}
	NioVariable_WriteArray(self, indices, value);
	Py_INCREF(Py_None);
	return Py_None;
}

/* Return typecode */

static PyObject *
NioVariableObject_typecode(NioVariableObject *self, PyObject *args) {
	char *t;

	t = typecode(self->type);
	return char_to_pystr(t, strlen(t));
}

/* Method table */

static PyMethodDef NioVariableObject_methods[] = { { "assign_value",
		(PyCFunction) NioVariableObject_assign, METH_VARARGS }, { "get_value",
		(PyCFunction) NioVariableObject_value, METH_NOARGS }, { "typecode",
		(PyCFunction) NioVariableObject_typecode, METH_NOARGS }, { NULL, NULL } /* sentinel */
};

/* Attribute access */

static int NioVariable_GetRank(NioVariableObject *var) {
	return var->nd;
}

static Py_ssize_t *
NioVariable_GetShape(NioVariableObject *var) {
	int i, j;
	if (check_if_open(var->file, -1)) {
		NclFile nfile = (NclFile) var->file->id;
		if (nfile->file.advanced_file_structure) {
			NclFileDimRecord* grpdimrec;
			NclFileDimNode* grpdimnode;
			NclFileDimRecord* dimrec;
			NclFileDimNode* dimnode;
			NclFileVarNode* varnode;
			NclFileGrpNode* grpnode;

			grpnode =
					((NclAdvancedFile) var->file->gnode)->advancedfile.grpnode;
			grpdimrec = grpnode->dim_rec;
			varnode = getVarFromGroup(grpnode, NrmStringToQuark(var->name));

			if (NULL != varnode) {
				dimrec = varnode->dim_rec;
				if (NULL != dimrec) {
					for (i = 0; i < var->nd; ++i) {
						dimnode = &(dimrec->dim_node[i]);
						var->dimensions[i] = (Py_ssize_t) dimnode->size;
						if (dimnode->is_unlimited) {
#if PY_MAJOR_VERSION < 3
							PyObject *size_ob = PyInt_FromSsize_t(
									var->dimensions[i]);
#else
							PyObject *size_ob = PyLong_FromSsize_t(
									var->dimensions[i]);
#endif
							for (j = 0; j < grpdimrec->n_dims; ++j) {
								grpdimnode = &(grpdimrec->dim_node[j]);
								if (grpdimnode->name == dimnode->name) {
									if (grpdimnode->size < dimnode->size) {
										grpdimnode->size = dimnode->size;

										/*
										 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
										 *                __PRETTY_FUNCTION__, __FILE__, __LINE__);
										 *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
										 *     (int)j, NrmQuarkToString(grpdimnode->name), (int)grpdimnode->size);
										 */
										break;
									}

									if (grpdimnode->size > dimnode->size) {
										dimnode->size = grpdimnode->size;
										/*
										 *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
										 *                   i, NrmQuarkToString(dimnode->name), (int)dimnode->size);
										 */
										var->dimensions[i] =
												(Py_ssize_t) dimnode->size;

										break;
									}
								}
							}

							DICT_SETITEMSTRING(var->file->dimensions,
									NrmQuarkToString(dimnode->name), size_ob);
							Py_DECREF(size_ob);
						}
					}
					return var->dimensions;
				}
			}
		} else {
			for (i = 0; i < var->nd; i++) {
				int dimid = _NclFileIsDim(nfile, var->qdims[i]);
				var->dimensions[i] =
						(Py_ssize_t) nfile->file.file_dim_info[dimid]->dim_size;
				if (dimid == var->file->recdim) {
#if PY_MAJOR_VERSION < 3
					PyObject *size_ob = PyInt_FromSsize_t(var->dimensions[i]);
#else
					PyObject *size_ob = PyLong_FromSsize_t(var->dimensions[i]);
#endif
					DICT_SETITEMSTRING(var->file->dimensions,
							NrmQuarkToString(var->qdims[i]), size_ob);
					Py_DECREF(size_ob);
				}
			}
			return var->dimensions;
		}
	}

	return NULL;
}

static Py_ssize_t NioVariable_GetSize(NioVariableObject *var) {
	int i, j;
	Py_ssize_t size = -1;
	if (check_if_open(var->file, -1)) {
		NclFile nfile = (NclFile) var->file->id;
		if (nfile->file.advanced_file_structure) {
			NclFileDimRecord* grpdimrec;
			NclFileDimNode* grpdimnode;
			NclFileDimRecord* dimrec;
			NclFileDimNode* dimnode;
			NclFileVarNode* varnode;
			NclFileGrpNode* grpnode;

			grpnode =
					((NclAdvancedFile) var->file->gnode)->advancedfile.grpnode;
			grpdimrec = grpnode->dim_rec;
			varnode = getVarFromGroup(grpnode, NrmStringToQuark(var->name));

			if (NULL != varnode) {
				size = _NclSizeOf(varnode->type);
				dimrec = varnode->dim_rec;
				if (NULL != dimrec) {
					for (i = 0; i < var->nd; ++i) {
						dimnode = &(dimrec->dim_node[i]);
						var->dimensions[i] = (Py_ssize_t) dimnode->size;
						size *= dimnode->size;
						if (dimnode->is_unlimited) {
#if PY_MAJOR_VERSION < 3
							PyObject *size_ob = PyInt_FromSsize_t(
									var->dimensions[i]);
#else
							PyObject *size_ob = PyLong_FromSsize_t(
									var->dimensions[i]);
#endif
							for (j = 0; j < grpdimrec->n_dims; ++j) {
								grpdimnode = &(grpdimrec->dim_node[j]);
								if (grpdimnode->name == dimnode->name) {
									if (grpdimnode->size < dimnode->size) {
										grpdimnode->size = dimnode->size;

										/*
										 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
										 *                __PRETTY_FUNCTION__, __FILE__, __LINE__);
										 *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
										 *     (int)j, NrmQuarkToString(grpdimnode->name), (int)grpdimnode->size);
										 */
										break;
									}

									if (grpdimnode->size > dimnode->size) {
										dimnode->size = grpdimnode->size;
										/*
										 *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
										 *                   i, NrmQuarkToString(dimnode->name), (int)dimnode->size);
										 */
										var->dimensions[i] =
												(Py_ssize_t) dimnode->size;

										break;
									}
								}
							}

							DICT_SETITEMSTRING(var->file->dimensions,
									NrmQuarkToString(dimnode->name), size_ob);
							Py_DECREF(size_ob);
						}
					}
					return size;
				}
			}
		} else {
			int varid = _NclFileIsVar(nfile, NrmStringToQuark(var->name));
			size = _NclSizeOf(nfile->file.var_info[varid]->data_type);
			for (i = 0; i < var->nd; i++) {
				int dimid = _NclFileIsDim(nfile, var->qdims[i]);
				var->dimensions[i] =
						(Py_ssize_t) nfile->file.file_dim_info[dimid]->dim_size;
				size *= (Py_ssize_t) nfile->file.file_dim_info[dimid]->dim_size;
				if (dimid == var->file->recdim) {
#if PY_MAJOR_VERSION < 3
					PyObject *size_ob = PyInt_FromSsize_t(var->dimensions[i]);
#else
					PyObject *size_ob = PyLong_FromSsize_t(var->dimensions[i]);
#endif
					DICT_SETITEMSTRING(var->file->dimensions,
							NrmQuarkToString(var->qdims[i]), size_ob);
					Py_DECREF(size_ob);
				}
			}
			return size;
		}
	}

	return -1;
}

static PyObject *
NioVariable_GetAttribute(NioVariableObject *self, char *name) {
	PyObject *value;
	if (strcmp(name, "name") == 0) {
		return char_to_pystr(self->name, strlen(self->name));
	}

	if (strcmp(name, "path") == 0) {

		py3_char *path_char = as_utf8_char(self->file->full_path);
		PyObject *result = NULL;

		if (!strcmp(path_char, "") || !strcmp(path_char, "/")) {
			/*return (PyString_FromFormat("/%s", self->name));*/
			result = pystr_to_utf8(PyUnicode_FromFormat("/%s", self->name));
		}
		else {
			/*return (PyString_FromFormat("%s/%s", path, self->name));*/
			result = pystr_to_utf8(PyUnicode_FromFormat("%s/%s", path_char, self->name));
		}

		free(path_char);

		return result;
	}

	if (strcmp(name, "shape") == 0) {
		PyObject *tuple;
		int i;
		if (check_if_open(self->file, -1)) {
			NioVariable_GetShape(self);
			tuple = PyTuple_New(self->nd);
			for (i = 0; i < self->nd; i++)
				PyTuple_SetItem(tuple, i,
#if PY_MAJOR_VERSION < 3
						PyInt_FromSsize_t(self->dimensions[i]));
#else
						PyLong_FromSsize_t(self->dimensions[i]));
#endif
			return tuple;
		} else
			return NULL;
	}

	if (strcmp(name, "rank") == 0) {
		int rank;
		if (check_if_open(self->file, -1)) {
			rank = NioVariable_GetRank(self);
			return Py_BuildValue("i", rank);
		} else
			return NULL;
	}

	if (strcmp(name, "size") == 0) {
		ng_size_t size;
		if (check_if_open(self->file, -1)) {
			size = NioVariable_GetSize(self);
			return Py_BuildValue("L", size);
		} else
			return NULL;
	}

	if (strcmp(name, "dimensions") == 0) {
		PyObject *tuple;
		char *dname;
		int i;
		if (check_if_open(self->file, -1)) {
			NclFile nfile = (NclFile) self->file->id;
			tuple = PyTuple_New(self->nd);
			if (nfile->file.advanced_file_structure) {
				for (i = 0; i < self->nd; i++) {
					dname = NrmQuarkToString(self->qdims[i]);
					PyTuple_SetItem(tuple, i, char_to_pystr(dname, strlen(dname)));
				}
			} else {
				for (i = 0; i < self->nd; i++) {
					int dimid = _NclFileIsDim(nfile, self->qdims[i]);
					dname = NrmQuarkToString(
							nfile->file.file_dim_info[dimid]->dim_name_quark);
					PyTuple_SetItem(tuple, i, char_to_pystr(dname, strlen(dname)));
				}
			}
			return tuple;
		} else
			return NULL;
	}
	if (strcmp(name, "__dict__") == 0) {
		Py_INCREF(self->attributes);
		return self->attributes;
	}
	value = PyDict_GetItemString(self->attributes, name);
	if (value != NULL) {
		Py_INCREF(value);
		return value;
	} else {
#if PY_MAJOR_VERSION < 3
		PyErr_Clear();
		return Py_FindMethod(NioVariableObject_methods, (PyObject *) self, name);
#else
		PyObject *name_obj = char_to_pystr(name, strlen(name));
		PyObject *result =  PyObject_GenericGetAttr((PyObject *) self, name_obj);
		Py_DECREF(name_obj);
		return result;
#endif
	}
}

static int NioVariable_SetAttribute(NioVariableObject *self, char *name,
		PyObject *value) {
	NclFile nfile = (NclFile) self->file->id;
	/*nio_ncerr = 0;*/
	if (check_if_open(self->file, 1)) {
		if (strcmp(name, "shape") == 0 || strcmp(name, "dimensions") == 0
				|| strcmp(name, "__dict__") == 0 || strcmp(name, "rank") == 0) {
			PyErr_SetString(PyExc_TypeError,
					"attempt to set read-only attribute");
			return -1;
		}
		define_mode(self->file, 1);
		if (nfile->file.advanced_file_structure)
			return set_advanced_variable_attribute(self->file, self,
					self->attributes, name, value);
		else
			return set_attribute(self->file, self->id, self->attributes, name,
					value);
	} else
		return -1;
}

/* Not actually used */
#if 0
int NioVariable_SetAttributeString(NioVariableObject *self, char *name,
		char *value) {
	PyObject *string = char_to_pystr(value, strlen(value));
	if (string != NULL)
		return NioVariable_SetAttribute(self, name, string);
	else
		return -1;
}
#endif

/* Subscripting */

static Py_ssize_t NioVariableObject_length(NioVariableObject *self) {
	if (self->nd > 0)
		return self->dimensions[0];
	else
		return 0;
}

NioIndex *
NioVariable_Indices(NioVariableObject *var) {
	NioIndex *indices = (NioIndex *) malloc(var->nd * sizeof(NioIndex));
	int i;
	if (indices != NULL)
		for (i = 0; i < var->nd; i++) {
			indices[i].start = 0;
			indices[i].stop = var->dimensions[i];
			indices[i].stride = 1;
			indices[i].item = 0;
			indices[i].no_start = indices[i].no_stop = 0;
			if (((NclFile) var->file->id)->file.advanced_file_structure) {
				indices[i].unlimited =
						advfile_is_unlimited(var->file,
								NrmQuarkToString(var->qdims[i])) > 0 ? 1 : 0;
			} else {
				indices[i].unlimited = (i == 0 && var->unlimited) ? 1 : 0;
			}
		}
	else
		PyErr_SetString(PyExc_MemoryError, "out of memory");
	return indices;
}

void _convertVLEN2Obj(PyArrayObject* array, obj* listids, ng_size_t nitems) {
	PyObject* pyobj;
	NclVar var;
	ng_size_t i = 0;
	NclList thelist = NULL;
	NclListObjList *tmp_list;
	NclMultiDValData md;
	npy_intp length;
	int itemsize = PyArray_ITEMSIZE(array);

	for (i = 0; i < nitems; ++i) {
		thelist = (NclList) _NclGetObj(listids[i]);
		tmp_list = thelist->list.last;
		var = (NclVar) _NclGetObj(tmp_list->obj_id);
		md = (NclMultiDValData) _NclGetObj(var->var.thevalue_id);
		length = md->multidval.totalelements;

		pyobj = PyArray_SimpleNewFromData(1, &length,
				data_type(md->multidval.data_type), md->multidval.val);

		PyArray_SETITEM(array, PyArray_BYTES(array) + i * itemsize, pyobj);

		Py_DECREF(pyobj);
	}
}

void _convertCOMPOUND2Obj(PyArrayObject* array, obj* listids, ng_size_t nitems,
		NclFileVarNode* varnode) {
	PyArrayObject* comparray;
	PyObject* pyobj;
	void* curval;
	NclVar var;
	ng_size_t i = 0;
	NclList thelist = NULL;
	NclListObjList *tmp_list;
	NclMultiDValData md;
	npy_intp length;
	int itemsize = PyArray_ITEMSIZE(array);
	int n;
	NclFileCompoundRecord *comprec = varnode->comprec;
	NclFileCompoundNode *compnode = NULL;

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *		__PRETTY_FUNCTION__, __FILE__, __LINE__);
	 *fprintf(stderr, "\titemsize = %d\n", itemsize);
	 */

	if ((NULL == comprec) || (1 > comprec->n_comps)) {
		fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
		fprintf(stderr,
				"\tThe compound record is NULL, there is nothing we can do then.\n");
		return;
	}

	for (i = 0; i < nitems; ++i) {
		thelist = (NclList) _NclGetObj(listids[i]);
		tmp_list = thelist->list.last;
		var = (NclVar) _NclGetObj(tmp_list->obj_id);
		md = (NclMultiDValData) _NclGetObj(var->var.thevalue_id);

		length = (npy_intp) comprec->n_comps;
		comparray = (PyArrayObject*) PyArray_SimpleNew(1, &length, NPY_OBJECT);
		if (NULL == comparray)
			PyErr_SetString(PyExc_MemoryError,
					"Problem creating PyArray in NioVariable_ReadAsArray for compound data.");

		curval = md->multidval.val;

		for (n = 0; n < comprec->n_comps; ++n) {
			compnode = &(comprec->compnode[n]);

			/*
			 *fprintf(stderr, "\n\tfile: %s, line: %d\n", __FILE__, __LINE__);
			 *fprintf(stderr, "\tcomponent [%d] name: <%s>\n", n, NrmQuarkToString(compnode->name));
			 *fprintf(stderr, "\toffset = %d, rank = %d, nvals = %d, type: %s\n",
			 *                   compnode->offset, compnode->rank, compnode->nvals,
			 *                   _NclBasicDataTypeToName(compnode->type));
			 */

			length = compnode->nvals;

			if (NCL_char == compnode->type) {
				PyObject* tmpobj;
				char* tmpstr = curval + compnode->offset;
				length = strlen(tmpstr);
				tmpobj = PyArray_SimpleNewFromData(1, &length, NPY_CHAR,
						tmpstr);
				pyobj = PyArray_ToString((PyArrayObject*) tmpobj, NPY_CORDER);
				Py_DECREF(tmpobj);
			} else if (NCL_string == compnode->type) {
				PyObject* tmpobj;
				char* tmpstr = NrmQuarkToString(
						*(NrmQuark*) ((char*) curval + compnode->offset));
				length = strlen(tmpstr);
				tmpobj = PyArray_SimpleNewFromData(1, &length, NPY_CHAR,
						tmpstr);
				pyobj = PyArray_ToString((PyArrayObject*) tmpobj, NPY_CORDER);
				Py_DECREF(tmpobj);
			} else
				pyobj = PyArray_SimpleNewFromData(1, &length,
						data_type(compnode->type), curval + compnode->offset);

			PyArray_SETITEM(comparray, PyArray_BYTES(comparray) + n * itemsize, pyobj);

			Py_DECREF(pyobj);
		}

		PyArray_SETITEM(array, PyArray_BYTES(array) + i * itemsize, (PyObject*) comparray);

		Py_DECREF(comparray);
	}
}

PyArrayObject *
NioVariable_ReadAsArray(NioVariableObject *self, NioIndex *indices) {
	int is_own;
	int ndims;
	npy_intp *dims;
	PyArrayObject *array = NULL;
	int i, d;
	ng_size_t nitems;
	int error = 0;
	int dir;
	NclFile nfile = self->file->id;

	if (nfile->file.advanced_file_structure)
		nfile = (NclFile) self->file->gnode;

	d = 0;
	nitems = 1;
	/*nio_ncerr = 0;*/

	ndims = self->nd;

	if (!check_if_open(self->file, -1)) {
		free(indices);
		return NULL;
	}
	define_mode(self->file, 0);
	if (self->nd == 0)
		dims = NULL;
	else {
		dims = (npy_intp *) malloc(self->nd * sizeof(npy_intp));
		if (dims == NULL) {
			free(indices);
			return (PyArrayObject *) PyErr_NoMemory();
		}
	}
	/* convert from Python to NCL indexing */
	/* negative stride in Python requires the start index to be greater than
	 the end index: in NCL negative stride reverses the direction
	 implied by the index start and stop.
	 */
	for (i = 0; i < self->nd; i++) {
		error = error || (indices[i].stride == 0);
		if (error)
			break;
		if (indices[i].stride < 0) {
			indices[i].stop += 1;
			indices[i].stride = -indices[i].stride;
			dir = -1;
		} else {
			indices[i].stop -= 1;
			dir = 1;
		}
		if (indices[i].start < 0)
			indices[i].start += self->dimensions[i];
		if (indices[i].start < 0)
			indices[i].start = 0;
		if (indices[i].start > self->dimensions[i] - 1)
			indices[i].start = self->dimensions[i] - 1;
		if (indices[i].item != 0) {
			indices[i].stop = indices[i].start;
			dims[d] = 1;
		} else {
			if (indices[i].stop < 0)
				indices[i].stop += self->dimensions[i];
			if (indices[i].stop < 0)
				indices[i].stop = 0;
			if (indices[i].stop > self->dimensions[i] - 1)
				indices[i].stop = self->dimensions[i] - 1;
			/* Python only creates a reverse-ordered return value if the stride is less than 0 */
			dims[d] = ((indices[i].stop - indices[i].start)
					/ (indices[i].stride * dir)) + 1;
			if (dims[d] < 0)
				dims[d] = 0;
			nitems *= dims[d];
			d++;
		}
	}
	if (error) {
		PyErr_SetString(PyExc_IndexError, "illegal index");
		if (dims != NULL)
			free(dims);
		if (indices != NULL)
			free(indices);
		return NULL;
	}

	if (nitems == 0) {
		array = (PyArrayObject *) PyArray_New(&PyArray_Type, d, dims,
				self->type, NULL, NULL, 0, 0, NULL);

	}
	if (nitems > 0 && self->type == NPY_STRING) {
		NclMultiDValData md;
		NclSelectionRecord *sel_ptr = NULL;

		if (self->nd > 0) {
			sel_ptr = (NclSelectionRecord*) malloc(sizeof(NclSelectionRecord));
			if (sel_ptr != NULL) {
				sel_ptr->n_entries = self->nd;
				for (i = 0; i < self->nd; i++) {
					sel_ptr->selection[i].sel_type = Ncl_SUBSCR;
					sel_ptr->selection[i].dim_num = i;
					sel_ptr->selection[i].u.sub.start = indices[i].start;
					sel_ptr->selection[i].u.sub.finish = indices[i].stop;
					sel_ptr->selection[i].u.sub.stride = indices[i].stride;
					sel_ptr->selection[i].u.sub.is_single = indices[i].item
							!= 0;
				}
			}
		}
		md = _NclFileReadVarValue(nfile, NrmStringToQuark(self->name), sel_ptr);
		if (!md) {
			nio_seterror(23);
			array = NULL;
		} else {
			int maxlen = 0;
			int tlen;
			NrmQuark qstr;
			py3_char *pqstr;
			PyObject *pystr;
			/* find the maximum string length */
			for (i = 0; i < nitems; i++) {
				qstr = ((NrmQuark *) md->multidval.val)[i];
				tlen = strlen(NrmQuarkToString(qstr));
				if (maxlen < tlen)
					maxlen = tlen;
			}
			array = (PyArrayObject *) PyArray_New(&PyArray_Type, self->nd, dims,
					self->type, NULL, NULL, maxlen, 0, NULL);
			if (array) {
				for (i = 0; i < nitems; i++) {
					qstr = ((NrmQuark *) md->multidval.val)[i];
					pqstr = NrmQuarkToString(qstr);
					pystr = char_to_pystr(pqstr, strlen(pqstr));
					PyArray_SETITEM(array, PyArray_BYTES(array) + i * PyArray_ITEMSIZE(array), pystr);
				}
			}
		}
		if (sel_ptr)
			free(sel_ptr);
	} else if (nitems > 0) {
		if (self->nd == 0) {
			NclMultiDValData md = _NclFileReadVarValue(nfile,
					NrmStringToQuark(self->name), NULL);
			if (!md) {
				nio_seterror(23);
				PyErr_SetString(PyExc_MemoryError,
						"Problem reading variable data.");
				array = NULL;
			} else {
				array = (PyArrayObject *) PyArray_New(&PyArray_Type, d, dims,
						self->type, NULL, md->multidval.val, 0, 0, NULL);
				md->multidval.val = NULL;
				_NclDestroyObj((NclObj) md);
			}
		} else {
			NclSelectionRecord *sel_ptr;
			sel_ptr = (NclSelectionRecord*) malloc(sizeof(NclSelectionRecord));
			if (sel_ptr != NULL) {
				NclMultiDValData md;
				sel_ptr->n_entries = self->nd;
				for (i = 0; i < self->nd; i++) {
					sel_ptr->selection[i].sel_type = Ncl_SUBSCR;
					sel_ptr->selection[i].dim_num = i;
					sel_ptr->selection[i].u.sub.start = indices[i].start;
					sel_ptr->selection[i].u.sub.finish = indices[i].stop;
					sel_ptr->selection[i].u.sub.stride = indices[i].stride;
					sel_ptr->selection[i].u.sub.is_single = indices[i].item
							!= 0;
				}
				md = _NclFileReadVarValue(nfile, NrmStringToQuark(self->name),
						sel_ptr);
				if (!md) {
					nio_seterror(23);
					PyErr_SetString(PyExc_MemoryError,
							"Problem reading variable data.");
					array = NULL;
				} else if (nfile->file.advanced_file_structure) {
					NclFileVarNode* varnode;
					NclFileGrpNode* grpnode;

					grpnode =
							((NclAdvancedFile) self->file->gnode)->advancedfile.grpnode;
					varnode = getVarFromGroup(grpnode,
							NrmStringToQuark(self->name));
					if (NULL == varnode) {
						array = NULL;
						PyErr_SetString(PyExc_MemoryError,
								"Problem reading variable information.");
					} else if (NCL_list == md->multidval.data_type) {
						if (NCL_UDT_vlen == varnode->udt_type) {
							/*
							 *fprintf(stderr, "\nFunction %s, in file: %s, line: %d\n",
							 *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
							 *fprintf(stderr, "\tNeed to work on read vlen\n");
							 *fprintf(stderr, "\tnitems = %ld\n", (long)nitems);
							 *for(i = 0; i < self->nd; ++i)
							 *    fprintf(stderr, "\tdims[%d] = %ld\n", i, (long)dims[i]);
							 */

							array = (PyArrayObject*) PyArray_SimpleNew(ndims,
									dims, NPY_OBJECT);
							if (NULL == array)
								PyErr_SetString(PyExc_MemoryError,
										"Problem to create PyArray in NioVariable_ReadAsArray for vlen data.");
							else
								_convertVLEN2Obj(array,
										(obj*) md->multidval.val, nitems);
						} else if (NCL_UDT_compound == varnode->udt_type) {
							/*
							 *fprintf(stderr, "\nFunction %s, in file: %s, line: %d\n",
							 *		__PRETTY_FUNCTION__, __FILE__, __LINE__);
							 *fprintf(stderr, "\tNeed to work on read compound\n");
							 *fprintf(stderr, "\tnitems = %ld\n", (long)nitems);
							 *for(i = 0; i < self->nd; ++i)
							 *	fprintf(stderr, "\tdims[%d] = %ld\n", i, (long)dims[i]);
							 */

							array = (PyArrayObject*) PyArray_SimpleNew(ndims,
									dims, NPY_OBJECT);
							if (NULL == array)
								PyErr_SetString(PyExc_MemoryError,
										"Problem to create PyArray in NioVariable_ReadAsArray for compound data.");
							else
								_convertCOMPOUND2Obj(array,
										(obj*) md->multidval.val, nitems,
										varnode);
						} else {
							fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__,
							__LINE__);
							fprintf(stderr,
									"\tDo not know anything about varnode->type. Return NULL.\n");
							array = NULL;
						}
					} else {
						array = (PyArrayObject *) PyArray_New(&PyArray_Type, d,
								dims, self->type, NULL, md->multidval.val, 0, 0,
								NULL);
					}

					/*Delete md will cause seg. fault. Wei, May 9, 2014.
					 *md->multidval.val = NULL;
					 *_NclDestroyObj((NclObj)md);
					 */
				} else {
					array = (PyArrayObject *) PyArray_New(&PyArray_Type, d,
							dims, self->type, NULL, md->multidval.val, 0, 0,
							NULL);
				}

				free(sel_ptr);
			}
		}
	}
	if (array) {
		is_own = PyArray_CHKFLAGS(array, NPY_ARRAY_OWNDATA);
		if (!is_own) {
			PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA);
		}
		PyArray_ENABLEFLAGS(array, NPY_ARRAY_C_CONTIGUOUS);
	}
	free(dims);
	free(indices);
	return array;
}

/* Defined but not used */
#if 0
static PyUnicodeObject *
NioVariable_ReadAsString(NioVariableObject *self)

{
	NclFile nfile = self->file->id;
	if (self->type != NPY_CHAR || self->nd != 1) {
		PyErr_SetString(get_NioError(), "not a string variable");
		return NULL;
	}

	if (nfile->file.advanced_file_structure)
		nfile = self->file->gnode;

	if (check_if_open(self->file, -1)) {
		char *tstr;
		PyObject *string;

		NclMultiDValData md = _NclFileReadVarValue(nfile,
				NrmStringToQuark(self->name), NULL);
		if (!md) {
			nio_seterror(23);
			return NULL;
		}
		/* all we care about is the actual value */
		tstr = NrmQuarkToString(*(NrmQuark *) md->multidval.val);
		_NclDestroyObj((NclObj) md);
		string = char_to_pystr(tstr, strlen(tstr));
		return (PyUnicodeObject *) string;
	} else
		return NULL;
}
#endif

void _convertObj2VLEN(PyObject* pyobj, obj* listids, NclBasicDataTypes type,
		ng_size_t n_dims, ng_size_t curdim, ng_usize_t* counter) {
	NclQuark dimname;
	ng_size_t dimsize = 0;
	PyArrayObject *array;
	PyObject* seq;
	PyObject* item;
	char buffer[16];
	NclVar var;
	int i, len;
	int processingdim = 1 + curdim;
	NclObj thelist = NULL;

	seq = PySequence_Fast(pyobj, "expected a sequence");
	len = PySequence_Size(pyobj);

	for (i = 0; i < len; ++i) {
		item = PySequence_Fast_GET_ITEM(seq, i);

		if (processingdim == n_dims) {
			array = (PyArrayObject *) PyArray_ContiguousFromAny(item,
					data_type(type), 0, 1);
			dimsize = PyArray_DIM(array, 0);
			sprintf(buffer, "list_%6.6d", (int) (*counter));
			dimname = NrmStringToQuark(buffer);
			var = _NclCreateVlenVar(buffer, PyArray_DATA(array), 1, &dimname,
					&dimsize, type);
			thelist = _NclGetObj(listids[*counter]);
			_NclListAppend((NclObj) thelist, (NclObj) var);

			*counter += 1;
		} else
			_convertObj2VLEN(item, listids, type, n_dims, processingdim,
					counter);
	}

	Py_DECREF(seq);
}

void _convertObj2COMPOUND(PyObject* pyobj, obj* listids,
		NclFileCompoundRecord *comprec, ng_size_t n_dims, ng_size_t curdim,
		ng_usize_t* counter) {
	NclQuark dimname;
	ng_size_t dimsize = 0;
	PyArrayObject *array;
	PyObject* seq;
	PyObject* item;
	char buffer[16];
	NclVar var;
	int i, len;
	int processingdim = 1 + curdim;
	NclObj thelist = NULL;

	seq = PySequence_Fast(pyobj, "expected a sequence");
	len = PySequence_Size(pyobj);

	/*
	 *fprintf(stderr, "\nFunc %s, in file: %s, line: %d\n",
	 *                __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 *fprintf(stderr, "\tlen = %d\n", len);
	 */

	for (i = 0; i < len; ++i) {
		item = PySequence_Fast_GET_ITEM(seq, i);

		if (processingdim == n_dims) {
			NclFileCompoundNode* compnode;
			PyObject* seq2;
			PyObject* item2;
			int n, len2;

			seq2 = PySequence_Fast(item, "expected a sequence");
			len2 = PySequence_Size(item);
			/*
			 *fprintf(stderr, "\tlen2 = %d\n", len2);
			 *fprintf(stderr, "\tcomprec->n_comps = %d\n", (int)comprec->n_comps);
			 */

			if (len2 != comprec->n_comps) {
				fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
				fprintf(stderr, "\tlen2 = %d\n", len2);
				fprintf(stderr, "\tcomprec->n_comps = %d\n",
						(int) comprec->n_comps);
				fprintf(stderr, "\tcomprec->n_comps and len2 do not equal.\n");
				return;
			}

			sprintf(buffer, "comp_%6.6d", (int) (*counter));
			dimname = NrmStringToQuark(buffer);

			for (n = 0; n < len2; ++n) {
				item2 = PySequence_Fast_GET_ITEM(seq2, n);
				compnode = &(comprec->compnode[n]);

				if (NCL_char == compnode->type) {
					char* tmpv = (char *) calloc(compnode->nvals, sizeof(char));
					if (NULL == tmpv) {
						fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__,
						__LINE__);
						fprintf(stderr,
								"\tFailed to allocate memory of %d char.\n",
								compnode->nvals);
						return;
					}
					array = (PyArrayObject *) PyArray_ContiguousFromAny(item2,
							NPY_STRING, 0, 1);
					dimsize = compnode->nvals;
					strcpy(tmpv, PyArray_BYTES(array));
					var = _NclCreateVlenVar(buffer, (void *) tmpv, 1, &dimname,
							&dimsize, compnode->type);
					/*
					 *fprintf(stderr, "\tcomp: %d, value: <%s>\n", n, tmpv);
					 *free(tmpv);
					 */
				} else {
					array = (PyArrayObject *) PyArray_ContiguousFromAny(item2,
							data_type(compnode->type), 0, 1);
					dimsize = PyArray_DIM(array, 0);
					var = _NclCreateVlenVar(buffer, PyArray_DATA(array), 1,
							&dimname, &dimsize, compnode->type);
				}
				thelist = _NclGetObj(listids[*counter]);
				_NclListAppend((NclObj) thelist, (NclObj) var);
			}

			*counter += 1;
		} else
			_convertObj2COMPOUND(item, listids, comprec, n_dims, processingdim,
					counter);
	}

	Py_DECREF(seq);
}

static int NioVariable_WriteArray(NioVariableObject *self, NioIndex *indices,
		PyObject *value) {
	ng_size_t *dims = NULL;
	PyArrayObject *array = NULL;
	int i, n_dims;
	Py_ssize_t nitems, var_el_count;
	int error = 0;
	int ret = 0;
	int dir, *dirs;
	NrmQuark qtype;
	ng_size_t scalar_size = 1;
	NclFile nfile = (NclFile) self->file->id;
	NclMultiDValData md = NULL;
	NhlErrorTypes nret;
	int select_all = 1;
	Py_ssize_t array_el_count = 1;
	int undefined_dim = -1, dim_undef = -1;

	NclFileVarNode* varnode = NULL;
	NclFileDimRecord* dimrec = NULL;

#if 0
	NclFileDimNode* dimnode = NULL;
#endif

	/* this code assumes ng_size_t and Py_ssize_t are basically equivalent */
	/* update shape */
	(void) NioVariable_GetShape(self);
	n_dims = 0;
	nitems = 1;
	var_el_count = 1;
	if (!check_if_open(self->file, 1)) {
		free(indices);
		return -1;
	}
	if (self->nd == 0) {
		dims = NULL;
		dirs = NULL;
	} else {
		dims = (ng_size_t *) malloc(self->nd * sizeof(ng_size_t));
		dirs = (int *) malloc(self->nd * sizeof(int));
		if (dims == NULL || dirs == NULL) {
			free(indices);
			PyErr_SetString(PyExc_MemoryError, "out of memory");
			return -1;
		}
	}
	define_mode(self->file, 0);

	/* Convert from Python to NCL indexing.
	 * Negative stride in Python requires the start index to be greater than
	 * the end index: in NCL negative stride reverses the direction
	 * implied by the index start and stop.
	 * Determination of unlimited dimensions needs to be deferred until it is
	 * clear how many elements will be added.
	 */

	/*
	 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *	  __PRETTY_FUNCTION__, __FILE__, __LINE__);
	 */
	if (nfile->file.advanced_file_structure) {
		NclFileGrpNode* grpnode;

		grpnode = ((NclAdvancedFile) self->file->gnode)->advancedfile.grpnode;
		varnode = getVarFromGroup(grpnode, NrmStringToQuark(self->name));
		if (varnode != NULL) {
			dimrec = varnode->dim_rec;
		}
	}
	for (i = 0; i < self->nd; i++) {
		if (indices[i].unlimited) {
			if ((indices[i].no_stop && indices[i].stride > 0)
					|| (indices[i].no_start && indices[i].stride < 0))
				undefined_dim = i;
		}
		if (i != undefined_dim && self->dimensions[i] > 0)
			var_el_count *= self->dimensions[i];
		error = error || (indices[i].stride == 0);
		if (error)
			break;
		if (indices[i].stride < 0) {
			indices[i].stop += 1;
			indices[i].stride = -indices[i].stride;
			dir = -1;
		} else {
			indices[i].stop -= 1;
			dir = 1;
		}
		if (indices[i].start < 0)
			indices[i].start += self->dimensions[i];
		if (indices[i].start < 0)
			indices[i].start = 0;
		if (indices[i].stop < 0)
			indices[i].stop += self->dimensions[i];
		if (indices[i].stop < 0)
			indices[i].stop = 0;
		if (!indices[i].unlimited) {
			if (indices[i].start > self->dimensions[i] - 1)
				indices[i].start = self->dimensions[i] - 1;
			if (indices[i].stop > self->dimensions[i] - 1)
				indices[i].stop = self->dimensions[i] - 1;
		}
		if (indices[i].item == 0) {
			dims[n_dims] = (ng_size_t) ((indices[i].stop - indices[i].start)
					/ (indices[i].stride * dir)) + 1;
			dirs[n_dims] = dir;
			if (dims[n_dims] < 0)
				dims[n_dims] = 0;
			if (i != undefined_dim)
				nitems *= dims[n_dims];
			else
				dim_undef = n_dims;
			n_dims++;
		} else {
			indices[i].stop = indices[i].start;
			dims[n_dims] = 1;
			dirs[n_dims] = dir;
		}
	}

	if (error) {
		PyErr_SetString(PyExc_IndexError, "illegal index");
		ret = -1;
		goto err_ret;
	}
	if (nitems == 0) {
		/* nothing to write; this is not an error; return 0 */
		ret = 0;
		goto err_ret;
	}

	if (!strcmp(value->ob_type->tp_name, "numpy.ndarray")
			|| !strcmp(value->ob_type->tp_name, "array")) {
		array = (PyArrayObject *) value;

		if ((!nfile->file.advanced_file_structure) && PyArray_DTYPE(array)->type == 'l'
				&& PyArray_ITEMSIZE(array) == 8) {
			PyArrayObject *array2 = (PyArrayObject *) PyArray_Cast(array,
					NPY_INT);
			sprintf(get_errbuf(),
					"output format does not support 8-byte integers; converting to 4-byte integer variable (%s): possible data loss due to overflow",
					self->name);
			PyErr_SetString(get_NioError(), get_errbuf());
			PyErr_Print();
			array = (PyArrayObject *) PyArray_ContiguousFromAny(
					(PyObject*) array2, self->type, 0, n_dims);
			Py_DECREF(array2);
		} else if (! strcmp(typecode(self->type),"c") &&  PyArray_DTYPE(array)->type == 'S' && PyArray_ITEMSIZE(array) > 1) {
			  PyArray_Descr tdescr;
			  PyArrayObject *array2,*array3;
			  PyArray_Dims pdims;
			  memcpy(&tdescr,PyArray_DTYPE(array),sizeof(PyArray_Descr));
			  tdescr.type = 'c';
			  tdescr.elsize = 1;
			  tdescr.type_num = self->type;
			  pdims.ptr = dims;
			  pdims.len = n_dims;
			  array2 = (PyArrayObject *)PyArray_View(array,&tdescr,NULL);
			  if (array2) {
				  if (*PyArray_DIMS(array2) != nitems) {
					  sprintf(get_errbuf(),"size of final dimension of character array variable (%s) must match length of longest string supplied as input",self->name);
					  PyErr_SetString(get_NioError(), get_errbuf());
					  ret = -1;
					  goto err_ret;
				  }
				  array3 = (PyArrayObject *) PyArray_Newshape(array2,&pdims,NPY_CORDER);
				  if (array3) {
					  array = (PyArrayObject *)PyArray_FromArray((PyArrayObject*)array3,&tdescr,0);
				  }
			  }
			  /*Py_DECREF(array2);*/
			  /*Py_DECREF(array3);*/
		}
		else {
			int single_el_dim_count = 0;
			/*
			 * Use numpy semantics.
			 * Numpy allows single element 'slow' dimensions to be discarded on assignment
			 */
			for (i = 0; i < PyArray_NDIM(array); i++) {
				if (PyArray_DIM(array, i) == 1)
					single_el_dim_count++;
				else
					break;
			}
			if (nfile->file.advanced_file_structure) {
				if (NCL_char != varnode->type)
					array = (PyArrayObject *) PyArray_ContiguousFromAny(value,
							self->type, 0, n_dims + single_el_dim_count);
			} else {
				array = (PyArrayObject *) PyArray_ContiguousFromAny(value,
						self->type, 0, n_dims + single_el_dim_count);
			}
		}
	} else {
		if (! strcmp(typecode(self->type),"c")) {
			array = (PyArrayObject *)PyArray_ContiguousFromAny(value,NPY_STRING,0,n_dims);
		}
		else {
			array = (PyArrayObject *) PyArray_ContiguousFromAny(value, self->type, 0, n_dims);
		}

		if (array && ! strcmp(typecode(self->type),"c") && PyArray_DTYPE(array)->type == 'S' && PyArray_ITEMSIZE(array) > 1) {
			PyArray_Descr tdescr;
			/*int tndim = 0;
			npy_intp tdims[NPY_MAXDIMS]; */
			PyArrayObject *array2=NULL,*array3=NULL,*array4=NULL;
			PyArray_Dims pdims;
			memcpy(&tdescr,PyArray_DTYPE(array),sizeof(PyArray_Descr));
			tdescr.type = 'c';
			tdescr.elsize = 1;
			tdescr.type_num = self->type;
			pdims.ptr = dims;
			pdims.len = n_dims;
			array2 = (PyArrayObject *)PyArray_View(array,&tdescr,NULL);
			if (array2) {
				if (*PyArray_DIMS(array2) != nitems) {
					sprintf(get_errbuf(),"size of final dimension of character array variable (%s) must match length of longest string supplied as input",self->name);
					PyErr_SetString(get_NioError(), get_errbuf());
					ret = -1;
					goto err_ret;
				}
				array3 = (PyArrayObject *) PyArray_Newshape(array2,&pdims,NPY_CORDER);
				if (array3) {
					array4 = (PyArrayObject *)PyArray_FromArray((PyArrayObject*)array3,&tdescr,0);
				}
			}
			if (array4) {
				Py_DECREF(array);
				array = array4;
			}
			else {
				Py_DECREF(array);
				array = NULL;
			}
		}
	}

	if (array == NULL) {
		sprintf(get_errbuf(),
				"type or dimensional mismatch writing to variable (%s)",
				self->name);
		PyErr_SetString(get_NioError(), get_errbuf());
		ret = -1;
		goto err_ret;
	}

	/*
	 * PyNIO does not support broadcasting except for the special case of a scalar
	 * applied to an array.
	 * However, it does ignore pre-pended single element dimensions either in the file
	 * variable or in the assigned array.
	 */

	if (PyArray_NDIM(array) == 0) {
		n_dims = 1;
	} else if (undefined_dim >= 0) {
		int adim_count = 0;
		int fdim_count = 0;
		int *adims, *fdims;
		adims = malloc(PyArray_NDIM(array) * sizeof(int));
		fdims = malloc(self->nd * sizeof(int));
		for (i = 0; i < PyArray_NDIM(array); i++) {
			if (PyArray_DIMS(array)[i] > 1) {
				adims[adim_count] = i;
				adim_count++;
			}
		}
		for (i = 0; i < n_dims; i++) {
			if (i == undefined_dim || dims[i] > 1) {
				fdims[fdim_count] = i;
				fdim_count++;
			}
		}
		if (fdim_count == adim_count) {
			ng_size_t undef_size = 0;
			for (i = 0; i < fdim_count; i++) {
				if (dims[fdims[i]] == PyArray_DIM(array, adims[i]))
					continue;
				if (fdims[i] == undefined_dim) {
					if (dirs[fdims[i]] == 1) {
						indices[fdims[i]].stop = indices[fdims[i]].start
								+ (PyArray_DIM(array, adims[i]) - 1)
										* indices[fdims[i]].stride;
					} else {
						indices[fdims[i]].start = indices[fdims[i]].stop
								+ (PyArray_DIM(array, adims[i]) - 1)
										* indices[fdims[i]].stride;
					}
					undef_size = PyArray_DIM(array, adims[i]);
					var_el_count *= PyArray_DIM(array, adims[i]);
				} else {
					sprintf(get_errbuf(),
							"Dimensional mismatch writing to variable (%s)",
							self->name);
					PyErr_SetString(get_NioError(), get_errbuf());
					ret = -1;
					goto err_ret;
				}
			}
			if (dim_undef > -1 && undef_size > 0) {
				dims[dim_undef] = undef_size;
				nitems *= undef_size;
			}
		}
	} else {
		int var_dim = 0;
		i = 0;
		while (i < PyArray_NDIM(array) && PyArray_DIMS(array)[i] == 1)
			i++;
		while (var_dim < n_dims && dims[var_dim] == 1)
			var_dim++;
		for (; i < PyArray_NDIM(array);) {
			if (PyArray_DIMS(array)[i] != dims[var_dim]) {
				if (dims[var_dim] == 1) {
					var_dim++;
					continue;
				}
				if (PyArray_DIMS(array)[i] == 1) {
					i++;
					continue;
				}
				sprintf(get_errbuf(),
						"Dimensional mismatch writing to variable (%s)",
						self->name);
				PyErr_SetString(get_NioError(), get_errbuf());
				ret = -1;
				goto err_ret;
			}
			array_el_count *= PyArray_DIMS(array)[i];
			var_dim++;
			i++;
		}
		while (var_dim < n_dims && dims[var_dim] == 1)
			var_dim++;
		if (var_dim != n_dims) {
			sprintf(get_errbuf(), "Dimensional mismatch writing to variable (%s)",
					self->name);
			PyErr_SetString(get_NioError(), get_errbuf());
			ret = -1;
			goto err_ret;
		}
		if (array_el_count < nitems) {
			/*
			 * This test should be redundant, but just in case.
			 */
			sprintf(get_errbuf(),
					"Not enough elements supplied for write to variable (%s)",
					self->name);
			PyErr_SetString(get_NioError(), get_errbuf());
			ret = -1;
			goto err_ret;
		}
	}

	if (nitems < var_el_count || self->unlimited) /* self->unlimited should be true if any dim is unlimited */
		select_all = 0;
	if (dirs != NULL) {
		for (i = 0; i < n_dims; i++) {
			if (dirs[i] == -1) {
				select_all = 0;
				break;
			}
		}
	}

	if (PyArray_DTYPE(array)->type == 'S' && PyArray_ITEMSIZE(array) == 1) {
		qtype = NrmStringToQuark("character");
	} else {
		qtype = nio_type_from_code(PyArray_DTYPE(array)->type);
	}

	if (nfile->file.advanced_file_structure) {
		if (NULL != varnode) {
			if ((NCL_vlen == varnode->type) || (NCL_list == varnode->type)) {
				obj *listids = NULL;
				int curdim = 0;
				ng_usize_t counter = 0;

				qtype = NrmStringToQuark("list");

				if (array) {
					NclObjTypes the_obj_type = Ncl_Typelist;

					listids = (obj*) NclMalloc(
							(ng_usize_t) (nitems * sizeof(obj)));
					assert(listids);

					_NclBuildArrayOfList(listids, n_dims, dims);

					_convertObj2VLEN((PyObject*) array, listids,
							varnode->base_type, n_dims, curdim, &counter);

					md = _NclCreateVal(NULL, NULL,
							((the_obj_type & NCL_VAL_TYPE_MASK) ?
									Ncl_MultiDValData : the_obj_type), 0,
							listids, NULL, n_dims, dims, TEMPORARY, NULL,
							(NclObjClass) (
									(the_obj_type & NCL_VAL_TYPE_MASK) ?
											_NclTypeEnumToTypeClass(
													the_obj_type) :
											NULL));
				}
			} else if (NCL_compound == varnode->type) {
				obj *listids = NULL;
				int curdim = 0;
				ng_usize_t counter = 0;

				qtype = NrmStringToQuark("compound");

				if (array) {
					NclObjTypes the_obj_type = Ncl_Typelist;

					listids = (obj*) NclMalloc(
							(ng_usize_t) (nitems * sizeof(obj)));
					assert(listids);

					_NclBuildArrayOfList(listids, n_dims, dims);

					_convertObj2COMPOUND((PyObject*) array, listids,
							varnode->comprec, n_dims, curdim, &counter);

					md = _NclCreateVal(NULL, NULL,
							((the_obj_type & NCL_VAL_TYPE_MASK) ?
									Ncl_MultiDValData : the_obj_type), 0,
							listids, NULL, n_dims, dims, TEMPORARY, NULL,
							(NclObjClass) (
									(the_obj_type & NCL_VAL_TYPE_MASK) ?
											_NclTypeEnumToTypeClass(
													the_obj_type) :
											NULL));
				}
			} else if (NCL_string == varnode->type) {
				NrmQuark *qval;
				char *cptr = NULL;
				char *cval = NULL;
				ng_size_t n_items = 1;
				ng_size_t i;

				for (i = 0; i < n_dims; i++)
					n_items *= dims[i];

				qval = (NrmQuark*) NclCalloc(n_items, sizeof(NrmQuark));

				varnode->type = NCL_string;
				/*
				 qval[0] = NrmStringToQuark((char*)array->data);
				 */
				cptr = PyArray_BYTES(array);
				for (i = 0; i < n_items; i++) {
					cval = cptr + i * PyArray_ITEMSIZE(array);
					/*fprintf(stderr, "\tcval[%ld] = %s\n", i, cval);*/
					qval[i] = NrmStringToQuark(cval);
				}

				md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0, qval,
						NULL, n_dims, PyArray_NDIM(array) == 0 ? &scalar_size : dims,
						TEMPORARY, NULL, _NclNameToTypeClass(qtype));
			} else
				md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
						PyArray_DATA(array), NULL, n_dims,
						PyArray_NDIM(array) == 0 ? &scalar_size : dims, TEMPORARY, NULL,
						_NclNameToTypeClass(qtype));
		}
	} else
		md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
				PyArray_DATA(array), NULL, n_dims,
				PyArray_NDIM(array) == 0 ? &scalar_size : dims, TEMPORARY, NULL,
				_NclNameToTypeClass(qtype));
	if (nfile->file.advanced_file_structure)
		nfile = (NclFile) self->file->gnode;
	if (!md) {
		nret = NhlFATAL;
	} else if (select_all) {
		nret = _NclFileWriteVar(nfile, NrmStringToQuark(self->name), md, NULL);
	} else {
		NclSelectionRecord *sel_ptr;
		sel_ptr = (NclSelectionRecord*) malloc(sizeof(NclSelectionRecord));
		if (sel_ptr == NULL) {
			nret = NhlFATAL;
		} else {
			sel_ptr->n_entries = self->nd;
			for (i = 0; i < self->nd; i++) {
				sel_ptr->selection[i].sel_type = Ncl_SUBSCR;
				sel_ptr->selection[i].dim_num = i;
				sel_ptr->selection[i].u.sub.start = indices[i].start;
				sel_ptr->selection[i].u.sub.finish = indices[i].stop;
				if (sel_ptr->selection[i].u.sub.finish < 0)
					sel_ptr->selection[i].u.sub.finish = 0;
				sel_ptr->selection[i].u.sub.stride = indices[i].stride;
				sel_ptr->selection[i].u.sub.is_single = indices[i].item != 0;
			}
			nret = _NclFileWriteVar(nfile, NrmStringToQuark(self->name), md,
					sel_ptr);
			free(sel_ptr);
		}
	}
	Py_DECREF(array);
	if (nret < NhlWARNING)
		ret = -1;

	err_ret:

	/* update shape */
	(void) NioVariable_GetShape(self);
	if (dims != NULL)
		free(dims);
	if (dirs != NULL)
		free(dirs);
	if (indices != NULL)
		free(indices);
	return ret;
}

/* Not actually used */
#if 0
static int NioVariable_WriteString(NioVariableObject *self,
		PyObject *value) {
	long len;
	py3_char *value_char;


	if (self->type != NPY_CHAR || self->nd != 1) {
		PyErr_SetString(get_NioError(), "not a string variable");
		return -1;
	}
	/*len = PyString_Size((PyObject *) value);*/
	value_char = as_utf8_char_and_size(value, &len);
	if (len > self->dimensions[0]) {
		PyErr_SetString(PyExc_ValueError, "string too long");
		free(value_char);
		return -1;
	}
	if (self->dimensions[0] > len)
		len++;
	if (check_if_open(self->file, 1)) {
		NclFile nfile = (NclFile) self->file->id;
		NclMultiDValData md;
		NhlErrorTypes nret;
		ng_size_t str_dim_size = 1;

		NrmQuark qstr = NrmStringToQuark(value_char);
		free(value_char);
		define_mode(self->file, 0);
		md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
				(void*) &qstr, NULL, 1, &str_dim_size, TEMPORARY, NULL,
				_NclNameToTypeClass(NrmStringToQuark("string")));
		if (!md) {
			nret = NhlFATAL;
		} else {
			nret = _NclFileWriteVar(nfile, NrmStringToQuark(self->name), md,
					NULL);
		}
		if (nret < NhlWARNING) {
			nio_seterror(23);
			return -1;
		}
		return 0;
	} else {
		return -1;
	}
}
#endif

static PyObject *
NioVariableObject_item(NioVariableObject *self, Py_ssize_t i) {
	NioIndex *indices;
	if (self->nd == 0) {
		PyErr_SetString(PyExc_TypeError, "Not a sequence");
		return NULL;
	}
	if (i >= self->dimensions[0] || i < -self->dimensions[0]) {
		PyErr_Format(PyExc_IndexError,
				"index %d is out of bounds for axis 0 with size %ld", (int) i,
				self->dimensions[0]);
		return NULL;
	}

	indices = NioVariable_Indices(self);
	if (indices != NULL) {
		indices[0].start = i;
		indices[0].stop = i + 1;
		indices[0].item = 1;
		return PyArray_Return(NioVariable_ReadAsArray(self, indices));
	}
	return NULL;
}

static PyObject *
NioVariableObject_slice(NioVariableObject *self, Py_ssize_t low,
		Py_ssize_t high) {
	NioIndex *indices;
	if (self->nd == 0) {
		PyErr_SetString(PyExc_TypeError, "Not a sequence");
		return NULL;
	}
	indices = NioVariable_Indices(self);
	if (indices != NULL) {
		indices[0].start = low;
		indices[0].stop = high;
		return PyArray_Return(NioVariable_ReadAsArray(self, indices));
	}
	return NULL;
}

static PyObject *
NioVariableObject_subscript(NioVariableObject *self, PyObject *index) {
	NioIndex *indices;
	if (PyLong_Check(index)) {
		Py_ssize_t i = (Py_ssize_t) PyLong_AsLong(index);
		return NioVariableObject_item(self, i);
	}
#if PY_MAJOR_VERSION < 3
	else if (PyInt_Check(index)) {
		Py_ssize_t i = (Py_ssize_t) PyInt_AsLong(index);
		return NioVariableObject_item(self, i);
	}
#endif
	if (self->nd == 0) {
		PyErr_SetString(PyExc_TypeError, "Not a sequence");
		return NULL;
	}
	indices = NioVariable_Indices(self);
	if (indices != NULL) {
		if (PySlice_Check(index)) {
			Py_ssize_t slicelen;
			PySliceObject *slice = (PySliceObject *) index;
#if PY_MAJOR_VERSION < 3
			if (PySlice_GetIndicesEx((PySliceObject *) index,
#else
			if (PySlice_GetIndicesEx(index,
#endif
					self->dimensions[0], &indices->start, &indices->stop,
					&indices->stride, &slicelen) < 0) {
				PyErr_SetString(PyExc_TypeError, "error in subscript");
				free(indices);
				return NULL;
			}
			if (slice->start == Py_None)
				indices->no_start = 1;
			if (slice->stop == Py_None)
				indices->no_stop = 1;
			return PyArray_Return(NioVariable_ReadAsArray(self, indices));
		}
		if (PyTuple_Check(index)) {
			int ni = PyTuple_Size(index);
			if (ni <= self->nd) {
				int d;
				Py_ssize_t i;
				d = 0;
				for (i = 0; i < ni; i++) {
					PyObject *subscript = PyTuple_GetItem(index, i);
					if (PyLong_Check(subscript)) {
						Py_ssize_t n = (Py_ssize_t) PyLong_AsLong(subscript);
						if (n >= self->dimensions[d]
								|| n < -self->dimensions[d]) {
							PyErr_Format(PyExc_IndexError,
									"index %d is out of bounds for axis %d with size %ld",
									(int) n, d, self->dimensions[d]);
							free(indices);
							return NULL;
						}
						indices[d].start = n;
						indices[d].stop = n + 1;
						indices[d].item = 1;
						d++;
					}
#if PY_MAJOR_VERSION < 3
					else if (PyInt_Check(subscript)) {
						Py_ssize_t n = (Py_ssize_t) PyInt_AsLong(subscript);
						if (n >= self->dimensions[d]
								|| n < -self->dimensions[d]) {
							PyErr_Format(PyExc_IndexError,
									"index %d is out of bounds for axis %d with size %ld",
									(int) n, d, self->dimensions[d]);
							free(indices);
							return NULL;
						}
						indices[d].start = n;
						indices[d].stop = n + 1;
						indices[d].item = 1;
						d++;
				}
#endif
					else if (PySlice_Check(subscript)) {
						Py_ssize_t slicelen;
						PySliceObject *slice = (PySliceObject *) subscript;
#if PY_MAJOR_VERSION < 3
						if (PySlice_GetIndicesEx((PySliceObject *) subscript,
#else
						if (PySlice_GetIndicesEx(subscript,
#endif
								self->dimensions[d], &indices[d].start,
								&indices[d].stop, &indices[d].stride, &slicelen)
								< 0) {
							PyErr_SetString(PyExc_TypeError,
									"error in subscript");
							free(indices);
							return NULL;
						}
						if (slice->start == Py_None)
							indices[d].no_start = 1;
						if (slice->stop == Py_None)
							indices[d].no_stop = 1;
						d++;
					} else if (subscript == Py_Ellipsis) {
						d = self->nd - ni + i + 1;
					} else {
						PyErr_SetString(PyExc_TypeError,
								"illegal subscript type");
						free(indices);
						return NULL;
					}
				}
				return PyArray_Return(NioVariable_ReadAsArray(self, indices));
			} else {
				PyErr_SetString(PyExc_IndexError, "too many subscripts");
				free(indices);
				return NULL;
			}
		}
		PyErr_SetString(PyExc_TypeError, "illegal subscript type");
		free(indices);
	}
	return NULL;
}

static int NioVariableObject_ass_item(NioVariableObject *self, Py_ssize_t i,
		PyObject *value) {
	NioIndex *indices;
	if (value == NULL) {
		PyErr_SetString(PyExc_ValueError, "Can't delete elements.");
		return -1;
	}
	if (self->nd == 0) {
		PyErr_SetString(PyExc_TypeError, "Not a sequence");
		return -1;
	}
	indices = NioVariable_Indices(self);
	if (indices != NULL) {
		indices[0].start = i;
		indices[0].stop = i + 1;
		indices[0].item = 1;
		return NioVariable_WriteArray(self, indices, value);
	}
	return -1;
}

static int NioVariableObject_ass_slice(NioVariableObject *self, Py_ssize_t low,
		Py_ssize_t high, PyObject *value) {
	NioIndex *indices;
	if (value == NULL) {
		PyErr_SetString(PyExc_ValueError, "Can't delete elements.");
		return -1;
	}
	if (self->nd == 0) {
		PyErr_SetString(PyExc_TypeError, "Not a sequence");
		return -1;
	}
	indices = NioVariable_Indices(self);
	if (indices != NULL) {
		indices[0].stop = high;
		indices[0].start = low;
		/* FIXME:  This is an overflow expression */
		if (indices[0].unlimited) {
			if (indices->start < PY_SSIZE_T_MIN * 100)
				indices->no_start = 1;
			if (indices->stop > PY_SSIZE_T_MAX / 100)
				indices->no_stop = 1;
		}
		return NioVariable_WriteArray(self, indices, value);
	}
	return -1;
}

static int NioVariableObject_ass_subscript(NioVariableObject *self,
		PyObject *index, PyObject *value) {
	NioIndex *indices;

	if (PyLong_Check(index)) {
		Py_ssize_t i = (Py_ssize_t) PyLong_AsLong(index);
		return NioVariableObject_ass_item(self, i, value);
	}
#if PY_MAJOR_VERSION < 3
	else if (PyInt_Check(index)) {
		Py_ssize_t i = (Py_ssize_t) PyInt_AsLong(index);
		return NioVariableObject_ass_item(self, i, value);
	}
#endif
	if (value == NULL) {
		PyErr_SetString(PyExc_ValueError, "Can't delete elements.");
		return -1;
	}
	if (self->nd == 0) {
		PyErr_SetString(PyExc_TypeError, "Not a sequence");
		return -1;
	}
	indices = NioVariable_Indices(self);
	if (indices != NULL) {
		if (PySlice_Check(index)) {
			Py_ssize_t slicelen;
			PySliceObject *slice = (PySliceObject *) index;
#if PY_MAJOR_VERSION < 3
			if (PySlice_GetIndicesEx((PySliceObject *) index,
#else
			if (PySlice_GetIndicesEx(index,
#endif
					self->dimensions[0], &indices->start, &indices->stop,
					&indices->stride, &slicelen) < 0) {
				PyErr_SetString(PyExc_TypeError, "error in subscript");
				free(indices);
				return -1;
			}
			/* Python slicing truncates to the closest valid index -- but we don't want that when the
			 * dimension is unlimited. So here we revert to the raw slice values.
			 */
			if (slice->start == Py_None)
				indices->no_start = 1;
			else if (indices->unlimited) {
				if (PyLong_Check(slice->start)) {
					indices->start = PyLong_AsLong(slice->start);
				}
#if PY_MAJOR_VERSION < 3
				else if (PyInt_Check(slice->start)) {
					indices->start = PyInt_AsLong(slice->start);
				}
#endif
				if (indices->start < PY_SSIZE_T_MIN * 100)
					indices->no_start = 1;
			}
			if (slice->stop == Py_None)
				indices->no_stop = 1;
			else if (indices->unlimited) {
				if (PyLong_Check(slice->stop)) {
					indices->stop = PyLong_AsLong(slice->stop);
				}
#if PY_MAJOR_VERSION < 3
				if (PyInt_Check(slice->stop)) {
					indices->stop = PyInt_AsLong(slice->stop);
				}
#endif
				if (indices->stop > PY_SSIZE_T_MAX / 100)
					indices->no_stop = 1;
			}
			return NioVariable_WriteArray(self, indices, value);
		}
		if (PyTuple_Check(index)) {
			int ni = PyTuple_Size(index);
			if (ni <= self->nd) {
				int i, d;
				d = 0;
				for (i = 0; i < ni; i++) {
					PyObject *subscript = PyTuple_GetItem(index, i);
					if (PyLong_Check(subscript)) {
						Py_ssize_t n = (Py_ssize_t) PyLong_AsLong(subscript);
						indices[d].start = n;
						indices[d].stop = n + 1;
						indices[d].item = 1;
						d++;
					}
#if PY_MAJOR_VERSION < 3
					else if (PyInt_Check(subscript)) {
						Py_ssize_t n = (Py_ssize_t) PyInt_AsLong(subscript);
						indices[d].start = n;
						indices[d].stop = n + 1;
						indices[d].item = 1;
						d++;
					}
#endif
					else if (PySlice_Check(subscript)) {
						Py_ssize_t slicelen;
						PySliceObject *slice = (PySliceObject *) subscript;
#if PY_MAJOR_VERSION < 3
						if (PySlice_GetIndicesEx((PySliceObject *) subscript,
#else
						if (PySlice_GetIndicesEx(subscript,
#endif
								self->dimensions[d], &indices[d].start,
								&indices[d].stop, &indices[d].stride, &slicelen)
								< 0) {
							PyErr_SetString(PyExc_TypeError,
									"error in subscript");
							free(indices);
							return -1;
						}
						/* Python slicing truncates to the closest valid index -- but we don't want that when the
						 * dimension is unlimited. So here we revert to the raw slice values.
						 */
						if (slice->start == Py_None)
							indices[d].no_start = 1;
						else if (indices[d].unlimited) {
							if (PyLong_Check(slice->start)) {
								indices[d].start = PyLong_AsLong(slice->start);
							}
#if PY_MAJOR_VERSION < 3
							if (PyInt_Check(slice->start)) {
								indices[d].start = PyInt_AsLong(slice->start);
							}
#endif
						}
						if (slice->stop == Py_None)
							indices[d].no_stop = 1;
						else if (indices[d].unlimited) {
							if (PyLong_Check(slice->stop)) {
								indices[d].stop = PyLong_AsLong(slice->stop);
							}
#if PY_MAJOR_VERSION < 3
							else if (PyInt_Check(slice->stop)) {
								indices[d].stop = PyInt_AsLong(slice->stop);
							}
#endif
						}
						d++;
					} else if (subscript == Py_Ellipsis) {
						d = self->nd - ni + i + 1;
					} else {
						PyErr_SetString(PyExc_TypeError,
								"illegal subscript type");
						free(indices);
						return -1;
					}
				}
				return NioVariable_WriteArray(self, indices, value);
			} else {
				PyErr_SetString(PyExc_IndexError, "too many subscripts");
				free(indices);
				return -1;
			}
		}
		PyErr_SetString(PyExc_TypeError, "illegal subscript type");
		free(indices);
	}
	return -1;
}

/* Type definition */

static PyObject *
NioVariableObject_error1(NioVariableObject *self, NioVariableObject *other) {
	PyErr_SetString(PyExc_TypeError, "can't concatenate NIO variables");
	return NULL;
}

static PyObject *
NioVariableObject_error2(NioVariableObject *self, Py_ssize_t n) {
	PyErr_SetString(PyExc_TypeError, "can't repeat NIO variables");
	return NULL;
}

#if PY_VERSION_HEX < 0x02050000

static PySequenceMethods NioVariableObject_as_sequence = {
		(inquiry) NioVariableObject_length, /*sq_length*/
		(binaryfunc) NioVariableObject_error1, /*sq_concat*/
		(intargfunc) NioVariableObject_error2, /*sq_repeat*/
		(intargfunc) NioVariableObject_item, /*sq_item*/
		(intintargfunc) NioVariableObject_slice, /*sq_slice*/
		(intobjargproc) NioVariableObject_ass_item, /*sq_ass_item*/
		(intintobjargproc) NioVariableObject_ass_slice, /*sq_ass_slice*/
};

static PyMappingMethods NioVariableObject_as_mapping = {
		(inquiry) NioVariableObject_length, /*mp_length*/
		(binaryfunc) NioVariableObject_subscript, /*mp_subscript*/
		(objobjargproc) NioVariableObject_ass_subscript, /*mp_ass_subscript*/
};

#else

static PySequenceMethods NioVariableObject_as_sequence =
{
	(lenfunc)NioVariableObject_length, /*sq_length*/
	(binaryfunc)NioVariableObject_error1, /*sq_concat*/
	(ssizeargfunc)NioVariableObject_error2, /*sq_repeat*/
	(ssizeargfunc)NioVariableObject_item, /*sq_item*/
	(ssizessizeargfunc)NioVariableObject_slice, /*sq_slice*/
	(ssizeobjargproc)NioVariableObject_ass_item, /*sq_ass_item*/
	(ssizessizeobjargproc)NioVariableObject_ass_slice, /*sq_ass_slice*/
};

static PyMappingMethods NioVariableObject_as_mapping =
{
	(lenfunc)NioVariableObject_length, /*mp_length*/
	(binaryfunc)NioVariableObject_subscript, /*mp_subscript*/
	(objobjargproc)NioVariableObject_ass_subscript, /*mp_ass_subscript*/
};

#endif

void printval(char *buf, NclBasicDataTypes type, void *val) {
	switch (type) {
	case NCL_double:
		sprintf(buf, "%4.16g", *(double *) val);
		return;
	case NCL_float:
		sprintf(buf, "%2.7g", *(float *) val);
		return;
	case NCL_int64:
		sprintf(buf, "%lld", *(long long *) val);
		return;
	case NCL_uint64:
		sprintf(buf, "%llu", *(unsigned long long *) val);
		return;
	case NCL_long:
		sprintf(buf, "%ld", *(long *) val);
		return;
	case NCL_ulong:
		sprintf(buf, "%lu", *(unsigned long *) val);
		return;
	case NCL_int:
		sprintf(buf, "%d", *(int *) val);
		return;
	case NCL_uint:
		sprintf(buf, "%u", *(unsigned int *) val);
		return;
	case NCL_short:
		sprintf(buf, "%d", (int) *(short *) val);
		return;
	case NCL_ushort:
		sprintf(buf, "%d", (int) *(unsigned short *) val);
		return;
	case NCL_string:
		strncat(buf, NrmQuarkToString(*(NrmQuark *) val), 1024);
		return;
	case NCL_char:
		sprintf(buf, "%c", *(char *) val);
		return;
	case NCL_byte:
		sprintf(buf, "%d", (int) *(char *) val);
		return;
	case NCL_ubyte:
		sprintf(buf, "%d", (int) *(unsigned char *) val);
		return;
	case NCL_logical:
		if (*(logical *) val == 0) {
			sprintf(buf, "False");
		} else {
			sprintf(buf, "True");
		}
		return;
	default:
		sprintf(buf, "-");
		return;
	}

}

/* Printed representation */
static PyObject *
NioVariableObject_str(NioVariableObject *var) {
	char tbuf[1024];
	char *buf;
	int len;
	int bufinc = 4096;
	int buflen = 0;
	int bufpos = 0;
	PyObject *pystr;
	NioFileObject *file = var->file;
	NclFile nfile = (NclFile) file->id;
	int i;
	NrmQuark varq = NrmStringToQuark(var->name);
	NclFileAttInfoList* step;
	char *vname = NULL;
	NioVariableObject *vobj;
	long long total;
	int j, dim_ix;
	int n_atts;
	NrmQuark qncl_scalar;

	if (!check_if_open(file, -1)) {
		PyErr_SetString(get_NioError(),
				"file has been closed: variable no longer valid");
		return NULL;
	}

	qncl_scalar = NrmStringToQuark("ncl_scalar");
	if (nfile->file.advanced_file_structure) {
		NclFileVarNode* varnode;
		NclAdvancedFile advfile = (NclAdvancedFile) file->id;

		varnode = getVarFromGroup(advfile->advancedfile.grpnode, varq);
		if (NULL == varnode) {
			PyErr_SetString(get_NioError(), "variable not found");
			return NULL;
		}
		buf = NioVarInfo2str(var, varnode);
		pystr = char_to_pystr(buf, strlen(buf));
		free(buf);
		return pystr;
	}
	buf = malloc(bufinc);
	buflen = bufinc;
	for (i = 0; i < nfile->file.n_vars; i++) {
		if (nfile->file.var_info[i]->var_name_quark != varq) {
			continue;
		}
		vname = NrmQuarkToString(nfile->file.var_info[i]->var_name_quark);
		break;
	}
	if (!vname) {
		PyErr_SetString(get_NioError(), "variable not found");
		return NULL;
	}
	vobj = (NioVariableObject *) PyDict_GetItemString(file->variables, vname);
	sprintf(tbuf, "Variable: %s\n", vname);
	BUF_INSERT(tbuf);
	total = 1;
	for (j = 0; j < nfile->file.var_info[i]->num_dimensions; j++) {
		total *=
				nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[j]]->dim_size;
	}
	sprintf(tbuf, "Type: %s\n",
			_NclBasicDataTypeToName(nfile->file.var_info[i]->data_type));
	BUF_INSERT(tbuf);
	sprintf(tbuf, "Total Size: %lld bytes\n",
			total * _NclSizeOf(nfile->file.var_info[i]->data_type));
	BUF_INSERT(tbuf);
	sprintf(tbuf, "            %lld values\n", total);
	BUF_INSERT(tbuf);
	if (nfile->file.var_info[i]->num_dimensions == 1
			&& nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[0]]->dim_name_quark
					== qncl_scalar) {
		sprintf(tbuf, "Number of Dimensions: %d\n", 0);
		BUF_INSERT(tbuf);
	} else {
		sprintf(tbuf, "Number of Dimensions: %d\n",
				nfile->file.var_info[i]->num_dimensions);
		BUF_INSERT(tbuf);
		sprintf(tbuf, "Dimensions and sizes:\t");
		BUF_INSERT(tbuf);
		for (j = 0; j < nfile->file.var_info[i]->num_dimensions; j++) {
			sprintf(tbuf, "[");
			BUF_INSERT(tbuf);
			sprintf(tbuf, "%s | ",
					NrmQuarkToString(
							nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[j]]->dim_name_quark));
			BUF_INSERT(tbuf);
			sprintf(tbuf, "%lld]",
					(long long) nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[j]]->dim_size);
			BUF_INSERT(tbuf);
			if (j != nfile->file.var_info[i]->num_dimensions - 1) {
				sprintf(tbuf, " x ");
				BUF_INSERT(tbuf);
			}
		}
		sprintf(tbuf, "\nCoordinates: \n");
		BUF_INSERT(tbuf);
		for (j = 0; j < nfile->file.var_info[i]->num_dimensions; j++) {
			NclVar tmp_var;
			dim_ix = nfile->file.var_info[i]->file_dim_num[j];
			if (_NclFileVarIsCoord(nfile,
					nfile->file.file_dim_info[dim_ix]->dim_name_quark) != -1) {
				sprintf(tbuf, "            ");
				BUF_INSERT(tbuf);
				sprintf(tbuf, "%s: [",
						NrmQuarkToString(
								nfile->file.file_dim_info[dim_ix]->dim_name_quark));
				BUF_INSERT(tbuf);
				tmp_var = _NclFileReadCoord(nfile,
						nfile->file.file_dim_info[dim_ix]->dim_name_quark,
						NULL);
				if (tmp_var != NULL) {
					NclMultiDValData tmp_md;

					tmp_md = (NclMultiDValData) _NclGetObj(
							tmp_var->var.thevalue_id);
					printval(tbuf, tmp_md->multidval.type->type_class.data_type,
							tmp_md->multidval.val);
					BUF_INSERT(tbuf);
					sprintf(tbuf, "..");
					BUF_INSERT(tbuf);
					printval(tbuf, tmp_md->multidval.type->type_class.data_type,
							(char *) tmp_md->multidval.val
									+ (tmp_md->multidval.totalelements - 1)
											* tmp_md->multidval.type->type_class.size);
					BUF_INSERT(tbuf);
					sprintf(tbuf, "]\n");
					BUF_INSERT(tbuf);
					if (tmp_var->obj.status != PERMANENT) {
						_NclDestroyObj((NclObj) tmp_var);
					}

				}
			} else {
				sprintf(tbuf, "            ");
				BUF_INSERT(tbuf);
				sprintf(tbuf, "%s: not a coordinate variable\n",
						NrmQuarkToString(
								nfile->file.file_dim_info[dim_ix]->dim_name_quark));
				BUF_INSERT(tbuf);
			}
		}
	}
	step = nfile->file.var_att_info[i];
	n_atts = 0;
	while (step != NULL) {
		n_atts++;
		step = step->next;
	}
	sprintf(tbuf, "Number of Attributes: %d\n", n_atts);
	BUF_INSERT(tbuf);
	step = nfile->file.var_att_info[i];
	while (step != NULL) {
		PyObject *att_val;
		char *aname = NrmQuarkToString(step->the_att->att_name_quark);
		sprintf(tbuf, "         %s :\t", aname);
		BUF_INSERT(tbuf);
		att_val = PyDict_GetItemString(vobj->attributes, aname);
		if (is_string_type(att_val)) {

			py3_char *att_val_char = as_utf8_char(att_val);

			BUF_INSERT(att_val_char);
			BUF_INSERT("\n");
			free(att_val_char);
		} else {
			int k;
			PyArrayObject *att_arr_val = (PyArrayObject *) att_val;
			if (PyArray_NDIM(att_arr_val) == 0 || PyArray_DIM(att_arr_val, 0) == 1) {
				PyObject *att = PyArray_GETITEM(att_arr_val, PyArray_DATA(att_arr_val));
				format_object(tbuf, att, PyArray_DTYPE(att_arr_val)->type);
				BUF_INSERT(tbuf);
				BUF_INSERT("\n");
				/*sprintf(tbuf,"%s\n",PyUnicode_AsUTF8(PyObject_Str(att)));*/
			} else {
				sprintf(tbuf, "[");
				BUF_INSERT(tbuf);
				for (k = 0; k < PyArray_DIM(att_arr_val, 0); k++) {
					PyObject *att = PyArray_GETITEM(att_arr_val, PyArray_BYTES(att_arr_val) + k * PyArray_ITEMSIZE(att_arr_val));
					format_object(tbuf, att, PyArray_DTYPE(att_arr_val)->type);
					/*sprintf(tbuf,"%s",PyUnicode_AsUTF8(PyObject_Str(att)));*/
					BUF_INSERT(tbuf);
					if (k < PyArray_DIM(att_arr_val, 0) - 1) {
						sprintf(tbuf, ", ");
					} else {
						sprintf(tbuf, "]\n");
					}
					BUF_INSERT(tbuf);
				}
			}
		}
		step = step->next;
	}
	pystr = char_to_pystr(buf, strlen(buf));
	free(buf);
	return pystr;
}

#if PY_MAJOR_VERSION < 3
static PyTypeObject NioVariable_Type =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	"_nio._NioVariable", /*tp_name*/
	sizeof(NioVariableObject), /*tp_basicsize*/
	0, /*tp_itemsize*/
	/* methods */
	(destructor)NioVariableObject_dealloc, /*tp_dealloc*/
	0, /*tp_print*/
	(getattrfunc)NioVariable_GetAttribute, /*tp_getattr*/
	(setattrfunc)NioVariable_SetAttribute, /*tp_setattr*/
	0, /*tp_compare*/
	0, /*tp_repr*/
	0, /*tp_as_number*/
	&NioVariableObject_as_sequence, /*tp_as_sequence*/
	&NioVariableObject_as_mapping, /*tp_as_mapping*/
	0, /*tp_hash*/
	0, /*tp_call*/
	(reprfunc)NioVariableObject_str, /*tp_str*/
	(getattrofunc)0, /*tp_getattro*/
	(setattrofunc)0, /*tp_setattro*/
	0, /*tp_as_buffer*/
	(Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE), /*tp_flags*/
	0, /*tp_doc*/
	(traverseproc)0, /*tp_traverse */
	(inquiry)0, /*tp_clear */
	(richcmpfunc)0, /*tp_richcompare */
	0, /*tp_weaklistoffset */

	/* Iterator support (use standard) */

	(getiterfunc) 0, /* tp_iter */
	(iternextfunc)0, /* tp_iternext */

	/* Sub-classing (new-style object) support */

	NioVariableObject_methods, /* tp_methods */
	0, /* tp_members */
	0, /* tp_getset */
	0, /* tp_base */
	0, /* tp_dict */
	0, /* tp_descr_get */
	0, /* tp_descr_set */
	0, /* tp_dictoffset */
	(initproc)0, /* tp_init */
	0, /* tp_alloc */
	0, /* tp_new */
	0, /* tp_free */
	0, /* tp_is_gc */
	0, /* tp_bases */
	0, /* tp_mro */
	0, /* tp_cache */
	0, /* tp_subclasses */
	0 /* tp_weaklist */
};
#else

#if 0
static PySequenceMethods NioVariableObject_as_sequence =
{
	(lenfunc)NioVariableObject_length, /*sq_length*/
	(binaryfunc)NioVariableObject_error1, /*sq_concat*/
	(ssizeargfunc)NioVariableObject_error2, /*sq_repeat*/
	(ssizeargfunc)NioVariableObject_item, /*sq_item*/
	(ssizessizeargfunc)NioVariableObject_slice, /*sq_slice*/
	(ssizeobjargproc)NioVariableObject_ass_item, /*sq_ass_item*/
	(ssizessizeobjargproc)NioVariableObject_ass_slice, /*sq_ass_slice*/
};

static PyMappingMethods NioVariableObject_as_mapping =
{
	(lenfunc)NioVariableObject_length, /*mp_length*/
	(binaryfunc)NioVariableObject_subscript, /*mp_subscript*/
	(objobjargproc)NioVariableObject_ass_subscript, /*mp_ass_subscript*/
};
#endif

static PyType_Slot niovar_slots[] = {
		{ Py_tp_dealloc, (destructor)NioVariableObject_dealloc },
		{ Py_tp_getattr, (getattrfunc)NioVariable_GetAttribute },
		{ Py_tp_setattr, (setattrfunc)NioVariable_SetAttribute },
		{ Py_tp_str, (reprfunc)NioVariableObject_str },
		{ Py_tp_methods, NioVariableObject_methods },
		{ Py_mp_length, (lenfunc)NioVariableObject_length },
		{ Py_mp_subscript, (binaryfunc)NioVariableObject_subscript },
		{ Py_mp_ass_subscript, (objobjargproc)NioVariableObject_ass_subscript },
		{ Py_sq_length, (lenfunc)NioVariableObject_length },
		{ Py_sq_concat, (binaryfunc)NioVariableObject_error1 },
		{ Py_sq_repeat, (ssizeargfunc)NioVariableObject_error2 },
		{ Py_sq_item, (ssizeargfunc)NioVariableObject_item },
		{ Py_sq_ass_item, (ssizeobjargproc)NioVariableObject_ass_item },
		{ 0 },
	};

static PyType_Spec niovar_spec = { "_nio._NioVariable", sizeof(NioVariableObject), 0,
		  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, niovar_slots };

#endif

static NrmQuark GetExtension(char * filename) {
	struct stat statbuf;
	char *cp;
	if (!strncmp(filename, "http://", 7)) {
		/* OPeNDAP files are handled by the NetCDF module */
		return NrmStringToQuark("nc");
	}
	cp = strrchr(filename, '.');
	if (cp == NULL || *(cp + 1) == '\0')
		return NrmNULLQUARK;

	/* for now only regular files are accepted */
	if (stat(filename, &statbuf) >= 0) {
		if (!(S_ISREG(statbuf.st_mode)))
			return NrmNULLQUARK;
	}
	return NrmStringToQuark(cp + 1);
}

void InitializeNioOptions(NrmQuark extq, int mode) {
	NclMultiDValData md;
	ng_size_t len_dims = 1;
	logical *lval;
	NrmQuark *qval;
	NrmQuark qnc = NrmStringToQuark("nc");
	NrmQuark qgrb = NrmStringToQuark("grb");

	if (_NclFormatEqual(extq, qnc)) {
		_NclFileSetOptionDefaults(qnc, NrmNULLQUARK);

		if (mode < 1) {
			/* if opened for writing default "definemode" to True */
			lval = (logical *) malloc(sizeof(logical));
			*lval = True;
			md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
					(void*) lval, NULL, 1, &len_dims, TEMPORARY, NULL,
					(NclTypeClass) nclTypelogicalClass);
			_NclFileSetOption(NULL, extq, NrmStringToQuark("definemode"), md);
			_NclDestroyObj((NclObj) md);
		}
		/* default "suppressclose" to True for files opened in all modes */
		lval = malloc(sizeof(logical));
		*lval = True;
		md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0, (void*) lval,
				NULL, 1, &len_dims, TEMPORARY, NULL,
				(NclTypeClass) nclTypelogicalClass);
		_NclFileSetOption(NULL, extq, NrmStringToQuark("suppressclose"), md);
		_NclDestroyObj((NclObj) md);
	} else if (_NclFormatEqual(extq, qgrb)) {
		_NclFileSetOptionDefaults(qgrb, NrmNULLQUARK);
		qval = (NrmQuark *) malloc(sizeof(NrmQuark));
		*qval = NrmStringToQuark("numeric");
		md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0, (void*) qval,
				NULL, 1, &len_dims, TEMPORARY, NULL,
				(NclTypeClass) nclTypestringClass);
		_NclFileSetOption(NULL, extq,
				NrmStringToQuark("initialtimecoordinatetype"), md);
		_NclDestroyObj((NclObj) md);
	}
}

void SetNioOptions(NrmQuark extq, int mode, PyObject *options,
		PyObject *option_defaults) {
	NclMultiDValData md = NULL;
	PyObject *keys = PyMapping_Keys(options);
	ng_size_t len_dims = 1;
	Py_ssize_t i;
	NrmQuark qsafe_mode = NrmStringToQuark("safemode");

	/*NrmQuark qnc = NrmStringToQuark("nc"); Not used...*/

	for (i = 0; i < PySequence_Length(keys); i++) {

		PyObject *key = PySequence_GetItem(keys, i);
		py3_char *keystr = as_utf8_char(key);
		NrmQuark qkeystr = NrmStringToQuark(keystr);
		PyObject *value = PyMapping_GetItemString(options, keystr);

		/* handle the PyNIO-defined "SafeMode" option */
		if (_NclFormatEqual(extq, NrmStringToQuark("nc"))
				&& (_NclGetLower(qkeystr) == qsafe_mode)) {
			/* 
			 * SafeMode == True:
			 *    DefineMode = False (if writable file)
			 *    SuppressClose = False
			 * SafeMode == False:
			 *    DefineMode = True (if writable file)
			 *    SuppressClose = True
			 */

			logical *lval;

			if (!PyBool_Check(value)) {
				char s[256] = "";
				strncat(s, keystr, 255);
				strncat(s, " value is an invalid type for option", 255);
				free(keystr);
				PyErr_SetString(get_NioError(), s);
				Py_DECREF(key);
				Py_DECREF(value);
				continue;
			}
			lval = (logical *) malloc(sizeof(logical));
			/* Note opposite */
			if (PyObject_RichCompareBool(value, Py_False, Py_EQ)) {
				*lval = True;
				/* printf("%s False\n",keystr);*/
			} else {
				*lval = False;
				/*printf("%s True\n",keystr); */
			}
			md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
					(void*) lval, NULL, 1, &len_dims, TEMPORARY, NULL,
					(NclTypeClass) nclTypelogicalClass);
			if (mode < 1)
				_NclFileSetOption(NULL, extq, NrmStringToQuark("DefineMode"),
						md);
			_NclFileSetOption(NULL, extq, NrmStringToQuark("SuppressClose"),
					md);
			_NclDestroyObj((NclObj) md);
			Py_DECREF(key);
			Py_DECREF(value);
			continue;
		} else if (!_NclFileIsOption(extq, qkeystr)) {
			if (options == option_defaults) {
				/* 
				 * this means we're setting the option defaults; more permissive:
				 * options for all file formats may be present
				 */
				continue;
			}
			if (PyDict_Contains(option_defaults, key)
					&& !_NclFileIsOption(NrmNULLQUARK, qkeystr)) {
				/* 
				 * Options that are user-defined are passed through, but not built-in options that
				 *  apply only to another file format
				 */
				continue;
			}
			char s[256] = "";
			strncat(s, keystr, 255);
			strncat(s, " is not a valid option for this file format", 255);
			PyErr_SetString(get_NioError(), s);
			Py_DECREF(key);
			continue;
		}
		if (is_string_type(value)) {
			py3_char *valstr = as_utf8_char(value);
			NrmQuark *qval = (NrmQuark *) malloc(sizeof(NrmQuark));
			*qval = NrmStringToQuark(valstr);
			md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
					(void*) qval, NULL, 1, &len_dims, TEMPORARY, NULL,
					(NclTypeClass) nclTypestringClass);
			free(valstr);
			/* printf("%s %s\n", keystr,valstr); */
		} else if (PyBool_Check(value)) {
			logical * lval = (logical *) malloc(sizeof(logical));
			if (PyObject_RichCompareBool(value, Py_False, Py_EQ)) {
				*lval = False;
				/* printf("%s False\n",keystr);*/
			} else {
				*lval = True;
				/*printf("%s True\n",keystr); */
			}
			md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
					(void*) lval, NULL, 1, &len_dims, TEMPORARY, NULL,
					(NclTypeClass) nclTypelogicalClass);
		} else if (PyLong_Check(value)) {
			int* ival = (int *) malloc(sizeof(int));
			*ival = (int) PyLong_AsLong(value);
			md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
					(void*) ival, NULL, 1, &len_dims, TEMPORARY, NULL,
					(NclTypeClass) nclTypeintClass);
			/* printf("%s %ld\n",keystr,PyLong_AsLong(value));*/
		}
#if PY_MAJOR_VERSION < 3
		else if (PyInt_Check(value)) {
			int* ival = (int *) malloc(sizeof(int));
			*ival = (int) PyInt_AsLong(value);
			md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
					(void*) ival, NULL, 1, &len_dims, TEMPORARY, NULL,
					(NclTypeClass) nclTypeintClass);
			/* printf("%s %ld\n",keystr,PyLong_AsLong(value));*/
		}
#endif
		else if (PyFloat_Check(value)) {
			float *fval = (float *) malloc(sizeof(float));
			*fval = (float) PyFloat_AsDouble(value);
			md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0,
					(void*) fval, NULL, 1, &len_dims, TEMPORARY, NULL,
					(NclTypeClass) nclTypefloatClass);
			/*printf("%s %lg\n",keystr,PyFloat_AsDouble(value));*/
		} else {
			char s[256] = "";
			strncat(s, keystr, 255);
			strncat(s, " value is an invalid type for option", 255);
			PyErr_SetString(get_NioError(), s);
			Py_DECREF(key);
			Py_DECREF(value);
			continue;
		}
		_NclFileSetOption(NULL, extq, qkeystr, md);
		_NclDestroyObj((NclObj) md);
		free(keystr);
		Py_DECREF(key);
		Py_DECREF(value);
	}
	Py_DECREF(keys);

	if (options == option_defaults && !PyDict_Contains(options, char_to_pystr("FileStructure", strlen("FileStructure")))) {
		NrmQuark *qval;
		qval = (NrmQuark *) malloc(sizeof(NrmQuark));
		if (_NclFormatEqual(extq, NrmStringToQuark("h5"))
				|| _NclFormatEqual(extq, NrmStringToQuark("he5"))) {
			*qval = NrmStringToQuark("advanced");
		} else {
			*qval = NrmStringToQuark("standard");
		}
		md = _NclCreateMultiDVal(NULL, NULL, Ncl_MultiDValData, 0, (void*) qval,
				NULL, 1, &len_dims, TEMPORARY, NULL,
				(NclTypeClass) nclTypestringClass);
		_NclFileSetOption(NULL, extq, NrmStringToQuark("FileStructure"), md);
	}
}

/* Creator for NioFile objects */

static PyObject*
NioFile(PyObject *self, PyObject *args, PyObject *kwds) {
	char *filepath;
	char *mode = "r";
	char *history = "";
	char *format = "";
	PyObject *options = Py_None;
	NioFileObject *file;
	NrmQuark extq;
	int crw;
	static char *argnames[] = { "filepath", "mode", "options", "history",
			"format", NULL };
	PyObject *option_defaults;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|sOss:open_file", argnames,
			&filepath, &mode, &options, &history, &format))
		return NULL;

	if (strlen(format) > 0 && strlen(format) < 16 && strlen(filepath) > 0) {
		char *tfile = filepath;
		filepath = malloc(strlen(filepath) + strlen(format) + 2);
		sprintf(filepath, "%s.%s", tfile, format);
	}
	extq = GetExtension(filepath);


	if (extq == NrmNULLQUARK) {
		PyErr_SetString(get_NioError(), "invalid extension or invalid file type");
		return NULL;
	}

	crw = GetNioMode(filepath, mode);
	if (crw < -1) {
		nio_seterror(25);
		return NULL;
	}
	/*
	 * Three step to handle options
	 * First set the hard-coded library defaults -- including a few module-specific values.
	 * Then set the option defaults (which are modifiable by the user and allow for user-specified keys)
	 * Finally set the options explicitly passed for this instance.
	 */

	InitializeNioOptions(extq, crw);

	option_defaults = PyObject_GetAttrString(get_Niomodule(), "option_defaults");

	SetNioOptions(extq, crw, option_defaults, option_defaults);

	if (options != Py_None) {
		PyObject *options_dict;
		if (!PyObject_HasAttrString(options, "__dict__")) {
			PyErr_SetString(get_NioError(),
					"options argument must be an NioOptions class instance");
		}
		options_dict = PyObject_GetAttrString(options, "__dict__");

		SetNioOptions(extq, crw, options_dict, option_defaults);

		Py_DECREF(options_dict);
	}
	Py_DECREF(option_defaults);

	file = NioFile_Open(filepath, mode);
	if (file == NULL) {
		nio_seterror(33); /* setting unknown error here */
		return NULL;
	}
	if (strlen(history) > 0) {
		if (check_if_open(file, 1)) {
			NioFile_AddHistoryLine(file, history);
		}
	}

	return (PyObject*) file;
}

static PyObject *
NioFile_Options(PyObject *self, PyObject *args) {

	PyObject *result = NULL;
	PyObject *class = NULL;
	PyObject *dict = PyDict_New();
	PyObject *bases = PyTuple_New(0);

	PyObject *pystr = char_to_pystr("NioOptions", strlen("NioOptions"));
	PyObject *modstr = char_to_pystr("__module__", strlen("__module__"));
	PyObject *modval = char_to_pystr("_nio", strlen("_nio"));
	PyObject *docstr = char_to_pystr("__doc__", strlen("__doc__"));
	PyObject *docval = char_to_pystr(option_class_doc, strlen(option_class_doc));

	PyDict_SetItem(dict, modstr, modval);
	PyDict_SetItem(dict, docstr, docval);

	/* Creating this instance dynamically by calling the C equivalent
	 * to 'type(class_name, bases, dict)' in Python with bases being (object,)
	 * */

	class = PyObject_CallFunctionObjArgs((PyObject*)&PyType_Type, pystr, bases, dict, NULL);
	result = PyObject_CallObject((PyObject *) class, NULL);

	Py_XDECREF(dict);
	Py_XDECREF(pystr);
	Py_XDECREF(bases);
	Py_XDECREF(modstr);
	Py_XDECREF(modval);
	Py_XDECREF(docstr);
	Py_XDECREF(docval);
	Py_XDECREF(class);

	return result;

}

static PyObject *
SetUpDefaultOptions(void) {
	PyObject *dict = PyDict_New();
	PyObject *opt, *val;

	opt = char_to_pystr("Format", strlen("Format"));
	val = char_to_pystr("classic", strlen("classic"));
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("HeaderReserveSpace", strlen("HeaderReserveSpace"));
#if PY_MAJOR_VERSION < 3
	val = PyInt_FromLong(0);
#else
	val = PyLong_FromLong(0);
#endif
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("MissingToFillValue", strlen("MissingToFillValue"));
	val = PyBool_FromLong(1);
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("PreFill", strlen("PreFill"));
	val = PyBool_FromLong(1);
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("SafeMode", strlen("SafeMode"));
	val = PyBool_FromLong(0);
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("CompressionLevel", strlen("CompressionLevel"));
#if PY_MAJOR_VERSION < 3
	val = PyInt_FromLong(-1);
#else
	val = PyLong_FromLong(-1);
#endif
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("DefaultNCEPPtable", strlen("DefaultNCEPPtable"));
	val = char_to_pystr("operational", strlen("operational"));
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("InitialTimeCoordinateType", strlen("InitialTimeCoordinateType"));
	val = char_to_pystr("numeric", strlen("numeric"));
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("SingleElementDimensions", strlen("SingleElementDimensions"));
	val = char_to_pystr("none", strlen("none"));
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("ThinnedGridInterpolation", strlen("ThinnedGridInterpolation"));
	val = char_to_pystr("cubic", strlen("cubic"));
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
	opt = char_to_pystr("TimePeriodSuffix", strlen("TimePeriodSuffix"));
	val = PyBool_FromLong(1);
	PyDict_SetItem(dict, opt, val);
	Py_DECREF(opt);
	Py_DECREF(val);
#if 0
	/* we don't want to set a default value for the FileStructure option, but the user can set it if they like */
	opt = char_to_pystr("FileStructure", strlen("FileStructure"));
	val = char_to_pystr("standard", strlen("standard"));
	PyDict_SetItem(dict,opt,val);
	Py_DECREF(opt);
	Py_DECREF(val);
#endif

	return dict;

}

/* Table of functions defined in the module */

static PyMethodDef nio_methods[] = {
		{ "open_file", (PyCFunction) NioFile, METH_KEYWORDS | METH_VARARGS, NULL },
		{ "options", (PyCFunction) NioFile_Options, METH_NOARGS, NULL },
		{ NULL, NULL } /* sentinel */
};


struct nio_state {
	PyObject *NioError;
	PyObject *NioFile_Type;
	PyObject *NioVariable_Type;
	PyObject *defaults;

	char err_buf[256];
};

#if PY_MAJOR_VERSION >= 3
#define INITERROR return NULL
#define GETSTATE(m) ((struct nio_state*)PyModule_GetState(m))
#else
#define INITERROR return
#endif

#if PY_MAJOR_VERSION >= 3
/*
static int nio_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->NioError);
    Py_VISIT(GETSTATE(m)->NioFile_Type);
    Py_VISIT(GETSTATE(m)->NioVariable_Type);
    Py_VISIT(GETSTATE(m)->defaults);

    return 0;
}

static int nio_clear(PyObject *m) {
	Py_CLEAR(GETSTATE(m)->NioError);
    Py_CLEAR(GETSTATE(m)->NioFile_Type);
    Py_CLEAR(GETSTATE(m)->NioVariable_Type);
    Py_CLEAR(GETSTATE(m)->defaults);

    return 0;
}
*/

static struct PyModuleDef nio_def = {
        PyModuleDef_HEAD_INIT,
        "_nio",
		NULL,
        sizeof(struct nio_state),
		nio_methods,
		NULL,
		NULL,
		/* TODO: Add support for cyclic garbage collection */
        /*(traverseproc)nio_traverse,*/
		NULL,
		/* TODO: Add support for cyclic garbage collection */
        /*(inquiry)nio_clear,*/
		NULL
};
#endif

static PyObject* get_NioError(void) {
#if PY_MAJOR_VERSION < 3
	return NIOError;
#else
	PyObject *m = PyState_FindModule(&nio_def);
	return GETSTATE(m)->NioError;
#endif
}

static PyObject* get_NioFileType(void) {
#if PY_MAJOR_VERSION < 3
	return (PyObject*) &NioFile_Type;
#else
	PyObject *m = PyState_FindModule(&nio_def);
	return GETSTATE(m)->NioFile_Type;
#endif
}

static PyObject* get_NioVariableType(void) {
#if PY_MAJOR_VERSION < 3
	return (PyObject*) &NioVariable_Type;
#else
	PyObject *m = PyState_FindModule(&nio_def);
	return GETSTATE(m)->NioVariable_Type;
#endif
}

static char* get_errbuf(void) {
#if PY_MAJOR_VERSION < 3
	return err_buf;
#else
	PyObject *m = PyState_FindModule(&nio_def);
	return GETSTATE(m)->err_buf;
#endif
}

static PyObject* get_Niomodule(void) {
#if PY_MAJOR_VERSION < 3
	return (PyObject*) Niomodule;
#else
	return PyState_FindModule(&nio_def);
#endif
}


/* Module initialization */

#if PY_MAJOR_VERSION < 3
void init_nio(void) {
	PyObject *m, *d;
	PyObject *def_opt_dict;
	/*static void *PyNIO_API[PyNIO_API_pointers];*/

	if (PyType_Ready(&(NioFile_Type)) < 0)
		INITERROR;
	if (PyType_Ready(&(NioVariable_Type)) < 0)
		INITERROR;

	/* Create the module and add the functions */
	m = Py_InitModule("_nio", nio_methods);
	if (m == NULL) {
		INITERROR;
	}

	Niomodule = m;

	/* Initialize type object headers */
	NioFile_Type.tp_doc = niofile_type_doc;
	NioVariable_Type.tp_doc = niovariable_type_doc;

	/* these cannot be initialized statically */
	nio_methods[0].ml_doc = open_file_doc;
	nio_methods[1].ml_doc = options_doc;
	NioFileObject_methods[0].ml_doc = close_doc;
	NioFileObject_methods[1].ml_doc = create_dimension_doc;
	NioFileObject_methods[2].ml_doc = create_chunk_dimension_doc;
	NioFileObject_methods[3].ml_doc = create_variable_doc;
	NioFileObject_methods[4].ml_doc = create_group_doc;
	NioFileObject_methods[5].ml_doc = create_vlen_doc;
	NioFileObject_methods[6].ml_doc = create_compound_type_doc;
	NioFileObject_methods[7].ml_doc = create_compound_doc;
	NioFileObject_methods[8].ml_doc = unlimited_doc;
	NioVariableObject_methods[0].ml_doc = assign_value_doc;
	NioVariableObject_methods[1].ml_doc = get_value_doc;
	NioVariableObject_methods[2].ml_doc = typecode_doc;


	/* Initialize the NIO library */

	NioInitialize();

	/* Import the array module */
	/*#ifdef import_array*/
	import_array();
	/*#endif*/

	/* Add error object to the module */
	d = PyModule_GetDict(m);
	/*NIOError = PyUnicode_DecodeUTF8("NIOError");*/
	NIOError = PyErr_NewException("_nio.NIOError", NULL, NULL);
	DICT_SETITEMSTRING(d, "NIOError", NIOError);

	/* make NioFile, NioGroup and NioVariable visible to the module for subclassing */
	/*FIXME: Breaking type punning rules */
	Py_INCREF(&NioFile_Type);
	PyModule_AddObject(m, "_NioFile", (PyObject *) &NioFile_Type);
	/*FIXME: Breaking type punning rules */
	Py_INCREF(&NioVariable_Type);
	PyModule_AddObject(m, "_NioVariable", (PyObject *) &NioVariable_Type);

	def_opt_dict = SetUpDefaultOptions();

	PyModule_AddObject(m, "option_defaults", (PyObject *) def_opt_dict);

	/* Check for errors */
	if (PyErr_Occurred()) {
		Py_FatalError("can't initialize module nio");
		INITERROR;
	}
}
#else
PyMODINIT_FUNC PyInit__nio(void) {
	PyObject *m;
	struct nio_state *module_state;

	/* Create the module and add the functions */
	m = PyModule_Create(&nio_def);

	if (m == NULL) {
		INITERROR;
	}

	/* these cannot be initialized statically */
	nio_methods[0].ml_doc = open_file_doc;
	nio_methods[1].ml_doc = options_doc;

	NioFileObject_methods[0].ml_doc = close_doc;
	NioFileObject_methods[1].ml_doc = create_dimension_doc;
	NioFileObject_methods[2].ml_doc = create_chunk_dimension_doc;
	NioFileObject_methods[3].ml_doc = create_variable_doc;
	NioFileObject_methods[4].ml_doc = create_group_doc;
	NioFileObject_methods[5].ml_doc = create_vlen_doc;
	NioFileObject_methods[6].ml_doc = create_compound_type_doc;
	NioFileObject_methods[7].ml_doc = create_compound_doc;
	NioFileObject_methods[8].ml_doc = unlimited_doc;
	NioVariableObject_methods[0].ml_doc = assign_value_doc;
	NioVariableObject_methods[1].ml_doc = get_value_doc;
	NioVariableObject_methods[2].ml_doc = typecode_doc;

	module_state = GETSTATE(m);

	module_state->NioFile_Type = PyType_FromSpec(&niofile_spec);

	((PyTypeObject*)module_state->NioFile_Type)->tp_weaklistoffset = offsetof(NioFileObject, weakreflist);

	module_state->NioVariable_Type = PyType_FromSpec(&niovar_spec);

	((PyTypeObject*)module_state->NioVariable_Type)->tp_as_sequence = &NioVariableObject_as_sequence;
	((PyTypeObject*)module_state->NioVariable_Type)->tp_as_mapping = &NioVariableObject_as_mapping;


	/* Initialize type object headers */
	((PyTypeObject*)module_state->NioFile_Type)->tp_doc = niofile_type_doc;
	((PyTypeObject*)module_state->NioVariable_Type)->tp_doc = niovariable_type_doc;


	/* Initialize the NIO library */

	NioInitialize();

	/* Import the array module */
	/*#ifdef import_array*/
	import_array();
	/*#endif*/

	/* Add error object to the module */
	module_state->NioError = PyErr_NewException("_nio.NIOError", NULL, NULL);
	Py_INCREF(module_state->NioError);
	PyModule_AddObject(m, "NIOError", module_state->NioError);

	/* make NioFile, NioGroup and NioVariable visible to the module for subclassing */
	/* PyModule_AddObject steals a reference, so need to bump the ref count here */

	Py_INCREF(module_state->NioFile_Type);
	PyModule_AddObject(m, "_NioFile", module_state->NioFile_Type);
	Py_INCREF(module_state->NioVariable_Type);
	PyModule_AddObject(m, "_NioVariable", module_state->NioVariable_Type);

	module_state->defaults = SetUpDefaultOptions();
	Py_INCREF(module_state->defaults);
	PyModule_AddObject(m, "option_defaults", module_state->defaults);

	/* Check for errors */
	if (PyErr_Occurred()) {
		Py_FatalError("can't initialize module nio");
		INITERROR;
	}

	return m;
}

#endif

