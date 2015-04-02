/*******************************************************
 * $Id$
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

#include "netcdf.h"
#include "Python.h"
#include "structmember.h"
#include "nio.h"
#include <numpy/arrayobject.h>

#include <sys/stat.h>
#include <unistd.h>

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
object as the 3rd (optional) argument to Nio.open_file:\n\
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
            of 0 - 9 (ignored unless Format is set to 'NetCDF4Classic').\n\
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
If 'f' is file object variable, get summary of contents including all\n\
dimensions, chunk_dimensions, ud_types, variables, groups, and attributes using:\n\
    print f\n\
Assign global file attributes to writable files using:\n\
    f.global_att = global_att_value\n\
Attributes (do not modify):\n\
    dimensions -- a dictionary with dimension names as keys and dimension.\n\
        lengths as values.\n\
    chunk_dimensions -- a dictionary with chunk_dimension names as keys and dimension.\n\
        lengths as values.\n\
    ud_types -- a dictionary with type names as keys and value.\n\
        objects as values.\n\
    variables -- a dictionary with variable names as keys and NioVariable.\n\
        objects as values.\n\
    groups -- a dictionary with group names as keys and NioFile.\n\
        objects as values.\n\
    attributes (or __dict__) -- a dictionary containing global attribute names and values.\n\
Methods:\n\
    close([history]) -- close the file.\n\
    create_dimension(name,length) -- create a dimension in the file.\n\
    create_chunk_dimension(name,length) -- create a chunk dimension in the file.\n\
    create_variable(name,type,dimensions) -- create a variable in the file.\n\
    create_group(name) -- create a group in the file.\n\
    create_vlen(name,type,dimensions) -- create a variable length array variable in the file.\n\
    create_compound_type(name, type)  -- create a user-defined compound type; with member names, sizes\n\
        and types as defined in the type sequence argument.\n\
    create_compound(name,type,dimensions) -- create a compound variable with the given type and dimensions.\n\
    unlimited(dimension_name) -- returns True if dimension_name refers to an unlimited dimension; False otherwise\n\n\
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
Get summary of variable contents including all dimensions,\n\
associated coordinate variables, and attributes using:\n\
    print f.variables['varname']\n\
Assign variable attributes for writable files using:\n\
    f.variables['varname'].attname = attvalue\n\
Get or assign variable values using Python slicing syntax:\n\
    val = f.variables['varname'][:]\n\
assigns the entire variable contents to variable 'val'.\n\
Attributes:(do not modify)\n\
    rank -- a scalar value indicating the number of dimensions\n\
    shape -- a tuple containing the number of elements in each dimension\n\
    size -- a scalar value indicating the size in bytes of the variable\n\
    dimensions -- a tuple containing the dimensions names in order\n\
    chunk_dimensions -- a tuple containing the chunk_dimensions names in order\n\
    attributes (or __dict__) -- a dictionary containing the variable attributes\n\
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

staticforward int nio_file_init(NioFileObject *self);
staticforward NioFileObject* nio_read_group(NioFileObject* file, NclFileGrpNode* grpnode);
staticforward NioFileObject* nio_create_group(NioFileObject* file, NrmQuark gname);
staticforward NioVariableObject* nio_read_advanced_variable(NioFileObject* file,
                                                            NclFileVarNode* varnode, int id);
staticforward NioVariableObject *nio_variable_new(NioFileObject *file, char *name, int id, 
	                                          int type, int ndims, NrmQuark *qdims, int nattrs);
void _convertObj2COMPOUND(PyObject* pyobj, obj* listids, NclFileCompoundRecord *comprec,
                          ng_size_t n_dims, ng_size_t curdim, ng_usize_t* counter);

static PyObject *Niomodule; /* the Nio Module object */

/* Error object and error messages for nio-specific errors */

static PyObject *NIOError;

static char *nio_errors[] = {
	"No Error",                    /* 0 */
	"Not a NIO id",
	"Too many NIO files open",
	"NIO file exists && NC_NOCLOBBER",
	"Invalid Argument",
	"Write to read only",
	"Operation not allowed in data mode",
	"Operation not allowed in define mode",
	"Coordinates out of Domain",
	"MAX_NC_DIMS exceeded",
	"String match to name in use",
	"Attribute not found",
	"MAX_NC_ATTRS exceeded",
	"Not a NIO data type",
	"Invalid dimension id",
	"NC_UNLIMITED in the wrong index",
	"MAX_NC_VARS exceeded",
	"Variable not found",
	"Action prohibited on NC_GLOBAL varid",
	"Not an NIO supported file",
	"In Fortran, string too short",
	"MAX_NC_NAME exceeded",
	"NC_UNLIMITED size already in use", /* 22 */
        "Memory allocation error",
	"attempt to set read-only attributes",
	"invalid mode specification",
	"", "", "", "", "", "",
	"XDR error" /* 32 */
};


static int nio_ncerr = 0;  
/* Set error string */
static void
nio_seterror(void)
{
  if (nio_ncerr != 0) {
    char *error = "Unknown error";
    if (nio_ncerr > 0 && nio_ncerr <= 32)
      error = nio_errors[nio_ncerr];
    PyErr_SetString(NIOError, error);
  }
}

static char err_buf[256];

/*
 * Python equivalents to NIO data types
 *
 * Attention: the following specification may not be fully portable.
 * The comments indicate the correct NIO specification. The assignment
 * of Python types assumes that 'short' is 16-bit and 'int' is 32-bit.
 */

#if 0
int data_types[] = {-1,  /* not used */

		    PyArray_SBYTE,  /* signed 8-bit int */
		    PyArray_CHAR,   /* 8-bit character */
		    PyArray_SHORT,  /* 16-bit signed int */
		    PyArray_INT,    /* 32-bit signed int */
		    PyArray_FLOAT,  /* 32-bit IEEE float */
		    PyArray_DOUBLE  /* 64-bit IEEE float */
};
#endif

int data_type(NclBasicDataTypes ntype)
{
	switch (ntype) {
	case NCL_short:
		return PyArray_SHORT;
	case NCL_ushort:
		return PyArray_USHORT;
	case NCL_int:
		return PyArray_INT;  /* netcdf 3.x has only a long type */
	case NCL_uint:
		return PyArray_UINT;  /* netcdf 3.x has only a long type */
	case NCL_long:
		return PyArray_LONG;
	case NCL_ulong:
		return PyArray_ULONG;
	case NCL_int64:
		return PyArray_LONGLONG;
	case NCL_uint64:
		return PyArray_ULONGLONG;
	case NCL_float:
		return PyArray_FLOAT;
	case NCL_double:
		return PyArray_DOUBLE;
        case NCL_byte:
		return PyArray_BYTE;
        case NCL_ubyte:
		return PyArray_UBYTE;
        case NCL_char:
		return PyArray_CHAR;
        case NCL_logical:
		return PyArray_BOOL;
        case NCL_string:
		return PyArray_STRING;
        case NCL_vlen:
                return PyArray_VLEN;
        case NCL_compound:
                return PyArray_COMPOUND;
        case NCL_enum:
                return PyArray_ENUM;
        case NCL_opaque:
                return PyArray_OPAQUE;
        case NCL_reference:
                return PyArray_REFERENCE;
        case NCL_list:
                return PyArray_LIST;
        case NCL_listarray:
                return PyArray_LISTARRAY;
	default:
		return PyArray_NOTYPE;
	}
	return PyArray_NOTYPE;
}



/* Utility functions */

static void
define_mode(NioFileObject *file, int define_flag)
{
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
typecode(int type)
{
   static char buf[3];

   memset(buf,0,3);


  switch(type) {
  case PyArray_BOOL:
	  buf[0] = PyArray_BOOLLTR;
	  break;
  case PyArray_BYTE:
	  buf[0] = PyArray_BYTELTR;
	  break;
  case PyArray_UBYTE:
	  buf[0]= PyArray_UBYTELTR;
	  break;
  case PyArray_SHORT:
	  buf[0] = PyArray_SHORTLTR;
	  break;
  case PyArray_USHORT:
	  buf[0] = PyArray_USHORTLTR;
	  break;
  case PyArray_INT:
	  buf[0] = PyArray_INTLTR;
	  break;
  case PyArray_UINT:
	  buf[0] = PyArray_UINTLTR;
	  break;
  case PyArray_LONG:
	  buf[0] = PyArray_LONGLTR;
	  break;
  case PyArray_ULONG:
	  buf[0] = PyArray_ULONGLTR;
	  break;
  case PyArray_LONGLONG:
	  buf[0] = PyArray_LONGLONGLTR;
	  break;
  case PyArray_ULONGLONG:
	  buf[0] = PyArray_ULONGLONGLTR;
	  break;
  case PyArray_FLOAT:
	  buf[0] = PyArray_FLOATLTR;
	  break;
  case PyArray_DOUBLE:
	  buf[0] = PyArray_DOUBLELTR;
	  break;
  case PyArray_STRING:
	  buf[0] = PyArray_STRINGLTR;
	  break;
#if 0
  case PyArray_CHAR:
	  buf[0] = PyArray_STRINGLTR;
	  break;
#else
  case PyArray_CHAR:
	  strcpy(buf,"S1");
	  break;
#endif
  case PyArray_VLEN:
          buf[0] = PyArray_VLENLTR;
          break;
  case PyArray_COMPOUND:
          buf[0] = PyArray_COMPOUNDLTR;
          break;
  case PyArray_ENUM:
          buf[0] = PyArray_ENUMLTR;
          break;
  case PyArray_OPAQUE:
          buf[0] = PyArray_OPAQUELTR;
          break;
  case PyArray_LIST:
          buf[0] = PyArray_LISTLTR;
          break;
  case PyArray_LISTARRAY:
          buf[0] = PyArray_LISTARRAYLTR;
          break;
  case PyArray_REFERENCE:
          buf[0] = PyArray_REFERENCELTR;
          break;
  default: 
	  buf[0] = ' ';
	  break;
  }
  return &buf[0];
}

static NrmQuark
nio_type_from_code(int code)
{
  NrmQuark type;
  switch((char)code) {
  case 'c':
	  type = NrmStringToQuark("character");
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
  case 'S':
	  type = NrmStringToQuark("string");
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

static void
collect_attributes(void *fileid, int varid, PyObject *attributes, int nattrs)
{
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
		  md = _NclFileReadAtt(file,att->att_name_quark,NULL);
	  }
	  else if (! (att_list && fvar)) {
		  PyErr_SetString(NIOError, "internal attribute or file variable error");
		  return;
	  }
	  else {
		  att = att_list->the_att;
		  name = NrmQuarkToString(att->att_name_quark);
		  md = _NclFileReadVarAtt(file,fvar->var_name_quark,att->att_name_quark,NULL);
		  att_list = att_list->next;
	  }
	  if (att->data_type == NCL_string) {
		  PyObject *string = NULL;
		  char *satt = NrmQuarkToString(*((NrmQuark *)md->multidval.val));
                  if (satt != NULL) {
		          string = PyString_FromString(satt);
                  }
                  else {
		          string = PyString_FromString("");
                  }
		  if (string != NULL) {
			  PyDict_SetItemString(attributes, name, string);
			  Py_DECREF(string);
		  }
	  }
	  else {
		  PyObject *array;
		  length = (npy_intp) md->multidval.totalelements;
		  py_type = data_type(att->data_type);
		  array = PyArray_SimpleNew(1, &length, py_type);
		  if (array != NULL) {
			  memcpy(((PyArrayObject *)array)->data,
				 md->multidval.val,(size_t)length * md->multidval.type->type_class.size);
			  array = PyArray_Return((PyArrayObject *)array);
			  if (array != NULL) {
				  PyDict_SetItemString(attributes, name, array);
				  Py_DECREF(array);
			  }
		  }
	  }
  }
}

static int
set_attribute(NioFileObject *file, int varid, PyObject *attributes,
	      char *name, PyObject *value)
{
  NclFile nfile = (NclFile) file->id;
  NhlErrorTypes ret;
  NclMultiDValData md = NULL;
  PyArrayObject *array = NULL;
  
  if (!value || value == Py_None) {
	  /* delete attribute */
	  if (varid == NC_GLOBAL) {
		  ret = _NclFileDeleteAtt(nfile,NrmStringToQuark(name));
	  }
	  else {
		  ret = _NclFileDeleteVarAtt(nfile,nfile->file.var_info[varid]->var_name_quark,
					    NrmStringToQuark(name));
	  }
	  PyObject_DelItemString(attributes,name);
	  return 0;
  }
	  

  if (PyString_Check(value)) {
	  ng_size_t len_dims = 1;
	  NrmQuark *qval = malloc(sizeof(NrmQuark));
	  qval[0] = NrmStringToQuark(PyString_AsString(value));
	  md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
				   (void*)qval,NULL,1,&len_dims,
				   TEMPORARY,NULL,(NclTypeClass)nclTypestringClass);
  }
  else {
	  ng_size_t dim_sizes = 1;
	  int n_dims;
	  NrmQuark qtype;
	  int pyarray_type = PyArray_NOTYPE;
	  PyArrayObject *tmparray = (PyArrayObject *)PyDict_GetItemString(attributes,name);
	  if (tmparray != NULL) {
		  pyarray_type = tmparray->descr->type_num;
	  }
	  array = (PyArrayObject *)PyArray_ContiguousFromAny(value, pyarray_type, 0, 1);
          if (array) {
	          n_dims = (array->nd == 0) ? 1 : array->nd;
	          qtype = nio_type_from_code(array->descr->type);
#if 0
	          if (array->descr->elsize == 8 && qtype == NrmStringToQuark("long")) {
                           PyArrayObject *array2 = (PyArrayObject *)
                                     PyArray_Cast(array, PyArray_INT);
                           Py_DECREF(array);
                           array = array2;
			   qtype = NrmStringToQuark("integer");
                  }
#endif
                  if (array) {
			   ng_size_t *dims;
			   void *data;
			   if (array->nd == 0) {
				   dims = &dim_sizes;
			   }
			   else  {
				   dims = (ng_size_t *) array->dimensions;
			   }
			   data = malloc(PyArray_NBYTES(array));
			   memcpy(data,PyArray_DATA(array),PyArray_NBYTES(array));
			   
	                   md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
		                                    (void*)data,NULL,n_dims,dims,
				                    TEMPORARY,NULL,_NclNameToTypeClass(qtype));
			   
                  }
           }
  }
  if (! md) {
	  nio_ncerr = 23;
	  nio_seterror();
	  return -1;
  }
  
  if (varid == NC_GLOBAL) {
	  ret = _NclFileWriteAtt(nfile,NrmStringToQuark(name),md,NULL);
  }
  else {
	  ret = _NclFileWriteVarAtt(nfile,nfile->file.var_info[varid]->var_name_quark,
			      NrmStringToQuark(name),md,NULL);
  }
  if (ret > NhlFATAL) {
	  if (PyString_Check(value)) {
		  PyDict_SetItemString(attributes, name, value);
	  }
	  else if (array) {
		  PyDict_SetItemString(attributes, name, (PyObject *)array);
	  }
  }
  return 0;
}

static int
check_if_open(NioFileObject *file, int mode)
{
  /* mode: -1 read, 1 write, 0 other */
  if (file->open) {
    if (mode != 1 || file->write) {
      return 1;
    }
    else {
      PyErr_SetString(NIOError, "write access to read-only file");
      return 0;
    }
  }
  else {
    PyErr_SetString(NIOError, "file has been closed");
    return 0;
  }
}

/*
 * NioFile object
 * (type declaration in niomodule.h)
 */


/* Destroy file object */

static void
NioFileObject_dealloc(NioFileObject *self)
{
  PyObject *keys, *key, *value;
  Py_ssize_t i, pos = 0;

  if (self->parent)  {/* group objects should just delete themselves */ 
	  PyObject_DEL(self);
	  return;
  }
  if (self->being_destroyed) {
	  /* already closed and mostly deallocated -- just del object and return */
	  Py_XDECREF(self->name);
	  Py_XDECREF(self->mode);
	  PyObject_DEL(self);
	  return;
  }
  else if (self->open) {
	  NioFile_Close(self);  /* calls this function back so we do not want to run it again */
	  return; 
  }

  /* destroy the contents of all groups except the root ('/') group */
  keys = PyDict_Keys(self->groups);
  if(NULL != keys) {
      for (i = 0; i < PyList_Size(keys); i++) {
	  key = PyList_GetItem(keys, i);
	  NioFileObject *g = (NioFileObject *) PyDict_GetItem(self->groups,key);
	  if (g != NULL && strcmp(PyString_AsString(g->name),"/")) {
		    _NclDestroyObj((NclObj)g->gnode);
	  }
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
	  PyDict_DelItem(self->groups,key);
      }
      Py_DECREF(keys);
  }
  if(NULL != self->id)
      _NclDestroyObj((NclObj)self->id);

  Py_CLEAR(self->variables);
  Py_CLEAR(self->dimensions);
  Py_CLEAR(self->chunk_dimensions);
  Py_CLEAR(self->ud_types);
  Py_CLEAR(self->groups);
  Py_CLEAR(self->attributes);
  Py_XDECREF(self->full_path);
  Py_XDECREF(self->type);

  /* The name and mode components are kept around for the benefit of the repr method */
  self->being_destroyed = 1;   /* indicates file is closed and data structures have been mostly torn down */
  return;
}

static int GetNioMode(char* filename,char *mode) 
{
	struct stat buf;
	int crw;

	if (mode == NULL)
		mode = "r";

	switch (mode[0]) {
	case 'a':
		if (stat(_NGResolvePath(filename),&buf) < 0)
			crw = -1;
		else 
			crw = 0;
		break;
	case 'c':
		crw = -1;
		break;
	case 'r':
		if (strlen(mode) > 1 && (mode[1] == '+' || mode[1] == 'w')) {
			if (stat(_NGResolvePath(filename),&buf) < 0)
				crw = -1;
			else 
				crw = 0;
		}
		else
			crw = 1;
		break;
	case 'w':
		if (stat(_NGResolvePath(filename),&buf) < 0)
			crw = -1;
		else
			crw = 0;
		break;
	default:
		crw = -2;
	}
	return crw;
}

/* Create file object */

NioFileObject *
NioFile_Open(char *filename, char *mode)
{
  NioFileObject *self = PyObject_NEW(NioFileObject,&NioFile_Type);
  NclFile file = NULL;
  int crw;
  char *name;

/*
 *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
 *	           __PRETTY_FUNCTION__, __FILE__, __LINE__);
 */

  nio_ncerr = 0;

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
  self->type = PyString_FromString("file");
  self->full_path = NULL;
  self->parent = NULL;
  self->top = NULL;
  self->open = 0;
  self->being_destroyed = 0;
  self->id = NULL;
  self->gnode = NULL;
  self->weakreflist = NULL;

  crw = GetNioMode(filename,mode);
  file = _NclCreateFile(NULL,NULL,Ncl_File,0,TEMPORARY,
			NrmStringToQuark(filename),crw);
  if (file) {
	  self->id = (void *) file;
	  self->define = 1;
	  self->open = 1;
	  self->write = (crw != 1);
	  name = strrchr(filename,'/');
	  if (name && strlen(name) > 1) {
		  self->name = PyString_FromString(name + 1);
		  self->full_path = PyString_FromString(filename);
	  }
	  else {
		  self->name = PyString_FromString(filename);
		  self->full_path = PyString_FromString(filename);
	  }
	  self->mode = PyString_FromString(mode);
	  nio_file_init(self); 
  }
  else {
	  NioFileObject_dealloc(self);
	  PyErr_SetString(NIOError,"Unable to open file");
	  return NULL;
  }
  return self;
}

/* Create variables from file */

static NclFileVarNode* getVarFromGroup(NclFileGrpNode* grpnode, NrmQuark vname)
{
    NclFileGrpRecord* grprec;
    NclFileVarRecord* varrec;
    NclFileVarNode*   varnode = NULL;
    int i;

    varrec = grpnode->var_rec;
    if(NULL != varrec)
    {
        for(i = 0; i < varrec->n_vars; ++i)
        {
            varnode = &(varrec->var_node[i]);
            if((vname == varnode->name) || (vname == varnode->real_name))
                return varnode;
        }
    }

    grprec = grpnode->grp_rec;
    if(NULL != grprec)
    {
        for(i = 0; i < grprec->n_grps; ++i)
        {
            varnode = getVarFromGroup(grprec->grp_node[i], vname);
            if(NULL != varnode)
                return varnode;
        }
    }

    return NULL;
}

/* Get user-define node from file */

static NclFileUDTNode* getUDTfromGroup(NclFileGrpNode* grpnode, NrmQuark name)
{
    NclFileGrpRecord* grprec;
    NclFileUDTRecord* udtrec;
    NclFileUDTNode*   udtnode = NULL;
    int i;

    udtrec = grpnode->udt_rec;
    if(NULL != udtrec)
    {
        for(i = 0; i < udtrec->n_udts; ++i)
        {
            udtnode = &(udtrec->udt_node[i]);
            if(name == udtnode->name)
                return udtnode;
        }
    }

    grprec = grpnode->grp_rec;
    if(NULL != grprec)
    {
        for(i = 0; i < grprec->n_grps; ++i)
        {
            udtnode = getUDTfromGroup(grprec->grp_node[i], name);
            if(NULL != udtnode)
                return udtnode;
        }
    }

    return NULL;
}

static int dimNvarInfofromGroup(NioFileObject *self, NclFileGrpNode* grpnode,
                                int* ndims, int* nvars, int* ngrps, int* ngattrs)
{
    NclFileDimRecord* dimrec;
    NclFileDimNode*   dimnode;

    NclFileVarRecord* varrec;
    NclFileVarNode*   varnode;

    NclFileGrpRecord* grprec;

    char* name;
    ng_size_t size;
    PyObject* size_ob;
    PyObject* str;

    NioVariableObject *variable;
    NioFileObject *group;

    int i;
    char *path;


    if(NULL == grpnode)
        return 0;


    if(NULL != grpnode->att_rec)
        *ngattrs += grpnode->att_rec->n_atts;

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *                  __PRETTY_FUNCTION__, __FILE__, __LINE__);
   */

    path = PyString_AsString(self->full_path);
    dimrec = grpnode->dim_rec;

    if(NULL != dimrec)
    {
        *ndims += dimrec->n_dims;
        for(i = 0; i < dimrec->n_dims; ++i)
        {
            dimnode = &(dimrec->dim_node[i]);
          /*if(dimnode->is_unlimited)*/
            if((dimnode->is_unlimited) && (0 > self->recdim))
            if(dimnode->is_unlimited)
                self->recdim = i;

            name = NrmQuarkToString(dimnode->name);
            size = dimnode->size;
            size_ob = PyInt_FromSize_t(size);
            PyDict_SetItemString(self->dimensions, name, size_ob);
	    if (! strcmp(path,"/")) {
		    str = PyString_FromFormat("%s",name);
		    PyDict_SetItem(self->top->dimensions,str, (PyObject *)size_ob);
	    }
	    else {
		    str = PyString_FromFormat("%s/%s",path,name);
		    PyDict_SetItem(self->top->dimensions,str, (PyObject *)size_ob);
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

    if(NULL != dimrec)
    {
        for(i = 0; i < dimrec->n_dims; ++i)
        {
            dimnode = &(dimrec->dim_node[i]);

            name = NrmQuarkToString(dimnode->name);
            size = dimnode->size;
            size_ob = PyInt_FromSize_t(size);
            PyDict_SetItemString(self->chunk_dimensions, name, size_ob);
            Py_DECREF(size_ob);

          /*
           *fprintf(stderr, "\tin file: %s, line: %d\n", __FILE__, __LINE__);
           *fprintf(stderr, "\tDim %d: name: <%s>, size: %ld\n", i, name, size);
           */
        }
    }

    varrec = grpnode->var_rec;
    if(NULL != varrec)
    {
        *nvars += varrec->n_vars;
        for(i = 0; i < varrec->n_vars; ++i)
        {
            varnode = &(varrec->var_node[i]);
            name = NrmQuarkToString(varnode->name);

          /*
           *fprintf(stderr, "\tin file: %s, line: %d\n", __FILE__, __LINE__);
           *fprintf(stderr, "\tVar %d: name: <%s>\n", i, name);
           */

            variable = nio_read_advanced_variable(self, varnode, i);
            PyDict_SetItemString(self->variables, name, (PyObject *)variable);
	    if (! strcmp(path,"/") || strlen(path) == 0) {
		    str = PyString_FromFormat("%s",name);
		    PyDict_SetItem(self->top->variables,str, (PyObject *)variable);
	    }
	    else {
		    str = PyString_FromFormat("%s/%s",path,name);
		    PyDict_SetItem(self->top->variables,str, (PyObject *)variable);
	    }
	    Py_DECREF(str);
            Py_DECREF(variable);
        }
    }

    grprec = grpnode->grp_rec;
    if(NULL != grprec)
    {
        *ngrps += grprec->n_grps;
        for(i = 0; i < grprec->n_grps; ++i)
        {
            name = NrmQuarkToString(grprec->grp_node[i]->name);

          /*
           *fprintf(stderr, "\tin file: %s, line: %d\n", __FILE__, __LINE__);
           *fprintf(stderr, "\tGrp %d: name: <%s>\n", i, name);
           */

            group = nio_read_group(self, grprec->grp_node[i]);
            PyDict_SetItemString((PyObject *)self->groups, name, (PyObject *)group);
	    if (! strcmp(path,"/") || strlen(path) == 0) {
		    str = PyString_FromFormat("%s",name);
		    PyDict_SetItem(self->top->groups,str, (PyObject *)group);
	    }
	    else {
		    str = PyString_FromFormat("%s/%s",path,name);
		    PyDict_SetItem(self->top->groups, str, (PyObject *)group);
	    }
	    Py_DECREF(str);
            Py_DECREF(group);

            /*dimNvarInfofromGroup(self, grprec->grp_node[i],
	      ndims, nvars, ngrps, ngattrs);*/
        }
    }

    return 0;
}

static void collect_advancedfile_attributes(NioFileObject *self, NclFileAttRecord* attrec, PyObject *attributes)
{
    NclFileAttNode*   attnode;
    char *name;
    npy_intp length;
    int py_type;
    int i;
    char *path = NULL;
    PyObject *attname;

    if(NULL == attrec)
        return;
  
    if (self) {
	    path = PyString_AsString(self->full_path);
    }

    for(i = 0; i < attrec->n_atts; ++i)
    {
        attnode = &(attrec->att_node[i]);
        name = NrmQuarkToString(attnode->name);

        if(attnode->type == NCL_string)
        {
            PyObject *astring = NULL;
            char *satt = NrmQuarkToString(*((NrmQuark *)attnode->value));
            if(satt != NULL)
                astring = PyString_FromString(satt);
            else
                astring = PyString_FromString("");

            if(astring != NULL)
            {
                PyDict_SetItemString(attributes, name, astring);
		if (path) {
			if (! strcmp(path,"/")) {
				attname = PyString_FromFormat("%s",name);
				PyDict_SetItem(self->top->attributes,attname, astring);
			}
			else {
				attname = PyString_FromFormat("%s/%s",path,name);
				PyDict_SetItem(self->top->attributes,attname, astring);
			}
			Py_DECREF(attname);
		}
            }
	    Py_DECREF(astring);
        }
        else
        {
            PyObject *array;
            length = (npy_intp) attnode->n_elem;
            py_type = data_type(attnode->type);
            array = PyArray_SimpleNew(1, &length, py_type);
            if(array != NULL)
            {
                memcpy(((PyArrayObject *)array)->data, attnode->value,
                      (size_t) length * _NclSizeOf(attnode->type));
                array = PyArray_Return((PyArrayObject *)array);
                if (array != NULL)
                {
                    PyDict_SetItemString(attributes, name, array);
		    if (path) {
			    if (! strcmp(path,"/")) {
				    attname = PyString_FromFormat("%s",name);
				    PyDict_SetItem(self->top->attributes,attname, array);
			    }
			    else {
				    attname = PyString_FromFormat("%s/%s",path,name);
				    PyDict_SetItem(self->top->attributes,PyString_FromFormat("%s/%s",path,name), array);
			    }
			    Py_DECREF(attname);
		    }
                    Py_DECREF(array);
                }
            }
        }
    }
}

static int
nio_file_init(NioFileObject *self)
{
	NclFile file = (NclFile) self->id;
	int ndims, nvars, ngrps, ngattrs;
	int i,j;
	int scalar_dim_ix = -1;
	NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
	self->dimensions = PyDict_New();
	self->chunk_dimensions = PyDict_New();
	self->variables = PyDict_New();
	self->ud_types = PyDict_New();
	self->groups = PyDict_New();
	self->attributes = PyDict_New();
	
		
	self->recdim = -1; /* for now */
	if(file->file.advanced_file_structure)
	{
		NclAdvancedFile advfile = (NclAdvancedFile) self->id;
		char *name = NrmQuarkToString(advfile->advancedfile.grpnode->name);
		NioFileObject *group;

		ndims = 0;
		nvars = 0;
		ngrps = 0;
		ngattrs = 0;

		self->gnode = advfile->advancedfile.grpnode;
		self->parent = NULL;
		self->top = self;
		group = nio_read_group(self, self->gnode);
		group->id = self->id;
		group->gnode = self->gnode;
		PyDict_SetItemString((PyObject *)self->groups, name, (PyObject *)group);
		Py_DECREF(group);
                /*dimNvarInfofromGroup(self, advfile->advancedfile.grpnode,
		  &ndims, &nvars, &ngrps, &ngattrs);*/
		/*collect_advancedfile_attributes(self,advfile->advancedfile.grpnode->att_rec,
		  self->attributes);*/
	}
	else
	{
		ndims = file->file.n_file_dims;
		nvars = file->file.n_vars;
		ngrps = 0;
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
				size_ob = PyInt_FromSize_t(size);
				PyDict_SetItemString(self->dimensions, name, size_ob);
				Py_DECREF(size_ob);
			}
			else {
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
			}
			else if (ndimensions > 0) {
				qdims = (NrmQuark *) malloc(ndimensions *sizeof(NrmQuark));
	
				if (qdims == NULL) {
					PyErr_NoMemory();
					return 0;
				}
				for (j = 0; j < ndimensions; j++) {
					qdims[j] = file->file.file_dim_info[fvar->file_dim_num[j]]->dim_name_quark;
				}
			}
			else
				qdims = NULL;
			variable = nio_variable_new(self, name, i, data_type(datatype),
					    	ndimensions, qdims, nattrs);
			PyDict_SetItemString(self->variables, name, (PyObject *)variable);
			Py_DECREF(variable);
		}

		collect_attributes(self->id, NC_GLOBAL, self->attributes, ngattrs);
	}

	return 1;
}


/* Create dimension */

int
NioFile_CreateDimension(NioFileObject *file, char *name, Py_ssize_t size)
{
  PyObject *size_ob;
  NrmQuark qname;
  if (check_if_open(file, 1)) {
	  NclFile nfile = (NclFile) file->id;
	  NhlErrorTypes ret;
	  
	  if (PyDict_GetItemString(file->dimensions,name)) {
		  printf("dimension (%s) exists: cannot create\n",name);
		  return 0;
	  }
	  if (size == 0 && file->recdim != -1) {
		if(! nfile->file.advanced_file_structure)
		{
			PyErr_SetString(NIOError, "there is already an unlimited dimension");
			return -1;
		}
	  }
	  define_mode(file, 1);
	  qname = NrmStringToQuark(name);
	  ret = _NclFileAddDim(nfile,qname,(ng_size_t)size,(size == 0 ? 1 : 0));
	  if (ret > NhlWARNING) {
		  if (size == 0) {
			  PyDict_SetItemString(file->dimensions, name, Py_None);
#if 0
		          if(nfile->file.advanced_file_structure)
			      ++file->recdim;
		          else
                          {
                              if(0 > file->recdim)
			          file->recdim = _NclFileIsDim(nfile,qname);
                          }
#endif
			  file->recdim = _NclFileIsDim(nfile,qname);
		  }
		  else {
			  size_ob = PyInt_FromSsize_t(size);
			  PyDict_SetItemString(file->dimensions, name, size_ob);
			  Py_DECREF(size_ob);
		  }
	  }
	  return 0;
  }
  else
	  return -1;
}

static PyObject *
NioFileObject_new_dimension(NioFileObject *self, PyObject *args)
{
  char *name;
  PyObject *size_ob;
  Py_ssize_t size;
  if (!PyArg_ParseTuple(args, "sO", &name, &size_ob))
    return NULL;
  if (size_ob == Py_None)
    size = 0;
  else if (PyInt_Check(size_ob))
    size = (Py_ssize_t)PyInt_AsSsize_t(size_ob);
  else {
    PyErr_SetString(PyExc_TypeError, "size must be None or integer");
    return NULL;
  }
  if (NioFile_CreateDimension(self, name, (ng_size_t) size) == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else
    return NULL;
}

/* Create chunk dimension */

int NioFile_CreateChunkDimension(NioFileObject *file, char *name, Py_ssize_t size)
{
    PyObject *size_ob;
    NrmQuark qname;

    NclFile nfile = (NclFile) file->id;
    NhlErrorTypes ret;

    if(! check_if_open(file, 1))
        return -1;
   
    if(PyDict_GetItemString(file->chunk_dimensions,name))
    {
        printf("chunk_dimension (%s) exists: cannot create\n",name);
        return 0;
    }

    define_mode(file, 1);
    qname = NrmStringToQuark(name);
    ret = _NclFileAddChunkDim(nfile,qname,(ng_size_t)size,(size == 0 ? 1 : 0));
    if(ret > NhlWARNING)
    {
        size_ob = PyInt_FromSsize_t(size);
        PyDict_SetItemString(file->chunk_dimensions, name, size_ob); 
        Py_DECREF(size_ob);
    }

    return 0;
}

static PyObject *NioFileObject_new_chunk_dimension(NioFileObject *self, PyObject *args)
{
    char *name;
    PyObject *size_ob;
    Py_ssize_t size;

    if (!PyArg_ParseTuple(args, "sO", &name, &size_ob))
        return NULL;

    if (size_ob == Py_None)
        size = 1;
    else if (PyInt_Check(size_ob))
        size = (Py_ssize_t)PyInt_AsSsize_t(size_ob);
    else
    {
        PyErr_SetString(PyExc_TypeError, "size must be None or integer");
        return NULL;
    }

    if(1 > size)
        size = 1;

    if (NioFile_CreateChunkDimension(self, name, (ng_size_t) size) == 0)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }
    else
        return NULL;
}

/* Create group */

statichere NioFileObject* nio_create_group(NioFileObject* niofileobj, NrmQuark qname)
{
    NioFileObject *self;
    NclFile nclfile = (NclFile) niofileobj->id;
    NclAdvancedFile advfilegroup = NULL;
    NhlErrorTypes ret = -1;
    char* name;
    char* buf;
    Py_ssize_t len;

    name = NrmQuarkToString(qname);

    if(! check_if_open(niofileobj, -1))
        return NULL;

    self = PyObject_NEW(NioFileObject, &NioFile_Type);
    if(self == NULL)
        return NULL;

    self->dimensions = PyDict_New();
    self->chunk_dimensions = PyDict_New();
    self->ud_types = PyDict_New();
    self->variables = PyDict_New();
    self->groups = PyDict_New();
    self->attributes = PyDict_New();
    self->recdim = -1; /* for now */

    self->open = niofileobj->open;
    self->write = niofileobj->write;
    self->define = niofileobj->define;
    self->name = PyString_FromString(name);
    self->mode = niofileobj->mode;
    self->type = PyString_FromString("group");
    self->parent = niofileobj;
    len = PyString_Size(niofileobj->full_path);
    buf = malloc(len +1);
    strcpy(buf,PyString_AsString(niofileobj->full_path));
    if (! strcmp(buf,"/")) {
	    self->full_path = PyString_FromString(name);
    }
    else {
	    self->full_path = PyString_FromFormat("%s/%s",PyString_AsString(niofileobj->full_path), name);
    }
    /*printf("path is %s\n",PyString_AsString(self->full_path));*/

  /*
   *fprintf(stderr, "\nfunc: %s, in file: %s, line: %d\n",
   *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
   */

    ret = _NclFileAddGrp(nclfile, qname);
    if(NhlNOERROR != ret)
    {
        sprintf(err_buf,"Can not add group (%s) to file", NrmQuarkToString(qname));
        PyErr_SetString(NIOError, err_buf);
    }

    advfilegroup = _NclAdvancedGroupCreate(NULL, NULL, Ncl_File, 0,
                                           TEMPORARY, nclfile, qname);

    self->id = (void *) advfilegroup;
    self->gnode = (void *) advfilegroup->advancedfile.grpnode;
    self->top = niofileobj->top;

    return self;
}

static NioFileObject* NioFileObject_new_group(NioFileObject *self, PyObject *args)
{
    NioFileObject *group, *pgroup;
    NrmQuark qname;
    char* name;
    char *path;

    if(! PyArg_ParseTuple(args, "s", &name))
        return NULL;

    if(! check_if_open(self, 1))
        return NULL;

    define_mode(self, 1);

    group = (NioFileObject *) PyDict_GetItemString((PyObject *)self->groups,name);
    if (group)
    {
        printf("group (%s) exists: cannot create\n",name);
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
    } 
    else {/* top file level object */
	    pgroup =(NioFileObject *) PyDict_GetItemString(self->groups,"/");
	    if (pgroup)
		    group = nio_create_group(pgroup, qname);
    }
		    
    if(group)
    {
	path = PyString_AsString(pgroup->full_path);
        PyDict_SetItemString((PyObject *)pgroup->groups, name, (PyObject *)group);
	if (! strcmp(path,"/") || strlen(path) == 0) {
		PyDict_SetItem(pgroup->top->groups,PyString_FromFormat("%s",name), (PyObject *)group);
	}
	else {
		PyDict_SetItem(pgroup->top->groups,PyString_FromFormat("%s/%s",path,name), (PyObject *)group);
	}

        return group;
    }
    else
        return NULL;
}

/* Return a variable object referring to an existing variable */

/* Create ud_type */

statichere NioVariableObject*
nio_create_advancedfile_ud_type(NioFileObject *file, char *name)
{
    NioVariableObject *self;
    NclAdvancedFile advfile = (NclAdvancedFile) file->id;
    NclFileGrpNode* grpnode = advfile->advancedfile.grpnode;
    NclFileUDTNode* udtnode;
    char cname[1024];
    NrmQuark qvar;

    if(!check_if_open(file, -1))
        return NULL;

    self = PyObject_NEW(NioVariableObject, &NioVariable_Type);
    if(self == NULL)
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
    self->dimensions = (Py_ssize_t *)malloc(sizeof(Py_ssize_t));
    self->dimensions[0] = 1;

    self->name = (char *)malloc(strlen(name)+1);
    if(self->name != NULL)
        strcpy(self->name, name);

    return self;
}

/* Create variable */

statichere NioVariableObject*
nio_create_advancedfile_variable(NioFileObject *file, char *name, int id, 
                                 int ndims, NrmQuark *qdims)
{
    NioVariableObject *self;
    NclAdvancedFile advfile = (NclAdvancedFile) file->id;
    NclFileGrpNode* grpnode = advfile->advancedfile.grpnode;
    NclFileVarNode* varnode;
    NrmQuark qvar = NrmStringToQuark(name);
    int i, j;

    if(!check_if_open(file, -1))
        return NULL;

    self = PyObject_NEW(NioVariableObject, &NioVariable_Type);
    if(self == NULL)
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

    if(ndims > 0)
    {
        NclFileDimRecord* dimrec = grpnode->dim_rec;
        NclFileDimNode*   dimnode;

        self->dimensions = (Py_ssize_t *)malloc(ndims*sizeof(Py_ssize_t));
        if((self->dimensions != NULL) && (NULL != dimrec))
        {
            int dimid = -1;
            for(i = 0; i < ndims; ++i)
            {
                dimid = -1;
                for(j = 0; j < dimrec->n_dims; ++j)
                {
                    dimnode = &(dimrec->dim_node[j]);
                    if(dimnode->name == qdims[i])
                    {
                        dimid = j;
                        self->dimensions[i] = (Py_ssize_t)dimnode->size;
                        if(dimnode->is_unlimited)
                            self->unlimited = 1;
                        break;
                    }
                }

                if(0 > dimid)
                {
                    sprintf(err_buf,"Dimension (%s) not found",NrmQuarkToString(qdims[i]));
                    PyErr_SetString(NIOError, err_buf);
                    return NULL;
                }
            }
        }
    }
    self->name = (char *)malloc(strlen(name)+1);
    if(self->name != NULL)
        strcpy(self->name, name);
    self->attributes = PyDict_New();
    collect_advancedfile_attributes(NULL,varnode->att_rec, self->attributes);
    return self;
}

NioVariableObject *
NioFile_CreateVariable( NioFileObject *file, char *name, 
			  int typecode, char **dimension_names, int ndim)
{

  if (check_if_open(file, 1)) {
	  NioVariableObject *variable;
	  int i,id;
	  NclFile nfile = (NclFile) file->id;
	  NrmQuark *qdims = NULL; 
	  int dimid;
	  NhlErrorTypes ret;
	  NrmQuark qvar;
	  NrmQuark qtype;
          int ncl_ndims = ndim;
	  char *path;
	  NioFileObject *pgroup;

	  define_mode(file, 1);

	  variable = (NioVariableObject *) PyDict_GetItemString(file->variables,name);
	  if (variable) {
		  printf("variable (%s) exists: cannot create\n",name);
		  return variable;
	  }
		  
	  if (ndim > 0) {
		  qdims = (NrmQuark *)malloc(ndim*sizeof(NrmQuark));
		  if (! qdims) {
			  return (NioVariableObject *)PyErr_NoMemory();
		  }
	  }
	  else if (ndim == 0) {
		  qdims = (NrmQuark *)malloc(sizeof(NrmQuark));
		  dimid = -1;
		  if (! qdims) {
			  return (NioVariableObject *)PyErr_NoMemory();
		  }
		  *qdims = NrmStringToQuark("ncl_scalar");
		  ncl_ndims = 1;
	  }
	  for (i = 0; i < ndim; i++) {
		  qdims[i] = NrmStringToQuark(dimension_names[i]);
		  dimid = _NclFileIsDim(nfile,qdims[i]);
		  if (dimid == -1) {
			  sprintf(err_buf,"Dimension (%s) not found",dimension_names[i]);
			  PyErr_SetString(NIOError, err_buf);
			  if (qdims != NULL) 
				  free(qdims);
			  return NULL;
		  }
	  }
	  qtype = nio_type_from_code(typecode);
	  qvar = NrmStringToQuark(name);
	  if(nfile->file.advanced_file_structure)
	  {
	     if (file->parent) { 
		     pgroup = file;
	     } 
	     else {/* top file level object */
		     pgroup =(NioFileObject *) PyDict_GetItemString(file->groups,"/");
	     }
	     variable = (NioVariableObject *) PyDict_GetItemString(pgroup->variables,name);
	     if (variable) {
		     printf("variable (%s) exists: cannot create\n",name);
		     return variable;
	     }
	      ret = _NclFileAddVar(nfile,qvar,qtype,ncl_ndims,qdims);
	      if (ret > NhlWARNING) {
	 	  id = _NclFileIsVar(nfile,qvar);
		  variable = nio_create_advancedfile_variable(pgroup, name, id, ndim, qdims);
	      }
	      else
	      {
		  sprintf(err_buf,"Error creating variable (%s)",name);
		  PyErr_SetString(NIOError, err_buf);
		  if (qdims != NULL) 
			  free(qdims);
		  return NULL;
	      }
	  }
	  else
	  {
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
	      ret = _NclFileAddVar(nfile,qvar,qtype,ncl_ndims,qdims);
	      if (ret > NhlWARNING) {
	 	  id = _NclFileIsVar(nfile,qvar);
		  variable = nio_variable_new(file, name, id, 
					      data_type(nfile->file.var_info[id]->data_type),
					      ndim, qdims, 0);
	      }
	      else
	      {
		  sprintf(err_buf,"Error creating variable (%s)",name);
		  PyErr_SetString(NIOError, err_buf);
		  if (qdims != NULL) 
			  free(qdims);
		  return NULL;
	      }
	  }

	  PyDict_SetItemString(file->variables, name, (PyObject *)variable);

	  if (nfile->file.advanced_file_structure) {
		  path = PyString_AsString(pgroup->full_path);
		  if (! strcmp(path,"/") || strlen(path) == 0) {
			  PyDict_SetItem(pgroup->top->variables,PyString_FromFormat("%s",name), (PyObject *)variable);
		  }
		  else {
			  PyDict_SetItem(pgroup->top->variables,PyString_FromFormat("%s/%s",path,name), (PyObject *)variable);
		  }
	  }
	  return variable;
  }
  else {
	  return NULL;
  }
}

static PyObject *
NioFileObject_new_variable(NioFileObject *self, PyObject *args)
{
  NioVariableObject *var;
  char **dimension_names;
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
			  ltype  = 'S';
		  }
		  else {
			  sprintf (errbuf,"Cannot create variable (%s): string arrays not yet supported on write",name);
			  PyErr_SetString(PyExc_TypeError, errbuf);
			  return NULL;
		  }
	  } 
	  else {
		  if (type[0] == 'S' && type[1] == '1') {
			  ltype  = 'c';
		  }
		  else {
			  sprintf (errbuf,"Cannot create variable (%s): string arrays not yet supported on write",name);
			  PyErr_SetString(PyExc_TypeError, errbuf);
			  return NULL;
		  }
	  }
  }
  ndim = PyTuple_Size(dim);
  if (ndim == 0)
    dimension_names = NULL;
  else {
    dimension_names = (char **)malloc(ndim*sizeof(char *));
    if (dimension_names == NULL) {
      PyErr_SetString(PyExc_MemoryError, "out of memory");
      return NULL;
    }
  }
  for (i = 0; i < ndim; i++) {
    item = PyTuple_GetItem(dim, i);
    if (PyString_Check(item))
      dimension_names[i] = PyString_AsString(item);
    else {
      PyErr_SetString(PyExc_TypeError, "dimension name must be a string");
      free(dimension_names);
      return NULL;
    }
  }
  var = NioFile_CreateVariable(self, name, (int) ltype, dimension_names, ndim);
  if (! var) {
	  sprintf(err_buf,"Failed to create variable (%s)",name);
	  PyErr_SetString(NIOError, err_buf);
  }

  if (dimension_names)
	  free(dimension_names);
  return (PyObject *)var;
}

NioVariableObject *NioFile_CreateVLEN(NioFileObject *file, char *name, 
                                      int typecode, char **dimension_names, int ndim)
{
    NioVariableObject *variable;
    int i;
    NclFile nfile = (NclFile) file->id;
    NrmQuark *qdims = NULL; 
    NhlErrorTypes ret;
    NrmQuark qvar;
    NrmQuark qtype;
    NioFileObject *pgroup;

    if(! check_if_open(file, 1))
        return NULL;

    define_mode(file, 1);

    if (file->parent) { 
	    pgroup = file;
    } 
    else {/* top file level object */
	    pgroup =(NioFileObject *) PyDict_GetItemString(file->groups,"/");
    }

    variable = (NioVariableObject *) PyDict_GetItemString(pgroup->variables, name);
    if (variable)
    {
        printf("variable (%s) exists: cannot create\n",name);
        return variable;
    }
    	  
    if (ndim > 0)
    {
        qdims = (NrmQuark *)malloc(ndim*sizeof(NrmQuark));
        if (! qdims)
	    return (NioVariableObject *)PyErr_NoMemory();
    }
    else if (ndim == 0)
    {
        qdims = (NrmQuark *)malloc(sizeof(NrmQuark));
        if (! qdims)
    	    return (NioVariableObject *)PyErr_NoMemory();

        *qdims = NrmStringToQuark("ncl_scalar");
    }

    for(i = 0; i < ndim; ++i)
        qdims[i] = NrmStringToQuark(dimension_names[i]);

    qtype = nio_type_from_code(typecode);
    qvar = NrmStringToQuark(name);
    ret = _NclFileAddVlen(nfile, NrmStringToQuark("vlen"), qvar, qtype, qdims, ndim);
    if (ret > NhlWARNING)
    {
	char *path;
        variable = nio_create_advancedfile_variable(file, name, 0, ndim, qdims);
	path = PyString_AsString(pgroup->full_path);
	if (! strcmp(path,"/") || strlen(path) == 0) {
		PyDict_SetItem(pgroup->top->variables,PyString_FromFormat("%s",name), (PyObject *)variable);
	}
	else {
		PyDict_SetItem(pgroup->top->variables,PyString_FromFormat("%s/%s",path,name), (PyObject *)variable);
	}

    	PyDict_SetItemString(pgroup->variables, name, (PyObject *)variable);
    	return variable;
    }
    else
    {
        sprintf(err_buf,"Error creating variable (%s)",name);
        PyErr_SetString(NIOError, err_buf);
        if (qdims != NULL) 
    	    free(qdims);
        return NULL;
    }
}

static PyObject *NioFileObject_new_vlen(NioFileObject *self, PyObject *args)
{
    NioVariableObject *var;
    char **dimension_names;
    PyObject *item, *dim;
    char *name;
    int ndim;
    char* type;
    int i;
    char errbuf[256];
    char ltype;

    if(!PyArg_ParseTuple(args, "ssO!", &name, &type, &PyTuple_Type, &dim))
        return NULL;

    ltype = type[0];
    if(strlen(type) > 1)
    {
        if (type[0] == 'S' && type[1] == '1')
            ltype  = 'c';
        else
        {
            sprintf (errbuf,"Cannot create vlen (%s): string arrays not yet supported on write",name);
            PyErr_SetString(PyExc_TypeError, errbuf);
            return NULL;
        }
    }
    ndim = PyTuple_Size(dim);
    if (ndim == 0)
        dimension_names = NULL;
    else
    {
        dimension_names = (char **)malloc(ndim*sizeof(char *));
        if (dimension_names == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "out of memory");
            return NULL;
        }
    }

    for(i = 0; i < ndim; ++i)
    {
        item = PyTuple_GetItem(dim, i);
        if (PyString_Check(item))
            dimension_names[i] = PyString_AsString(item);
        else
        {
            PyErr_SetString(PyExc_TypeError, "dimension name must be a string");
            free(dimension_names);
            return NULL;
        }
    }

    var = NioFile_CreateVLEN(self, name, (int) ltype, dimension_names, ndim);
    if (! var)
    {
        sprintf(err_buf,"Failed to create vlen (%s)",name);
        PyErr_SetString(NIOError, err_buf);
    }

    if (dimension_names)
        free(dimension_names);
    return (PyObject *)var;
}

NioVariableObject *NioFile_CreateCOMPOUNDtype(NioFileObject *file, char *name, 
                                              char **memb_names, int *memb_types,
                                              int *memb_sizes, int nmemb)
{
    NioVariableObject *variable;
    int i;
    NclFile nfile = (NclFile) file->id;
    NhlErrorTypes ret;
    char     cname[1024];
    NrmQuark qvar;
    NrmQuark* memqname = NULL;
    NrmQuark* memqtype = NULL;

    if(! check_if_open(file, 1))
        return NULL;

    define_mode(file, 1);

    variable = (NioVariableObject *) PyDict_GetItemString(file->ud_types, name);
    if (variable)
    {
        printf("ud_types (%s) exists: cannot create\n",name);
        return variable;
    }
    	  
    if(nmemb > 0)
    {
        memqname = (NrmQuark *)malloc(nmemb*sizeof(NrmQuark));
        if(! memqname)
            return (NioVariableObject *)PyErr_NoMemory();
        memqtype = (NrmQuark *)malloc(nmemb*sizeof(NrmQuark));
        if(! memqtype)
            return (NioVariableObject *)PyErr_NoMemory();
    }
    else
    {
        fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
        fprintf(stderr, "\tnmemb = %d\n", nmemb);
        fprintf(stderr, "\tnumber of compound components must great than 0.\n");
        return NULL;
    }

    for(i = 0; i < nmemb; ++i)
    {
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

    if (ret > NhlWARNING)
    {
        variable = nio_create_advancedfile_ud_type(file, name);

    	PyDict_SetItemString(file->ud_types, name, (PyObject *)variable);
    	return variable;
    }
    else
    {
        sprintf(err_buf,"Error creating ud_type (%s)",name);
        PyErr_SetString(NIOError, err_buf);
        return NULL;
    }
}

NioVariableObject *NioFile_CreateCOMPOUND(NioFileObject *file, char *name, 
                                          char **dimension_names, int ndim,
                                          char **memb_names, int *memb_types,
                                          int *memb_sizes, int nmemb)
{
    NioVariableObject *variable;
    int i;
    NclFile nfile = (NclFile) file->id;
    NrmQuark *qdims = NULL; 
    NhlErrorTypes ret;
    char     cname[1024];
    NrmQuark qvar;
    NrmQuark* memqname = NULL;
    NrmQuark* memqtype = NULL;
    NioFileObject *pgroup;

    if(! check_if_open(file, 1))
        return NULL;

    define_mode(file, 1);

    if (file->parent) { 
	    pgroup = file;
    } 
    else {/* top file level object */
	    pgroup =(NioFileObject *) PyDict_GetItemString(file->groups,"/");
    }

    variable = (NioVariableObject *) PyDict_GetItemString(pgroup->variables, name);
    if (variable)
    {
        printf("variable (%s) exists: cannot create\n",name);
        return variable;
    }
    	  
    if (ndim > 0)
    {
        qdims = (NrmQuark *)malloc(ndim*sizeof(NrmQuark));
        if (! qdims)
	    return (NioVariableObject *)PyErr_NoMemory();
    }
    else if (ndim == 0)
    {
        qdims = (NrmQuark *)malloc(sizeof(NrmQuark));
        if (! qdims)
    	    return (NioVariableObject *)PyErr_NoMemory();

        *qdims = NrmStringToQuark("ncl_scalar");
    }

    for(i = 0; i < ndim; ++i)
        qdims[i] = NrmStringToQuark(dimension_names[i]);

    if(nmemb > 0)
    {
        memqname = (NrmQuark *)malloc(nmemb*sizeof(NrmQuark));
        if(! memqname)
            return (NioVariableObject *)PyErr_NoMemory();
        memqtype = (NrmQuark *)malloc(nmemb*sizeof(NrmQuark));
        if(! memqtype)
            return (NioVariableObject *)PyErr_NoMemory();
    }
    else
    {
        fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
        fprintf(stderr, "\tnmemb = %d\n", nmemb);
        fprintf(stderr, "\tnumber of compound components must great than 0.\n");
        return NULL;
    }

    for(i = 0; i < nmemb; ++i)
    {
        memqname[i] = NrmStringToQuark(memb_names[i]);
        memqtype[i] = nio_type_from_code(memb_types[i]);
    }

    qvar = NrmStringToQuark(name);
    strcpy(cname, name);
    strcat(cname, "_compound_type");
    ret = _NclFileAddCompound(nfile, NrmStringToQuark(cname), qvar,
                              ndim, qdims, nmemb, memqname, memqtype, memb_sizes);

    free(memqname);
    free(memqtype);

    if (ret > NhlWARNING)
    {
	char *path;
        variable = nio_create_advancedfile_variable(pgroup, name, 0, ndim, qdims);
	path = PyString_AsString(pgroup->full_path);
	if (! strcmp(path,"/") || strlen(path) == 0) {
		PyDict_SetItem(pgroup->top->variables,PyString_FromFormat("%s",name), (PyObject *)variable);
	}
	else {
		PyDict_SetItem(pgroup->top->variables,PyString_FromFormat("%s/%s",path,name), (PyObject *)variable);
	}

    	PyDict_SetItemString(pgroup->variables, name, (PyObject *)variable);
    	return variable;
    }
    else
    {
        sprintf(err_buf,"Error creating variable (%s)",name);
        PyErr_SetString(NIOError, err_buf);
        if (qdims != NULL) 
    	    free(qdims);
        return NULL;
    }
}

static PyObject *NioFileObject_new_compound_type(NioFileObject *self, PyObject *args)
{
    NioVariableObject *var;
    char **memb_names;
    int   *memb_types;
    int   *memb_sizes;
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
    char *typestr;

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
   */

    if(!PyArg_ParseTuple(args, "sO", &name, &type))
        return NULL;

    nmemb = PySequence_Size(type);

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
   *fprintf(stderr, "\tnmemb = %d\n", nmemb);
   */

    if (nmemb == 0)
    {
        memb_names = NULL;
        memb_types = NULL;
        memb_sizes = NULL;
    }
    else
    {
        memb_names = (char **)malloc(nmemb*sizeof(char *));
        if (memb_names == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "out of memory to define compound member name");
            return NULL;
        }
        memb_types = (int *)malloc(nmemb*sizeof(int));
        if (memb_types == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "out of memory to define compound member type");
            return NULL;
        }
        memb_sizes = (int *)malloc(nmemb*sizeof(int));
        if (memb_sizes == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "out of memory to define compound member size");
            return NULL;
        }
    }

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
   */

    seq = PySequence_Fast(type, "expected a sequence");
    for(i = 0; i < nmemb; ++i)
    {
        item = PySequence_Fast_GET_ITEM(seq, i);
        nitem = PySequence_Size(item);
      /*
       *fprintf(stderr, "\tItem %d has %d elements\n", i, nitem);
       */

        seq2 = PySequence_Fast(item,  "expected a sequence");

        item2 = PySequence_Fast_GET_ITEM(seq2, 0);
        if(PyString_Check(item2))
        {
            memb_names[i] = PyString_AsString(item2);
          /*
           *fprintf(stderr, "\tmemb_names[%d] = <%s>\n", i, memb_names[i]);
           */
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "memb name must be a string");
            free(memb_names);
            return NULL;
        }

        item2 = PySequence_Fast_GET_ITEM(seq2, 1);
        if(PyString_Check(item2))
        {
            typestr = PyString_AsString(item2);
            ltype = typestr[0];
            if(strlen(typestr) > 1)
            {
                if (typestr[0] == 'S' && typestr[1] == '1')
                    ltype = 'c';
                else
                {
                    sprintf (errbuf,"Cannot create compound (%s): string arrays not yet supported on write",name);
                    PyErr_SetString(PyExc_TypeError, errbuf);
                    return NULL;
                }
            }
            memb_types[i] = ltype;
          /*
           *fprintf(stderr, "\tmemb_types[%d] = <%c>\n", i, ltype);
           */
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "memb type must be a string");
            free(memb_names);
            return NULL;
        }

        memb_sizes[i] = 1;
        if(3 <= nitem)
        {
            item2 = PySequence_Fast_GET_ITEM(seq2, 2);
            if(PyInt_Check(item2))
            {
                memb_sizes[i] = (int)PyInt_AsLong(item2);
            }
            else if(PyLong_Check(item2))
            {
                memb_sizes[i] = (int)PyLong_AsLong(item2);
            }
            else if(PyString_Check(item2))
            {
                typestr = PyString_AsString(item2);
                sscanf(typestr, "%d", &(memb_sizes[i]));
              /*
               *fprintf(stderr, "\tmemb_size[%d] = <%s>\n", i, typestr);
               *fprintf(stderr, "\tmemb_size[%d] = %d\n", i, memb_sizes[i]);
               */
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "memb size must be a PyLong");
                free(memb_names);
                return NULL;
            }
        }
      /*
       *fprintf(stderr, "\tmemb_sizes[%d] = %d\n", i, memb_sizes[i]);
       */
    }

    var = NioFile_CreateCOMPOUNDtype(self, name, memb_names, memb_types, memb_sizes, nmemb);
    if (! var)
    {
        sprintf(errbuf, "Failed to create compound (%s)",name);
        PyErr_SetString(NIOError, errbuf);
    }

    if (memb_names)
        free(memb_names);
    if (memb_types)
        free(memb_types);
    if (memb_sizes)
        free(memb_sizes);
    return (PyObject *)var;
}

static PyObject *NioFileObject_new_compound(NioFileObject *self, PyObject *args)
{
    NioVariableObject *var;
    char **dimension_names;
    char **memb_names;
    int   *memb_types;
    int   *memb_sizes;
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
    char *typestr;

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
   */

    if(!PyArg_ParseTuple(args, "sOO!", &name, &type, &PyTuple_Type, &dim))
        return NULL;

    nmemb = PySequence_Size(type);

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
   *fprintf(stderr, "\tnmemb = %d\n", nmemb);
   */

    if (nmemb == 0)
    {
        memb_names = NULL;
        memb_types = NULL;
        memb_sizes = NULL;
    }
    else
    {
        memb_names = (char **)malloc(nmemb*sizeof(char *));
        if (memb_names == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "out of memory to define compound member name");
            return NULL;
        }
        memb_types = (int *)malloc(nmemb*sizeof(int));
        if (memb_types == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "out of memory to define compound member type");
            return NULL;
        }
        memb_sizes = (int *)malloc(nmemb*sizeof(int));
        if (memb_sizes == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "out of memory to define compound member size");
            return NULL;
        }
    }

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
   */

    seq = PySequence_Fast(type, "expected a sequence");
    for(i = 0; i < nmemb; ++i)
    {
        item = PySequence_Fast_GET_ITEM(seq, i);
        nitem = PySequence_Size(item);
      /*
       *fprintf(stderr, "\tItem %d has %d elements\n", i, nitem);
       */

        seq2 = PySequence_Fast(item,  "expected a sequence");

        item2 = PySequence_Fast_GET_ITEM(seq2, 0);
        if(PyString_Check(item2))
        {
            memb_names[i] = PyString_AsString(item2);
          /*
           *fprintf(stderr, "\tmemb_names[%d] = <%s>\n", i, memb_names[i]);
           */
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "memb name must be a string");
            free(memb_names);
            return NULL;
        }

        item2 = PySequence_Fast_GET_ITEM(seq2, 1);
        if(PyString_Check(item2))
        {
            typestr = PyString_AsString(item2);
            ltype = typestr[0];
            if(strlen(typestr) > 1)
            {
                if (typestr[0] == 'S' && typestr[1] == '1')
                    ltype = 'c';
                else
                {
                    sprintf (errbuf,"Cannot create compound (%s): string arrays not yet supported on write",name);
                    PyErr_SetString(PyExc_TypeError, errbuf);
                    return NULL;
                }
            }
            memb_types[i] = ltype;
          /*
           *fprintf(stderr, "\tmemb_types[%d] = <%c>\n", i, ltype);
           */
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "memb type must be a string");
            free(memb_names);
            return NULL;
        }

        memb_sizes[i] = 1;
        if(3 <= nitem)
        {
            item2 = PySequence_Fast_GET_ITEM(seq2, 2);
            if(PyInt_Check(item2))
            {
                memb_sizes[i] = (int)PyInt_AsLong(item2);
            }
            else if(PyLong_Check(item2))
            {
                memb_sizes[i] = (int)PyLong_AsLong(item2);
            }
            else if(PyString_Check(item2))
            {
                typestr = PyString_AsString(item2);
                sscanf(typestr, "%d", &(memb_sizes[i]));
              /*
               *fprintf(stderr, "\tmemb_size[%d] = <%s>\n", i, typestr);
               *fprintf(stderr, "\tmemb_size[%d] = %d\n", i, memb_sizes[i]);
               */
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "memb size must be a PyLong");
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
    else
    {
        dimension_names = (char **)malloc(ndim*sizeof(char *));
        if (dimension_names == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "out of memory");
            return NULL;
        }
    }

    for(i = 0; i < ndim; ++i)
    {
        item = PyTuple_GetItem(dim, i);
        if (PyString_Check(item))
            dimension_names[i] = PyString_AsString(item);
        else
        {
            PyErr_SetString(PyExc_TypeError, "dimension name must be a string");
            free(dimension_names);
            return NULL;
        }
    }

    var = NioFile_CreateCOMPOUND(self, name, dimension_names, ndim,
                                 memb_names, memb_types, memb_sizes, nmemb);
    if (! var)
    {
        sprintf(errbuf, "Failed to create compound (%s)",name);
        PyErr_SetString(NIOError, errbuf);
    }

    if (dimension_names)
        free(dimension_names);
    if (memb_names)
        free(memb_names);
    if (memb_types)
        free(memb_types);
    if (memb_sizes)
        free(memb_sizes);
    return (PyObject *)var;
}


static PyObject *
NioFileObject_Unlimited(NioFileObject *self, PyObject *args)
{
  PyObject *obj = NULL;
  int i;
  NrmQuark qstr;
  char *str = NULL;
  NclFile nfile = (NclFile) self->id;

  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;

  if (PyString_Check(obj)) {
	  str = PyString_AsString(obj);
	  if (! str) {
	      PyErr_SetString(PyExc_MemoryError, "out of memory");
	      return NULL;
	  }
	  qstr = NrmStringToQuark(str);
	  if(nfile->file.advanced_file_structure)
	  {
		  NclFileGrpNode* grpnode;
		  NclFileDimRecord* dimrec;

		  if (self->parent) {
			  grpnode = (NclFileGrpNode*)self->gnode;
			  dimrec = grpnode->dim_rec;
		  }
		  else {
			  char *cp = NULL;
			  char *dimname = NULL;
			  char *dimpath = NULL;
			  NioFileObject *grp = NULL;
			  cp = strrchr(str,'/');
			  if (cp >= str) {
				  dimname = cp + 1;
				  *cp = '\0';
				  if (strlen(str) > 0) {
					  dimpath = str[0] == '/' ? &(str[1]) : str;
				  }
			  }
			  else {
				  dimname = str;
				  dimpath = "/";
			  }
			  grp = (NioFileObject *)PyDict_GetItemString(self->groups,dimpath);
			  grpnode = grp->gnode;
			  dimrec = grpnode->dim_rec;
			  qstr = NrmStringToQuark(dimname);
		  }
		  if (dimrec) {
			  NclFileDimNode*   dimnode;
			  int i;
			  for (i = 0; i < dimrec->n_dims; i++) {
				  dimnode =  &(dimrec->dim_node[i]);
				  if (qstr == dimnode->name) {
					  return (dimnode->is_unlimited == 0 ? Py_False : Py_True);
				  }
			  }
		  }
		  PyErr_SetString(NIOError, self->parent == NULL ? "dimension name not found in file" : "dimension name not found in group");
		  return NULL;
	  }
	  else {
		  for (i = 0; i < nfile->file.n_file_dims; i++) {
			  if (nfile->file.file_dim_info[i]->dim_name_quark != qstr)
				  continue;
			  return (nfile->file.file_dim_info[i]->is_unlimited == 0 ? Py_False : Py_True);


		  }
		  PyErr_SetString(NIOError, "dimension name not found in file");
		  return NULL;
	  }
  }
  PyErr_SetString(PyExc_TypeError, "Invalid type");
  return NULL;
}


/* Return a variable object referring to an existing variable */

static NioVariableObject *
NioFile_GetVariable(NioFileObject *file, char *name)
{
  return (NioVariableObject *)PyDict_GetItemString(file->variables, name);
}

/* Synchronize output */

#if 0
int
NioFile_Sync(NioFileObject *file)
{
  if (check_if_open(file, 0)) {
#if 0
    define_mode(file, 0);
    if (ncsync(file->id) == -1) {
      nio_seterror();
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
  if (NioFile_Sync(self) == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else
    return NULL;
}
#endif

/* Close file */

int
NioFile_Close(NioFileObject *file)
{
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
	  PyObject *keys, *key, *value;
	  Py_ssize_t i, pos = 0;

	  /* it is necessary to delete the variable dictionary here because the
	     variables contain references to the file object */
	  file->open = 0;
	  /* DecREF references to the file object */

	  keys = PyDict_Keys(file->variables);
	  for (i = 0; i < PyList_Size(keys); i++) {
		  NioVariableObject *v = (NioVariableObject *) PyDict_GetItem(file->variables,PyList_GetItem(keys, i));
		  Py_DECREF(v->file);
	  }
	  Py_DECREF(keys);

	  keys = PyDict_Keys(file->groups);
	  for (i = 0; i < PyList_Size(keys); i++) {
		  NioFileObject *g = (NioFileObject *) PyDict_GetItem(file->groups,PyList_GetItem(keys, i));
		  Py_XDECREF(g->top);
		  Py_XDECREF(g->parent);
	  }
	  Py_DECREF(keys);

	  /* deleting the NIO file structures happens in the deallocation routine */
	  NioFileObject_dealloc(file);
	  return 0;
  }
  else {
	  return -1;
  }
}

static PyObject *
NioFileObject_close(NioFileObject *self, PyObject *args)
{
  char *history = NULL;
  if (!PyArg_ParseTuple(args, "|s", &history))
    return NULL;
  if (history && strlen(history) > 0) {
	  if (check_if_open(self,1)) {
		  NioFile_AddHistoryLine(self, history);
	  }
  }

  if (NioFile_Close(self) == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else
    return NULL;
}



/* Method table */

static PyMethodDef NioFileObject_methods[] = {
  {"close", (PyCFunction)NioFileObject_close, METH_VARARGS},
/* {"sync", (PyCFunction)NioFileObject_sync, 1}, */
  {"create_dimension", (PyCFunction)NioFileObject_new_dimension, METH_VARARGS},
  {"create_chunk_dimension", (PyCFunction)NioFileObject_new_chunk_dimension, METH_VARARGS},
  {"create_variable", (PyCFunction)NioFileObject_new_variable, METH_VARARGS},
  {"create_group", (PyCFunction)NioFileObject_new_group, METH_VARARGS},
  {"create_vlen", (PyCFunction)NioFileObject_new_vlen, METH_VARARGS},
  {"create_compound_type", (PyCFunction)NioFileObject_new_compound_type, METH_VARARGS},
  {"create_compound", (PyCFunction)NioFileObject_new_compound, METH_VARARGS},
  {"unlimited", (PyCFunction) NioFileObject_Unlimited,METH_VARARGS},
  {NULL, NULL}		/* sentinel */
};

static PyMemberDef NioFileObject_members[] = {
  {"dimensions", T_OBJECT, NhlOffset(NioFileObject,dimensions),READONLY,NULL},
  {"variables", T_OBJECT, NhlOffset(NioFileObject,variables),READONLY,NULL},
  {"attributes", T_OBJECT, NhlOffset(NioFileObject,attributes),READONLY,NULL},
  {"groups", T_OBJECT, NhlOffset(NioFileObject,groups),READONLY,NULL},
  {"name", T_OBJECT, NhlOffset(NioFileObject,name),READONLY,NULL},
  {"path", T_OBJECT, NhlOffset(NioFileObject,full_path),READONLY,NULL},
  {NULL,0, 0,0, NULL}		/* sentinel */
};


/* Attribute access */

PyObject *
NioFile_GetAttribute(NioFileObject *self, char *name)
{
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
      return (PyObject *)self->groups;
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
      return (PyObject *)self->full_path;
    }
    if (strcmp(name, "name") == 0) {
      Py_INCREF(self->name);
      return (PyObject *)self->name;
    }
    value = PyDict_GetItemString(self->attributes, name);
    if (value != NULL) {
      Py_INCREF(value);
      return value;
    }
    else {
        PyErr_Clear();
        return Py_FindMethod(NioFileObject_methods, (PyObject *)self, name);
    }
  }
  else
    return NULL;
}

static NclMultiDValData createAttMD(NclFile nfile, PyObject *attributes, char *name,
                                    PyObject *value)
{
    NclMultiDValData md = NULL;
    PyArrayObject *array = NULL;
    
    if (!value || value == Py_None)
    {
        return md;
    }
        
    if (PyString_Check(value))
    {
        ng_size_t len_dims = 1;
        NrmQuark *qval = malloc(sizeof(NrmQuark));
        qval[0] = NrmStringToQuark(PyString_AsString(value));
        md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
                     (void*)qval,NULL,1,&len_dims,
                     TEMPORARY,NULL,(NclTypeClass)nclTypestringClass);
    }
    else
    {
        ng_size_t dim_sizes = 1;
        int n_dims;
        NrmQuark qtype;
        int pyarray_type = PyArray_NOTYPE;
        PyArrayObject *tmparray = (PyArrayObject *)PyDict_GetItemString(attributes,name);
        if (tmparray != NULL)
            pyarray_type = tmparray->descr->type_num;

        array = (PyArrayObject *)PyArray_ContiguousFromAny(value, pyarray_type, 0, 1);
        if (array)
        {
            n_dims = (array->nd == 0) ? 1 : array->nd;
            qtype = nio_type_from_code(array->descr->type);
#if 1
            if (array->descr->elsize == 8 && qtype == NrmStringToQuark("long"))
            {
                 PyArrayObject *array2 = (PyArrayObject *) PyArray_Cast(array, PyArray_INT);
                 Py_DECREF(array);
                 array = array2;
                 qtype = NrmStringToQuark("integer");
            }
#endif
            if(NrmStringToQuark("object") == qtype)
	    {
		fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
				__PRETTY_FUNCTION__, __FILE__, __LINE__);
		fprintf(stderr, "\tNeed to handle object array, here, probably a compound attribute.\n");
		if(nfile->file.advanced_file_structure)
                {
                    NclAdvancedFile   advfile = (NclAdvancedFile) nfile;
                    NclFileVarNode*   varnode;

                    obj *listids = NULL;
                    ng_size_t n, n_dims;
                    ng_size_t nitems = 1;
                    ng_size_t curdim = 0;
                    ng_usize_t counter = 0;
                    ng_size_t *dims;

		    varnode = getVarFromGroup(advfile->advancedfile.grpnode, NrmStringToQuark(name));

                    if(array)
                    {
                        NclObjTypes the_obj_type = Ncl_Typelist;

                        n_dims = (ng_size_t) array->nd;
                        dims = (ng_size_t*) malloc(n_dims * sizeof(ng_size_t));
                        assert(dims);
                        for(n = 0; n < n_dims; ++n)
                        {
                            dims[n] = (ng_size_t)array->dimensions[n];
                            nitems *= dims[n];
                        }
      
                        listids = (obj*)NclMalloc((ng_usize_t)(nitems * sizeof(obj)));
                        assert(listids);

                        _NclBuildArrayOfList(listids, n_dims, dims);

                        _convertObj2COMPOUND((PyObject*)array, listids, varnode->comprec, n_dims, curdim, &counter);

                        md = _NclCreateVal(NULL,NULL,((the_obj_type & NCL_VAL_TYPE_MASK) ?
                                           Ncl_MultiDValData:the_obj_type),0,listids,
                                           NULL,n_dims,dims,TEMPORARY,NULL,
                                           (NclObjClass)((the_obj_type & NCL_VAL_TYPE_MASK) ?
                                           _NclTypeEnumToTypeClass(the_obj_type):NULL));

                        Py_DECREF(array);
                    }
                }
                else
                {
		    fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
		    fprintf(stderr, "\tCompound attributes only implemented for advanced file structure.\n");
		    return NULL;
                }
	    }
            else if (array)
            {
                ng_size_t *dims;
                void *data;
                if (array->nd == 0)
                {
                    dims = &dim_sizes;
                }
                else
                {
                    dims = (ng_size_t *) array->dimensions;
                }
                data = malloc(PyArray_NBYTES(array));
                memcpy(data,PyArray_DATA(array),PyArray_NBYTES(array));
                 
                md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
                                         (void*)data,NULL,n_dims,dims,
                                         TEMPORARY,NULL,_NclNameToTypeClass(qtype));
                 
            }
        }
    }

    return md;
}

static int set_advanced_file_attribute(NioFileObject *file, PyObject *attributes,
                                       char *name, PyObject *value)
{
    NclFile nfile = (NclFile) file->id;
    NhlErrorTypes ret;
    NclMultiDValData md = NULL;
    
    if (!value || value == Py_None)
    {
     /* delete attribute */
        ret = _NclFileDeleteAtt(nfile,NrmStringToQuark(name));
        PyObject_DelItemString(attributes,name);
        return 0;
    }
        
    md = createAttMD(nfile, attributes, name, value);

    if (! md)
    {
        nio_ncerr = 23;
        nio_seterror();
        return -1;
    }
    
    ret = _NclFileWriteAtt(nfile,NrmStringToQuark(name),md,NULL);

    if (ret > NhlFATAL)
    {
        if (PyString_Check(value))
        {
            PyDict_SetItemString(attributes, name, value);
        }
        else
        {
            PyArrayObject *array = NULL;
            int pyarray_type = PyArray_NOTYPE;
            PyArrayObject *tmparray = (PyArrayObject *)PyDict_GetItemString(attributes,name);
            if (tmparray != NULL)
                pyarray_type = tmparray->descr->type_num;
            array = (PyArrayObject *)PyArray_ContiguousFromAny(value, pyarray_type, 0, 1);
            PyDict_SetItemString(attributes, name, (PyObject *)array);
        }
    }

    return 0;
}

static int set_advanced_variable_attribute(NioFileObject *file, NioVariableObject *self,
                                           PyObject *attributes, char *name, PyObject *value)
{
    NclFile nfile = (NclFile) file->id;
    NhlErrorTypes ret;
    NclMultiDValData md = NULL;
    
    if (!value || value == Py_None)
    {
        /* delete attribute */
        ret = _NclFileDeleteVarAtt(nfile, NrmStringToQuark(self->name),
                                   NrmStringToQuark(name));
        PyObject_DelItemString(attributes,name);
        return 0;
    }
        
    md = createAttMD(nfile, attributes, name, value);

    if (! md)
    {
        nio_ncerr = 23;
        nio_seterror();
        return -1;
    }
    
    ret = _NclFileWriteVarAtt(nfile, NrmStringToQuark(self->name),
                              NrmStringToQuark(name),md,NULL);

    if (ret > NhlFATAL)
    {
        if (PyString_Check(value))
        {
            PyDict_SetItemString(attributes, name, value);
        }
        else
        {
            PyArrayObject *array = NULL;
            int pyarray_type = PyArray_NOTYPE;
            PyArrayObject *tmparray = (PyArrayObject *)PyDict_GetItemString(attributes,name);
            if (tmparray != NULL)
                pyarray_type = tmparray->descr->type_num;
            array = (PyArrayObject *)PyArray_ContiguousFromAny(value, pyarray_type, 0, 1);
            PyDict_SetItemString(attributes, name, (PyObject *)array);
        }
    }

    return 0;
}

int
NioFile_SetAttribute(NioFileObject *self, char *name, PyObject *value)
{
  NclFile nfile = (NclFile) self->id;
  nio_ncerr = 0;
  if (check_if_open(self, 1)) {
	  if (strcmp(name, "dimensions") == 0 ||
	      strcmp(name, "chunk_dimensions") == 0 ||
	      strcmp(name, "ud_types") == 0 ||
	      strcmp(name, "variables") == 0 ||
	      strcmp(name, "groups") == 0 ||
	      strcmp(name, "attributes") == 0 ||
	      strcmp(name, "__dict__") == 0) {
		  PyErr_SetString(PyExc_TypeError, "attempt to set read-only attribute");
		  return -1;
	  }
	  define_mode(self, 1);
	  if(nfile->file.advanced_file_structure) {
		  return set_advanced_file_attribute(self, self->attributes, name, value);
	  }
	  else
		  return set_attribute(self, NC_GLOBAL, self->attributes, name, value);
  }
  else
	  return -1;
}

int
NioFile_SetAttributeString(NioFileObject *self, char *name, char *value)
{
  PyObject *string = PyString_FromString(value);
  if (string != NULL)
    return NioFile_SetAttribute(self, name, string);
  else
    return -1;
}

int
NioFile_AddHistoryLine(NioFileObject *self, char *text)
{
  static char *history = "history";
  int alloc, old, new, new_alloc;
  PyStringObject *new_string;
  PyObject *h = NioFile_GetAttribute(self, history);
  if (h == NULL) {
    PyErr_Clear();
    alloc = 0;
    old = 0;
    new = strlen(text);
  }
  else {
    alloc = PyString_Size(h);
    old = strlen(PyString_AsString(h));
    new = old + strlen(text) + 1;
  }
  new_alloc = (new <= alloc) ? alloc : new + 500;
  new_string = (PyStringObject *)PyString_FromStringAndSize(NULL, new_alloc);
  if (new_string) {
    char *s = new_string->ob_sval;
    int len, ret;
    memset(s, 0, new_alloc+1);
    if (h == NULL)
      len = -1;
    else {
      strcpy(s, PyString_AsString(h));
      len = strlen(s);
      s[len] = '\n';
    }
    strcpy(s+len+1, text);
    ret = NioFile_SetAttribute(self, history, (PyObject *)new_string);
    Py_XDECREF(h);
    Py_DECREF(new_string);
    return ret;
  }
  else
    return -1;
}

/* Printed representation */
static PyObject *
NioFileObject_repr(NioFileObject *file)
{
  char buf[512];
  sprintf(buf, "<%s NioFile object '%.256s', mode '%.10s' at %lx>",
	  file->open ? "open" : "closed",
	  PyString_AsString(file->name),
	  PyString_AsString(file->mode),
	  (long)file);
  return PyString_FromString(buf);
}

#define BUF_INSERT(tbuf) \
	len = strlen(tbuf); \
	while (bufpos + len > buflen - 2) { \
		buf = realloc(buf,buflen + bufinc); \
		buflen += bufinc; \
	} \
	strcpy(&(buf[bufpos]),tbuf); \
	bufpos += len;

void
format_object(char *buf,PyObject *obj,int code)
{

	switch(code) {
	case 'i':
		sprintf(buf,"%d",(int)PyInt_AsLong(obj));
		break;
	case 'l':
		sprintf(buf,"%lld",PyLong_AsLongLong(obj));
		break;
	case 'f':
		sprintf(buf,"%.7g",PyFloat_AsDouble(obj));
		break;
	case 'd':
		sprintf(buf,"%.16g",PyFloat_AsDouble(obj));
		break;
	default:
		sprintf(buf,"%s",PyString_AsString(PyObject_Str(obj)));
	}
}

static void insert2buf(char* tbuf, char** buf, int *bufpos, int *buflen, int bufinc)
{
    int len = strlen(tbuf);

    while((*bufpos + len) > (*buflen - 2))
    {
        *buflen += bufinc;
        *buf = realloc(*buf, *buflen);
    }

    strcpy(&((*buf)[*bufpos]),tbuf);		\
    *bufpos += len;
}

static void attrec2buf(PyObject *attributes, NclFileAttRecord* attrec,
                       char** buf, int *bufpos, int *buflen, int bufinc,
                       char* title, char* titlename, char* attname)
{
    NclFileAttNode*   attnode;
    PyObject *att_val;
    char tbuf[1024];
    char* name;
    int i;

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *    __PRETTY_FUNCTION__, __FILE__, __LINE__);
   */

    if(strlen(title) > 1)
        sprintf(tbuf,"%s:\t%.510s\n", title, titlename);
    else
        sprintf(tbuf,"\t%.510s\n", titlename);
    insert2buf(tbuf, buf, bufpos, buflen, bufinc);


/*    sprintf(tbuf,"   %s:\n", attname);
      insert2buf(tbuf, buf, bufpos, buflen, bufinc);*/
   

    if(NULL == attrec)
        return;

    for(i = 0; i < attrec->n_atts; ++i)
    {
        attnode = &(attrec->att_node[i]);

        name = NrmQuarkToString(attnode->name);

        sprintf(tbuf,"      %s : ", name);
        insert2buf(tbuf, buf, bufpos, buflen, bufinc);
        att_val = PyDict_GetItemString(attributes, name);

	if (! att_val) {
		continue;
	}
        if (PyString_Check(att_val))
        {
            insert2buf(PyString_AsString(att_val), buf, bufpos, buflen, bufinc);
            insert2buf("\n", buf, bufpos, buflen, bufinc);
        }
        else
        {
            int k;
            PyArrayObject *att_arr_val = (PyArrayObject *)att_val;
            if(att_arr_val->nd == 0 || att_arr_val->dimensions[0] == 1)
            {
                PyObject *att = att_arr_val->descr->f->getitem(PyArray_DATA(att_val),att_val);

                format_object(tbuf,att,att_arr_val->descr->type);
                insert2buf(tbuf, buf, bufpos, buflen, bufinc);
                insert2buf("\n", buf, bufpos, buflen, bufinc);
            }
            else
            {
                insert2buf("[", buf, bufpos, buflen, bufinc);
                for(k = 0; k < att_arr_val->dimensions[0]; ++k)
                {
                    PyObject *att = att_arr_val->descr->f->getitem(att_arr_val->data + 
                                               k * att_arr_val->descr->elsize,att_val);
                    format_object(tbuf,att,att_arr_val->descr->type);
                    insert2buf(tbuf, buf, bufpos, buflen, bufinc);
                    if (k < att_arr_val->dimensions[0] - 1)
                        sprintf(tbuf,", ");
                    else
                        sprintf(tbuf,"]\n");
                    insert2buf(tbuf, buf, bufpos, buflen, bufinc);
                }
            }
        }
    }
}
char* NioVarInfo2str(NioVariableObject *var, NclFileVarNode* varnode)
{
    char *buf[1];
    char *name, *vname;
    char tbuf[1024];
    int bufinc = 8192;
    int buflen = 0;
    int bufpos = 0;
    NioFileObject *file = var->file;
    NclFileAttRecord* attrec;
    NclFileDimRecord* dimrec;
    NclFileDimNode*   dimnode;
    NioVariableObject *vobj;
    NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
    int j;

    *buf = malloc(bufinc);
    buflen = bufinc;

    vname = NrmQuarkToString(varnode->name);

    dimrec = varnode->dim_rec;

    if(NULL != dimrec)
    {
	    dimnode = &(dimrec->dim_node[0]);
	    if((1 == dimrec->n_dims) && (dimnode->name == scalar_dim))
	    {
                    sprintf(tbuf,"   %s %s\n",
			    _NclBasicDataTypeToName(varnode->type), vname);
                    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
	    }
	    else
	    {
                    sprintf(tbuf,"   %s %s [ ",
			    _NclBasicDataTypeToName(varnode->type), vname);
                    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

                    for(j = 0; j < dimrec->n_dims; ++j)
                    {
			    dimnode = &(dimrec->dim_node[j]);
			    name = NrmQuarkToString(dimnode->name);

			    if(j)
				    sprintf(tbuf,", %s|%ld", name, (long)dimnode->size);
			    else
				    sprintf(tbuf,"%s|%ld", name, (long)dimnode->size);
			    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
                    }
                    sprintf(tbuf," ]");
                    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
	    }
    }

    vobj = (NioVariableObject *) PyDict_GetItemString(file->variables,vname);

    attrec = varnode->att_rec;
    attrec2buf(vobj->attributes, attrec, buf, &bufpos, &buflen, bufinc, " ", " ", vname);
    sprintf(tbuf,"\n");
    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

    return *buf;
}

char* NioGroupInfo2str(NioFileObject *file, NclFileGrpNode* grpnode, char* title, char* attname)
{
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
    NclFileDimNode*   dimnode;

    NclFileVarRecord* varrec;
    NclFileVarNode*   varnode;

    NclFileGrpRecord* grprec;

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *        __PRETTY_FUNCTION__, __FILE__, __LINE__);
   */

    *buf = malloc(bufinc);
    buflen = bufinc;

    sprintf(titlename,"%.510s", PyString_AsString(file->name));

    if (file->parent == NULL) { /* top level file object */
	   file = (NioFileObject *) PyDict_GetItemString(file->groups,"/");
    }

    attrec = grpnode->att_rec;
    attrec2buf(file->attributes, attrec, buf, &bufpos, &buflen, bufinc, title, titlename, attname);

    sprintf(tbuf,"   dimensions:\n");
    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

    dimrec = grpnode->dim_rec;

    if(NULL != dimrec)
    {
        for(i = 0; i < dimrec->n_dims; ++i)
        {
            dimnode = &(dimrec->dim_node[i]);
            if(dimnode->name == scalar_dim)
                continue;

            name = NrmQuarkToString(dimnode->name);

            if(dimnode->is_unlimited)
                sprintf(tbuf,"      %s = %ld // unlimited\n", name, dimnode->size);
            else
                sprintf(tbuf,"      %s = %ld\n", name, dimnode->size);
            insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
        }        
    }

    dimrec = grpnode->chunk_dim_rec;

    if(NULL != dimrec)
    {
        sprintf(tbuf,"   chunk dimensions:\n");
        insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

        for(i = 0; i < dimrec->n_dims; ++i)
        {
            dimnode = &(dimrec->dim_node[i]);
            if(dimnode->name == scalar_dim)
                continue;

            name = NrmQuarkToString(dimnode->name);

            sprintf(tbuf,"      %s = %ld\n", name, dimnode->size);
            insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
        }
    }

    sprintf(tbuf,"   variables:\n");
    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

    varrec = grpnode->var_rec;
    if(NULL != varrec)
    {
        NioVariableObject *vobj;
        char* vname;
        int j;

        for(i = 0; i < varrec->n_vars; ++i)
        {
            varnode = &(varrec->var_node[i]);
            vname = NrmQuarkToString(varnode->name);

            dimrec = varnode->dim_rec;

            if(NULL != dimrec)
            {
                dimnode = &(dimrec->dim_node[0]);
	        if((1 == dimrec->n_dims) && (dimnode->name == scalar_dim))
                {
                    sprintf(tbuf,"   %s %s\n",
                        _NclBasicDataTypeToName(varnode->type), vname);
                    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
                }
                else
                {
                    sprintf(tbuf,"   %s %s [ ",
                        _NclBasicDataTypeToName(varnode->type), vname);
                    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);

                    for(j = 0; j < dimrec->n_dims; ++j)
                    {
                        dimnode = &(dimrec->dim_node[j]);

                        name = NrmQuarkToString(dimnode->name);

                        if(j)
                            sprintf(tbuf,", %s|%ld", name, (long)dimnode->size);
                        else
                            sprintf(tbuf,"%s|%ld", name, (long)dimnode->size);
                        insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
                    }
                    sprintf(tbuf," ]");
                    insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
                }
            }

            vobj = (NioVariableObject *) PyDict_GetItemString(file->variables,vname);

            attrec = varnode->att_rec;
#if 1
            attrec2buf(vobj->attributes, attrec, buf, &bufpos, &buflen, bufinc, " ", " ", vname);
#else
            sprintf(titlename,"%s", vname);
            attrec2buf(vobj->attributes, attrec, buf, &bufpos, &buflen, bufinc, "Variable", titlename, vname);
#endif
            sprintf(tbuf,"\n");
            insert2buf(tbuf, buf, &bufpos, &buflen, bufinc);
        }
	
    }

    grprec = grpnode->grp_rec;

    if(NULL != grprec)
    {
        char titlebuf[256];
        char attribuf[256];
        char* grpbuf;

        for(i = 0; i < grprec->n_grps; ++i)
        {
	    NioFileObject *group;
            name = NrmQuarkToString(grprec->grp_node[i]->name);
            sprintf(titlebuf,"Nio group <%s>", name);
            sprintf(attribuf,"group attributes");
	    group = (NioFileObject *) PyDict_GetItemString(file->groups,name);

            grpbuf = NioGroupInfo2str(group, grprec->grp_node[i], titlebuf, attribuf);

            insert2buf(grpbuf, buf, &bufpos, &buflen, bufinc);
            insert2buf("\n", buf, &bufpos, &buflen, bufinc);
            free(grpbuf);
        }
    }

    return *buf;
}

/* Printed representation */
static PyObject *
NioFileObject_str(NioFileObject *file)
{
	char *buf = NULL;
	NclFile nfile = (NclFile) file->id;
	PyObject *pystr;

	if (! check_if_open(file, -1)) {
		PyErr_Clear();
		return NioFileObject_repr(file);
	}

	if(nfile->file.advanced_file_structure)
	{
		NclFileGrpNode *grpnode = (NclFileGrpNode *) file->gnode;
		char *name, *attname;
		NioFileObject *group;

              /*
	       *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	       *		__PRETTY_FUNCTION__, __FILE__, __LINE__);
               */
		if (file->parent == NULL) {
			name = "Nio file";
			attname = "file attributes";
			group = (NioFileObject *)PyDict_GetItemString(file->groups,"/");
			group = file;
		}
		else {
			name = "Nio group";
			attname = "group attributes";
			group = file;
		}
		buf = NioGroupInfo2str(group, grpnode,
                                          name, attname);
	}
	else
	{
	char tbuf[1024];
	int len;
	int bufinc = 4096;
	int buflen = 0;
	int bufpos = 0;
	int i;

	NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
	PyObject *att_val;

      /*
       *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
       *		__PRETTY_FUNCTION__, __FILE__, __LINE__);
       */

	buf = malloc(bufinc);
	buflen = bufinc;
	sprintf(tbuf,"Nio file:\t%.510s\n",PyString_AsString(file->name));
	BUF_INSERT(tbuf);
	sprintf(tbuf,"   global attributes:\n");
	BUF_INSERT(tbuf);
	/* The problem with just printing the Python dictionary is that it 
	 * has no consistent ordering. Since we want an order, we will need
	 * to use the nio library's records at least to get the names of atts, dims, and vars.
	 * On the other hand, use Nio python tools to get the attribute values, because this will
	 * format array values in a Pythonic way.
	 */

	for (i = 0; i < nfile->file.n_file_atts; i++) {
		char *attname;
		if (! nfile->file.file_atts[i])
			continue;
		attname = NrmQuarkToString(nfile->file.file_atts[i]->att_name_quark);
		sprintf(tbuf,"      %s : ",attname);
		BUF_INSERT(tbuf);
		att_val = PyDict_GetItemString(file->attributes,attname);
		if (PyString_Check(att_val)) {
			BUF_INSERT(PyString_AsString(att_val));
			BUF_INSERT("\n");
		}
		else {
			int k;
			PyArrayObject *att_arr_val = (PyArrayObject *)att_val;
			if (att_arr_val->nd == 0 || att_arr_val->dimensions[0] == 1) {
				PyObject *att = att_arr_val->descr->f->getitem(PyArray_DATA(att_val),att_val);

				format_object(tbuf,att,att_arr_val->descr->type);
				BUF_INSERT(tbuf);
				BUF_INSERT("\n");
			}
			else {
				sprintf(tbuf,"[");
				BUF_INSERT(tbuf);
				for (k = 0; k < att_arr_val->dimensions[0]; k++) {
					PyObject *att = att_arr_val->descr->f->getitem(att_arr_val->data + 
										       k * att_arr_val->descr->elsize,att_val);
					format_object(tbuf,att,att_arr_val->descr->type);
					BUF_INSERT(tbuf);
					if (k < att_arr_val->dimensions[0] - 1) {
						sprintf(tbuf,", ");
					}
					else {
						sprintf(tbuf,"]\n");
					}
					BUF_INSERT(tbuf);
				}
			}
		}
	}
	sprintf(tbuf,"   dimensions:\n");
	BUF_INSERT(tbuf);
	for (i = 0; i < nfile->file.n_file_dims; i++) {
		char *dim;
		if (! nfile->file.file_dim_info[i])
			continue;
		if (nfile->file.file_dim_info[i]->dim_name_quark == scalar_dim)
		    continue;
		dim = NrmQuarkToString(nfile->file.file_dim_info[i]->dim_name_quark);
		if (nfile->file.file_dim_info[i]->is_unlimited) {
			sprintf(tbuf,"      %s = %ld // unlimited\n",dim, nfile->file.file_dim_info[i]->dim_size);
		}
		else {
			sprintf(tbuf,"      %s = %ld\n",dim, nfile->file.file_dim_info[i]->dim_size);
		}		
		BUF_INSERT(tbuf);
	}

	sprintf(tbuf,"   variables:\n");
	BUF_INSERT(tbuf);

	for(i = 0; i < nfile->file.n_vars; i++) {
		NclFileAttInfoList* step;
		char *vname;
		NioVariableObject *vobj;
		int j, dim_ix;
 
		if(nfile->file.var_info[i] == NULL) {
			continue;
		}
		vname = NrmQuarkToString(nfile->file.var_info[i]->var_name_quark);
		if (nfile->file.var_info[i]->num_dimensions == 1 && 
		    nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[0]]->dim_name_quark == scalar_dim) {
			sprintf(tbuf,"      %s %s\n",
				_NclBasicDataTypeToName(nfile->file.var_info[i]->data_type),vname);
			BUF_INSERT(tbuf);
		}
		else {
			sprintf(tbuf,"      %s %s [ ",
				_NclBasicDataTypeToName(nfile->file.var_info[i]->data_type),vname);
			BUF_INSERT(tbuf);
			for(j=0; j < nfile->file.var_info[i]->num_dimensions; j++) {
				dim_ix = nfile->file.var_info[i]->file_dim_num[j];
				if (j != nfile->file.var_info[i]->num_dimensions - 1) {
					sprintf(tbuf,"%s, ",NrmQuarkToString(nfile->file.file_dim_info[dim_ix]->dim_name_quark));
				}
				else {
					sprintf(tbuf,"%s ]\n",NrmQuarkToString(nfile->file.file_dim_info[dim_ix]->dim_name_quark));
				}
				BUF_INSERT(tbuf);
			}
		}
		vobj = (NioVariableObject *) PyDict_GetItemString(file->variables,vname);
		step = nfile->file.var_att_info[i];
		while(step != NULL) {
			char *aname = NrmQuarkToString(step->the_att->att_name_quark);
			sprintf(tbuf,"         %s :\t", aname);
			BUF_INSERT(tbuf);
			att_val = PyDict_GetItemString(vobj->attributes,aname);
			if (PyString_Check(att_val)) {
				BUF_INSERT(PyString_AsString(att_val));
				BUF_INSERT("\n");
			}
			else {
				int k;
				PyArrayObject *att_arr_val = (PyArrayObject *)att_val;
				if (att_arr_val->nd == 0 || att_arr_val->dimensions[0] == 1) {
					PyObject *att = att_arr_val->descr->f->getitem(PyArray_DATA(att_val),att_val);
					format_object(tbuf,att,att_arr_val->descr->type);
					/*sprintf(tbuf,"%s\n",PyString_AsString(PyObject_Str(att)));*/
					BUF_INSERT(tbuf);
					BUF_INSERT("\n");
				}
				else {
					sprintf(tbuf,"[");
					BUF_INSERT(tbuf);
					for (k = 0; k < att_arr_val->dimensions[0]; k++) {
						PyObject *att = 
							att_arr_val->descr->f->getitem(att_arr_val->data + 
										       k * att_arr_val->descr->elsize,att_val);

						format_object(tbuf,att,att_arr_val->descr->type);
						/*sprintf(tbuf,"%s",PyString_AsString(PyObject_Str(att)));*/
						BUF_INSERT(tbuf);
						if (k < att_arr_val->dimensions[0] - 1) {
							sprintf(tbuf,", ");
						}
						else {
							sprintf(tbuf,"]\n");
						}
						BUF_INSERT(tbuf);
					}
				}
			}
			step = step->next;
		}
	}
	}

	pystr = PyString_FromString(buf);
	free(buf);

	return pystr;
}


    


/* Type definition */

statichere PyTypeObject NioFile_Type = {
  PyObject_HEAD_INIT(NULL)
  0,		/*ob_size*/
  "_Nio._NioFile",	/*tp_name*/
  sizeof(NioFileObject),	/*tp_basicsize*/
  0,		/*tp_itemsize*/
  /* methods */
  (destructor)NioFileObject_dealloc, /*tp_dealloc*/
  0,			/*tp_print*/
  (getattrfunc)NioFile_GetAttribute, /*tp_getattr*/
  (setattrfunc)NioFile_SetAttribute, /*tp_setattr*/
  0,			/*tp_compare*/
  (reprfunc)NioFileObject_repr,   /*tp_repr*/
  0,			/*tp_as_number*/
  0,			/*tp_as_sequence*/
  0,			/*tp_as_mapping*/
  0,			/*tp_hash*/
  0,                    /*tp_call*/
  (reprfunc)NioFileObject_str,   /*tp_str*/
  0,			/*tp_getattro*/
  0,                    /*tp_setattro*/
  0,			/*tp_as_buffer*/
  (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE),  /*tp_flags*/
  0,                     /*tp_doc*/
  (traverseproc)0,                          /*tp_traverse */
  (inquiry)0,                               /*tp_clear */
  (richcmpfunc)0,           /*tp_richcompare */
  offsetof(NioFileObject,weakreflist),     /*tp_weaklistoffset */

  /* Iterator support (use standard) */

  (getiterfunc) 0,                  /* tp_iter */
  (iternextfunc)0,                          /* tp_iternext */

  /* Sub-classing (new-style object) support */

  NioFileObject_methods,                    /* tp_methods */
  NioFileObject_members,                    /* tp_members */
  0,                                        /* tp_getset */
  0,                                        /* tp_base */
  0,                                        /* tp_dict */
  0,                                        /* tp_descr_get */
  0,                                        /* tp_descr_set */
  0,                                        /* tp_dictoffset */
  (initproc)0,                              /* tp_init */
  0,                                        /* tp_alloc */
  0,                                        /* tp_new */
  0,                                        /* tp_free */
  0,                                        /* tp_is_gc */
  0,                                        /* tp_bases */
  0,                                        /* tp_mro */
  0,                                        /* tp_cache */
  0,                                        /* tp_subclasses */
  0                                         /* tp_weaklist */
};

/*
 * NIOVariable object
 * (type declaration in niomodule.h)
 */

/* Destroy variable object */

static void
NioVariableObject_dealloc(NioVariableObject *self)
{
  PyObject *keys;
  Py_ssize_t i;

  if (self->qdims != NULL)
    free(self->qdims);
  if (self->dimensions != NULL)
    free(self->dimensions);
  if (self->name != NULL)
    free(self->name);
  /*
  keys = PyDict_Keys(self->attributes);
  for (i = 0; i < PyList_Size(keys); i++) {
	  PyObject *a = (PyObject *) PyDict_GetItem(self->attributes,PyList_GetItem(keys, i));
	  PyDict_DelItem(self->attributes,PyList_GetItem(keys, i));
  }
  Py_DECREF(keys);
  */

  Py_XDECREF(self->attributes);
  /*Py_XDECREF(self->file);*/
  PyObject_DEL(self);
  /*Py_XDECREF(self);*/
}

/* Create variable object */

statichere NioVariableObject* nio_read_advanced_variable(NioFileObject* file,
                                                         NclFileVarNode* varnode, int id)
{
    NioVariableObject *self;
    NclFileDimRecord* dimrec = varnode->dim_rec;
    NclFileDimNode*   dimnode;
    NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
    NrmQuark* qdims = NULL;
    char* name = NrmQuarkToString(varnode->name);
    int i;

    if(! check_if_open(file, -1))
        return NULL;

    self = PyObject_NEW(NioVariableObject, &NioVariable_Type);
    if(self == NULL)
       return NULL;

    self->file = file;
    Py_INCREF(file);
    self->id = id;
    self->type = data_type(varnode->type);
    self->nd = 0;
    self->qdims = NULL;
    self->unlimited = 0;
    self->dimensions = NULL;
    self->name = (char *)malloc(strlen(name)+1);
    if(self->name != NULL)
        strcpy(self->name, name);
    self->attributes = PyDict_New();
    collect_advancedfile_attributes(NULL,varnode->att_rec, self->attributes);

    /* shouldn't this be an error */
    if(NULL == dimrec)
        return self;
    /* --------------- */

    if (dimrec->n_dims == 1 &&  dimrec->dim_node[0].name == scalar_dim) {
	    return self;
    }
    self->dimensions = (Py_ssize_t *)malloc(dimrec->n_dims*sizeof(Py_ssize_t));
    if(NULL == self->dimensions)
    {
        PyErr_NoMemory();
        return self;
    }

    qdims = (NrmQuark *) malloc(dimrec->n_dims *sizeof(NrmQuark));

    if(NULL == qdims)
    {
        PyErr_NoMemory();
        return self;
    }

    for(i = 0; i < dimrec->n_dims; ++i)
    {
        dimnode = &(dimrec->dim_node[i]);
        if(dimnode->id < 0)
        {
            sprintf(err_buf,"Dimension (%s) not found",NrmQuarkToString(qdims[i]));
            PyErr_SetString(NIOError, err_buf);
            return self;
        }
        qdims[i] = dimnode->name;
        self->dimensions[i] = (Py_ssize_t)dimnode->size;
        if(dimnode->is_unlimited)
            self->unlimited = 1;
    }

    self->nd = dimrec->n_dims;
    self->qdims = qdims;

    return self;
}

statichere NioVariableObject *
nio_variable_new(NioFileObject *file, char *name, int id, 
		 int type, int ndims, NrmQuark *qdims, int nattrs)
{
  NioVariableObject *self;
  NclFile nfile = (NclFile) file->id;
  NclAdvancedFile   advfile = (NclAdvancedFile) file->id;
  NclFileVarNode*   varnode;
  int i;

  if (check_if_open(file, -1)) {
    self = PyObject_NEW(NioVariableObject, &NioVariable_Type);
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
	    self->dimensions = (Py_ssize_t *)malloc(ndims*sizeof(Py_ssize_t));
	    if (self->dimensions != NULL) {
		if(nfile->file.advanced_file_structure)
		{
                    NclFileDimRecord* dimrec;
                    NclFileDimNode*   dimnode;
		    varnode = getVarFromGroup(advfile->advancedfile.grpnode, NrmStringToQuark(name));
		    if(NULL != varnode)
		    {
		        dimrec = varnode->dim_rec;
		        if(NULL != dimrec)
		        {
		            for(i = 0; i < dimrec->n_dims; ++i)
			    {
			        dimnode = &(dimrec->dim_node[i]);
			        self->dimensions[i] = (Py_ssize_t)dimnode->size;
			        if(dimnode->is_unlimited)
				    self->unlimited = 1;
		            }
		        }
		    }
		}
		else
		{
		    for (i = 0; i < ndims; i++) {
			    int dimid = _NclFileIsDim(nfile,qdims[i]);
			    if (dimid < 0) {
				    sprintf(err_buf,"Dimension (%s) not found",NrmQuarkToString(qdims[i]));
				    PyErr_SetString(NIOError, err_buf);
				    return NULL;
			    }
			    self->dimensions[i] = (Py_ssize_t)nfile->file.file_dim_info[dimid]->dim_size;
			    if (nfile->file.file_dim_info[dimid]->is_unlimited)
				    self->unlimited = 1;
		    }
		}
	    }
    }

    if(nfile->file.advanced_file_structure)
    {
        varnode = getVarFromGroup(advfile->advancedfile.grpnode, NrmStringToQuark(name));
        if(NULL != varnode)
        {
            self->attributes = PyDict_New();
            collect_advancedfile_attributes(NULL, varnode->att_rec, self->attributes);
        }
    }
    else
    {
        self->attributes = PyDict_New();
        collect_attributes(file->id, self->id, self->attributes, nattrs);
    }

    self->name = (char *)malloc(strlen(name)+1);
    if (self->name != NULL)
      strcpy(self->name, name);
    return self;
  }
  else
    return NULL;
}

/* Create group object */

statichere NioFileObject* nio_read_group(NioFileObject* niofileobj, NclFileGrpNode* grpnode)
{
    NioFileObject *self;
    NclFile nclfile = (NclFile) niofileobj->id;
    NclAdvancedFile advfilegroup = NULL;
    int ndims, nvars, ngrps, ngattrs;
    char* name;
    char* buf;
    Py_ssize_t len;
    char *full_path;

    name = NrmQuarkToString(grpnode->name);

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
   *fprintf(stderr, "\tgroup name: <%s>\n", name);
   */

    if(! check_if_open(niofileobj, -1))
        return NULL;

    if(NULL == grpnode)
        return NULL;

    self = PyObject_NEW(NioFileObject, &NioFile_Type);
    if(self == NULL)
        return NULL;

    self->dimensions = PyDict_New();
    self->chunk_dimensions = PyDict_New();
    self->ud_types = PyDict_New();
    self->variables = PyDict_New();
    self->groups = PyDict_New();
    self->parent = niofileobj;
    Py_INCREF(niofileobj);
    self->top = niofileobj->top;
    Py_INCREF(self->top);
    self->attributes = PyDict_New();
    self->weakreflist = NULL;
    self->recdim = -1; /* for now */

    self->open = niofileobj->open;
    self->write = niofileobj->write;
    self->define = niofileobj->define;
    self->name = PyString_FromString(name);
    self->mode = niofileobj->mode;
    Py_INCREF(self->mode);
    self->type = PyString_FromString("group");
    len = PyString_Size(niofileobj->full_path);
    buf = malloc(len + 1);
    strcpy(buf,PyString_AsString(niofileobj->full_path));
    len = strlen(buf);
    while (--len >= 0 && buf[len] == '/') {
	    buf[len] = '\0';
    }
    if (! strcmp(name,"/") || strlen(buf) == 0) {
	    self->full_path = PyString_FromFormat("%s",name);
    }
    else {
	    self->full_path = PyString_FromFormat("%s/%s",buf, name);
    }
    free(buf);
    /*printf("path is %s\n",PyString_AsString(self->full_path));*/
    full_path = PyString_AsString(self->full_path);
    advfilegroup = _NclAdvancedGroupCreate(NULL, NULL, Ncl_File, 0,
                                           TEMPORARY, nclfile, NrmStringToQuark(full_path));

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
NioVariableObject_value(NioVariableObject *self, PyObject *args)
{
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
NioVariableObject_assign(NioVariableObject *self, PyObject *args)
{
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
NioVariableObject_typecode(NioVariableObject *self, PyObject *args)
{
  char *t;

  t = typecode(self->type);
  return PyString_FromStringAndSize(t, strlen(t));
}

/* Method table */

static PyMethodDef NioVariableObject_methods[] = {
  {"assign_value", (PyCFunction)NioVariableObject_assign, METH_VARARGS},
  {"get_value", (PyCFunction)NioVariableObject_value, METH_NOARGS},
  {"typecode", (PyCFunction)NioVariableObject_typecode, METH_NOARGS},
  {NULL, NULL}		/* sentinel */
};

/* Attribute access */

static int
NioVariable_GetRank( NioVariableObject *var)
{
  return var->nd;
}

static Py_ssize_t *
NioVariable_GetShape(NioVariableObject *var)
{
  int i, j;
  if (check_if_open(var->file, -1))
  {
      NclFile nfile = (NclFile) var->file->id;
      if(nfile->file.advanced_file_structure)
      {
          NclFileDimRecord* grpdimrec;
          NclFileDimNode*   grpdimnode;
          NclFileDimRecord* dimrec;
          NclFileDimNode*   dimnode;
          NclFileVarNode*   varnode;
	  NclFileGrpNode*   grpnode;

	  grpnode = (NclFileGrpNode*)var->file->gnode;
          grpdimrec = grpnode->dim_rec;
          varnode = getVarFromGroup(grpnode, NrmStringToQuark(var->name));

          if(NULL != varnode)
          {
              dimrec = varnode->dim_rec;
              if(NULL != dimrec)
              {
                  for(i = 0; i < dimrec->n_dims; ++i)
                  {
                      dimnode = &(dimrec->dim_node[i]);
                      var->dimensions[i] = (Py_ssize_t) dimnode->size;
                      if(dimnode->is_unlimited)
                      {
                          PyObject *size_ob = PyInt_FromSsize_t(var->dimensions[i]);
                          for(j = 0; j < grpdimrec->n_dims; ++j)
                          {
                              grpdimnode = &(grpdimrec->dim_node[j]);
                              if(grpdimnode->name == dimnode->name)
                              {
                                  if(grpdimnode->size < dimnode->size)
                                  {
                                      grpdimnode->size = dimnode->size;

	                            /*
	                             *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
			             *                __PRETTY_FUNCTION__, __FILE__, __LINE__);
                                     *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
                                     *     (int)j, NrmQuarkToString(grpdimnode->name), (int)grpdimnode->size);
                                     */
                                      break;
                                  }

                                  if(grpdimnode->size > dimnode->size)
                                  {
                                      dimnode->size = grpdimnode->size;
                                    /*
                                     *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
                                     *                   i, NrmQuarkToString(dimnode->name), (int)dimnode->size);
                                     */
                                      var->dimensions[i] = (Py_ssize_t) dimnode->size;

                                      break;
                                  }
                              }
                          }

                          PyDict_SetItemString(var->file->dimensions, NrmQuarkToString(dimnode->name), size_ob);
                          Py_DECREF(size_ob);
                      }
                  }
                  return var->dimensions;
              }
          }
      }
      else
      {
	  for (i = 0; i < var->nd; i++) {
		  int dimid = _NclFileIsDim(nfile,var->qdims[i]);
		  var->dimensions[i] = (Py_ssize_t) nfile->file.file_dim_info[dimid]->dim_size;
		  if (dimid == var->file->recdim) {
			  PyObject *size_ob = PyInt_FromSsize_t(var->dimensions[i]);
			  PyDict_SetItemString(var->file->dimensions,NrmQuarkToString(var->qdims[i]),size_ob);
			  Py_DECREF(size_ob);
		  }
	  }
          return var->dimensions;
      }
  }

  return NULL;
}

static Py_ssize_t
NioVariable_GetSize(NioVariableObject *var)
{
  int i, j;
  Py_ssize_t size = -1;
  if (check_if_open(var->file, -1))
  {
      NclFile nfile = (NclFile) var->file->id;
      if(nfile->file.advanced_file_structure)
      {
          NclFileDimRecord* grpdimrec;
          NclFileDimNode*   grpdimnode;
          NclFileDimRecord* dimrec;
          NclFileDimNode*   dimnode;
          NclFileVarNode*   varnode;
	  NclFileGrpNode*   grpnode;

	  grpnode = (NclFileGrpNode*)var->file->gnode;
          grpdimrec = grpnode->dim_rec;
          varnode = getVarFromGroup(grpnode, NrmStringToQuark(var->name));

          if(NULL != varnode)
          {
	      size = _NclSizeOf(varnode->type);
              dimrec = varnode->dim_rec;
              if(NULL != dimrec)
              {
                  for(i = 0; i < dimrec->n_dims; ++i)
                  {
                      dimnode = &(dimrec->dim_node[i]);
                      var->dimensions[i] = (Py_ssize_t) dimnode->size;
		      size *= dimnode->size;
                      if(dimnode->is_unlimited)
                      {
                          PyObject *size_ob = PyInt_FromSsize_t(var->dimensions[i]);
                          for(j = 0; j < grpdimrec->n_dims; ++j)
                          {
                              grpdimnode = &(grpdimrec->dim_node[j]);
                              if(grpdimnode->name == dimnode->name)
                              {
                                  if(grpdimnode->size < dimnode->size)
                                  {
                                      grpdimnode->size = dimnode->size;

	                            /*
	                             *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
			             *                __PRETTY_FUNCTION__, __FILE__, __LINE__);
                                     *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
                                     *     (int)j, NrmQuarkToString(grpdimnode->name), (int)grpdimnode->size);
                                     */
                                      break;
                                  }

                                  if(grpdimnode->size > dimnode->size)
                                  {
                                      dimnode->size = grpdimnode->size;
                                    /*
                                     *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
                                     *                   i, NrmQuarkToString(dimnode->name), (int)dimnode->size);
                                     */
                                      var->dimensions[i] = (Py_ssize_t) dimnode->size;

                                      break;
                                  }
                              }
                          }

                          PyDict_SetItemString(var->file->dimensions, NrmQuarkToString(dimnode->name), size_ob);
                          Py_DECREF(size_ob);
                      }
                  }
                  return size;
              }
          }
      }
      else
      {
	  int varid = _NclFileIsVar(nfile,NrmStringToQuark(var->name));
	  size = _NclSizeOf(nfile->file.var_info[varid]->data_type);
	  for (i = 0; i < var->nd; i++) {
		  int dimid = _NclFileIsDim(nfile,var->qdims[i]);
		  var->dimensions[i] = (Py_ssize_t) nfile->file.file_dim_info[dimid]->dim_size;
		  size *= (Py_ssize_t) nfile->file.file_dim_info[dimid]->dim_size;
		  if (dimid == var->file->recdim) {
			  PyObject *size_ob = PyInt_FromSsize_t(var->dimensions[i]);
			  PyDict_SetItemString(var->file->dimensions,NrmQuarkToString(var->qdims[i]),size_ob);
			  Py_DECREF(size_ob);
		  }
	  }
          return size;
      }
  }

  return -1;
}

static PyObject *
NioVariable_GetAttribute(NioVariableObject *self, char *name)
{
  PyObject *value;
  if (strcmp(name, "name") == 0) {
	  return( PyString_FromString(self->name));
  }
  if (strcmp(name, "path") == 0) {
	  char *path = PyString_AsString(self->file->full_path);
	  if (! strcmp(path,"") || ! strcmp(path,"/"))
		  return( PyString_FromFormat("/%s",self->name));
	  else
		  return( PyString_FromFormat("%s/%s",path,self->name));
  }
  if (strcmp(name, "shape") == 0) {
    PyObject *tuple;
    int i;
    if (check_if_open(self->file, -1)) {
      NioVariable_GetShape(self);
      tuple = PyTuple_New(self->nd);
      for (i = 0; i < self->nd; i++)
	PyTuple_SetItem(tuple, i, PyInt_FromSsize_t(self->dimensions[i]));
      return tuple;
    }
    else
      return NULL;
  }
  if (strcmp(name, "rank") == 0) {
    int rank;
    if (check_if_open(self->file, -1)) {
      rank = NioVariable_GetRank(self);
      return Py_BuildValue("i",rank);
    }
    else
      return NULL;
  }
  if (strcmp(name, "size") == 0) {
    ng_size_t size;
    if (check_if_open(self->file, -1)) {
      size = NioVariable_GetSize(self);
      return Py_BuildValue("L",size);
    }
    else
      return NULL;
  }
  if (strcmp(name, "dimensions") == 0) {
    PyObject *tuple;
    char *dname;
    int i;
    if (check_if_open(self->file, -1)) {
      NclFile nfile = (NclFile) self->file->id;
      tuple = PyTuple_New(self->nd);
      if(nfile->file.advanced_file_structure)
      {
          for(i = 0; i < self->nd; i++)
          {
	      dname = NrmQuarkToString(self->qdims[i]);
	      PyTuple_SetItem(tuple, i, PyString_FromString(dname));
          }
      }
      else
      {
          for (i = 0; i < self->nd; i++) {
	      int dimid = _NclFileIsDim(nfile,self->qdims[i]);
	      dname = NrmQuarkToString(nfile->file.file_dim_info[dimid]->dim_name_quark);
	      PyTuple_SetItem(tuple, i, PyString_FromString(dname));
          }
      }
      return tuple;
    }
    else
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
  }
  else {
    PyErr_Clear();
    return Py_FindMethod(NioVariableObject_methods, (PyObject *)self, name);
  }
}

static int
NioVariable_SetAttribute(NioVariableObject *self, char *name, PyObject *value)
{
  NclFile nfile = (NclFile) self->file->id;
  nio_ncerr = 0;
  if (check_if_open(self->file, 1)) {
    if (strcmp(name, "shape") == 0 ||
	strcmp(name, "dimensions") == 0 ||
	strcmp(name, "__dict__") == 0 ||
	strcmp(name, "rank") == 0) {
      PyErr_SetString(PyExc_TypeError, "attempt to set read-only attribute");
      return -1;
    }
    define_mode(self->file, 1);
    if(nfile->file.advanced_file_structure)
        return set_advanced_variable_attribute(self->file, self, self->attributes, name, value);
    else
        return set_attribute(self->file, self->id, self->attributes, name, value);
  }
  else
    return -1;
}

int
NioVariable_SetAttributeString(NioVariableObject *self, char *name, char *value)
{
  PyObject *string = PyString_FromString(value);
  if (string != NULL)
    return NioVariable_SetAttribute(self, name, string);
  else
    return -1;
}


/* Subscripting */

static Py_ssize_t
NioVariableObject_length(NioVariableObject *self)
{
  if (self->nd > 0)
    return self->dimensions[0];
  else
    return 0;
}

NioIndex *
NioVariable_Indices(NioVariableObject *var)
{
  NioIndex *indices = 
    (NioIndex *)malloc(var->nd*sizeof(NioIndex));
  int i;
  if (indices != NULL)
    for (i = 0; i < var->nd; i++) {
      indices[i].start = 0;
      indices[i].stop = var->dimensions[i];
      indices[i].stride = 1;
      indices[i].item = 0;
      indices[i].unlimited = (i == 0 && var->unlimited) ? 1 : 0;
      indices[i].no_start = indices[i].no_stop = 0;
    }
  else
    PyErr_SetString(PyExc_MemoryError, "out of memory");
  return indices;
}

void _convertVLEN2Obj(PyArrayObject* array, obj* listids, ng_size_t nitems)
{
    PyObject* pyobj;
    NclVar var;
    ng_size_t i = 0;
    NclList thelist = NULL;
    NclListObjList *tmp_list;
    NclMultiDValData md;
    npy_intp length;
    int itemsize = PyArray_ITEMSIZE(array);

    for(i = 0; i < nitems; ++i)
    {
        thelist = (NclList)_NclGetObj(listids[i]);
        tmp_list = thelist->list.last;
        var = (NclVar)_NclGetObj(tmp_list->obj_id);
        md = (NclMultiDValData)_NclGetObj(var->var.thevalue_id);
        length = md->multidval.totalelements;

        pyobj = PyArray_SimpleNewFromData(1, &length, data_type(md->multidval.data_type), md->multidval.val);

        PyArray_SETITEM(array, array->data + i * itemsize, pyobj);

        Py_DECREF(pyobj);
    }
}

void _convertCOMPOUND2Obj(PyArrayObject* array, obj* listids, ng_size_t nitems, NclFileVarNode* varnode)
{
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
    NclFileCompoundNode   *compnode = NULL;

  /*
   *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
   *		__PRETTY_FUNCTION__, __FILE__, __LINE__);
   *fprintf(stderr, "\titemsize = %d\n", itemsize);
   */

    if((NULL == comprec) || (1 > comprec->n_comps))
    {
        fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
        fprintf(stderr, "\tThe compound record is NULL, there is nothing we can do then.\n");
        return;
    }

    for(i = 0; i < nitems; ++i)
    {
        thelist = (NclList)_NclGetObj(listids[i]);
        tmp_list = thelist->list.last;
        var = (NclVar)_NclGetObj(tmp_list->obj_id);
        md = (NclMultiDValData)_NclGetObj(var->var.thevalue_id);

        length = (npy_intp) comprec->n_comps;
        comparray = (PyArrayObject*)PyArray_SimpleNew(1, &length, NPY_OBJECT);
        if(NULL == comparray)
            PyErr_SetString(PyExc_MemoryError,
                           "Problem to create PyArray in NioVariable_ReadAsArray for compound data.");

        curval = md->multidval.val;

        for(n = 0; n < comprec->n_comps; ++n)
        {
            compnode = &(comprec->compnode[n]);

          /*
           *fprintf(stderr, "\n\tfile: %s, line: %d\n", __FILE__, __LINE__);
           *fprintf(stderr, "\tcomponent [%d] name: <%s>\n", n, NrmQuarkToString(compnode->name));
           *fprintf(stderr, "\toffset = %d, rank = %d, nvals = %d, type: %s\n",
           *                   compnode->offset, compnode->rank, compnode->nvals,
           *                   _NclBasicDataTypeToName(compnode->type));
           */

            length = compnode->nvals;

            if(NCL_char == compnode->type)
            {
                PyObject* tmpobj;
                char* tmpstr = curval + compnode->offset;
                length = strlen(tmpstr);
                tmpobj = PyArray_SimpleNewFromData(1, &length, PyArray_CHAR, tmpstr);
                pyobj = PyArray_ToString((PyArrayObject*)tmpobj, NPY_CORDER);
                Py_DECREF(tmpobj);
            }
            else
                pyobj = PyArray_SimpleNewFromData(1, &length, data_type(compnode->type), curval + compnode->offset);

            PyArray_SETITEM(comparray, comparray->data + n * itemsize, pyobj);

            Py_DECREF(pyobj);
        }

        PyArray_SETITEM(array, array->data + i * itemsize, comparray);

        Py_DECREF(comparray);
    }
}

PyArrayObject *
NioVariable_ReadAsArray(NioVariableObject *self,NioIndex *indices)
{
  int is_own;
  int ndims;
  npy_intp *dims;
  PyArrayObject *array = NULL;
  int i, d;
  ng_size_t nitems;
  int error = 0;
  int dir;

  d = 0;
  nitems = 1;
  nio_ncerr = 0;

  ndims = self->nd;

  if (!check_if_open(self->file, -1)) {
    free(indices);
    return NULL;
  }
  define_mode(self->file, 0);
  if (self->nd == 0)
    dims = NULL;
  else {
    dims = (npy_intp *)malloc(self->nd*sizeof(npy_intp));
    if (dims == NULL) {
      free(indices);
      return (PyArrayObject *)PyErr_NoMemory();
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
    }
    else {
	    indices[i].stop -= 1;
	    dir = 1;
    }
    if (indices[i].start < 0)
      indices[i].start += self->dimensions[i];
    if (indices[i].start < 0)
      indices[i].start = 0;
    if (indices[i].start > self->dimensions[i] -1)
      indices[i].start = self->dimensions[i] -1;
    if (indices[i].item != 0) {
      indices[i].stop = indices[i].start;
      dims[d] = 1;
    }
    else {
      if (indices[i].stop < 0)
	indices[i].stop += self->dimensions[i];
      if (indices[i].stop < 0)
	indices[i].stop = 0;
      if (indices[i].stop > self->dimensions[i] -1)
	indices[i].stop = self->dimensions[i] -1;
      /* Python only creates a reverse-ordered return value if the stride is less than 0 */   
      dims[d] = ((indices[i].stop-indices[i].start)/(indices[i].stride * dir))+1;
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
	  array =(PyArrayObject *)  PyArray_New(&PyArray_Type,d,
						dims,self->type,NULL,NULL,0,0,NULL);
				  
  }
  if (nitems > 0 && self->type == PyArray_STRING) {
	  NclMultiDValData md;
	  NclSelectionRecord *sel_ptr = NULL;
	  NclFile nfile = (NclFile) self->file->id;
	  if (self->nd > 0) {
		  sel_ptr = (NclSelectionRecord*)malloc(sizeof(NclSelectionRecord));
		  if (sel_ptr != NULL) {
			  sel_ptr->n_entries = self->nd;
			  for (i = 0; i < self->nd; i++) {
				  sel_ptr->selection[i].sel_type = Ncl_SUBSCR;
				  sel_ptr->selection[i].dim_num = i;
				  sel_ptr->selection[i].u.sub.start = indices[i].start;
				  sel_ptr->selection[i].u.sub.finish = indices[i].stop;
				  sel_ptr->selection[i].u.sub.stride = indices[i].stride;
				  sel_ptr->selection[i].u.sub.is_single = 
					  indices[i].item != 0;
			  }
		  }
	  }
	  md = _NclFileReadVarValue(nfile,NrmStringToQuark(self->name),sel_ptr);
	  if (! md) {
		  nio_ncerr = 23;
		  nio_seterror();
	  }
	  else {
		  int maxlen = 0;
		  int tlen;
		  NrmQuark qstr;
		  PyObject *pystr;
		  /* find the maximum string length */
		  for (i = 0; i < nitems; i++) {
			  qstr = ((NrmQuark *)md->multidval.val)[i];
			  tlen = strlen(NrmQuarkToString(qstr));
			  if (maxlen < tlen)
				  maxlen = tlen;
		  }
		  array = (PyArrayObject *)PyArray_New(&PyArray_Type, self->nd, dims, 
						       self->type, NULL,NULL,maxlen,0,NULL);
		  if (array) {
			  for (i = 0; i < nitems; i++) {
				  qstr = ((NrmQuark *)md->multidval.val)[i];
				  pystr = PyString_FromString(NrmQuarkToString(qstr));
				  array->descr->f->setitem(pystr,array->data + i * array->descr->elsize,array);
			  }
		  }
	  }		  
	  if (sel_ptr)
		  free(sel_ptr);
  }
  else if (nitems > 0) {
	  if (self->nd == 0) {
		  NclFile nfile = (NclFile) self->file->id;
		  NclMultiDValData md = _NclFileReadVarValue
			  (nfile,NrmStringToQuark(self->name),NULL);
		  if (! md) {
			  nio_ncerr = 23;
			  nio_seterror();
			  Py_DECREF(array);
			  array = NULL;
		  }
		  array =(PyArrayObject *)
			  PyArray_New(&PyArray_Type,d,
				      dims,self->type,NULL,md->multidval.val,
				      0,0,NULL);
		  md->multidval.val = NULL;
		  _NclDestroyObj((NclObj)md);
	  }
	  else {
		  NclSelectionRecord *sel_ptr;
		  sel_ptr = (NclSelectionRecord*)malloc(sizeof(NclSelectionRecord));
		  if (sel_ptr != NULL) {
			  NclFile nfile = (NclFile) self->file->id;
			  NclMultiDValData md;
			  sel_ptr->n_entries = self->nd;
			  for (i = 0; i < self->nd; i++) {
				  sel_ptr->selection[i].sel_type = Ncl_SUBSCR;
				  sel_ptr->selection[i].dim_num = i;
				  sel_ptr->selection[i].u.sub.start = indices[i].start;
				  sel_ptr->selection[i].u.sub.finish = indices[i].stop;
				  sel_ptr->selection[i].u.sub.stride = indices[i].stride;
				  sel_ptr->selection[i].u.sub.is_single = 
					  indices[i].item != 0;
			  }
			  md = _NclFileReadVarValue
				  (nfile,NrmStringToQuark(self->name),sel_ptr);
			  if (! md) {
				  nio_ncerr = 23;
				  nio_seterror();
				  Py_DECREF(array);
				  array = NULL;
			  }

			  if(nfile->file.advanced_file_structure)
			  {
			  	NclFileVarNode* varnode;
				NclFileGrpNode* grpnode;

				grpnode = (NclFileGrpNode*)self->file->gnode;
				varnode = getVarFromGroup(grpnode, NrmStringToQuark(self->name));
			  	if(NULL == varnode)
					return NULL;

			  	if(NCL_list == md->multidval.data_type)
			  	{
			  		if(NCL_UDT_vlen == varnode->udt_type)
			  		{
                            			/*
                             			*fprintf(stderr, "\nFunction %s, in file: %s, line: %d\n",
                             			*                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
                             			*fprintf(stderr, "\tNeed to work on read vlen\n");
                             			*fprintf(stderr, "\tnitems = %ld\n", (long)nitems);
                             			*for(i = 0; i < self->nd; ++i)
                             			*    fprintf(stderr, "\tdims[%d] = %ld\n", i, (long)dims[i]);
                             			*/

                              			array = (PyArrayObject*)PyArray_SimpleNew(ndims, dims, NPY_OBJECT);
                              			if(NULL == array)
                                  			PyErr_SetString(PyExc_MemoryError,
                                                  			"Problem to create PyArray in NioVariable_ReadAsArray for vlen data.");

                              			_convertVLEN2Obj(array, (obj*)md->multidval.val, nitems);
			  		}
					else if(NCL_UDT_compound == varnode->udt_type)
                          		{
                            			/*
                              			*fprintf(stderr, "\nFunction %s, in file: %s, line: %d\n",
                                               	*		__PRETTY_FUNCTION__, __FILE__, __LINE__);
                              			*fprintf(stderr, "\tNeed to work on read compound\n");
                              			*fprintf(stderr, "\tnitems = %ld\n", (long)nitems);
                              			*for(i = 0; i < self->nd; ++i)
                                  		*	fprintf(stderr, "\tdims[%d] = %ld\n", i, (long)dims[i]);
                             			*/

                              			array = (PyArrayObject*)PyArray_SimpleNew(ndims, dims, NPY_OBJECT);
                              			if(NULL == array)
                                  			PyErr_SetString(PyExc_MemoryError,
                                                  			"Problem to create PyArray in NioVariable_ReadAsArray for compound data.");

                              			_convertCOMPOUND2Obj(array, (obj*)md->multidval.val, nitems, varnode);
			  		}
			  		else
                          		{
                              			fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
                              			fprintf(stderr, "\tDo not know anything about varnode->type. Return NULL.\n");
                              			return NULL;
			  		}
			  	}
			  	else
			  	{
			      		array =(PyArrayObject *) PyArray_New(&PyArray_Type,d,
					      				dims,self->type,NULL,md->multidval.val,
					      				0,0,NULL);
			  	}

				/*Delete md will cause seg. fault. Wei, May 9, 2014.
			 	*md->multidval.val = NULL;
			 	*_NclDestroyObj((NclObj)md);
			 	*/
			  }
			  else
			  {
			      array =(PyArrayObject *) PyArray_New(&PyArray_Type,d,
					      dims,self->type,NULL,md->multidval.val,
					      0,0,NULL);
			  }

			  free(sel_ptr);
		  }
	  }
  }
  is_own = PyArray_CHKFLAGS(array,NPY_OWNDATA);
  if (!is_own) {
	  array->flags |= NPY_OWNDATA;
  }
  array->flags |= NPY_CARRAY;
  free(dims);
  free(indices);
  return array;
}

static PyStringObject *
NioVariable_ReadAsString(NioVariableObject *self)
{
  if (self->type != PyArray_CHAR || self->nd != 1) {
    PyErr_SetString(NIOError, "not a string variable");
    return NULL;
  }
  if (check_if_open(self->file, -1)) {
	  char *tstr;
	  PyObject *string;
	  NclFile nfile = (NclFile) self->file->id;
	  NclMultiDValData md = _NclFileReadVarValue
		  (nfile,NrmStringToQuark(self->name),NULL);
	  if (! md) {
		  nio_seterror();
		  return NULL;
	  }
	  /* all we care about is the actual value */
	  tstr = NrmQuarkToString(*(NrmQuark *) md->multidval.val);
	  _NclDestroyObj((NclObj)md);
	  string = PyString_FromString(tstr);
	  return (PyStringObject *)string;
  }
  else
    return NULL;
}

void _convertObj2VLEN(PyObject* pyobj, obj* listids, NclBasicDataTypes type,
                      ng_size_t n_dims, ng_size_t curdim, ng_usize_t* counter)
{
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

    for(i = 0; i < len; ++i)
    {
        item = PySequence_Fast_GET_ITEM(seq, i);

        if(processingdim == n_dims)
        {
            array = (PyArrayObject *)PyArray_ContiguousFromAny(item,data_type(type),0,1);
            dimsize = array->dimensions[0];
            sprintf(buffer, "list_%6.6d", (int)(*counter));
            dimname = NrmStringToQuark(buffer);
            var = _NclCreateVlenVar(buffer, (void *)array->data, 1, &dimname, &dimsize, type);
            thelist = _NclGetObj(listids[*counter]);
            _NclListAppend((NclObj)thelist, (NclObj)var);

            *counter += 1;
        }
        else
            _convertObj2VLEN(item, listids, type, n_dims, processingdim, counter);
    }

    Py_DECREF(seq);
}

void _convertObj2COMPOUND(PyObject* pyobj, obj* listids, NclFileCompoundRecord *comprec,
                          ng_size_t n_dims, ng_size_t curdim, ng_usize_t* counter)
{
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

    for(i = 0; i < len; ++i)
    {
        item = PySequence_Fast_GET_ITEM(seq, i);

        if(processingdim == n_dims)
        {
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

            if(len2 != comprec->n_comps)
            {
                fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
                fprintf(stderr, "\tlen2 = %d\n", len2);
                fprintf(stderr, "\tcomprec->n_comps = %d\n", (int)comprec->n_comps);
                fprintf(stderr, "\tcomprec->n_comps and len2 do not equal.\n");
                return;
            }

            sprintf(buffer, "comp_%6.6d", (int)(*counter));
            dimname = NrmStringToQuark(buffer);

            for(n = 0; n < len2; ++n)
            {
                item2 = PySequence_Fast_GET_ITEM(seq2, n);
                compnode = &(comprec->compnode[n]);

                if(NCL_char == compnode->type)
                {
                    char* tmpv = (char *)calloc(compnode->nvals, sizeof(char));
                    if(NULL == tmpv)
                    {
                        fprintf(stderr, "\nfile: %s, line: %d\n", __FILE__, __LINE__);
                        fprintf(stderr, "\tFailed to allocate memory of %d char.\n", compnode->nvals);
                        return;
                    }
                    array = (PyArrayObject *)PyArray_ContiguousFromAny(item2,PyArray_STRING,0,1);
                    dimsize = compnode->nvals;
                    strcpy(tmpv, (char*)array->data);
                    var = _NclCreateVlenVar(buffer, (void *)tmpv, 1, &dimname, &dimsize, compnode->type);
                  /*
                   *fprintf(stderr, "\tcomp: %d, value: <%s>\n", n, tmpv);
                   *free(tmpv);
                   */
                }
                else
                {
                    array = (PyArrayObject *)PyArray_ContiguousFromAny(item2,data_type(compnode->type),0,1);
                    dimsize = array->dimensions[0];
                    var = _NclCreateVlenVar(buffer, (void *)array->data, 1, &dimname, &dimsize, compnode->type);
                }
                thelist = _NclGetObj(listids[*counter]);
                _NclListAppend((NclObj)thelist, (NclObj)var);
            }

            *counter += 1;
        }
        else
            _convertObj2COMPOUND(item, listids, comprec, n_dims, processingdim, counter);
    }

    Py_DECREF(seq);
}

static int
NioVariable_WriteArray(NioVariableObject *self, NioIndex *indices, PyObject *value)
{
  ng_size_t *dims = NULL;
  PyArrayObject *array = NULL;
  int i, n_dims;
  Py_ssize_t nitems,var_el_count;
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

  NclFileDimRecord* dimrec  = NULL;
  NclFileDimNode*   dimnode = NULL;
  NclFileVarNode*   varnode = NULL;

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
  }
  else {
    dims = (ng_size_t *)malloc(self->nd*sizeof(ng_size_t));
    dirs = (int *)malloc(self->nd*sizeof(int));
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
  if(nfile->file.advanced_file_structure)
  {
      NclFileGrpNode*   grpnode;

      grpnode = (NclFileGrpNode*)self->file->gnode;
      varnode = getVarFromGroup(grpnode, NrmStringToQuark(self->name));
      if(NULL != varnode)
      {
          dimrec = varnode->dim_rec;
      }
  }
  for (i = 0; i < self->nd; i++) {
	  if (indices[i].unlimited) {
		  if ((indices[i].no_stop && indices[i].stride > 0) ||
		      (indices[i].no_start && indices[i].stride < 0))
			  undefined_dim = i;
	  }
	  if (i != undefined_dim)
		  var_el_count *= self->dimensions[i];
	  error = error || (indices[i].stride == 0);
          if (error)
		  break;
	  if (indices[i].stride < 0) {
		  indices[i].stop += 1;
		  indices[i].stride = -indices[i].stride;
		  dir = -1;
	  }
	  else {
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
	  if (i > 0 || !self->unlimited) {
		  if (indices[i].start > self->dimensions[i] -1)
			  indices[i].start = self->dimensions[i] - 1;
		  if (indices[i].stop > self->dimensions[i] - 1)
			  indices[i].stop = self->dimensions[i] - 1;
	  }
	  if (indices[i].item == 0) {
		  dims[n_dims] = (ng_size_t)((indices[i].stop-indices[i].start)/(indices[i].stride * dir))+1;
		  dirs[n_dims] = dir;
		  if (dims[n_dims] < 0) 
			  dims[n_dims] = 0;
		  if (i != undefined_dim) 
			  nitems *= dims[n_dims];
		  else
			  dim_undef = n_dims;
		  n_dims++;
	  }
	  else {
		  indices[i].stop = indices[i].start;
	  }

	  if (indices[i].stop >= self->dimensions[i])
	      indices[i].stop  = self->dimensions[i] - 1;

        /*
         *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *          __PRETTY_FUNCTION__, __FILE__, __LINE__);
         *fprintf(stderr, "\tdims[%d] = %d\n", i, (int)dims[i]);
         */
          if(nfile->file.advanced_file_structure)
          {
              if(NULL != varnode)
              {
                  if(NULL != dimrec)
                  {
                      dimnode = &(dimrec->dim_node[i]);
		      if(dims[i] != dimnode->size)
                      {
		         dims[i] = dimnode->size;
		         indices[i].stop = dims[i] - 1;
                      }
                  }
              }
          }
        /*
         *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
	 *          __PRETTY_FUNCTION__, __FILE__, __LINE__);
         *fprintf(stderr, "\tdims[%d] = %d\n", i, (int)dims[i]);
         */
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

  if (! strcmp(value->ob_type->tp_name,"numpy.ndarray") || !strcmp(value->ob_type->tp_name,"array")) {
	  array = (PyArrayObject *) value;

	  if ((! nfile->file.advanced_file_structure) && array->descr->type == 'l' && array->descr->elsize == 8) {
		  PyArrayObject *array2 = (PyArrayObject *)  PyArray_Cast(array, PyArray_INT);
		  sprintf(err_buf,"coercing 8-byte long data to 4-byte integer variable (%s): possible data loss due to overflow",
			  self->name);
		  PyErr_SetString(NIOError, err_buf);
		  PyErr_Print();
		  array = (PyArrayObject *)PyArray_ContiguousFromAny((PyObject*)array2,self->type,0,n_dims);
		  Py_DECREF(array2);
	  }
	  else
          {
		  int single_el_dim_count = 0;
		  /* 
		   * Use numpy semantics.
		   * Numpy allows single element 'slow' dimensions to be discarded on assignment
		   */
		  for (i = 0; i < array->nd; i++) {
			  if (array->dimensions[i] == 1)
				  single_el_dim_count++;
			  else
				  break;
		  }
		  if(nfile->file.advanced_file_structure) {
			  if (NCL_char != varnode->type)
				  array = (PyArrayObject *)PyArray_ContiguousFromAny(value,self->type,0,n_dims + single_el_dim_count);
		  }
		  else {
			  array = (PyArrayObject *)PyArray_ContiguousFromAny(value,self->type,0,n_dims + single_el_dim_count);
		  }
	  }
  }
  else {
	  array = (PyArrayObject *)PyArray_ContiguousFromAny(value,self->type,0,n_dims);
  }

  if (array == NULL) {
	  sprintf(err_buf,"type or dimensional mismatch writing to variable (%s)",self->name);
	  PyErr_SetString(NIOError, err_buf);
	  ret = -1;
	  goto err_ret;
  }

  /*
   * PyNIO does not support broadcasting except for the special case of a scalar 
   * applied to an array.
   * However, it does ignore pre-pended single element dimensions either in the file
   * variable or in the assigned array.
   */
	  
  if (array->nd == 0) {
	  n_dims = 1;
  }
  else if (undefined_dim >= 0) {
	  int adim_count = 0;
	  int fdim_count = 0;
	  int *adims, *fdims;
          adims = malloc (array->nd * sizeof(int));
          fdims = malloc (self->nd * sizeof(int));
	  for (i = 0; i < array->nd; i++) {
		  if (array->dimensions[i] > 1) {
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
			  if (dims[fdims[i]] == array->dimensions[adims[i]])
				  continue;
			  if (fdims[i] == undefined_dim) {
				  if (dirs[fdims[i]] == 1) {
					  indices[fdims[i]].stop = indices[fdims[i]].start + (array->dimensions[adims[i]] - 1) * indices[fdims[i]].stride;
					  var_el_count *= array->dimensions[adims[i]];
					  undef_size = array->dimensions[adims[i]];
				  }
				  else {
					  indices[fdims[i]].start = indices[fdims[i]].stop + (array->dimensions[adims[i]] - 1) * indices[fdims[i]].stride;
					  var_el_count *= array->dimensions[adims[i]];
					  undef_size = array->dimensions[adims[i]];
				  }
				  undef_size =  array->dimensions[adims[i]];
			  }
			  else {
				  sprintf(err_buf,"Dimensional mismatch writing to variable (%s)",self->name);
				  PyErr_SetString(NIOError, err_buf);
				  ret = -1;
				  goto err_ret;
			  }
		  }
		  if (dim_undef > -1 && undef_size > 0) {
			  dims[dim_undef] = undef_size;
			  nitems *= undef_size;
		  }
	  }
  }
  else {
	  int var_dim = 0;
#if 0
          if (indices[var_dim].unlimited && dims[var_dim] == 0)
#else
	/*Added this paragraph to check the indices,
	 *Where, the stop somehow become pretty wild, which has the INT64MAX.
	 *Wei, April 7, 2014.
	 */
          if(indices[var_dim].unlimited)
	  {
              if((indices[var_dim].stop >= array->dimensions[var_dim]) ||
                 (indices[var_dim].stop != (array->dimensions[var_dim] - 1)))
              {
                  NclFile nfile = (NclFile) self->file->id;
                  if(nfile->file.advanced_file_structure)
                  {
                      NclAdvancedFile advfile = (NclAdvancedFile) nfile;
                      NclFileDimRecord* grpdimrec;
                      NclFileDimNode*   grpdimnode;
                      NclFileDimRecord* dimrec;
                      NclFileDimNode*   dimnode;
                      NclFileVarNode*   varnode;

                      varnode = getVarFromGroup(advfile->advancedfile.grpnode, NrmStringToQuark(self->name));
                      if(NULL != varnode)
                      {
                          dimrec = varnode->dim_rec;
                          if(NULL != dimrec)
                          {
                            /*
                             *fprintf(stderr, "\nEnter %s, in file: %s, line: %d\n",
                             *                 __PRETTY_FUNCTION__, __FILE__, __LINE__);
                             */
                              dimnode = &(dimrec->dim_node[var_dim]);
                            /*
                             *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
                             *                   (int)var_dim, NrmQuarkToString(dimnode->name), (int)dimnode->size);
                             */
                              dimnode->size = (ng_size_t) array->dimensions[var_dim];
                            /*
                             *fprintf(stderr, "\tDim %d, name: %s, size: %d\n",
                             *                   (int)var_dim, NrmQuarkToString(dimnode->name), (int)dimnode->size);
                             */

                              grpdimrec = advfile->advancedfile.grpnode->dim_rec;
	                      for(i = 0; i < grpdimrec->n_dims; ++i)
	                      {
	                          grpdimnode = &(grpdimrec->dim_node[i]);
                                  if(grpdimnode->name == dimnode->name)
                                  {
                                      if(grpdimnode->size != dimnode->size)
                                         grpdimnode->size = dimnode->size;
	                          }
	                      }
                          }
                      }
                  }

                  indices[var_dim].stop = array->dimensions[var_dim] - 1;
                  dims[var_dim] = array->dimensions[var_dim];
                  self->dimensions[var_dim] = array->dimensions[var_dim];
                  nitems = 1;
	          for(i = 0; i < self->nd; ++i)
                  {
	              nitems *= self->dimensions[i];
                  }
              }
	  }
#endif		  
	  i = 0;
	  while (i < array->nd && array->dimensions[i] == 1)
		  i++;
	  while (var_dim < n_dims && dims[var_dim] == 1)
		  var_dim++;
	  for ( ; i < array->nd; ) {
		  if (array->dimensions[i] != dims[var_dim]) {
			  if (dims[var_dim] == 1) {
				  var_dim++;
				  continue;
			  }
			  if (array->dimensions[i] == 1) {
				  i++;
				  continue;
			  }
			  sprintf(err_buf,"Dimensional mismatch writing to variable (%s)",self->name);
			  PyErr_SetString(NIOError, err_buf);
			  ret = -1;
			  goto err_ret;
		  }
		  array_el_count *= array->dimensions[i];
		  var_dim++;
		  i++;
	  }
	  while (var_dim < n_dims && dims[var_dim] == 1)
		  var_dim++;
	  if (var_dim != n_dims) {
		  sprintf(err_buf,"Dimensional mismatch writing to variable (%s)",self->name);
		  PyErr_SetString(NIOError, err_buf);
		  ret = -1;
		  goto err_ret;
	  }
	  if (array_el_count < nitems) {
		  /*
		   * This test should be redundant, but just in case.
		   */
		  sprintf(err_buf,"Not enough elements supplied for write to variable (%s)",self->name);
		  PyErr_SetString(NIOError, err_buf);
		  ret = -1;
		  goto err_ret;
	  }
  }

  if (nitems < var_el_count || self->unlimited)
	  select_all = 0;
  if (dirs != NULL) {
	  for (i = 0; i < n_dims; i++) {
		  if (dirs[i] == -1) {
			  select_all = 0;
			  break;
		  }
	  }
  }
  qtype = nio_type_from_code(array->descr->type);

  if(nfile->file.advanced_file_structure)
  {
      if(NULL != varnode)
      {
          if((NCL_vlen == varnode->type) || (NCL_list == varnode->type))
          {
              obj *listids = NULL;
              int curdim = 0;
              ng_usize_t counter = 0;

              qtype = NrmStringToQuark("list");

              if(array)
              {
                  NclObjTypes the_obj_type = Ncl_Typelist;

                  listids = (obj*)NclMalloc((ng_usize_t)(nitems * sizeof(obj)));
                  assert(listids);

                  _NclBuildArrayOfList(listids, n_dims, dims);

                  _convertObj2VLEN((PyObject*)array, listids, varnode->base_type, n_dims, curdim, &counter);

                  md = _NclCreateVal(NULL,NULL,((the_obj_type & NCL_VAL_TYPE_MASK) ?
                                     Ncl_MultiDValData:the_obj_type),0,listids,
                                     NULL,n_dims,dims,TEMPORARY,NULL,
                                     (NclObjClass)((the_obj_type & NCL_VAL_TYPE_MASK) ?
                                     _NclTypeEnumToTypeClass(the_obj_type):NULL));
              }
          }
          else if(NCL_compound == varnode->type)
          {
              obj *listids = NULL;
              int curdim = 0;
              ng_usize_t counter = 0;

              qtype = NrmStringToQuark("compound");

              if(array)
              {
                  NclObjTypes the_obj_type = Ncl_Typelist;

                  listids = (obj*)NclMalloc((ng_usize_t)(nitems * sizeof(obj)));
                  assert(listids);

                  _NclBuildArrayOfList(listids, n_dims, dims);

                  _convertObj2COMPOUND((PyObject*)array, listids, varnode->comprec, n_dims, curdim, &counter);

                  md = _NclCreateVal(NULL,NULL,((the_obj_type & NCL_VAL_TYPE_MASK) ?
                                     Ncl_MultiDValData:the_obj_type),0,listids,
                                     NULL,n_dims,dims,TEMPORARY,NULL,
                                     (NclObjClass)((the_obj_type & NCL_VAL_TYPE_MASK) ?
                                     _NclTypeEnumToTypeClass(the_obj_type):NULL));
              }
          }
          else if(NCL_string == varnode->type)
          {
              NrmQuark *qval;
              char *cptr = NULL;
              char *cval = NULL;
              ng_size_t n_items = 1;
              ng_size_t i;

              for(i = 0; i < n_dims; i++)
                  n_items *= dims[i];

              qval = (NrmQuark*)NclCalloc(n_items, sizeof(NrmQuark));

	      varnode->type = NCL_string;
             /*
              qval[0] = NrmStringToQuark((char*)array->data);
	     */
              cptr = (char*)array->data;
              for(i = 0; i < n_items; i++)
	      {
                  cval = cptr + i*array->descr->elsize;
                  /*fprintf(stderr, "\tcval[%ld] = %s\n", i, cval);*/
                  qval[i] = NrmStringToQuark(cval);
	      }

              md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
			   		qval,NULL,n_dims,
			   		array->nd == 0 ? &scalar_size : dims,
					TEMPORARY,NULL,_NclNameToTypeClass(qtype));
          }
          else
              md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
			   		(void*)array->data,NULL,n_dims,
			   		array->nd == 0 ? &scalar_size : dims,
					TEMPORARY,NULL,_NclNameToTypeClass(qtype));
      }
  }
  else
      md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
			   (void*)array->data,NULL,n_dims,
			   array->nd == 0 ? &scalar_size : dims,
			   TEMPORARY,NULL,_NclNameToTypeClass(qtype));
  if (! md) {
	  nret = NhlFATAL;
  }
  else if (select_all) {
	  nret = _NclFileWriteVar(nfile,NrmStringToQuark(self->name),md,NULL);
  }
  else {
	  NclSelectionRecord *sel_ptr;
	  sel_ptr = (NclSelectionRecord*)malloc(sizeof(NclSelectionRecord));
	  if (sel_ptr == NULL) {
		  nret = NhlFATAL;
	  }
	  else {
		  sel_ptr->n_entries = self->nd;
		  for (i = 0; i < self->nd; i++) {
			  sel_ptr->selection[i].sel_type = Ncl_SUBSCR;
			  sel_ptr->selection[i].dim_num = i;
			  sel_ptr->selection[i].u.sub.start = indices[i].start;
			  sel_ptr->selection[i].u.sub.finish = indices[i].stop;
			  if(sel_ptr->selection[i].u.sub.finish < 0)
			     sel_ptr->selection[i].u.sub.finish = 0;
			  sel_ptr->selection[i].u.sub.stride = indices[i].stride;
			  sel_ptr->selection[i].u.sub.is_single = 
				  indices[i].item != 0;
		  }
		  nret = _NclFileWriteVar(nfile,NrmStringToQuark(self->name),md,sel_ptr);
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


static int
NioVariable_WriteString(NioVariableObject *self, PyStringObject *value)
{
  long len;

  if (self->type != PyArray_CHAR || self->nd != 1) {
	  PyErr_SetString(NIOError, "not a string variable");
	  return -1;
  }
  len = PyString_Size((PyObject *)value);
  if (len > self->dimensions[0]) {
	  PyErr_SetString(PyExc_ValueError, "string too long");
	  return -1;
  }
  if (self->dimensions[0] > len)
	  len++;
  if (check_if_open(self->file, 1)) {
	  NclFile nfile = (NclFile) self->file->id;
	  NclMultiDValData md;
	  NhlErrorTypes nret;
	  ng_size_t str_dim_size = 1;
	  NrmQuark qstr = NrmStringToQuark(PyString_AsString((PyObject *)value));
	  define_mode(self->file, 0);
	  md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
				   (void*)&qstr,NULL,1,
				   &str_dim_size,
				   TEMPORARY,NULL,_NclNameToTypeClass(NrmStringToQuark("string")));
	  if (! md) {
		  nret = NhlFATAL;
	  }
	  else {
		  nret = _NclFileWriteVar(nfile,NrmStringToQuark(self->name),md,NULL);
	  }
	  if (nret < NhlWARNING) {
		  nio_seterror();
		  return -1;
	  }
	  return 0;
  }
  else {
	  return -1;
  }
}


static PyObject *
NioVariableObject_item(NioVariableObject *self, Py_ssize_t i)
{
  NioIndex *indices;
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return NULL;
  }
  indices = NioVariable_Indices(self);
  if (indices != NULL) {
    indices[0].start = i;
    indices[0].stop = i+1;
    indices[0].item = 1;
    return PyArray_Return(NioVariable_ReadAsArray(self, indices));
  }
  return NULL;
}

static PyObject *
NioVariableObject_slice(NioVariableObject *self, Py_ssize_t low, Py_ssize_t high)
{
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
NioVariableObject_subscript(NioVariableObject *self, PyObject *index)
{
  NioIndex *indices;
  if (PyInt_Check(index)) {
    Py_ssize_t i = (Py_ssize_t)PyInt_AsLong(index);
    return NioVariableObject_item(self, i);
  }
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return NULL;
  }
  indices = NioVariable_Indices(self);
  if (indices != NULL) {
    if (PySlice_Check(index)) {
	    Py_ssize_t slicelen;
	    PySliceObject *slice = (PySliceObject *)index;
	    if (PySlice_GetIndicesEx((PySliceObject *)index, self->dimensions[0],
				   &indices->start, &indices->stop, &indices->stride, &slicelen) < 0) {
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
	  if (PyInt_Check(subscript)) {
	    Py_ssize_t n = (Py_ssize_t)PyInt_AsLong(subscript);
	    indices[d].start = n;
	    indices[d].stop = n+1;
	    indices[d].item = 1;
	    d++;
	  }
	  else if (PySlice_Check(subscript)) {
	    Py_ssize_t slicelen;
	    PySliceObject *slice = (PySliceObject *)subscript;
	    if (PySlice_GetIndicesEx((PySliceObject *)subscript, self->dimensions[d],
			       &indices[d].start, &indices[d].stop,
			       &indices[d].stride, &slicelen) < 0) {
	        PyErr_SetString(PyExc_TypeError, "error in subscript");
	        free(indices);
	        return NULL;
	    }
	    if (slice->start == Py_None)
		    indices[d].no_start = 1;
	    if (slice->stop == Py_None)
		    indices[d].no_stop = 1;
	    d++;
	  }
	  else if (subscript == Py_Ellipsis) {
	    d = self->nd - ni + i + 1;
	  }
	  else {
	    PyErr_SetString(PyExc_TypeError, "illegal subscript type");
	    free(indices);
	    return NULL;
	  }
	}
	return PyArray_Return(NioVariable_ReadAsArray(self, indices));
      }
      else {
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

static int
NioVariableObject_ass_item(NioVariableObject *self, Py_ssize_t i, PyObject *value)
{
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
    indices[0].stop = i+1;
    indices[0].item = 1;
    return NioVariable_WriteArray(self, indices, value);
  }
  return -1;
}

static int
NioVariableObject_ass_slice(NioVariableObject *self, Py_ssize_t low, Py_ssize_t high, PyObject *value)
{
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
    indices[0].start = low;
    indices[0].stop = high;
    return NioVariable_WriteArray(self, indices, value);
  }
  return -1;
}

static int
NioVariableObject_ass_subscript(NioVariableObject *self, PyObject *index, PyObject *value)
{
  NioIndex *indices;
  if (PyInt_Check(index)) {
    Py_ssize_t i = (Py_ssize_t) PyInt_AsLong(index);
    return NioVariableObject_ass_item(self, i, value);
  }
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
	    PySliceObject *slice = (PySliceObject *)index;
	    if (PySlice_GetIndicesEx((PySliceObject *)index, self->dimensions[0],
				   &indices->start, &indices->stop, &indices->stride, &slicelen) < 0) {
	        PyErr_SetString(PyExc_TypeError, "error in subscript");
	        free(indices);
	        return -1;
	    }
	    if (slice->start == Py_None)
		    indices->no_start = 1;
	    if (slice->stop == Py_None)
		    indices->no_stop = 1;
	    return NioVariable_WriteArray(self, indices, value);
    }
    if (PyTuple_Check(index)) {
      int ni = PyTuple_Size(index);
      if (ni <= self->nd) {
	int i, d;
	d = 0;
	for (i = 0; i < ni; i++) {
	  PyObject *subscript = PyTuple_GetItem(index, i);
	  if (PyInt_Check(subscript)) {
	    Py_ssize_t n = (Py_ssize_t)PyInt_AsLong(subscript);
	    indices[d].start = n;
	    indices[d].stop = n+1;
	    indices[d].item = 1;
	    d++;
	  }
	  else if (PySlice_Check(subscript)) {
	    Py_ssize_t slicelen;
	    PySliceObject *slice = (PySliceObject *)subscript;
	    if (PySlice_GetIndicesEx((PySliceObject *)subscript, self->dimensions[d],
				   &indices[d].start, &indices[d].stop,
				   &indices[d].stride, &slicelen) < 0) {
	        PyErr_SetString(PyExc_TypeError, "error in subscript");
	        free(indices);
	        return -1;
	    }
	    if (slice->start == Py_None)
		    indices[d].no_start = 1;
	    if (slice->stop == Py_None)
		    indices[d].no_stop = 1;
	    d++;
	  }
	  else if (subscript == Py_Ellipsis) {
	    d = self->nd - ni + i + 1;
	  }
	  else {
	    PyErr_SetString(PyExc_TypeError, "illegal subscript type");
	    free(indices);
	    return -1;
	  }
	}
	return NioVariable_WriteArray(self, indices, value);
      }
      else {
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
NioVariableObject_error1(NioVariableObject *self, NioVariableObject *other)
{
  PyErr_SetString(PyExc_TypeError, "can't concatenate NIO variables");
  return NULL;
}

static PyObject *
NioVariableObject_error2(NioVariableObject *self,  Py_ssize_t n)
{
  PyErr_SetString(PyExc_TypeError, "can't repeat NIO variables");
  return NULL;
}

#if PY_VERSION_HEX < 0x02050000

static PySequenceMethods NioVariableObject_as_sequence = {
  (inquiry)NioVariableObject_length,		/*sq_length*/
  (binaryfunc)NioVariableObject_error1,       /*sq_concat*/
  (intargfunc)NioVariableObject_error2,       /*sq_repeat*/
  (intargfunc)NioVariableObject_item,		/*sq_item*/
  (intintargfunc)NioVariableObject_slice,	/*sq_slice*/
  (intobjargproc)NioVariableObject_ass_item,	/*sq_ass_item*/
  (intintobjargproc)NioVariableObject_ass_slice,   /*sq_ass_slice*/
};

static PyMappingMethods NioVariableObject_as_mapping = {
  (inquiry)NioVariableObject_length,		/*mp_length*/
  (binaryfunc)NioVariableObject_subscript,	      /*mp_subscript*/
  (objobjargproc)NioVariableObject_ass_subscript,   /*mp_ass_subscript*/
};

#else

static PySequenceMethods NioVariableObject_as_sequence = {
  (lenfunc)NioVariableObject_length,		/*sq_length*/
  (binaryfunc)NioVariableObject_error1,       /*sq_concat*/
  (ssizeargfunc)NioVariableObject_error2,       /*sq_repeat*/
  (ssizeargfunc)NioVariableObject_item,		/*sq_item*/
  (ssizessizeargfunc)NioVariableObject_slice,	/*sq_slice*/
  (ssizeobjargproc)NioVariableObject_ass_item,	/*sq_ass_item*/
  (ssizessizeobjargproc)NioVariableObject_ass_slice,   /*sq_ass_slice*/
};

static PyMappingMethods NioVariableObject_as_mapping = {
  (lenfunc)NioVariableObject_length,		/*mp_length*/
  (binaryfunc)NioVariableObject_subscript,	      /*mp_subscript*/
  (objobjargproc)NioVariableObject_ass_subscript,   /*mp_ass_subscript*/
};

#endif

void printval(char *buf,NclBasicDataTypes type,void *val)
{
	switch(type) {
	case NCL_double:
		sprintf(buf,"%4.16g",*(double *)val);
		return;
	case NCL_float:
		sprintf(buf,"%2.7g", *(float *)val);
		return;
	case NCL_int64:
		sprintf(buf,"%lld",*(long long *)val);
		return;
	case NCL_uint64:
		sprintf(buf,"%llu",*(unsigned long long *)val);
		return;
	case NCL_long:
		sprintf(buf,"%ld",*(long *)val);
		return;
	case NCL_ulong:
		sprintf(buf,"%lu",*(unsigned long *)val);
		return;
	case NCL_int:
		sprintf(buf,"%d",*(int *)val);
		return;
	case NCL_uint:
		sprintf(buf,"%u",*(unsigned int *)val);
		return;
	case NCL_short:
		sprintf(buf,"%d",(int)*(short *)val);
		return;
	case NCL_ushort:
		sprintf(buf,"%d",(int)*(unsigned short *)val);
		return;
	case NCL_string:
		strncat(buf,NrmQuarkToString(*(NrmQuark *)val),1024);
		return;
	case NCL_char:
		sprintf(buf,"%c",*(char *)val);
		return;
	case NCL_byte:
		sprintf(buf,"%d",(int)*(char *)val);
		return;
	case NCL_ubyte:
		sprintf(buf,"%d",(int)*(unsigned char *)val);
		return;
	case NCL_logical:
		if (*(logical *)val == 0) {
			sprintf(buf,"False");
		}
		else {
			sprintf(buf,"True");
		}
		return;
	default:
		sprintf(buf,"-");
		return;
	}

}

/* Printed representation */
static PyObject *
NioVariableObject_str(NioVariableObject *var)
{
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

	if (! check_if_open(file, -1)) {
		PyErr_SetString(NIOError, "file has been closed: variable no longer valid");
		return NULL;
	}

	qncl_scalar = NrmStringToQuark("ncl_scalar");
	if(nfile->file.advanced_file_structure)
	{
		NclFileVarNode* varnode;
		NclAdvancedFile advfile = (NclAdvancedFile) file->id;

		varnode = getVarFromGroup(advfile->advancedfile.grpnode, varq);
		if(NULL == varnode) {
			PyErr_SetString(NIOError, "variable not found");
			return NULL;
		}
		buf = NioVarInfo2str(var,varnode);
		pystr = PyString_FromString(buf);
		free(buf);
		return pystr;
	}
	buf = malloc(bufinc);
	buflen = bufinc;
	for(i = 0; i < nfile->file.n_vars; i++) {
		if(nfile->file.var_info[i]->var_name_quark != varq) {
			continue;
		}
		vname = NrmQuarkToString(nfile->file.var_info[i]->var_name_quark);
		break;
	}
	if (! vname) {
		PyErr_SetString(NIOError, "variable not found");
		return NULL;
	}
	vobj = (NioVariableObject *) PyDict_GetItemString(file->variables,vname);
	sprintf(tbuf,"Variable: %s\n",vname);
	BUF_INSERT(tbuf);
	total = 1;
	for (j = 0; j <  nfile->file.var_info[i]->num_dimensions;j++) {
		total *= nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[j]]->dim_size;
	}
	sprintf(tbuf,"Type: %s\n",_NclBasicDataTypeToName(nfile->file.var_info[i]->data_type));
	BUF_INSERT(tbuf);
	sprintf(tbuf,"Total Size: %lld bytes\n",total * _NclSizeOf(nfile->file.var_info[i]->data_type));
	BUF_INSERT(tbuf);
	sprintf(tbuf,"            %lld values\n",total);
	BUF_INSERT(tbuf);
	if (nfile->file.var_info[i]->num_dimensions == 1 && 
	    nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[0]]->dim_name_quark == qncl_scalar) {
		sprintf(tbuf,"Number of Dimensions: %d\n",0);
		BUF_INSERT(tbuf);
	}
	else {
		sprintf(tbuf,"Number of Dimensions: %d\n",nfile->file.var_info[i]->num_dimensions);
		BUF_INSERT(tbuf);
		sprintf(tbuf,"Dimensions and sizes:\t");
		BUF_INSERT(tbuf);
		for(j = 0; j < nfile->file.var_info[i]->num_dimensions;j++) {
			sprintf(tbuf,"[");
			BUF_INSERT(tbuf);
			sprintf(tbuf,"%s | ",
				NrmQuarkToString(nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[j]]->dim_name_quark));
			BUF_INSERT(tbuf);
			sprintf(tbuf,"%lld]",
				(long long) nfile->file.file_dim_info[nfile->file.var_info[i]->file_dim_num[j]]->dim_size);
			BUF_INSERT(tbuf);
			if(j != nfile->file.var_info[i]->num_dimensions-1) {
				sprintf(tbuf," x ");
				BUF_INSERT(tbuf);
			}
		}	
		sprintf(tbuf,"\nCoordinates: \n");
		BUF_INSERT(tbuf);
		for(j = 0; j < nfile->file.var_info[i]->num_dimensions;j++) {
			NclVar tmp_var;
			dim_ix = nfile->file.var_info[i]->file_dim_num[j];
			if(_NclFileVarIsCoord(nfile,
					      nfile->file.file_dim_info[dim_ix]->dim_name_quark)!= -1) {
				sprintf(tbuf,"            ");
				BUF_INSERT(tbuf);
				sprintf(tbuf,"%s: [", 
					NrmQuarkToString(nfile->file.file_dim_info[dim_ix]->dim_name_quark));
				BUF_INSERT(tbuf);
				tmp_var = _NclFileReadCoord(nfile,nfile->file.file_dim_info[dim_ix]->dim_name_quark,NULL);
				if(tmp_var != NULL) {
					NclMultiDValData tmp_md;

					tmp_md = (NclMultiDValData)_NclGetObj(tmp_var->var.thevalue_id);
					printval(tbuf,tmp_md->multidval.type->type_class.data_type,tmp_md->multidval.val);
					BUF_INSERT(tbuf);
					sprintf(tbuf,"..");
					BUF_INSERT(tbuf);
					printval(tbuf,tmp_md->multidval.type->type_class.data_type,(char *) tmp_md->multidval.val +
						 (tmp_md->multidval.totalelements -1)*tmp_md->multidval.type->type_class.size);
					BUF_INSERT(tbuf);
					sprintf(tbuf,"]\n");
					BUF_INSERT(tbuf);
					if(tmp_var->obj.status != PERMANENT) {
						_NclDestroyObj((NclObj)tmp_var);
					}

				}
			}
			else {
				sprintf(tbuf,"            ");
				BUF_INSERT(tbuf);
				sprintf(tbuf,"%s: not a coordinate variable\n",
					NrmQuarkToString(nfile->file.file_dim_info[dim_ix]->dim_name_quark));
 				BUF_INSERT(tbuf);
			}
		}	
	}
	step = nfile->file.var_att_info[i];
	n_atts = 0;
	while(step != NULL) {
		n_atts++;
		step = step->next;
	}
	sprintf(tbuf,"Number of Attributes: %d\n",n_atts);	
	BUF_INSERT(tbuf);
	step = nfile->file.var_att_info[i];
	while(step != NULL) {
		PyObject *att_val;
		char *aname = NrmQuarkToString(step->the_att->att_name_quark);
		sprintf(tbuf,"         %s :\t", aname);
		BUF_INSERT(tbuf);
		att_val = PyDict_GetItemString(vobj->attributes,aname);
		if (PyString_Check(att_val)) {
			BUF_INSERT(PyString_AsString(att_val));
			BUF_INSERT("\n");
		}
		else {
			int k;
			PyArrayObject *att_arr_val = (PyArrayObject *)att_val;
			if (att_arr_val->nd == 0 || att_arr_val->dimensions[0] == 1) {
				PyObject *att = att_arr_val->descr->f->getitem(PyArray_DATA(att_val),att_val);
				format_object(tbuf,att,att_arr_val->descr->type);
				BUF_INSERT(tbuf);
				BUF_INSERT("\n");
				/*sprintf(tbuf,"%s\n",PyString_AsString(PyObject_Str(att)));*/
			}
			else {
				sprintf(tbuf,"[");
				BUF_INSERT(tbuf);
				for (k = 0; k < att_arr_val->dimensions[0]; k++) {
					PyObject *att = att_arr_val->descr->f->getitem(att_arr_val->data + 
										       k * att_arr_val->descr->elsize,att_val);
					format_object(tbuf,att,att_arr_val->descr->type);
					/*sprintf(tbuf,"%s",PyString_AsString(PyObject_Str(att)));*/
					BUF_INSERT(tbuf);
					if (k < att_arr_val->dimensions[0] - 1) {
						sprintf(tbuf,", ");
					}
					else {
						sprintf(tbuf,"]\n");
					}
					BUF_INSERT(tbuf);
				}
			}
		}
		step = step->next;
	}
	pystr = PyString_FromString(buf);
	free(buf);
	return pystr;
}

statichere PyTypeObject NioVariable_Type = {
  PyObject_HEAD_INIT(NULL)
  0,		     /*ob_size*/
  "_Nio._NioVariable",  /*tp_name*/
  sizeof(NioVariableObject),	     /*tp_basicsize*/
  0,		     /*tp_itemsize*/
  /* methods */
  (destructor)NioVariableObject_dealloc, /*tp_dealloc*/
  0,			/*tp_print*/
  (getattrfunc)NioVariable_GetAttribute, /*tp_getattr*/
  (setattrfunc)NioVariable_SetAttribute, /*tp_setattr*/
  0,			/*tp_compare*/
  0,			/*tp_repr*/
  0,			/*tp_as_number*/
  &NioVariableObject_as_sequence,	/*tp_as_sequence*/
  &NioVariableObject_as_mapping,	/*tp_as_mapping*/
  0,			/*tp_hash*/
  0,                    /*tp_call*/
  (reprfunc)NioVariableObject_str,   /*tp_str*/ 
  (getattrofunc)0,                          /*tp_getattro*/
  (setattrofunc)0,                          /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE),  /*tp_flags*/
  0,                     /*tp_doc*/
  (traverseproc)0,                          /*tp_traverse */
  (inquiry)0,                               /*tp_clear */
  (richcmpfunc)0,           /*tp_richcompare */
  0,     /*tp_weaklistoffset */

  /* Iterator support (use standard) */

  (getiterfunc) 0,                  /* tp_iter */
  (iternextfunc)0,                          /* tp_iternext */

  /* Sub-classing (new-style object) support */

  NioVariableObject_methods,                    /* tp_methods */
  0,                                        /* tp_members */
  0,                                        /* tp_getset */
  0,                                        /* tp_base */
  0,                                        /* tp_dict */
  0,                                        /* tp_descr_get */
  0,                                        /* tp_descr_set */
  0,                                        /* tp_dictoffset */
  (initproc)0,                              /* tp_init */
  0,                                        /* tp_alloc */
  0,                                        /* tp_new */
  0,                                        /* tp_free */
  0,                                        /* tp_is_gc */
  0,                                        /* tp_bases */
  0,                                        /* tp_mro */
  0,                                        /* tp_cache */
  0,                                        /* tp_subclasses */
  0                                         /* tp_weaklist */
};


static NrmQuark 
GetExtension ( char * filename)
{
	struct stat statbuf;
	char *cp = strrchr(filename,'.');
	if (cp == NULL || *(cp+1) == '\0')
		return NrmNULLQUARK;

	/* for now only regular files are accepted */
	if (stat(filename,&statbuf) >= 0) {
		if (! (S_ISREG(statbuf.st_mode)))
			return NrmNULLQUARK;
	}
	return NrmStringToQuark(cp+1);
}
	

void InitializeNioOptions(NrmQuark extq, int mode)
{
	NclMultiDValData md;
	ng_size_t len_dims = 1;
	logical *lval;
	NrmQuark *qval;
	NrmQuark qnc = NrmStringToQuark("nc");
	NrmQuark qgrb = NrmStringToQuark("qgrb");

	if (_NclFormatEqual(extq,qnc)) {
		_NclFileSetOptionDefaults(qnc,NrmNULLQUARK);

		if (mode < 1) {
			/* if opened for writing default "definemode" to True */
			lval = (logical *)malloc( sizeof(logical));
			*lval = True;
			md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
						 (void*)lval,NULL,1,&len_dims,
						 TEMPORARY,NULL,(NclTypeClass)nclTypelogicalClass);
			_NclFileSetOption(NULL,extq,NrmStringToQuark("definemode"),md);
			_NclDestroyObj((NclObj)md);
		}
		/* default "suppressclose" to True for files opened in all modes */
		lval = malloc( sizeof(logical));
		*lval = True;
		md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
					 (void*)lval,NULL,1,&len_dims,
					 TEMPORARY,NULL,(NclTypeClass)nclTypelogicalClass);
		_NclFileSetOption(NULL,extq,NrmStringToQuark("suppressclose"),md);
		_NclDestroyObj((NclObj)md);
	}
	else if (_NclFormatEqual(extq,qgrb)) {
		_NclFileSetOptionDefaults(qgrb,NrmNULLQUARK);
		qval = (NrmQuark *) malloc(sizeof(NrmQuark));
		*qval = NrmStringToQuark("numeric");
		md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
					 (void*)qval,NULL,1,&len_dims,
					 TEMPORARY,NULL,(NclTypeClass)nclTypestringClass);
		_NclFileSetOption(NULL,extq,NrmStringToQuark("initialtimecoordinatetype"),md);
		_NclDestroyObj((NclObj)md);
	}
}


void SetNioOptions(NrmQuark extq, int mode, PyObject *options, PyObject *option_defaults)
{
	NclMultiDValData md = NULL;
	PyObject *keys = PyMapping_Keys(options);
	ng_size_t len_dims = 1;
	Py_ssize_t i;
	NrmQuark qsafe_mode = NrmStringToQuark("safemode");

	for (i = 0; i < PySequence_Length(keys); i++) {


		PyObject *key = PySequence_GetItem(keys,i);
		char *keystr = PyString_AsString(key);
		NrmQuark qkeystr = NrmStringToQuark(keystr);
		PyObject *value = PyMapping_GetItemString(options,keystr);

		/* handle the PyNIO-defined "SafeMode" option */
		if (_NclFormatEqual(extq,NrmStringToQuark("nc")) &&
		    (_NclGetLower(qkeystr) == qsafe_mode)) {
			/* 
			 * SafeMode == True:
			 *    DefineMode = False (if writable file)
			 *    SuppressClose = False
			 * SafeMode == False:
			 *    DefineMode = True (if writable file)
			 *    SuppressClose = True
			 */
			        
			logical *lval;
			if (! PyBool_Check(value)) {
				char s[256] = "";
				strncat(s,keystr,255);
				strncat(s," value is an invalid type for option",255);
				PyErr_SetString(NIOError,s);
				Py_DECREF(key);
				Py_DECREF(value);
				continue;
			}
			lval = (logical *) malloc(sizeof(logical));
			/* Note opposite */
			if (PyObject_RichCompareBool(value,Py_False,Py_EQ)) {
				*lval = True;
				/* printf("%s False\n",keystr);*/
			}
			else {
				*lval = False;
				/*printf("%s True\n",keystr); */
			}
			md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
						 (void*)lval,NULL,1,&len_dims,
						 TEMPORARY,NULL,(NclTypeClass)nclTypelogicalClass);
			if (mode < 1)
				_NclFileSetOption(NULL,extq,NrmStringToQuark("DefineMode"),md);
			_NclFileSetOption(NULL,extq,NrmStringToQuark("SuppressClose"),md);
			_NclDestroyObj((NclObj)md);
			Py_DECREF(key);
			Py_DECREF(value);
			continue;
		}
		else if (! _NclFileIsOption(extq,qkeystr)) {
			if (options == option_defaults) { 
				/* 
				 * this means we're setting the option defaults; more permissive:
				 * options for all file formats may be present
				 */
				continue;
			}
			if (PyDict_Contains(option_defaults,key) && ! _NclFileIsOption(NrmNULLQUARK,qkeystr)) {
				/* 
				 * Options that are user-defined are passed through, but not built-in options that
				 *  apply only to another file format
				 */
				continue;
			}
			char s[256] = "";
			strncat(s,keystr,255);
			strncat(s," is not a valid option for this file format",255);
			PyErr_SetString(NIOError,s);
			Py_DECREF(key);
			continue;
		}
		if (PyString_Check(value)) {
			char *valstr = PyString_AsString(value);
			NrmQuark *qval = (NrmQuark *) malloc(sizeof(NrmQuark));
			*qval = NrmStringToQuark(valstr);
			md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
						 (void*)qval,NULL,1,&len_dims,
						 TEMPORARY,NULL,(NclTypeClass)nclTypestringClass);
			/* printf("%s %s\n", keystr,valstr); */
		}
		else if (PyBool_Check(value)) {
			logical * lval = (logical *) malloc(sizeof(logical));
			if (PyObject_RichCompareBool(value,Py_False,Py_EQ)) {
				*lval = False;
				/* printf("%s False\n",keystr);*/
			}
			else {
				*lval = True;
				/*printf("%s True\n",keystr); */
			}
			md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
						 (void*)lval,NULL,1,&len_dims,
						 TEMPORARY,NULL,(NclTypeClass)nclTypelogicalClass);
		}
		else if (PyInt_Check(value)) {
			int* ival = (int *) malloc(sizeof(int));
			*ival = (int) PyInt_AsLong(value);
			md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
						 (void*)ival,NULL,1,&len_dims,
						 TEMPORARY,NULL,(NclTypeClass)nclTypeintClass);
			/* printf("%s %ld\n",keystr,PyInt_AsLong(value));*/
		}
		else if (PyFloat_Check(value)) {
			float *fval = (float *) malloc(sizeof(float));
			*fval = (float) PyFloat_AsDouble(value);
			md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
						 (void*)fval,NULL,1,&len_dims,
						 TEMPORARY,NULL,(NclTypeClass)nclTypefloatClass);
			/*printf("%s %lg\n",keystr,PyFloat_AsDouble(value));*/
		}
		else {
			char s[256] = "";
			strncat(s,keystr,255);
			strncat(s," value is an invalid type for option",255);
			PyErr_SetString(NIOError,s);
			Py_DECREF(key);
			Py_DECREF(value);
			continue;
		}
		_NclFileSetOption(NULL,extq,qkeystr,md);
		_NclDestroyObj((NclObj)md);
		Py_DECREF(key);
		Py_DECREF(value);
	}
	Py_DECREF(keys);
}
	
/* Creator for NioFile objects */

static PyObject *
NioFile(PyObject *self, PyObject *args,PyObject *kwds)
{
  char *filepath;
  char *mode = "r";
  char *history = "";
  char *format = "";
  PyObject *options = Py_None;
  NioFileObject *file;
  NrmQuark extq;
  int crw;
  static char *argnames[] = { "filepath", "mode", "options", "history", "format", NULL };
  PyObject *option_defaults;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|sOss:open_file", argnames, &filepath, &mode, &options, &history,&format))
    return NULL;

  if (strlen(format) > 0 && strlen(format) < 16 && strlen(filepath) > 0) {
	  char *tfile = filepath;
	  filepath = malloc(strlen(filepath) + strlen(format) + 2);
	  sprintf(filepath,"%s.%s",tfile,format);
  }	  
  extq = GetExtension(filepath);

  if (extq == NrmNULLQUARK) {
	  PyErr_SetString(NIOError,"invalid extension or invalid file type");
	  return NULL;
  }

  crw = GetNioMode(filepath, mode);
  if (crw < -1) {
	  nio_ncerr = 25;
	  nio_seterror();
	  return NULL;
  }
  /* 
   * Three step to handle options
   * First set the hard-coded library defaults -- including a few module-specific values.
   * Then set the option defaults (which are modifiable by the user and allow for user-specified keys)
   * Finally set the options explicitly passed for this instance.
   */

  InitializeNioOptions(extq,crw);
  
  option_defaults = PyObject_GetAttrString(Niomodule,"option_defaults");
  SetNioOptions(extq,crw,option_defaults,option_defaults);
  if (options != Py_None) {
	  PyObject *options_dict;
	  if ( !(PyInstance_Check(options) && PyObject_HasAttrString(options,"__dict__"))) {
		  PyErr_SetString(NIOError, "options argument must be an NioOptions class instance");
	  }
	  options_dict = PyObject_GetAttrString(options,"__dict__");
	  SetNioOptions(extq,crw,options_dict,option_defaults);
	  Py_DECREF(options_dict);
  }
  Py_DECREF(option_defaults);

  file = NioFile_Open(filepath, mode);
  if (file == NULL) {
    nio_seterror();
    return NULL;
  }
  if (strlen(history) > 0) {
	  if (check_if_open(file,1)) {
		  NioFile_AddHistoryLine(file, history);
	  }
  }
  return (PyObject *)file;
}



static PyObject *
NioFile_Options(PyObject *self, PyObject *args)
{
	PyObject *class;
	PyObject *dict = PyDict_New();
	PyObject *pystr = PyString_FromString("NioOptions");
	PyObject *modstr = PyString_FromString("__module__");
	PyObject *modval = PyString_FromString("_Nio");
	PyObject *docstr = PyString_FromString("__doc__");
	PyObject *docval = PyString_FromString(option_class_doc);

	PyDict_SetItem(dict,modstr,modval);
	PyDict_SetItem(dict,docstr,docval);
	class = PyClass_New(NULL,dict,pystr);
	return PyInstance_New(class,NULL,NULL);
}
	

static PyObject *
SetUpDefaultOptions( void )
{
	PyObject *dict = PyDict_New();
	PyObject *opt, *val;

	opt = PyString_FromString("Format");
	val = PyString_FromString("classic");
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("HeaderReserveSpace");
	val = PyInt_FromLong(0);
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("MissingToFillValue");
	val = PyBool_FromLong(1);
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("PreFill");
	val = PyBool_FromLong(1);
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("SafeMode");
	val = PyBool_FromLong(0);
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("CompressionLevel");
	val = PyInt_FromLong(-1);
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("DefaultNCEPPtable");
	val = PyString_FromString("operational");
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("InitialTimeCoordinateType");
	val = PyString_FromString("numeric");
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("SingleElementDimensions");
	val = PyString_FromString("none");
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("ThinnedGridInterpolation");
	val = PyString_FromString("cubic");
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("TimePeriodSuffix");
	val = PyBool_FromLong(1);
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
	opt = PyString_FromString("FileStructure");
	val = PyString_FromString("standard");
	PyDict_SetItem(dict,opt,val);
 	Py_DECREF(opt);
	Py_DECREF(val);
		
	return dict;

}

/* Table of functions defined in the module */

static PyMethodDef nio_methods[] = {
	{"open_file",	(PyCFunction) NioFile, METH_KEYWORDS,NULL},
	{"options",   (PyCFunction)NioFile_Options,METH_NOARGS},
	{NULL,		NULL}		/* sentinel */
};

/* Module initialization */

PyMODINIT_FUNC
initnio(void)
{
  PyObject *m, *d;
  PyObject *def_opt_dict;
  static void *PyNIO_API[PyNIO_API_pointers];

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
  
  if (PyType_Ready(&NioFile_Type) < 0)
	  return;
  if (PyType_Ready(&NioVariable_Type) < 0)
	  return;
  
  /* Create the module and add the functions */
  m = Py_InitModule("nio", nio_methods);
  Niomodule = m;

  /* Initialize the NIO library */

  NioInitialize();
 
  /* Import the array module */
/*#ifdef import_array*/
  import_array();
/*#endif*/

  /* Add error object to the module */
  d = PyModule_GetDict(m);
  /*NIOError = PyString_FromString("NIOError");*/
  NIOError = PyErr_NewException("Nio.NIOError",NULL,NULL);
  PyDict_SetItemString(d, "NIOError", NIOError);
 
  /* make NioFile, NioGroup and NioVariable visible to the module for subclassing */ 
  Py_INCREF(&NioFile_Type);
  PyModule_AddObject(m,"_NioFile", (PyObject *) &NioFile_Type);
  Py_INCREF(&NioVariable_Type);
  PyModule_AddObject(m,"_NioVariable", (PyObject *) &NioVariable_Type);
  Py_INCREF(m);
  PyModule_AddObject(m,"_Nio", (PyObject *) m);

  def_opt_dict = SetUpDefaultOptions();

  PyModule_AddObject(m,"option_defaults", (PyObject *) def_opt_dict);

  /* Initialize C API pointer array and store in module */
  PyNIO_API[NioFile_Type_NUM] = (void *)&NioFile_Type;
  PyNIO_API[NioVariable_Type_NUM] = (void *)&NioVariable_Type;
  PyNIO_API[NioFile_Open_NUM] = (void *)&NioFile_Open;
  PyNIO_API[NioFile_Close_NUM] = (void *)&NioFile_Close;
/*
  PyNIO_API[NioFile_Sync_NUM] = (void *)&NioFile_Sync;
*/
  PyNIO_API[NioFile_CreateDimension_NUM] =
    (void *)&NioFile_CreateDimension;
  PyNIO_API[NioFile_CreateChunkDimension_NUM] =
    (void *)&NioFile_CreateChunkDimension;
  PyNIO_API[NioFile_CreateVariable_NUM] =
    (void *)&NioFile_CreateVariable;
  PyNIO_API[NioFile_GetVariable_NUM] =
    (void *)&NioFile_GetVariable;
  PyNIO_API[NioVariable_GetRank_NUM] =
    (void *)&NioVariable_GetRank;
  PyNIO_API[NioVariable_GetShape_NUM] =
    (void *)&NioVariable_GetShape;
  PyNIO_API[NioVariable_Indices_NUM] =
    (void *)&NioVariable_Indices;
  PyNIO_API[NioVariable_ReadAsArray_NUM] =
    (void *)&NioVariable_ReadAsArray;
  PyNIO_API[NioVariable_ReadAsString_NUM] =
    (void *)&NioVariable_ReadAsString;
  PyNIO_API[NioVariable_WriteArray_NUM] =
    (void *)&NioVariable_WriteArray;
  PyNIO_API[NioVariable_WriteString_NUM] =
    (void *)&NioVariable_WriteString;
  PyNIO_API[NioFile_GetAttribute_NUM] =
    (void *)&NioFile_GetAttribute;
  PyNIO_API[NioFile_SetAttribute_NUM] =
    (void *)&NioFile_SetAttribute;
  PyNIO_API[NioFile_SetAttributeString_NUM] =
    (void *)&NioFile_SetAttributeString;
  PyNIO_API[NioVariable_GetAttribute_NUM] =
    (void *)&NioVariable_GetAttribute;
  PyNIO_API[NioVariable_SetAttribute_NUM] =
    (void *)&NioVariable_SetAttribute;
  PyNIO_API[NioVariable_SetAttributeString_NUM] =
    (void *)&NioVariable_SetAttributeString;
  PyNIO_API[NioFile_AddHistoryLine_NUM] =
    (void *)&NioFile_AddHistoryLine;
  PyDict_SetItemString(d, "_C_API",
		       PyCObject_FromVoidPtr((void *)PyNIO_API, NULL));

  PyNIO_API[NioFile_CreateVLEN_NUM] = (void *)&NioFile_CreateVLEN;
  PyNIO_API[NioFile_CreateCOMPOUND_NUM] = (void *)&NioFile_CreateCOMPOUND;
  PyNIO_API[NioFile_CreateCOMPOUNDtype_NUM] = (void *)&NioFile_CreateCOMPOUNDtype;
  PyNIO_API[NioVariable_GetSize_NUM] = (void *)&NioVariable_GetSize;

  /* Check for errors */
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module nio");
}

