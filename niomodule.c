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
#ifdef NUMPY
#define PY_ARRAY_TYPES_PREFIX NUMPY_
#include <numpy/arrayobject.h>
typedef NUMPY_intp intp;
#else
#include <Numeric/arrayobject.h>
#endif

#include <sys/stat.h>
#include <unistd.h>


#define _NIO_MODULE
#include "niomodule.h"

/*
 * global used in NclMultiDValData
 */
short NCLnoPrintElem = 0;

staticforward int nio_file_init(NioFileObject *self);
staticforward NioVariableObject *nio_variable_new(
	NioFileObject *file, char *name, int id, 
	int type, int ndims, int *dimids, int nattrs);


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
	case NCL_int:
		return PyArray_INT;  /* netcdf 3.x has only a long type */
	case NCL_long:
		return PyArray_LONG;
	case NCL_float:
		return PyArray_FLOAT;
	case NCL_double:
		return PyArray_DOUBLE;
        case NCL_char:
		return PyArray_CHAR;
        case NCL_byte:
		return PyArray_UBYTE;
        case NCL_string:
#ifdef NUMPY
		return PyArray_STRING;
#else
		return PyArray_OBJECT; /* this works for read-only */
#endif
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


static char
typecode(int type)
{
  char t;
  switch(type) {
#ifdef NUMPY
  case PyArray_BYTE:
	  t = PyArray_BYTELTR;
	  break;
  case PyArray_UBYTE:
	  t = PyArray_UBYTELTR;
	  break;
  case PyArray_SHORT:
	  t = PyArray_SHORTLTR;
	  break;
  case PyArray_INT:
	  t = PyArray_INTLTR;
	  break;
  case PyArray_LONG:
	  t = PyArray_LONGLTR;
	  break;
  case PyArray_FLOAT:
	  t = PyArray_FLOATLTR;
	  break;
  case PyArray_DOUBLE:
	  t = PyArray_DOUBLELTR;
	  break;
  case PyArray_STRING:
	  t = PyArray_STRINGLTR;
	  break;
  default: 
	  t = ' ';
#else
  case PyArray_CHAR:
	  t = 'c';
	  break;
  case PyArray_SBYTE:
	  t = '1';
	  break;
  case PyArray_UBYTE:
	  t = 'b';
	  break;
  case PyArray_SHORT:
	  t = 's';
	  break;
  case PyArray_INT:
	  t = 'i';
	  break;
  case PyArray_LONG:
	  t = 'l';
	  break;
  case PyArray_FLOAT:
	  t = 'f';
	  break;
  case PyArray_DOUBLE:
	  t = 'd';
	  break;
  default: 
	  t = ' ';
#endif
  }
  return t;
}

#ifdef NUMPY
static NrmQuark
nio_type_from_code(char code)
{
  int type;
  switch(code) {
  case 'c':
  case '1':
	  type = NrmStringToQuark("character");
	  break;
  case 'B':
	  type = NrmStringToQuark("byte");
	  break;
  case 'h':
	  type = NrmStringToQuark("short");
	  break;
  case 'i':
	  type = NrmStringToQuark("integer");
	  break;
  case 'l':
	  type = NrmStringToQuark("long");
	  break;
  case 'f':
	  type = NrmStringToQuark("float");
	  break;
  case 'd':
	  type = NrmStringToQuark("double");
	  break;
  default:
	  type = NrmNULLQUARK;
  }
  return type;
}
#else
static NrmQuark
nio_type_from_code(char code)
{
  int type;
  switch(code) {
  case 'c':
	  type = NrmStringToQuark("character");
	  break;
  case 'b':
  case '1':
	  type = NrmStringToQuark("byte");
	  break;
  case 's':
	  type = NrmStringToQuark("short");
	  break;
  case 'i':
	  type = NrmStringToQuark("integer");
	  break;
  case 'l':
	  type = NrmStringToQuark("long");
	  break;
  case 'f':
	  type = NrmStringToQuark("float");
	  break;
  case 'd':
	  type = NrmStringToQuark("double");
	  break;
  default:
	  type = NrmNULLQUARK;
  }
  return type;
}
#endif

static void
collect_attributes(void *fileid, int varid, PyObject *attributes, int nattrs)
{
  NclFile file = (NclFile) fileid;
  NclFileAttInfoList *att_list = NULL;
  NclFAttRec *att;
  NclFVarRec *fvar = NULL;
  char *name;
  int length;
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
		  PyObject *string;
		  char *satt = NrmQuarkToString(*((NrmQuark *)md->multidval.val));
		  string = PyString_FromString(satt);
		  if (string != NULL) {
			  PyDict_SetItemString(attributes, name, string);
			  Py_DECREF(string);
		  }
	  }
	  else {
		  PyObject *array;
		  length = md->multidval.totalelements;
		  py_type = data_type(att->data_type);
		  array = PyArray_FromDims(1, &length, py_type);
		  if (array != NULL) {
			  memcpy(((PyArrayObject *)array)->data,
				 md->multidval.val,length * md->multidval.type->type_class.size);
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
  
  if (!value) {
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
	  int len_dims = 1;
	  NrmQuark *qval = malloc(sizeof(NrmQuark));
	  qval[0] = NrmStringToQuark(PyString_AsString(value));
	  md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
				   (void*)qval,NULL,1,&len_dims,
				   TEMPORARY,NULL,(NclTypeClass)nclTypestringClass);
  }
  else {
	  int n_dims;
	  int dim_sizes = 1;
	  NrmQuark qtype;
	  int pyarray_type = PyArray_NOTYPE;
	  PyArrayObject *tmparray = (PyArrayObject *)PyDict_GetItemString(attributes,name);
	  if (tmparray != NULL) {
		  pyarray_type = tmparray->descr->type_num;
	  }
	  array = (PyArrayObject *)PyArray_ContiguousFromObject(value, pyarray_type, 0, 1);
          if (array) {
	          n_dims = (array->nd == 0) ? 1 : array->nd;
	          qtype = nio_type_from_code(array->descr->type);
	          if (array->descr->elsize == 8 && qtype == NrmStringToQuark("long")) {
                           PyArrayObject *array2 = (PyArrayObject *)
                                     PyArray_Cast(array, PyArray_INT);
                           Py_DECREF(array);
                           array = array2;
			   qtype = NrmStringToQuark("integer");
                  }
                  if (array) {
	                   md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
		                                    (void*)array->data,NULL,n_dims,
				                    array->nd == 0 ? &dim_sizes : array->dimensions,
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
/*
  if (self->open)
    NioFile_Close(self);
*/
  Py_XDECREF(self->dimensions);
  Py_XDECREF(self->variables);
  Py_XDECREF(self->attributes);
  Py_XDECREF(self->name);
  Py_XDECREF(self->mode);
  PyMem_DEL(self);
}

static int GetNioMode(char* filename,char *mode) 
{
	struct stat buf;
	int crw;

	if (mode == NULL)
		mode = "r";

	switch (mode[0]) {
	case 'a':
		if (stat(filename,&buf) < 0)
			crw = -1;
		else 
			crw = 0;
		break;
	case 'c':
		crw = -1;
		break;
	case 'r':
		if (strlen(mode) > 1 && (mode[1] == '+' || mode[1] == 'w')) {
			if (stat(filename,&buf) < 0)
				crw = -1;
			else 
				crw = 0;
		}
		else
			crw = 1;
		break;
	case 'w':
		if (stat(filename,&buf) < 0)
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
  nio_ncerr = 0;
  int crw;

  if (self == NULL)
    return NULL;
  self->dimensions = NULL;
  self->variables = NULL;
  self->attributes = NULL;
  self->name = NULL;
  self->mode = NULL;
  self->open = 0;

  crw = GetNioMode(filename,mode);
  file = _NclCreateFile(NULL,NULL,Ncl_File,0,TEMPORARY,
			NrmStringToQuark(filename),crw);
  if (file) {
	  self->id = (void *) file;
	  self->define = 1;
	  self->open = 1;
	  self->write = (crw != 1);
	  nio_file_init(self); 
  }
  else {
	  NioFileObject_dealloc(self);
	  return NULL;
  }
  self->name = PyString_FromString(filename);
  self->mode = PyString_FromString(mode);
  return self;
}

/* Create variables from file */

static int
nio_file_init(NioFileObject *self)
{
  NclFile file = (NclFile) self->id;
  int ndims, nvars, ngattrs;
  int i,j;
  int scalar_dim_ix = -1;
  NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
  self->dimensions = PyDict_New();
  self->variables = PyDict_New();
  self->attributes = PyDict_New();
  ndims = file->file.n_file_dims;
  nvars = file->file.n_vars;
  ngattrs = file->file.n_file_atts;
  self->recdim = -1; /* for now */
  for (i = 0; i < ndims; i++) {
    char *name;
    long size;
    PyObject *size_ob;
    NclFDimRec *fdim = file->file.file_dim_info[i];
    if (fdim->dim_name_quark != scalar_dim) {
	    name = NrmQuarkToString(fdim->dim_name_quark);
	    size = fdim->dim_size;
	    size_ob = PyInt_FromLong(size);
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
    int *dimids;
    NioVariableObject *variable;
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
	    dimids = NULL;
    }
    else if (ndimensions > 0) {
      dimids = (int *)malloc(ndimensions*sizeof(int));
      if (dimids == NULL) {
	PyErr_NoMemory();
	return 0;
      }
      for (j = 0; j < ndimensions; j++)
	      dimids[j] = fvar->file_dim_num[j];
    }
    else
      dimids = NULL;
    variable = nio_variable_new(self, name, i, data_type(datatype),
				   ndimensions, dimids, nattrs);
    PyDict_SetItemString(self->variables, name, (PyObject *)variable);
    Py_DECREF(variable);
  }

  collect_attributes(self->id, NC_GLOBAL, self->attributes, ngattrs);

  return 1;
}


/* Create dimension */

int
NioFile_CreateDimension(NioFileObject *file, char *name, long size)
{
  PyObject *size_ob;
  NrmQuark qname;
  if (check_if_open(file, 1)) {
	  NclFile nfile = (NclFile) file->id;
	  NhlErrorTypes ret;
	  
	  if (PyDict_GetItemString(file->dimensions,name)) {
		  printf("dimension %s exists: cannot create\n",name);
		  return 0;
	  }
	  if (size == 0 && file->recdim != -1) {
		  PyErr_SetString(NIOError, "there is already an unlimited dimension");
		  return -1;
	  }
	  define_mode(file, 1);
	  qname = NrmStringToQuark(name);
	  ret = _NclFileAddDim(nfile,qname,(int)size,(size == 0 ? 1 : 0));
	  if (ret > NhlWARNING) {
		  if (size == 0) {
			  NclFile nfile = (NclFile) file->id;
			  PyDict_SetItemString(file->dimensions, name, Py_None);
			  file->recdim = _NclFileIsDim(nfile,qname);
		  }
		  else {
			  size_ob = PyInt_FromLong(size);
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
  long size;
  if (!PyArg_ParseTuple(args, "sO", &name, &size_ob))
    return NULL;
  if (size_ob == Py_None)
    size = 0;
  else if (PyInt_Check(size_ob))
    size = PyInt_AsLong(size_ob);
  else {
    PyErr_SetString(PyExc_TypeError, "size must be None or integer");
    return NULL;
  }
  if (NioFile_CreateDimension(self, name, size) == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  else
    return NULL;
}

/* Create variable */

NioVariableObject *
NioFile_CreateVariable( NioFileObject *file, char *name, 
			  int typecode, char **dimension_names, int ndim)
{

  if (check_if_open(file, 1)) {
	  NioVariableObject *variable;
	  int i,id;
	  NclFile nfile = (NclFile) file->id;
	  int *dimids = NULL;
	  NrmQuark *qdims = NULL; 
	  NhlErrorTypes ret;
	  NrmQuark qvar;
	  NrmQuark qtype;
          int ncl_ndims = ndim;
	  define_mode(file, 1);

	  variable = (NioVariableObject *) PyDict_GetItemString(file->variables,name);
	  if (variable) {
		  printf("variable %s exists: cannot create\n",name);
		  return variable;
	  }
		  
	  if (ndim > 0) {
		  qdims = (NrmQuark *)malloc(ndim*sizeof(NrmQuark));
		  dimids = (int *) malloc(ndim*sizeof(NrmQuark));
		  if (! (qdims && dimids)) {
			  return (NioVariableObject *)PyErr_NoMemory();
		  }
	  }
	  else if (ndim == 0) {
		  qdims = (NrmQuark *)malloc(sizeof(NrmQuark));
		  dimids = NULL;
		  if (! qdims) {
			  return (NioVariableObject *)PyErr_NoMemory();
		  }
		  *qdims = NrmStringToQuark("ncl_scalar");
		  ncl_ndims = 1;
	  }
	  for (i = 0; i < ndim; i++) {
		  qdims[i] = NrmStringToQuark(dimension_names[i]);
		  dimids[i] = _NclFileIsDim(nfile,qdims[i]);
                  /*
		  for (j = 0; j < nfile->file.n_file_dims; j++) {
			  dimids[i] = -1;
			  if (nfile->file.file_dim_info[j]->dim_name_quark != qdims[i])
				  continue;
			  dimids[i] = j;
			  break;
		  }
		  */
		  if (dimids[i] == -1) {
			  nio_seterror();
			  if (qdims != NULL) 
				  free(qdims);
			  if (dimids != NULL)
				  free(dimids);
			  return NULL;
		  }
	  }
	  qtype = nio_type_from_code(typecode);
          if (sizeof(long) > 4 && qtype == NrmStringToQuark("long")) {
	          qtype = NrmStringToQuark("integer");
          }
	  qvar = NrmStringToQuark(name);
	  ret = _NclFileAddVar(nfile,qvar,qtype,ncl_ndims,qdims);
	  if (ret > NhlWARNING) {
		  id = _NclFileIsVar(nfile,qvar);
		  variable = nio_variable_new(file, name, id, 
					      data_type(nfile->file.var_info[id]->data_type),
					      ndim, dimids, 0);
		  PyDict_SetItemString(file->variables, name, (PyObject *)variable);
		  if (qdims != NULL) 
			  free(qdims);
		  return variable;
	  }
	  else {
		  if (qdims != NULL) 
			  free(qdims);
		  if (dimids != NULL)
			  free(dimids);
		  return NULL;
	  }
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
  char *name;
  int ndim;
  char type;
  int i;
  if (!PyArg_ParseTuple(args, "scO!", &name, &type, &PyTuple_Type, &dim))
    return NULL;
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
  var = NioFile_CreateVariable(self, name, type, dimension_names, ndim);
  free(dimension_names);
  return (PyObject *)var;
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
  if (check_if_open(file, 0)) {
	  _NclDestroyObj((NclObj)file->id);
	  file->open = 0;
	  return 0;
  }
  else
	  return -1;
}

static PyObject *
NioFileObject_close(NioFileObject *self, PyObject *args)
{
  char *history = NULL;
  if (!PyArg_ParseTuple(args, "|s", &history))
    return NULL;
  if (history != NULL) {
	  if (check_if_open(self,1)) {
		  NioFile_AddHistoryLine(self, history);
	  }
	  else {
		  PyErr_Print();
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
  {"close", (PyCFunction)NioFileObject_close, 1},
/* {"sync", (PyCFunction)NioFileObject_sync, 1}, */
  {"create_dimension", (PyCFunction)NioFileObject_new_dimension, 1},
  {"create_variable", (PyCFunction)NioFileObject_new_variable, 1},
  {NULL, NULL}		/* sentinel */
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
    if (strcmp(name, "variables") == 0) {
      Py_INCREF(self->variables);
      return self->variables;
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
      return Py_FindMethod(NioFileObject_methods, (PyObject *)self, name);
    }
  }
  else
    return NULL;
}

int
NioFile_SetAttribute(NioFileObject *self, char *name, PyObject *value)
{
  nio_ncerr = 0;
  if (check_if_open(self, 1)) {
    if (strcmp(name, "dimensions") == 0 ||
	strcmp(name, "variables") == 0 ||
	strcmp(name, "__dict__") == 0) {
	    PyErr_SetString(PyExc_TypeError, "attempt to set read-only attribute");
	    return -1;
    }
    define_mode(self, 1);
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
  char buf[300];
  sprintf(buf, "<%s Nio file '%.256s', mode '%.10s' at %lx>",
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

/* Printed representation */
static PyObject *
NioFileObject_str(NioFileObject *file)
{
	char tbuf[1024];
	char *buf;
	int len;
	int bufinc = 4096;
	int buflen = 0;
	int bufpos = 0;
	PyObject *pystr;
	NclFile nfile = (NclFile) file->id;
	int i;
	NrmQuark scalar_dim = NrmStringToQuark("ncl_scalar");
	PyObject *att_val;

	if (! check_if_open(file, -1)) {
		PyErr_Clear();
		return NioFileObject_repr(file);
	}

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
#ifdef NUMPY
				PyObject *att = att_arr_val->descr->f->getitem(PyArray_DATA(att_val),att_val);
#else
				PyObject *att = att_arr_val->descr->getitem(att_arr_val->data);
#endif

				format_object(tbuf,att,att_arr_val->descr->type);
				BUF_INSERT(tbuf);
				BUF_INSERT("\n");
			}
			else {
				sprintf(tbuf,"[");
				BUF_INSERT(tbuf);
				for (k = 0; k < att_arr_val->dimensions[0]; k++) {
#ifdef NUMPY
					PyObject *att = att_arr_val->descr->f->getitem(att_arr_val->data + 
										       k * att_arr_val->descr->elsize,att_val);
#else
					PyObject *att = att_arr_val->descr->getitem(att_arr_val->data + 
										    k * att_arr_val->descr->elsize);
#endif
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
#ifdef NUMPY
					PyObject *att = att_arr_val->descr->f->getitem(PyArray_DATA(att_val),att_val);
#else
					PyObject *att = att_arr_val->descr->getitem(att_arr_val->data);
#endif
					format_object(tbuf,att,att_arr_val->descr->type);
					/*sprintf(tbuf,"%s\n",PyString_AsString(PyObject_Str(att)));*/
					BUF_INSERT(tbuf);
					BUF_INSERT("\n");
				}
				else {
					sprintf(tbuf,"[");
					BUF_INSERT(tbuf);
					for (k = 0; k < att_arr_val->dimensions[0]; k++) {
#ifdef NUMPY
						PyObject *att = 
							att_arr_val->descr->f->getitem(att_arr_val->data + 
										       k * att_arr_val->descr->elsize,att_val);
#else
						PyObject *att = att_arr_val->descr->getitem(att_arr_val->data + 
											    k * att_arr_val->descr->elsize);
#endif

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

	pystr = PyString_FromString(buf);
	free(buf);
	return pystr;
}

/* Type definition */

statichere PyTypeObject NioFile_Type = {
  PyObject_HEAD_INIT(NULL)
  0,		/*ob_size*/
  "NioFile",	/*tp_name*/
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
  (reprfunc)NioFileObject_str   /*tp_str*/
};

/*
 * NIOVariable object
 * (type declaration in niomodule.h)
 */

/* Destroy variable object */

static void
NioVariableObject_dealloc(NioVariableObject *self)
{
  if (self->dimids != NULL)
    free(self->dimids);
  if (self->name != NULL)
    free(self->name);
  Py_XDECREF(self->file);
  PyMem_DEL(self);
}

/* Create variable object */

statichere NioVariableObject *
nio_variable_new(NioFileObject *file, char *name, int id, 
		 int type, int ndims, int *dimids, int nattrs)
{
  NioVariableObject *self;
  NclFile nfile = (NclFile) file->id;
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
    self->dimids = dimids;
    self->unlimited = 0;
    if (ndims > 0) {
	    self->dimensions = (size_t *)malloc(ndims*sizeof(size_t));
	    if (self->dimensions != NULL) {
		    for (i = 0; i < ndims; i++) {
			    self->dimensions[i] = nfile->file.file_dim_info[dimids[i]]->dim_size;
			    if (nfile->file.file_dim_info[dimids[i]]->is_unlimited)
				    self->unlimited = 1;
		    }
	    }
    }
    self->name = (char *)malloc(strlen(name)+1);
    if (self->name != NULL)
      strcpy(self->name, name);
    self->attributes = PyDict_New();
    collect_attributes(file->id, self->id, self->attributes, nattrs);
    return self;
  }
  else
    return NULL;
}

/* Return value */

static PyObject *
NioVariableObject_value(NioVariableObject *self, PyObject *args)
{
  NioIndex *indices;
  if (!PyArg_ParseTuple(args, ""))
    return NULL;
  if (self->nd == 0)
    indices = NULL;
  else
    indices = NioVariable_Indices(self);
  return PyArray_Return(NioVariable_ReadAsArray(self, indices));
}

/* Assign value */

static PyObject *
NioVariableObject_assign(NioVariableObject *self, PyObject *args)
{
  PyObject *value;
  NioIndex *indices;
  if (!PyArg_ParseTuple(args, "O", &value))
    return NULL;
  if (self->nd == 0)
    indices = NULL;
  else
    indices = NioVariable_Indices(self);
  NioVariable_WriteArray(self, indices, value);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Return typecode */

static PyObject *
NioVariableObject_typecode(NioVariableObject *self, PyObject *args)
{
  char t;
  if (!PyArg_ParseTuple(args, ""))
    return NULL;
  t = typecode(self->type);
  return PyString_FromStringAndSize(&t, 1);
}

/* Method table */

static PyMethodDef NioVariableObject_methods[] = {
  {"assign_value", (PyCFunction)NioVariableObject_assign, 1},
  {"get_value", (PyCFunction)NioVariableObject_value, 1},
  {"typecode", (PyCFunction)NioVariableObject_typecode, 1},
  {NULL, NULL}		/* sentinel */
};

/* Attribute access */

static int
NioVariable_GetRank( NioVariableObject *var)
{
  return var->nd;
}

static size_t *
NioVariable_GetShape(NioVariableObject *var)
{
  int i;
  if (check_if_open(var->file, -1)) {
	  NclFile nfile = (NclFile) var->file->id;
	  for (i = 0; i < var->nd; i++) {
		  var->dimensions[i] = nfile->file.file_dim_info[var->dimids[i]]->dim_size;
	  }
	  return var->dimensions;
  }
  else
	  return NULL;
}

static PyObject *
NioVariable_GetAttribute(NioVariableObject *self, char *name)
{
  PyObject *value;
  if (strcmp(name, "shape") == 0) {
    PyObject *tuple;
    int i;
    if (check_if_open(self->file, -1)) {
      NioVariable_GetShape(self);
      tuple = PyTuple_New(self->nd);
      for (i = 0; i < self->nd; i++)
	PyTuple_SetItem(tuple, i, PyInt_FromLong(self->dimensions[i]));
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
  if (strcmp(name, "dimensions") == 0) {
    PyObject *tuple;
    char *name;
    int i;
    if (check_if_open(self->file, -1)) {
      NclFile nfile = (NclFile) self->file->id;
      tuple = PyTuple_New(self->nd);
      for (i = 0; i < self->nd; i++) {
	      name = NrmQuarkToString(nfile->file.file_dim_info[self->dimids[i]]->dim_name_quark);
	      PyTuple_SetItem(tuple, i, PyString_FromString(name));
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
    return set_attribute(self->file, self->id, self->attributes,
			 name, value);
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

static int
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
    }
  else
    PyErr_SetString(PyExc_MemoryError, "out of memory");
  return indices;
}

PyArrayObject *
NioVariable_ReadAsArray(NioVariableObject *self,NioIndex *indices)
{
  int *dims;
  PyArrayObject *array;
  int i, d;
  int nitems;
  int error = 0;
  d = 0;
  nitems = 1;
  nio_ncerr = 0;
  if (!check_if_open(self->file, -1)) {
    free(indices);
    return NULL;
  }
  define_mode(self->file, 0);
  if (self->nd == 0)
    dims = NULL;
  else {
    dims = (int *)malloc(self->nd*sizeof(int));
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
    if (indices[i].stride < 0) {
	    indices[i].stop += 1;
	    indices[i].stride = -indices[i].stride;
    }
    else {
	    indices[i].stop -= 1;
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
      dims[d] = abs((indices[i].stop-indices[i].start)/indices[i].stride)+1;
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

#ifdef NUMPY
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
		  _NclDestroyObj((NclObj)md);
		  if (sel_ptr)
			  free(sel_ptr);

	  }		  
  }
#else
  if (nitems > 0 && self->type == PyArray_OBJECT) {
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
		  NrmQuark qstr;
		  PyObject *pystr;
		  array = (PyArrayObject *)PyArray_FromDims(d, dims, self->type);
		  for (i = 0; i < nitems; i++) {
			  qstr = ((NrmQuark *)md->multidval.val)[i];
			  pystr = PyString_FromString(NrmQuarkToString(qstr));
			  if (pystr != NULL) {
				  array->descr->setitem(pystr,array->data + i * array->descr->elsize);
			  }
		  }
		  _NclDestroyObj((NclObj)md);
		  if (sel_ptr)
			  free(sel_ptr);

	  }		  
  }
#endif
  else if (nitems > 0) {
	  array = (PyArrayObject *)PyArray_FromDims(d, dims, self->type);
	  if (array && self->nd == 0) {
		  NclFile nfile = (NclFile) self->file->id;
		  NclMultiDValData md = _NclFileReadVarValue
			  (nfile,NrmStringToQuark(self->name),NULL);
		  if (! md) {
			  nio_ncerr = 23;
			  nio_seterror();
			  Py_DECREF(array);
			  array = NULL;
		  }
		  /* all we care about is the actual value */
		  array->data = md->multidval.val;
		  md->multidval.val = NULL;
		  _NclDestroyObj((NclObj)md);
	  }
	  else if (array) {
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
		    
			  /* all we care about is the actual value */
			  array->data = md->multidval.val;
			  md->multidval.val = NULL;
			  _NclDestroyObj((NclObj)md);
			  free(sel_ptr);
		  }
	  }
  }
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

static int
NioVariable_WriteArray(NioVariableObject *self, NioIndex *indices, PyObject *value)
{
  int *dims;
  PyArrayObject *array;
  int i, d;
  int nitems,var_el_count;
  int error = 0;
  int ret = 0;

  /* update shape */
  (void) NioVariable_GetShape(self);
  d = 0;
  nitems = 1;
  var_el_count = 1;
  if (!check_if_open(self->file, 1)) {
    free(indices);
    return -1;
  }
  if (self->nd == 0)
    dims = NULL;
  else {
    dims = (int *)malloc(self->nd*sizeof(int));
    if (dims == NULL) {
      free(indices);
      PyErr_SetString(PyExc_MemoryError, "out of memory");
      return -1;
    }
  }
  define_mode(self->file, 0);
  /* convert from Python to NCL indexing */
  /* negative stride in Python requires the start index to be greater than
     the end index: in NCL negative stride reverses the direction 
     implied by the index start and stop.
  */

  for (i = 0; i < self->nd; i++) {
	  var_el_count *= self->dimensions[i];
	  error = error || (indices[i].stride == 0);
	  if (indices[i].stride < 0) {
		  indices[i].stop += 1;
		  indices[i].stride = -indices[i].stride;
	  }
	  else {
		  indices[i].stop -= 1;
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
		  dims[d] = abs((indices[i].stop-indices[i].start)/indices[i].stride)+1;
		  if (dims[d] < 0)
			  dims[d] = 0;
		  nitems *= dims[d];
		  d++;
	  }
	  else
		  indices[i].stop = indices[i].start;
  }
  if (error) {
    PyErr_SetString(PyExc_IndexError, "illegal index");
    if (dims != NULL)
      free(dims);
    if (indices != NULL)
      free(indices);
    return -1;
  }
  array = (PyArrayObject *)PyArray_ContiguousFromObject(value,self->type,0,d);
  if (array != NULL) {
	  NrmQuark qtype;
	  int n_dims;
	  int scalar_size = 1;
	  NclFile nfile = (NclFile) self->file->id;
	  NclMultiDValData md;
	  NhlErrorTypes nret;
	  int select_all = 1;

	  n_dims = array->nd;
	  if (array->nd == 0) {
		  n_dims = 1;
	  }
	  /*
	  if (array->nd != self->nd)
		  select_all = 0;
	  else {
		  for (i = 0; i < self->nd; i++) {
			  if (dims[i] == array->dimensions[i])
				  continue;
			  select_all = 0;
			  break;
		  }
	  }
	  */
	  if (nitems < var_el_count || self->unlimited)
		  select_all = 0;
	  qtype = nio_type_from_code(array->descr->type);
	  md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
				   (void*)array->data,NULL,n_dims,
				   array->nd == 0 ? &scalar_size : array->dimensions,
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
  }
  /* update shape */
  (void) NioVariable_GetShape(self);
  free(dims);
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
	  int str_dim_size = 1;
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
NioVariableObject_item(NioVariableObject *self, int i)
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
NioVariableObject_slice(NioVariableObject *self, int low, int high)
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
    int i = PyInt_AsLong(index);
    return NioVariableObject_item(self, i);
  }
  if (self->nd == 0) {
    PyErr_SetString(PyExc_TypeError, "Not a sequence");
    return NULL;
  }
  indices = NioVariable_Indices(self);
  if (indices != NULL) {
    if (PySlice_Check(index)) {
      PySlice_GetIndices((PySliceObject *)index, self->dimensions[0],
			 &indices->start, &indices->stop, &indices->stride);
      return PyArray_Return(NioVariable_ReadAsArray(self, indices));
    }
    if (PyTuple_Check(index)) {
      int ni = PyTuple_Size(index);
      if (ni <= self->nd) {
	int i, d;
	d = 0;
	for (i = 0; i < ni; i++) {
	  PyObject *subscript = PyTuple_GetItem(index, i);
	  if (PyInt_Check(subscript)) {
	    int n = PyInt_AsLong(subscript);
	    indices[d].start = n;
	    indices[d].stop = n+1;
	    indices[d].item = 1;
	    d++;
	  }
	  else if (PySlice_Check(subscript)) {
	    PySlice_GetIndices((PySliceObject *)subscript, self->dimensions[d],
			       &indices[d].start, &indices[d].stop,
			       &indices[d].stride);
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
NioVariableObject_ass_item(NioVariableObject *self, int i, PyObject *value)
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
NioVariableObject_ass_slice(NioVariableObject *self, int low, int high, PyObject *value)
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
    int i = PyInt_AsLong(index);
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
      PySlice_GetIndices((PySliceObject *)index, self->dimensions[0],
			 &indices->start, &indices->stop, &indices->stride);
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
	    int n = PyInt_AsLong(subscript);
	    indices[d].start = n;
	    indices[d].stop = n+1;
	    indices[d].item = 1;
	    d++;
	  }
	  else if (PySlice_Check(subscript)) {
	    PySlice_GetIndices((PySliceObject *)subscript, self->dimensions[d],
			       &indices[d].start, &indices[d].stop,
			       &indices[d].stride);
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
  PyErr_SetString(PyExc_TypeError, "can't add NIO variables");
  return NULL;
}

static PyObject *
NioVariableObject_error2(NioVariableObject *self,  int n)
{
  PyErr_SetString(PyExc_TypeError, "can't multiply NIO variables");
  return NULL;
}


static PySequenceMethods NioVariableObject_as_sequence = {
  (inquiry)NioVariableObject_length,		/*sq_length*/
  (binaryfunc)NioVariableObject_error1,       /*nb_add*/
  (intargfunc)NioVariableObject_error2,       /*nb_multiply*/
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

void printval(char *buf,NclBasicDataTypes type,void *val)
{
	switch(type) {
	case NCL_double:
		sprintf(buf,"%4.16g",*(double *)val);
		return;
	case NCL_float:
		sprintf(buf,"%2.7g", *(float *)val);
		return;
	case NCL_long:
		sprintf(buf,"%ld",*(long *)val);
		return;
	case NCL_int:
		sprintf(buf,"%d",*(int *)val);
		return;
	case NCL_short:
		sprintf(buf,"%d",*(short *)val);
		return;
	case NCL_string:
		strncat(buf,NrmQuarkToString(*(NrmQuark *)val),1024);
		return;
	case NCL_char:
		sprintf(buf,"%c",*(char *)val);
		return;
	case NCL_byte:
		sprintf(buf,"0x%x",*(unsigned char *)val);
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
	char *vname;
	NioVariableObject *vobj;
	long long total;
	int j, dim_ix;
	int n_atts;

	buf = malloc(bufinc);
	buflen = bufinc;
	for(i = 0; i < nfile->file.n_vars; i++) {
		if(nfile->file.var_info[i]->var_name_quark != varq) {
			continue;
		}
		vname = NrmQuarkToString(nfile->file.var_info[i]->var_name_quark);
		break;
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
#ifdef NUMPY
				PyObject *att = att_arr_val->descr->f->getitem(PyArray_DATA(att_val),att_val);
#else
				PyObject *att = att_arr_val->descr->getitem(att_arr_val->data);
#endif
				format_object(tbuf,att,att_arr_val->descr->type);
				BUF_INSERT(tbuf);
				BUF_INSERT("\n");
				/*sprintf(tbuf,"%s\n",PyString_AsString(PyObject_Str(att)));*/
			}
			else {
				sprintf(tbuf,"[");
				BUF_INSERT(tbuf);
				for (k = 0; k < att_arr_val->dimensions[0]; k++) {
#ifdef NUMPY
					PyObject *att = att_arr_val->descr->f->getitem(att_arr_val->data + 
										       k * att_arr_val->descr->elsize,att_val);
#else
					PyObject *att = att_arr_val->descr->getitem(att_arr_val->data + 
										    k * att_arr_val->descr->elsize);
#endif
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
  "NioVariable",  /*tp_name*/
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
  (reprfunc)NioVariableObject_str   /*tp_str*/ 
};


static NrmQuark 
GetExtension ( char * filename)
{
	char *cp = strrchr(filename,'.');
	if (cp == NULL || *(cp+1) == '\0')
		return NrmNULLQUARK;

	return NrmStringToQuark(cp+1);
}
	

void SetNioOptionDefaults(NrmQuark extq, int mode)
{
	NclMultiDValData md;
	int len_dims = 1;
	logical *lval;
	NrmQuark *qval;
	
	if (_NclFormatEqual(extq,NrmStringToQuark("nc"))) {
		_NclFileSetOptionDefaults(NrmStringToQuark("nc"),NrmNULLQUARK);
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
	else if (_NclFormatEqual(extq,NrmStringToQuark("grb"))) {
		_NclFileSetOptionDefaults(NrmStringToQuark("grb"),NrmNULLQUARK);
		qval = (NrmQuark *) malloc(sizeof(NrmQuark));
		*qval = NrmStringToQuark("numeric");
		md = _NclCreateMultiDVal(NULL,NULL,Ncl_MultiDValData,0,
					 (void*)qval,NULL,1,&len_dims,
					 TEMPORARY,NULL,(NclTypeClass)nclTypestringClass);
		_NclFileSetOption(NULL,extq,NrmStringToQuark("initialtimecoordinatetype"),md);
		_NclDestroyObj((NclObj)md);
	}
}
	
/* Creator for NioFile objects */

static PyObject *
NioFile(PyObject *self, PyObject *args)
{
  char *filename;
  char *mode = NULL;
  char *history = NULL;
  PyObject *options = NULL;
  NioFileObject *file;
  NrmQuark extq;
  int crw;

  if (!PyArg_ParseTuple(args, "s|sOs:open_file", &filename, &mode, &options, &history))
    return NULL;

  extq = GetExtension(filename);

  if (extq == NrmNULLQUARK) {
	  nio_seterror();
	  return NULL;
  }

  if (! mode)
	  mode = "r";
  crw = GetNioMode(filename, mode);
  if (crw < -1) {
	  nio_ncerr = 25;
	  nio_seterror();
	  return NULL;
  }
  SetNioOptionDefaults(extq,crw);
  
  if (options && !(PyInstance_Check(options) && PyObject_HasAttrString(options,"__dict__"))) {
	  if (options != Py_None) {
		  PyErr_SetString(NIOError, "options argument must be an Nio.options class instance");
		  PyErr_Print();
	  }
  }
  else if (options) {
	  NclMultiDValData md;
	  PyObject *dict = PyObject_GetAttrString(options,"__dict__");
	  PyObject *keys = PyMapping_Keys(dict);
	  int len_dims = 1;
	  int i;

	  for (i = 0; i < PySequence_Length(keys); i++) {
		  PyObject *key = PySequence_GetItem(keys,i);
		  char *keystr = PyString_AsString(key);
		  if (! _NclFileIsOption(extq,NrmStringToQuark(keystr))) {
			  printf("%s is not an option for this format\n",keystr);
			  Py_DECREF(key);
			  continue;
		  }
		  PyObject *value = PyMapping_GetItemString(dict,keystr);
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
			  if (PyObject_RichCompare(value,Py_False,Py_EQ)) {
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
		  _NclFileSetOption(NULL,extq,NrmStringToQuark(keystr),md);
		  _NclDestroyObj((NclObj)md);
		  Py_DECREF(key);
		  Py_DECREF(value);
	  }
  }
  file = NioFile_Open(filename, mode);
  if (file == NULL) {
    nio_seterror();
    return NULL;
  }
  if (history != NULL) {
	  if (check_if_open(file,1)) {
		  NioFile_AddHistoryLine(file, history);
	  }
	  else {
		  PyErr_Print();
	  }
  }
  return (PyObject *)file;
}

static PyObject *
NioFile_Option(PyObject *self, PyObject *args)
{
	PyObject *class;
	PyObject *dict = PyDict_New();
	PyObject *pystr = PyString_FromFormat("option");
	PyObject *modstr = PyString_FromFormat("__module__");
	PyObject *modval = PyString_FromFormat("Nio");

	PyDict_SetItem(dict,modstr,modval);
	class = PyClass_New(NULL,dict,pystr);
	return PyInstance_New(class,NULL,NULL);
}

/* Table of functions defined in the module */

static PyMethodDef nio_methods[] = {
  {"open_file",	NioFile, 1},
  {"options",   NioFile_Option,1},
  {NULL,		NULL}		/* sentinel */
};

/* Module initialization */

void
initNio(void)
{
  PyObject *m, *d;
  static void *PyNIO_API[PyNIO_API_pointers];


  /* Initialize type object headers */
  NioFile_Type.ob_type = &PyType_Type;
  NioVariable_Type.ob_type = &PyType_Type;

  /* Create the module and add the functions */
  m = Py_InitModule("Nio", nio_methods);

  NioInitialize();
  
  /* Import the array module */
/*#ifdef import_array*/
/*#ifdef NUMPY*/
  import_array();
/*#endif*/

  /* Add error object the module */
  d = PyModule_GetDict(m);
  NIOError = PyString_FromString("NIOError");
  PyDict_SetItemString(d, "NIOError", NIOError);

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

  /* Check for errors */
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module nio");
}

