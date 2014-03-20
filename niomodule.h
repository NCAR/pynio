/*******************************************************
 * $Id$
 *******************************************************/

#ifndef Py_NIOMODULE_H
#define Py_NIOMODULE_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Include file for NIO files and variables.
 *
 * adapted from code written by Konrad Hinsen
 * last revision: 2006-03-07
 */


#include <stdio.h>
#include "Python.h"

/* NIOFile object */

typedef struct {
  PyObject_HEAD
  PyObject *dimensions;   /* dictionary */
  PyObject *variables;    /* dictionary */
  PyObject *attributes;   /* dictionary */
  PyObject *name;         /* string */
  PyObject *mode;         /* string */
  void *id;
  char open;
  char define;
  char write;
  int recdim;
} NioFileObject;

/* NIOVariable object */

typedef struct {
  PyObject_HEAD
  NioFileObject *file;
  PyObject *attributes;   /* dictionary */
  char *name;
  NrmQuark *qdims;
  Py_ssize_t *dimensions;
  int type;               /* same as array types */
  int nd;
  int id;
  char unlimited;
} NioVariableObject;


/* Variable index structure */

typedef struct {
  Py_ssize_t start;
  Py_ssize_t stop;
  Py_ssize_t stride;
  short item;
  short unlimited;
  short no_start; /* start is not None */
  short no_stop;  /* stop is not None */
} NioIndex;

/*
 * C API functions
 */

/* Type definitions */
#define NioFile_Type_NUM 0
#define NioVariable_Type_NUM 1

/* Open a NIO file (i.e. create a new file object) */
#define NioFile_Open_RET NioFileObject *
#define NioFile_Open_PROTO Py_PROTO((char *filename, char *mode))
#define NioFile_Open_NUM 2

/* Close a NIO file. Returns -1 if there was an error. */
#define NioFile_Close_RET int
#define NioFile_Close_PROTO Py_PROTO((NioFileObject *file))
#define NioFile_Close_NUM 3

/* Ensure that all data is written to the disk file.
   Returns 0 if there was an error. */
/*
#define NioFile_Sync_RET int
#define NioFile_Sync_PROTO Py_PROTO((NioFileObject *file))
#define NioFile_Sync_NUM 4
*/
/* Create a new dimension. Returns -1 if there was an error. */
#define NioFile_CreateDimension_RET int
#define NioFile_CreateDimension_PROTO \
        Py_PROTO((NioFileObject *file, char *name, Py_ssize_t size))
#define NioFile_CreateDimension_NUM 5

/* Create a NIO variable and return the variable object */
#define NioFile_CreateVariable_RET NioVariableObject *
#define NioFile_CreateVariable_PROTO \
      Py_PROTO((NioFileObject *file, char *name, int typecode, \
                char **dimension_names, int ndim))
#define NioFile_CreateVariable_NUM 6

/* Return an object referring to an existing variable */
#define NioFile_GetVariable_RET NioVariableObject *
#define NioFile_GetVariable_PROTO \
	  Py_PROTO((NioFileObject *file, char *name))
#define NioFile_GetVariable_NUM 7

/* Get variable rank */
#define NioVariable_GetRank_RET int
#define NioVariable_GetRank_PROTO Py_PROTO((NioVariableObject *var))
#define NioVariable_GetRank_NUM 8

/* Get variable shape */
#define NioVariable_GetShape_RET Py_ssize_t *
#define NioVariable_GetShape_PROTO Py_PROTO((NioVariableObject *var))
#define NioVariable_GetShape_NUM 9

/* Allocate and initialize index structures for reading/writing data */
#define NioVariable_Indices_RET NioIndex *
#define NioVariable_Indices_PROTO Py_PROTO((NioVariableObject *var))
#define NioVariable_Indices_NUM 10

/* Read data and return an array object */
#define NioVariable_ReadAsArray_RET PyArrayObject *
#define NioVariable_ReadAsArray_PROTO \
	  Py_PROTO((NioVariableObject *var, NioIndex *indices))
#define NioVariable_ReadAsArray_NUM 11

/* Write array. Returns -1 if there was an error.  */
#define NioVariable_WriteArray_RET int
#define NioVariable_WriteArray_PROTO \
	  Py_PROTO((NioVariableObject *var, NioIndex *indices, \
		    PyObject *array))
#define NioVariable_WriteArray_NUM 12

/* Get file attribute */
#define NioFile_GetAttribute_RET PyObject *
#define NioFile_GetAttribute_PROTO \
	  Py_PROTO((NioFileObject *var, char *name))
#define NioFile_GetAttribute_NUM 13

/* Set file attribute */
#define NioFile_SetAttribute_RET int
#define NioFile_SetAttribute_PROTO \
	  Py_PROTO((NioFileObject *var, char *name, PyObject *value))
#define NioFile_SetAttribute_NUM 14

/* Set file attribute to string value */
#define NioFile_SetAttributeString_RET int
#define NioFile_SetAttributeString_PROTO \
	  Py_PROTO((NioFileObject *var, char *name, char *value))
#define NioFile_SetAttributeString_NUM 15

/* Get variable attribute */
#define NioVariable_GetAttribute_RET PyObject *
#define NioVariable_GetAttribute_PROTO \
	  Py_PROTO((NioVariableObject *var, char *name))
#define NioVariable_GetAttribute_NUM 16

/* Set variable attribute */
#define NioVariable_SetAttribute_RET int
#define NioVariable_SetAttribute_PROTO \
	  Py_PROTO((NioVariableObject *var, char *name, PyObject *value))
#define NioVariable_SetAttribute_NUM 17

/* Set variable attribute to string value */
#define NioVariable_SetAttributeString_RET int
#define NioVariable_SetAttributeString_PROTO \
	  Py_PROTO((NioVariableObject *var, char *name, char *value))
#define NioVariable_SetAttributeString_NUM 18

/* Add entry to the history */
#define NioFile_AddHistoryLine_RET int
#define NioFile_AddHistoryLine_PROTO \
	  Py_PROTO((NioFileObject *self, char *text))
#define NioFile_AddHistoryLine_NUM 19

/* Write string. Returns -1 if there was an error.  */
#define NioVariable_WriteString_RET int
#define NioVariable_WriteString_PROTO \
	  Py_PROTO((NioVariableObject *var, PyStringObject *value))
#define NioVariable_WriteString_NUM 20

/* Read string  */
#define NioVariable_ReadAsString_RET PyStringObject *
#define NioVariable_ReadAsString_PROTO \
	  Py_PROTO((NioVariableObject *var))
#define NioVariable_ReadAsString_NUM 21

/* Total number of C API pointers */
#define PyNIO_API_pointers 22



#ifdef _NIO_MODULE

/* Type object declarations */
staticforward PyTypeObject NioFile_Type;
staticforward PyTypeObject NioVariable_Type;

/* Type check macros */
#define NioFile_Check(op) ((op)->ob_type == &NioFile_Type)
#define NioVariable_Check(op) ((op)->ob_type == &NioVariable_Type)

/* C API function declarations */
static NioFile_Open_RET NioFile_Open NioFile_Open_PROTO;
static NioFile_Close_RET NioFile_Close NioFile_Close_PROTO;
/*
static NioFile_Sync_RET NioFile_Sync NioFile_Sync_PROTO;
*/
static NioFile_CreateDimension_RET NioFile_CreateDimension \
  NioFile_CreateDimension_PROTO;
static NioFile_CreateVariable_RET NioFile_CreateVariable \
  NioFile_CreateVariable_PROTO;
static NioFile_GetVariable_RET NioFile_GetVariable \
  NioFile_GetVariable_PROTO;
static NioVariable_GetRank_RET NioVariable_GetRank \
  NioVariable_GetRank_PROTO;
static NioVariable_GetShape_RET NioVariable_GetShape \
  NioVariable_GetShape_PROTO;
static NioVariable_Indices_RET NioVariable_Indices \
  NioVariable_Indices_PROTO;
static NioVariable_ReadAsArray_RET NioVariable_ReadAsArray \
  NioVariable_ReadAsArray_PROTO;
static NioVariable_ReadAsString_RET NioVariable_ReadAsString \
  NioVariable_ReadAsString_PROTO;
static NioVariable_WriteArray_RET NioVariable_WriteArray \
  NioVariable_WriteArray_PROTO;
static NioVariable_WriteString_RET NioVariable_WriteString \
  NioVariable_WriteString_PROTO;
static NioFile_GetAttribute_RET NioFile_GetAttribute \
  NioFile_GetAttribute_PROTO;
static NioFile_SetAttribute_RET NioFile_SetAttribute \
  NioFile_SetAttribute_PROTO;
static NioFile_SetAttributeString_RET NioFile_SetAttributeString \
  NioFile_SetAttributeString_PROTO;
static NioVariable_GetAttribute_RET NioVariable_GetAttribute \
  NioVariable_GetAttribute_PROTO;
static NioVariable_SetAttribute_RET NioVariable_SetAttribute \
  NioVariable_SetAttribute_PROTO;
static NioVariable_SetAttributeString_RET \
  NioVariable_SetAttributeString \
  NioVariable_SetAttributeString_PROTO;
static NioFile_AddHistoryLine_RET NioFile_AddHistoryLine \
  NioFile_AddHistoryLine_PROTO;

#else

/* C API address pointer */ 
static void **PyNIO_API;

/* Type check macros */
#define NioFile_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyNIO_API[NioFile_Type_NUM])
#define NioVariable_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyNIO_API[NioVariable_Type_NUM])

/* C API function declarations */
#define NioFile_Open \
  (*(NioFile_Open_RET (*)NioFile_Open_PROTO) \
   PyNIO_API[NioFile_Open_NUM])
#define NioFile_Close \
  (*(NioFile_Close_RET (*)NioFile_Close_PROTO) \
   PyNIO_API[NioFile_Close_NUM])
#define NioFile_Sync \
  (*(NioFile_Sync_RET (*)NioFile_Sync_PROTO) \
   PyNIO_API[NioFile_Sync_NUM])
#define NioFile_CreateDimension \
  (*(NioFile_CreateDimension_RET (*)NioFile_CreateDimension_PROTO) \
   PyNIO_API[NioFile_CreateDimension_NUM])
#define NioFile_CreateVariable \
  (*(NioFile_CreateVariable_RET (*)NioFile_CreateVariable_PROTO) \
   PyNIO_API[NioFile_CreateVariable_NUM])
#define NioFile_GetVariable \
  (*(NioFile_GetVariable_RET (*)NioFile_GetVariable_PROTO) \
   PyNIO_API[NioFile_GetVariable_NUM])
#define NioVariable_GetRank \
  (*(NioVariable_GetRank_RET (*)NioVariable_GetRank_PROTO) \
   PyNIO_API[NioVariable_GetRank_NUM])
#define NioVariable_GetShape \
  (*(NioVariable_GetShape_RET (*)NioVariable_GetShape_PROTO) \
   PyNIO_API[NioVariable_GetShape_NUM])
#define NioVariable_Indices \
  (*(NioVariable_Indices_RET (*)NioVariable_Indices_PROTO) \
   PyNIO_API[NioVariable_Indices_NUM])
#define NioVariable_ReadAsArray \
  (*(NioVariable_ReadAsArray_RET (*)NioVariable_ReadAsArray_PROTO) \
   PyNIO_API[NioVariable_ReadAsArray_NUM])
#define NioVariable_ReadAsString \
  (*(NioVariable_ReadAsString_RET (*)NioVariable_ReadAsString_PROTO) \
   PyNIO_API[NioVariable_ReadAsString_NUM])
#define NioVariable_WriteArray \
  (*(NioVariable_WriteArray_RET (*)NioVariable_WriteArray_PROTO) \
   PyNIO_API[NioVariable_WriteArray_NUM])
#define NioVariable_WriteString \
  (*(NioVariable_WriteString_RET (*)NioVariable_WriteString_PROTO) \
   PyNIO_API[NioVariable_WriteString_NUM])
#define NioFile_GetAttribute \
  (*(NioFile_GetAttribute_RET (*)NioFile_GetAttribute_PROTO) \
   PyNIO_API[NioFile_GetAttribute_NUM])
#define NioFile_SetAttribute \
  (*(NioFile_SetAttribute_RET (*)NioFile_SetAttribute_PROTO) \
   PyNIO_API[NioFile_SetAttribute_NUM])
#define NioFile_SetAttributeString \
  (*(NioFile_SetAttributeString_RET \
     (*)NioFile_SetAttributeString_PROTO) \
   PyNIO_API[NioFile_SetAttributeString_NUM])
#define NioVariable_GetAttribute \
  (*(NioVariable_GetAttribute_RET (*)NioVariable_GetAttribute_PROTO) \
   PyNIO_API[NioVariable_GetAttribute_NUM])
#define NioVariable_SetAttribute \
  (*(NioVariable_SetAttribute_RET (*)NioVariable_SetAttribute_PROTO) \
   PyNIO_API[NioVariable_SetAttribute_NUM])
#define NioVariable_SetAttributeString \
  (*(NioVariable_SetAttributeString_RET \
     (*)NioVariable_SetAttributeString_PROTO) \
   PyNIO_API[NioVariable_SetAttributeString_NUM])
#define NioFile_AddHistoryLine \
  (*(NioFile_AddHistoryLine_RET \
     (*)NioFile_AddHistoryLine_PROTO) \
   PyNIO_API[NioFile_AddHistoryLine_NUM])

#define import_NIO() \
{ \
  PyObject *module = PyImport_ImportModule("nio"); \
  if (module != NULL) { \
    PyObject *module_dict = PyModule_GetDict(module); \
    PyObject *c_api_object = PyDict_GetItemString(module_dict, "_C_API"); \
    if (PyCObject_Check(c_api_object)) { \
      PyNIO_API = (void **)PyCObject_AsVoidPtr(c_api_object); \
    } \
  } \
}

#endif



#ifdef __cplusplus
}
#endif
#endif /* Py_NIOMODULE_H */
