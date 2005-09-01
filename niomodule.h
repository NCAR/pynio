#ifndef Py_NIOMODULE_H
#define Py_NIOMODULE_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Include file for NIO files and variables.
 *
 * Written by Konrad Hinsen
 * last revision: 2001-5-4
 */


#include <stdio.h>

/* NIOFile object */

typedef struct {
  PyObject_HEAD
  PyObject *dimensions;   /* dictionary */
  PyObject *variables;    /* dictionary */
  PyObject *attributes;   /* dictionary */
  PyObject *name;         /* string */
  PyObject *mode;         /* string */
  int id;
  char open;
  char define;
  char write;
  int recdim;
} PyNIOFileObject;


/* NIOVariable object */

typedef struct {
  PyObject_HEAD
  PyNIOFileObject *file;
  PyObject *attributes;   /* dictionary */
  char *name;
  int *dimids;
  size_t *dimensions;
  int type;               /* same as array types */
  int nd;
  int id;
  char unlimited;
} PyNIOVariableObject;


/* Variable index structure */

typedef struct {
  int start;
  int stop;
  int stride;
  int item;
} PyNIOIndex;

/*
 * C API functions
 */

/* Type definitions */
#define PyNIOFile_Type_NUM 0
#define PyNIOVariable_Type_NUM 1

/* Open a NIO file (i.e. create a new file object) */
#define PyNIOFile_Open_RET PyNIOFileObject *
#define PyNIOFile_Open_PROTO Py_PROTO((char *filename, char *mode))
#define PyNIOFile_Open_NUM 2

/* Close a NIO file. Returns -1 if there was an error. */
#define PyNIOFile_Close_RET int
#define PyNIOFile_Close_PROTO Py_PROTO((PyNIOFileObject *file))
#define PyNIOFile_Close_NUM 3

/* Ensure that all data is written to the disk file.
   Returns 0 if there was an error. */
#define PyNIOFile_Sync_RET int
#define PyNIOFile_Sync_PROTO Py_PROTO((PyNIOFileObject *file))
#define PyNIOFile_Sync_NUM 4

/* Create a new dimension. Returns -1 if there was an error. */
#define PyNIOFile_CreateDimension_RET int
#define PyNIOFile_CreateDimension_PROTO \
        Py_PROTO((PyNIOFileObject *file, char *name, long size))
#define PyNIOFile_CreateDimension_NUM 5

/* Create a NIO variable and return the variable object */
#define PyNIOFile_CreateVariable_RET PyNIOVariableObject *
#define PyNIOFile_CreateVariable_PROTO \
      Py_PROTO((PyNIOFileObject *file, char *name, int typecode, \
                char **dimension_names, int ndim))
#define PyNIOFile_CreateVariable_NUM 6

/* Return an object referring to an existing variable */
#define PyNIOFile_GetVariable_RET PyNIOVariableObject *
#define PyNIOFile_GetVariable_PROTO \
	  Py_PROTO((PyNIOFileObject *file, char *name))
#define PyNIOFile_GetVariable_NUM 7

/* Get variable rank */
#define PyNIOVariable_GetRank_RET int
#define PyNIOVariable_GetRank_PROTO Py_PROTO((PyNIOVariableObject *var))
#define PyNIOVariable_GetRank_NUM 8

/* Get variable shape */
#define PyNIOVariable_GetShape_RET size_t *
#define PyNIOVariable_GetShape_PROTO Py_PROTO((PyNIOVariableObject *var))
#define PyNIOVariable_GetShape_NUM 9

/* Allocate and initialize index structures for reading/writing data */
#define PyNIOVariable_Indices_RET PyNIOIndex *
#define PyNIOVariable_Indices_PROTO Py_PROTO((PyNIOVariableObject *var))
#define PyNIOVariable_Indices_NUM 10

/* Read data and return an array object */
#define PyNIOVariable_ReadAsArray_RET PyArrayObject *
#define PyNIOVariable_ReadAsArray_PROTO \
	  Py_PROTO((PyNIOVariableObject *var, PyNIOIndex *indices))
#define PyNIOVariable_ReadAsArray_NUM 11

/* Write array. Returns -1 if there was an error.  */
#define PyNIOVariable_WriteArray_RET int
#define PyNIOVariable_WriteArray_PROTO \
	  Py_PROTO((PyNIOVariableObject *var, PyNIOIndex *indices, \
		    PyObject *array))
#define PyNIOVariable_WriteArray_NUM 12

/* Get file attribute */
#define PyNIOFile_GetAttribute_RET PyObject *
#define PyNIOFile_GetAttribute_PROTO \
	  Py_PROTO((PyNIOFileObject *var, char *name))
#define PyNIOFile_GetAttribute_NUM 13

/* Set file attribute */
#define PyNIOFile_SetAttribute_RET int
#define PyNIOFile_SetAttribute_PROTO \
	  Py_PROTO((PyNIOFileObject *var, char *name, PyObject *value))
#define PyNIOFile_SetAttribute_NUM 14

/* Set file attribute to string value */
#define PyNIOFile_SetAttributeString_RET int
#define PyNIOFile_SetAttributeString_PROTO \
	  Py_PROTO((PyNIOFileObject *var, char *name, char *value))
#define PyNIOFile_SetAttributeString_NUM 15

/* Get variable attribute */
#define PyNIOVariable_GetAttribute_RET PyObject *
#define PyNIOVariable_GetAttribute_PROTO \
	  Py_PROTO((PyNIOVariableObject *var, char *name))
#define PyNIOVariable_GetAttribute_NUM 16

/* Set variable attribute */
#define PyNIOVariable_SetAttribute_RET int
#define PyNIOVariable_SetAttribute_PROTO \
	  Py_PROTO((PyNIOVariableObject *var, char *name, PyObject *value))
#define PyNIOVariable_SetAttribute_NUM 17

/* Set variable attribute to string value */
#define PyNIOVariable_SetAttributeString_RET int
#define PyNIOVariable_SetAttributeString_PROTO \
	  Py_PROTO((PyNIOVariableObject *var, char *name, char *value))
#define PyNIOVariable_SetAttributeString_NUM 18

/* Add entry to the history */
#define PyNIOFile_AddHistoryLine_RET int
#define PyNIOFile_AddHistoryLine_PROTO \
	  Py_PROTO((PyNIOFileObject *self, char *text))
#define PyNIOFile_AddHistoryLine_NUM 19

/* Write string. Returns -1 if there was an error.  */
#define PyNIOVariable_WriteString_RET int
#define PyNIOVariable_WriteString_PROTO \
	  Py_PROTO((PyNIOVariableObject *var, PyStringObject *value))
#define PyNIOVariable_WriteString_NUM 20

/* Read string  */
#define PyNIOVariable_ReadAsString_RET PyStringObject *
#define PyNIOVariable_ReadAsString_PROTO \
	  Py_PROTO((PyNIOVariableObject *var))
#define PyNIOVariable_ReadAsString_NUM 21

/* Total number of C API pointers */
#define PyNIO_API_pointers 22



#ifdef _NIO_MODULE

/* Type object declarations */
staticforward PyTypeObject PyNIOFile_Type;
staticforward PyTypeObject PyNIOVariable_Type;

/* Type check macros */
#define PyNIOFile_Check(op) ((op)->ob_type == &PyNIOFile_Type)
#define PyNIOVariable_Check(op) ((op)->ob_type == &PyNIOVariable_Type)

/* C API function declarations */
static PyNIOFile_Open_RET PyNIOFile_Open PyNIOFile_Open_PROTO;
static PyNIOFile_Close_RET PyNIOFile_Close PyNIOFile_Close_PROTO;
static PyNIOFile_Sync_RET PyNIOFile_Sync PyNIOFile_Sync_PROTO;
static PyNIOFile_CreateDimension_RET PyNIOFile_CreateDimension \
  PyNIOFile_CreateDimension_PROTO;
static PyNIOFile_CreateVariable_RET PyNIOFile_CreateVariable \
  PyNIOFile_CreateVariable_PROTO;
static PyNIOFile_GetVariable_RET PyNIOFile_GetVariable \
  PyNIOFile_GetVariable_PROTO;
static PyNIOVariable_GetRank_RET PyNIOVariable_GetRank \
  PyNIOVariable_GetRank_PROTO;
static PyNIOVariable_GetShape_RET PyNIOVariable_GetShape \
  PyNIOVariable_GetShape_PROTO;
static PyNIOVariable_Indices_RET PyNIOVariable_Indices \
  PyNIOVariable_Indices_PROTO;
static PyNIOVariable_ReadAsArray_RET PyNIOVariable_ReadAsArray \
  PyNIOVariable_ReadAsArray_PROTO;
static PyNIOVariable_ReadAsString_RET PyNIOVariable_ReadAsString \
  PyNIOVariable_ReadAsString_PROTO;
static PyNIOVariable_WriteArray_RET PyNIOVariable_WriteArray \
  PyNIOVariable_WriteArray_PROTO;
static PyNIOVariable_WriteString_RET PyNIOVariable_WriteString \
  PyNIOVariable_WriteString_PROTO;
static PyNIOFile_GetAttribute_RET PyNIOFile_GetAttribute \
  PyNIOFile_GetAttribute_PROTO;
static PyNIOFile_SetAttribute_RET PyNIOFile_SetAttribute \
  PyNIOFile_SetAttribute_PROTO;
static PyNIOFile_SetAttributeString_RET PyNIOFile_SetAttributeString \
  PyNIOFile_SetAttributeString_PROTO;
static PyNIOVariable_GetAttribute_RET PyNIOVariable_GetAttribute \
  PyNIOVariable_GetAttribute_PROTO;
static PyNIOVariable_SetAttribute_RET PyNIOVariable_SetAttribute \
  PyNIOVariable_SetAttribute_PROTO;
static PyNIOVariable_SetAttributeString_RET \
  PyNIOVariable_SetAttributeString \
  PyNIOVariable_SetAttributeString_PROTO;
static PyNIOFile_AddHistoryLine_RET PyNIOFile_AddHistoryLine \
  PyNIOFile_AddHistoryLine_PROTO;

#else

/* C API address pointer */ 
static void **PyNIO_API;

/* Type check macros */
#define PyNIOFile_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyNIO_API[PyNIOFile_Type_NUM])
#define PyNIOVariable_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyNIO_API[PyNIOVariable_Type_NUM])

/* C API function declarations */
#define PyNIOFile_Open \
  (*(PyNIOFile_Open_RET (*)PyNIOFile_Open_PROTO) \
   PyNIO_API[PyNIOFile_Open_NUM])
#define PyNIOFile_Close \
  (*(PyNIOFile_Close_RET (*)PyNIOFile_Close_PROTO) \
   PyNIO_API[PyNIOFile_Close_NUM])
#define PyNIOFile_Sync \
  (*(PyNIOFile_Sync_RET (*)PyNIOFile_Sync_PROTO) \
   PyNIO_API[PyNIOFile_Sync_NUM])
#define PyNIOFile_CreateDimension \
  (*(PyNIOFile_CreateDimension_RET (*)PyNIOFile_CreateDimension_PROTO) \
   PyNIO_API[PyNIOFile_CreateDimension_NUM])
#define PyNIOFile_CreateVariable \
  (*(PyNIOFile_CreateVariable_RET (*)PyNIOFile_CreateVariable_PROTO) \
   PyNIO_API[PyNIOFile_CreateVariable_NUM])
#define PyNIOFile_GetVariable \
  (*(PyNIOFile_GetVariable_RET (*)PyNIOFile_GetVariable_PROTO) \
   PyNIO_API[PyNIOFile_GetVariable_NUM])
#define PyNIOVariable_GetRank \
  (*(PyNIOVariable_GetRank_RET (*)PyNIOVariable_GetRank_PROTO) \
   PyNIO_API[PyNIOVariable_GetRank_NUM])
#define PyNIOVariable_GetShape \
  (*(PyNIOVariable_GetShape_RET (*)PyNIOVariable_GetShape_PROTO) \
   PyNIO_API[PyNIOVariable_GetShape_NUM])
#define PyNIOVariable_Indices \
  (*(PyNIOVariable_Indices_RET (*)PyNIOVariable_Indices_PROTO) \
   PyNIO_API[PyNIOVariable_Indices_NUM])
#define PyNIOVariable_ReadAsArray \
  (*(PyNIOVariable_ReadAsArray_RET (*)PyNIOVariable_ReadAsArray_PROTO) \
   PyNIO_API[PyNIOVariable_ReadAsArray_NUM])
#define PyNIOVariable_ReadAsString \
  (*(PyNIOVariable_ReadAsString_RET (*)PyNIOVariable_ReadAsString_PROTO) \
   PyNIO_API[PyNIOVariable_ReadAsString_NUM])
#define PyNIOVariable_WriteArray \
  (*(PyNIOVariable_WriteArray_RET (*)PyNIOVariable_WriteArray_PROTO) \
   PyNIO_API[PyNIOVariable_WriteArray_NUM])
#define PyNIOVariable_WriteString \
  (*(PyNIOVariable_WriteString_RET (*)PyNIOVariable_WriteString_PROTO) \
   PyNIO_API[PyNIOVariable_WriteString_NUM])
#define PyNIOFile_GetAttribute \
  (*(PyNIOFile_GetAttribute_RET (*)PyNIOFile_GetAttribute_PROTO) \
   PyNIO_API[PyNIOFile_GetAttribute_NUM])
#define PyNIOFile_SetAttribute \
  (*(PyNIOFile_SetAttribute_RET (*)PyNIOFile_SetAttribute_PROTO) \
   PyNIO_API[PyNIOFile_SetAttribute_NUM])
#define PyNIOFile_SetAttributeString \
  (*(PyNIOFile_SetAttributeString_RET \
     (*)PyNIOFile_SetAttributeString_PROTO) \
   PyNIO_API[PyNIOFile_SetAttributeString_NUM])
#define PyNIOVariable_GetAttribute \
  (*(PyNIOVariable_GetAttribute_RET (*)PyNIOVariable_GetAttribute_PROTO) \
   PyNIO_API[PyNIOVariable_GetAttribute_NUM])
#define PyNIOVariable_SetAttribute \
  (*(PyNIOVariable_SetAttribute_RET (*)PyNIOVariable_SetAttribute_PROTO) \
   PyNIO_API[PyNIOVariable_SetAttribute_NUM])
#define PyNIOVariable_SetAttributeString \
  (*(PyNIOVariable_SetAttributeString_RET \
     (*)PyNIOVariable_SetAttributeString_PROTO) \
   PyNIO_API[PyNIOVariable_SetAttributeString_NUM])
#define PyNIOFile_AddHistoryLine \
  (*(PyNIOFile_AddHistoryLine_RET \
     (*)PyNIOFile_AddHistoryLine_PROTO) \
   PyNIO_API[PyNIOFile_AddHistoryLine_NUM])

#define import_NIO() \
{ \
  PyObject *module = PyImport_ImportModule("NIO"); \
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
