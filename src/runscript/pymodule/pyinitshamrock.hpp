#pragma once

#include "pylib.hpp"

#include "pymethods.hpp"
#include "pyshamrockcontext.hpp"

static PyModuleDef Pyshamrock_Module = {
    PyModuleDef_HEAD_INIT, "shamrock", NULL, -1, Pyshamrock_Methods,
    NULL, NULL, NULL, NULL
};

static PyObject* PyInit_shamrock(void) {


    PyObject *m;
    if (PyType_Ready(&PyShamCtxType) < 0)
        return NULL;

    m = PyModule_Create(&Pyshamrock_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyShamCtxType);
    if (PyModule_AddObject(m, "Context", (PyObject *) &PyShamCtxType) < 0) {
        Py_DECREF(&PyShamCtxType);
        Py_DECREF(m);
        return NULL;
    }


    //return PyModule_Create(&Pyshamrock_Module);
    return m;


    //return PyModule_Create(&Pyshamrock_Module);
}