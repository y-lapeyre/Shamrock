#pragma once

#include "models/sph/models/basic_sph_gas.hpp"
#include "pylib.hpp"

#include "pymethods.hpp"
#include "pyshamrockcontext.hpp"

static PyModuleDef Pyshamrock_Module = {
    PyModuleDef_HEAD_INIT, "shamrock", NULL, -1, Pyshamrock_Methods,
    NULL, NULL, NULL, NULL
};

static PyObject* PyInit_shamrock(void) {


    PyObject *m;
    

    m = PyModule_Create(&Pyshamrock_Module);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&PyShamCtxType) < 0)
            return NULL;
            
    Py_INCREF(&PyShamCtxType);
    if (PyModule_AddObject(m, "Context", (PyObject *) &PyShamCtxType) < 0) {
        Py_DECREF(&PyShamCtxType);
        Py_DECREF(m);
        return NULL;
    }
    

    for(auto fct : init_python_binding_lst){
        fct(m);
    }


    //return PyModule_Create(&Pyshamrock_Module);
    return m;


    //return PyModule_Create(&Pyshamrock_Module);
}