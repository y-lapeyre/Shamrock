// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "models/sph/models/basic_sph_gas.hpp"
#include "pylib.hpp"

#include "pymethods.hpp"
#include "pyshamrockcontext.hpp"

/**

typedef struct PyModuleDef{
  PyModuleDef_Base m_base;
  const char* m_name;
  const char* m_doc;
  Py_ssize_t m_size;
  PyMethodDef *m_methods;
  struct PyModuleDef_Slot* m_slots;
  traverseproc m_traverse;
  inquiry m_clear;
  freefunc m_free;
} PyModuleDef;



*/


static PyModuleDef Pyshamrock_Module = {
    PyModuleDef_HEAD_INIT, 
    "shamrock", 
    NULL, 
    -1, 
    Pyshamrock_Methods,
    NULL, 
    NULL, 
    NULL, 
    NULL
};

static PyModuleDef Pyshamrock_Model_Module = {
    PyModuleDef_HEAD_INIT, 
    "models", 
    NULL, 
    -1, 
    NULL,
    NULL, 
    NULL, 
    NULL, 
    NULL
};

static PyObject* PyInit_shamrock(void) {

    //create shamrock module
    PyObject *m;
    m = PyModule_Create(&Pyshamrock_Module);
    if (m == NULL)
        return NULL;


    //PyObject *m_models;
    //m_models = PyModule_Create(&Pyshamrock_Model_Module);
    //if (m_models == NULL)
    //    return NULL;
    //
    //Py_INCREF(m_models);                                                                                                 
    //if (PyModule_AddObject(m, "models", (PyObject *)m_models) < 0) {                                            
    //    Py_DECREF(m_models);                                                                                             
    //    Py_DECREF(m);                                                                                              
    //    return NULL;                                                                             
    //}

    //for(auto fct : init_python_binding_lst){
    //    fct(m_models);
    //}

    for(auto fct : init_python_binding_lst){
        fct(m);
    }

    return m;

}