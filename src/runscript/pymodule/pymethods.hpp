#pragma once

#include "aliases.hpp"
#include "core/patch/patchdata_layout.hpp"
#include "pylib.hpp"
#include <memory>
#include <object.h>
#include <unicodeobject.h>



static PyObject* pyshamrock_get_version(PyObject *self, PyObject *args) {
    if(!PyArg_ParseTuple(args, ":numargs")) {
        return NULL;
    }

    return PyUnicode_FromString(git_info_str.c_str());
}



inline std::unique_ptr<PatchDataLayout> current_pdl;

static PyObject* pyshamrock_pdata_layout_reset(PyObject *self, PyObject *args) {
    if(!PyArg_ParseTuple(args, ":numargs")) {
        return NULL;
    }

    current_pdl = std::make_unique<PatchDataLayout>();

    return Py_None;
}

static PyObject* pyshamrock_pdata_layout_add_field(PyObject *self, PyObject *args) {

    if (!current_pdl) {
        std::cout << "patchdata layout uninitialized" << std::endl;
        return NULL;
    }

    char * fname;
    int fnvar;
    char *ftype;

    if(!PyArg_ParseTuple(args, "sis",&fname,&fnvar,&ftype)) {
        return NULL;
    }


    std::string stype = std::string(ftype);
    std::string sname = std::string(fname);

    if(fnvar < 1){
        std::cout << "field must have at least nvar = 1" << std::endl;
        return NULL;
    }

    u32 nvar = fnvar;

    if (stype == "f32"){
        current_pdl->add_field<f32>(sname, nvar);
    }else if (stype == "f32_2"){
        current_pdl->add_field<f32_2>(sname, nvar);
    }else if (stype == "f32_3"){
        current_pdl->add_field<f32_3>(sname, nvar);
    }else if (stype == "f32_4"){
        current_pdl->add_field<f32_4>(sname, nvar);
    }else if (stype == "f32_8"){
        current_pdl->add_field<f32_8>(sname, nvar);
    }else if (stype == "f32_16"){
        current_pdl->add_field<f32_16>(sname, nvar);
    }else if (stype == "f64"){
        current_pdl->add_field<f64>(sname, nvar);
    }else if (stype == "f64_2"){
        current_pdl->add_field<f64_2>(sname, nvar);
    }else if (stype == "f64_3"){
        current_pdl->add_field<f64_3>(sname, nvar);
    }else if (stype == "f64_4"){
        current_pdl->add_field<f64_4>(sname, nvar);
    }else if (stype == "f64_8"){
        current_pdl->add_field<f64_8>(sname, nvar);
    }else if (stype == "f64_16"){
        current_pdl->add_field<f64_16>(sname, nvar);
    }else if (stype == "u32"){
        current_pdl->add_field<u32>(sname, nvar);
    }else if (stype == "u64"){
        current_pdl->add_field<u64>(sname, nvar);
    }else{
        return NULL;
    }



    return Py_None;
}

static PyObject* pyshamrock_pdata_layout_get_str(PyObject *self, PyObject *args) {

    if (!current_pdl) {
        std::cout << "patchdata layout uninitialized" << std::endl;
        return NULL;
    }

    if(!PyArg_ParseTuple(args, ":numargs")) {
        return NULL;
    }

    return PyUnicode_FromString(current_pdl->get_description_str().c_str());
}







static PyMethodDef Pyshamrock_Methods[] = {
    {"get_version", pyshamrock_get_version, METH_VARARGS, "get git commit vers"},
    {"pdata_layout_reset", pyshamrock_pdata_layout_reset, METH_VARARGS, "get git commit vers"},
    {"pdata_layout_add_field", pyshamrock_pdata_layout_add_field, METH_VARARGS, "get git commit vers"},
    {"pdata_layout_get_str", pyshamrock_pdata_layout_get_str, METH_VARARGS, "get git commit vers"},
    {NULL, NULL, 0, NULL}
};


static PyModuleDef Pyshamrock_Module = {
    PyModuleDef_HEAD_INIT, "shamrock", NULL, -1, Pyshamrock_Methods,
    NULL, NULL, NULL, NULL
};
static PyObject* PyInit_shamrock(void) {
    return PyModule_Create(&Pyshamrock_Module);
}
