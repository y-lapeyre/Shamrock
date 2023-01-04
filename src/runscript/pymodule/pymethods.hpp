// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamrock/patch/base/patchdata_layout.hpp"
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











static PyMethodDef Pyshamrock_Methods[] = {
    {"get_version", pyshamrock_get_version, METH_VARARGS, "get git commit vers"},
    {NULL, NULL, 0, NULL}
};




