// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "pylib.hpp"
#include "runscript/shamrockapi.hpp"
#include <array>
#include <exception>
#include <memory>

struct PySHAMROCKContext{
    PyObject_HEAD
    /* Type-specific fields go here. */
    ShamrockCtx* ctx;
};


inline PyTypeObject * PyShamCtxType_ptr = nullptr;



template<class T> bool test_cast(PyObject* o, T & val);