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
