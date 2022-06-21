#pragma once

#include "pylib.hpp"
#include "runscript/shamrockapi.hpp"
#include <exception>
#include <memory>

struct PySHAMROCKContext{
    PyObject_HEAD
    /* Type-specific fields go here. */
    std::unique_ptr<ShamrockCtx> ctx;
};



static void
PySHAMROCKContext_dealloc(PySHAMROCKContext *self)
{

    std::cout << "free Shamrock Context : " << self << std::endl;
    

    self->ctx.reset();

    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
PySHAMROCKContext_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySHAMROCKContext *self;
    self = (PySHAMROCKContext *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ctx = std::make_unique<ShamrockCtx>();
    }

    std::cout << "new Shamrock Context : " << self << std::endl;

    return (PyObject *) self;
}


#define PySHamExcHandle(a)           \
    try {           \
        a;          \
    } catch (ShamAPIException const & e) {          \
        PyErr_SetString(PyExc_TypeError, e.what());          \
        return NULL;          \
    }



static PyObject *
PySHAMROCKContext_pdata_layout_new(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored))
{
    std::cout << "resetting the layout" << std::endl;

    PySHamExcHandle(self->ctx->pdata_layout_new());

    return Py_None;
}

static PyObject *
PySHAMROCKContext_init_sched(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored))
{
    std::cout << "init the scheduler" << std::endl;

    PySHamExcHandle(self->ctx->init_sched(100000,1));

    return Py_None;
}

static PyObject *
PySHAMROCKContext_close_sched(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored))
{
    std::cout << "closing the scheduler" << std::endl;

    PySHamExcHandle(self->ctx->close_sched());

    return Py_None;
}



static PyMethodDef PySHAMROCKContext_methods[] = {
    {"pdata_layout_new", (PyCFunction) PySHAMROCKContext_pdata_layout_new, METH_NOARGS,
     "Initialise a new patchdata layout"
    },
    {"init_sched", (PyCFunction) PySHAMROCKContext_init_sched, METH_NOARGS,
     "Initialise a new patchdata layout"
    },{"close_sched", (PyCFunction) PySHAMROCKContext_close_sched, METH_NOARGS,
     "Initialise a new patchdata layout"
    },
    {NULL}  /* Sentinel */
};


static PyTypeObject PyShamCtxType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "shamrock.Context",
    .tp_doc = PyDoc_STR("Shamrock context"),
    .tp_basicsize = sizeof(PySHAMROCKContext),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PySHAMROCKContext_new,
    //.tp_init = (initproc) PySHAMROCKContext_init,
    .tp_dealloc = (destructor) PySHAMROCKContext_dealloc,
    .tp_methods = PySHAMROCKContext_methods,
};