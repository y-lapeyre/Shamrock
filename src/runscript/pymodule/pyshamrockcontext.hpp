#pragma once

#include "pylib.hpp"
#include "runscript/shamrockapi.hpp"
#include <array>
#include <exception>
#include <memory>

struct PySHAMROCKContext{
    PyObject_HEAD
    /* Type-specific fields go here. */
    std::unique_ptr<ShamrockCtx> ctx;
};

class PySHAMROCKContextImpl {public:

    static void dealloc(PySHAMROCKContext *self){

        std::cout << "free Shamrock Context : " << self << std::endl;
        

        self->ctx.reset();

        Py_TYPE(self)->tp_free((PyObject *) self);
    }

    static PyObject * objnew(PyTypeObject *type, PyObject *args, PyObject *kwds){
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
            PyErr_SetString(PyExc_RuntimeError, e.what());          \
            return NULL;          \
        }

    static PyObject * reset(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)){
        std::cout << "resetting context" << std::endl;

        self->ctx.reset();
        self->ctx = std::make_unique<ShamrockCtx>();

        return Py_None;
    }

    static PyObject * pdata_layout_new(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)){
        PySHamExcHandle(self->ctx->pdata_layout_new());
        return Py_None;
    }

    static PyObject * init_sched(PySHAMROCKContext *self, PyObject *args){

        i32 split_crit , merge_crit;

        if(!PyArg_ParseTuple(args, "ii",&split_crit,&merge_crit)) {
            return NULL;
        }

        PySHamExcHandle(self->ctx->init_sched(split_crit,merge_crit));

        return Py_None;
    }

    static PyObject * close_sched(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)){
        PySHamExcHandle(self->ctx->close_sched());
        return Py_None;
    }





};

static PyMethodDef methods [] = {
    {"reset", (PyCFunction) PySHAMROCKContextImpl::reset, METH_NOARGS,"doc str"},
    {"pdata_layout_new", (PyCFunction) PySHAMROCKContextImpl::pdata_layout_new, METH_NOARGS,"doc str"},
    {"init_sched", (PyCFunction) PySHAMROCKContextImpl::init_sched, METH_VARARGS,"doc str"},
    {"close_sched", (PyCFunction) PySHAMROCKContextImpl::close_sched, METH_NOARGS,"doc str"},
    {NULL}  /* Sentinel */
};


static PyTypeObject PyShamCtxType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "shamrock.Context",
    .tp_doc = PyDoc_STR("Shamrock context"),
    .tp_basicsize = sizeof(PySHAMROCKContext),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PySHAMROCKContextImpl::objnew,
    //.tp_init = (initproc) PySHAMROCKContext_init,
    .tp_dealloc = (destructor) PySHAMROCKContextImpl::dealloc,
    .tp_methods = methods,
};