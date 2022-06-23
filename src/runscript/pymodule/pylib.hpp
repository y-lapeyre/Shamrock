#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <vector>

class CPyObject {
  private:
    PyObject *p;

  public:
    CPyObject() : p(nullptr) {}

    explicit CPyObject(PyObject *_p) : p(_p) {}

    ~CPyObject() { Release(); }

    PyObject *getObject() { return p; }

    PyObject *setObject(PyObject *_p) { return (p = _p); }

    PyObject *AddRef() {
        if (p) {
            Py_INCREF(p);
        }
        return p;
    }

    void Release() {
        if (p) {
            Py_DECREF(p);
        }

        p = nullptr;
    }

    PyObject *operator->() { return p; }

    bool is() { return p ? true : false; }

    explicit operator PyObject *() { return p; }

    PyObject *operator=(PyObject *pp) {
        p = pp;
        return p;
    }

    explicit operator bool() { return p ? true : false; }
};



#define __ADD_METHODS__                            \
                            \
static void dealloc(Type *self){                            \
    self->model.reset();                            \
    Py_TYPE(self)->tp_free((PyObject *) self);                            \
}                            \
                            \
static PyObject * objnew(PyTypeObject *type, PyObject *args, PyObject *kwds){                            \
    Type *self;                            \
    self = (Type *) type->tp_alloc(type, 0);                            \
    if (self != NULL) {                            \
        self->model = std::make_unique<IntType>();                            \
    }                            \
    return (PyObject *) self;                            \
}



#define __ADD_PYBIND__     {                       \
                            \
std::string tp_name_str = "shamrock."+name;                            \
                            \
static PyTypeObject pytype = {                            \
    PyVarObject_HEAD_INIT(NULL, 0)                            \
    .tp_name = tp_name_str.c_str(),                            \
    .tp_doc = PyDoc_STR(descriptor.c_str()),                            \
    .tp_basicsize = sizeof(Type),                            \
    .tp_itemsize = 0,                            \
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                            \
    .tp_new = objnew,                            \
    .tp_dealloc = (destructor) dealloc,                            \
    .tp_methods = methods,                            \
};                            \
                            \
std::string notready_str = "[pybind] " + name + " not ready";                            \
std::string failedtype   = "[pybind] " + name + " failed to be added";                            \
std::string typeready    = "[pybind] " + name + " type ready";                            \
                            \
if (PyType_Ready(&pytype) < 0)                            \
    throw ShamAPIException(notready_str);                            \
                                    \
Py_INCREF(&pytype);                            \
if (PyModule_AddObject(module, name.c_str(), (PyObject *) &pytype) < 0) {                            \
    Py_DECREF(&pytype);                            \
    Py_DECREF(module);                            \
    throw ShamAPIException(failedtype);                            \
}                            \
                            \
std::cout << typeready << std::endl;   \        
}






inline std::vector<void (*)(PyObject *)> init_python_binding_lst;

class PyBindAdder {public:
    PyBindAdder(void (*tmp)(PyObject *)){
        init_python_binding_lst.push_back(tmp);
    }
};

#define addpybinding(name)\
void test_func_##name (PyObject * module);\
void (*test_func_ptr_##name)(PyObject *) = test_func_##name;\
PyBindAdder class_pybind_obj_##name (test_func_ptr_##name);\
void test_func_##name (PyObject * module)