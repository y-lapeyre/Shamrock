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
    self = (Type *) type->tp_alloc(type, 0);                                         \
    return (PyObject *) self;                            \
}



#define __ADD_PYBIND__     {                       \
                            \
static std::string name = get_name();                            \
                            \
static PyTypeObject pytype = get_pytype();                            \
                            \
static std::string notready_str = "[pybind] " + name + " not ready";                            \
static std::string failedtype   = "[pybind] " + name + " failed to be added";                            \
static std::string typeready    = "[pybind] " + name + " type ready";                            \
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