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









#define MakeContainer(Container_name, ...)                                                                                  \
    struct Container_name {                                                                                                 \
        using InternalType = __VA_ARGS__;                                                                                   \
        InternalType *ptr  = nullptr;                                                                                       \
                                                                                                                            \
        bool is_allocated() { return ptr != nullptr; }                                                                      \
                                                                                                                            \
        void alloc() {                                                                                                      \
            if (is_allocated()) {                                                                                           \
                throw ShamAPIException("container already allocated");                                                      \
            }                                                                                                               \
                                                                                                                            \
            ptr = new InternalType();                                                                                       \
        }                                                                                                                   \
                                                                                                                            \
        void dealloc() {                                                                                                    \
                                                                                                                            \
            if (!is_allocated()) {                                                                                          \
                throw ShamAPIException("container already deallocated");                                                    \
            }                                                                                                               \
                                                                                                                            \
            delete ptr;                                                                                                     \
            ptr = nullptr;                                                                                                  \
        }                                                                                                                   \
                                                                                                                            \
        void reset() {                                                                                                      \
            if (is_allocated()) {                                                                                           \
                dealloc();                                                                                                  \
                alloc();                                                                                                    \
            } else {                                                                                                        \
                alloc();                                                                                                    \
            }                                                                                                               \
        }                                                                                                                   \
    }

#define MakePyContainer(PyContainer_name, ...)                                                                              \
    struct PyContainer_name {                                                                                               \
        using ContainerType = __VA_ARGS__;                                                                                  \
        PyObject_HEAD ContainerType container;                                                                              \
    };

#define AddPyContainer_methods(PyContainer_name)                                                                            \
    static void dealloc(PyContainer_name *self) {                                                                           \
        if (self->container.is_allocated()) {                                                                               \
            self->container.dealloc();                                                                                      \
        }                                                                                                                   \
        Py_TYPE(self)->tp_free((PyObject *)self);                                                                           \
    }                                                                                                                       \
    static PyObject *objnew(PyTypeObject *type, PyObject *args, PyObject *kwds) {                                           \
        PyContainer_name *self;                                                                                             \
        self = (PyContainer_name *)type->tp_alloc(type, 0);                                                                 \
                                                                                                                            \
        if (self != NULL) {                                                                                                 \
            self->container.alloc();                                                                                        \
        }                                                                                                                   \
        return (PyObject *)self;                                                                                            \
    }                                                                                                                       \
                                                                                                                            \
    static PyObject *reset(PyContainer_name *self, PyObject *Py_UNUSED(ignored)) {                                          \
        self->container.reset();                                                                                            \
        return Py_None;                                                                                                     \
    }                                                                                                                       \
                                                                                                                            \
    static PyObject *clear(PyContainer_name *self, PyObject *Py_UNUSED(ignored)) {                                          \
        if (self->container.is_allocated()) {                                                                               \
            self->container.dealloc();                                                                                      \
        }                                                                                                                   \
        return Py_None;                                                                                                     \
    }



    



#define __ADD_METHODS__(field)                            \
                            \
static void dealloc(Type *self){                            \
    if(self->field != nullptr){\
        delete self->field;\
        self->field = nullptr;\
    }                            \
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