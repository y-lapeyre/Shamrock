#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

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


