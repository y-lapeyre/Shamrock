#include "pyshamrockcontext.hpp"

#define PySHamExcHandle(a)                                                                                                  \
    try {                                                                                                                   \
        a;                                                                                                                  \
    } catch (ShamAPIException const &e) {                                                                                   \
        PyErr_SetString(PyExc_RuntimeError, e.what());                                                                      \
        return NULL;                                                                                                        \
    }


class PySHAMROCKContextImpl {
  public:
    static void dealloc(PySHAMROCKContext *self) {

        std::cout << "free Shamrock Context : " << self << std::endl;

        if(self->ctx != nullptr){
            delete self->ctx;
        }

        Py_TYPE(self)->tp_free((PyObject *)self);
    }

    static PyObject *objnew(PyTypeObject *type, PyObject *args, PyObject *kwds) {
        PySHAMROCKContext *self;
        self = (PySHAMROCKContext *)type->tp_alloc(type, 0);

        self->ctx = new ShamrockCtx();

        std::cout << "new Shamrock Context : " << self << std::endl;

        return (PyObject *)self;
    }

    static PyObject *reset(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)) {
        std::cout << "resetting context" << std::endl;


        if(self->ctx != nullptr){
            delete self->ctx;
        }
        self->ctx = new ShamrockCtx();

        return Py_None;
    }

    static PyObject *pdata_layout_new(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)) {
        PySHamExcHandle(self->ctx->pdata_layout_new());
        return Py_None;
    }


    static PyObject *pdata_layout_add_field(PySHAMROCKContext *self, PyObject* args){

        if (!self->ctx->pdl) {
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
            self->ctx->pdl->add_field<f32>(sname, nvar);
        }else if (stype == "f32_2"){
            self->ctx->pdl->add_field<f32_2>(sname, nvar);
        }else if (stype == "f32_3"){
            self->ctx->pdl->add_field<f32_3>(sname, nvar);
        }else if (stype == "f32_4"){
            self->ctx->pdl->add_field<f32_4>(sname, nvar);
        }else if (stype == "f32_8"){
            self->ctx->pdl->add_field<f32_8>(sname, nvar);
        }else if (stype == "f32_16"){
            self->ctx->pdl->add_field<f32_16>(sname, nvar);
        }else if (stype == "f64"){
            self->ctx->pdl->add_field<f64>(sname, nvar);
        }else if (stype == "f64_2"){
            self->ctx->pdl->add_field<f64_2>(sname, nvar);
        }else if (stype == "f64_3"){
            self->ctx->pdl->add_field<f64_3>(sname, nvar);
        }else if (stype == "f64_4"){
            self->ctx->pdl->add_field<f64_4>(sname, nvar);
        }else if (stype == "f64_8"){
            self->ctx->pdl->add_field<f64_8>(sname, nvar);
        }else if (stype == "f64_16"){
            self->ctx->pdl->add_field<f64_16>(sname, nvar);
        }else if (stype == "u32"){
            self->ctx->pdl->add_field<u32>(sname, nvar);
        }else if (stype == "u64"){
            self->ctx->pdl->add_field<u64>(sname, nvar);
        }else{
            return NULL;
        }

        return Py_None;
    }


    static PyObject* pdata_layout_get_str(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)) {

        if (!self->ctx->pdl) {
            std::cout << "patchdata layout uninitialized" << std::endl;
            return NULL;
        }

        return PyUnicode_FromString(self->ctx->pdl->get_description_str().c_str());
    }


    static PyObject *init_sched(PySHAMROCKContext *self, PyObject *args) {

        i32 split_crit, merge_crit;

        if (!PyArg_ParseTuple(args, "ii", &split_crit, &merge_crit)) {
            return NULL;
        }

        PySHamExcHandle(self->ctx->init_sched(split_crit, merge_crit));

        return Py_None;
    }

    static PyObject *close_sched(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)) {
        PySHamExcHandle(self->ctx->close_sched());
        return Py_None;
    }
};

static PyMethodDef methods[] = {
    {"reset", (PyCFunction)PySHAMROCKContextImpl::reset, METH_NOARGS, "doc str"},
    {"pdata_layout_new", (PyCFunction)PySHAMROCKContextImpl::pdata_layout_new, METH_NOARGS, "doc str"},
    {"pdata_layout_add_field", (PyCFunction)PySHAMROCKContextImpl::pdata_layout_add_field, METH_VARARGS, "doc str"},
    {"pdata_layout_get_str", (PyCFunction)PySHAMROCKContextImpl::pdata_layout_get_str, METH_NOARGS, "doc str"},
    {"init_sched", (PyCFunction)PySHAMROCKContextImpl::init_sched, METH_VARARGS, "doc str"},
    {"close_sched", (PyCFunction)PySHAMROCKContextImpl::close_sched, METH_NOARGS, "doc str"},
    {NULL} /* Sentinel */
};

static PyTypeObject PyShamCtxType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "shamrock.Context",
    .tp_doc                                = PyDoc_STR("Shamrock context"),
    .tp_basicsize                          = sizeof(PySHAMROCKContext),
    .tp_itemsize                           = 0,
    .tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new                                = PySHAMROCKContextImpl::objnew,
    //.tp_init = (initproc) PySHAMROCKContext_init,
    .tp_dealloc = (destructor)PySHAMROCKContextImpl::dealloc,
    .tp_methods = methods,
};

addpybinding(shamctx) {

    static std::string name         = "Context";
    static std::string notready_str = "[pybind] " + name + " not ready";
    static std::string failedtype   = "[pybind] " + name + " failed to be added";
    static std::string typeready    = "[pybind] " + name + " type ready";

    PyShamCtxType_ptr = &PyShamCtxType;
    if (PyType_Ready(PyShamCtxType_ptr) < 0)
        throw ShamAPIException(notready_str);

    Py_INCREF(PyShamCtxType_ptr);
    if (PyModule_AddObject(module, name.c_str(), (PyObject *)PyShamCtxType_ptr) < 0) {
        Py_DECREF(PyShamCtxType_ptr);
        Py_DECREF(module);
        throw ShamAPIException(failedtype);
    }

    std::cout << typeready << std::endl;
}