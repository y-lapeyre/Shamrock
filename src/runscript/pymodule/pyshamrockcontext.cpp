// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "pyshamrockcontext.hpp"
#include "shamrock/patch/base/patchdata_field.hpp"
#include "shamsys/legacy/mpi_handler.hpp"
#include <floatobject.h>
#include <longobject.h>
#include <map>
#include <vector>

#define PySHamExcHandle(a)                                                                                                  \
    try {                                                                                                                   \
        a;                                                                                                                  \
    } catch (ShamAPIException const &e) {                                                                                   \
        PyErr_SetString(PyExc_RuntimeError, e.what());                                                                      \
        return NULL;                                                                                                        \
    }


template<class T> PyObject* convert(T val);

template<> PyObject* convert(f32 val){
    return PyFloat_FromDouble(f64(val));
}

template<> PyObject* convert(f64 val){
    return PyFloat_FromDouble(f64(val));
}

template<> PyObject* convert(u64 val){
    return PyLong_FromLong(long(val));
}

template<> PyObject* convert(u32 val){
    return PyLong_FromLong(long(val));
}


template<> PyObject* convert(f32_2 val){
    return Py_BuildValue("[f,f]",val.x(),val.y());
}

template<> PyObject* convert(f64_2 val){
    return Py_BuildValue("[d,d]",val.x(),val.y());
}

template<> PyObject* convert(f32_3 val){
    return Py_BuildValue("[f,f,f]",val.x(),val.y(),val.z());
}

template<> PyObject* convert(f64_3 val){
    return Py_BuildValue("[d,d,d]",val.x(),val.y(),val.z());
}

template<> PyObject* convert(f32_4 val){
    return Py_BuildValue("[f,f,f,f]",val.x(),val.y(),val.z(),val.w());
}

template<> PyObject* convert(f64_4 val){
    return Py_BuildValue("[d,d,d,d]",val.x(),val.y(),val.z(),val.w());
}

template<> PyObject* convert(f32_8 val){
    return Py_BuildValue("[f,f,f,f,f,f,f,f]",
    val.s0() ,val.s1(),val.s2(),val.s3(),
    val.s4() ,val.s5(),val.s6(),val.s7()
    );
}

template<> PyObject* convert(f64_8 val){
    return Py_BuildValue("[d,d,d,d,d,d,d,d]",
    val.s0() ,val.s1(),val.s2(),val.s3(),
    val.s4() ,val.s5(),val.s6(),val.s7()
    );
}

template<> PyObject* convert(f32_16 val){
    return Py_BuildValue("[f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f]",
    val.s0() ,val.s1(),val.s2(),val.s3(),
    val.s4() ,val.s5(),val.s6(),val.s7(),
    val.s8() ,val.s9(),val.sA(),val.sB(),
    val.sC() ,val.sD(),val.sE(),val.sF()
    );
}

template<> PyObject* convert(f64_16 val){
    return Py_BuildValue("[d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d]",
    val.s0() ,val.s1(),val.s2(),val.s3(),
    val.s4() ,val.s5(),val.s6(),val.s7(),
    val.s8() ,val.s9(),val.sA(),val.sB(),
    val.sC() ,val.sD(),val.sE(),val.sF()
    );
}



template<> bool test_cast(PyObject* o, f32 & val){

    if(!PyFloat_Check(o)){
        return false;
    }

    f64 ret = PyFloat_AS_DOUBLE(o);
    val = ret;
    return true;
}

template<> bool test_cast(PyObject* o, f64 & val){

    if(!PyFloat_Check(o)){
        return false;
    }
    
    f64 ret = PyFloat_AS_DOUBLE(o);
    val = ret;
    return true;
}


template<class T> void append_to_map(std::vector<PatchDataField<T>> & pfields, std::map<std::string, PyObject*> map_app){
    for(PatchDataField<T> & field : pfields){

        auto & refobj = map_app[field.get_name()];

        std::cout << "appending " << field.get_name() << " (" << field.size() << " elements)" << std::endl;

        {

            auto & buf = field.get_buf();

            sycl::host_accessor acc{ *buf, sycl::read_only};

            for (u32 i = 0 ; i < field.size(); i++) {
                PyList_Append(refobj,convert(acc[i]));
            }

        }

        
    }
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

    

    static PyObject* collect_data(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)){

        auto data = self->ctx->allgather_data();

        std::cout << "collected : " << data.size() << " patches" << std::endl;

        auto dic = PyDict_New();

        std::map<std::string, PyObject*> fields;

        for (auto fname : self->ctx->pdl->get_field_names()) {
            fields.insert({fname,PyList_New(0)});
        }

        

        for (auto & pdat : data) {

            append_to_map(pdat->fields_f32, fields);
            append_to_map(pdat->fields_f32_2, fields);
            append_to_map(pdat->fields_f32_3, fields);
            append_to_map(pdat->fields_f32_4, fields);
            append_to_map(pdat->fields_f32_8, fields);
            append_to_map(pdat->fields_f32_16, fields);
            append_to_map(pdat->fields_f64, fields);
            append_to_map(pdat->fields_f64_2, fields);
            append_to_map(pdat->fields_f64_3, fields);
            append_to_map(pdat->fields_f64_4, fields);
            append_to_map(pdat->fields_f64_8, fields);
            append_to_map(pdat->fields_f64_16, fields);
            append_to_map(pdat->fields_u32, fields);
            append_to_map(pdat->fields_u64, fields);
        }



        for (auto & [k,v] : fields) {
            PyDict_SetItemString(dic, k.c_str(), v);
        }
        

        return dic;
    }

    static PyObject* set_box_size(PySHAMROCKContext *self, PyObject * args){

        f64 xm,xM,ym,yM,zm,zM;

        if(!PyArg_ParseTuple(args, "((ddd)(ddd))",&xm,&ym,&zm,&xM,&yM,&zM)) {
            return NULL;
        }

        self->ctx->set_box_size({{xm,ym,zm},{xM,yM,zM}});

        return Py_None;

    }

    static PyObject* get_world_size(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)){
        return PyLong_FromLong(mpi_handler::world_size);
    }

    static PyObject* get_world_rank(PySHAMROCKContext *self, PyObject *Py_UNUSED(ignored)){
        return PyLong_FromLong(mpi_handler::world_rank);
    }

};

static PyMethodDef methods[] = {
    {"reset", (PyCFunction)PySHAMROCKContextImpl::reset, METH_NOARGS, "doc str"},
    {"pdata_layout_new", (PyCFunction)PySHAMROCKContextImpl::pdata_layout_new, METH_NOARGS, "doc str"},
    {"pdata_layout_add_field", (PyCFunction)PySHAMROCKContextImpl::pdata_layout_add_field, METH_VARARGS, "doc str"},
    {"pdata_layout_get_str", (PyCFunction)PySHAMROCKContextImpl::pdata_layout_get_str, METH_NOARGS, "doc str"},
    {"init_sched", (PyCFunction)PySHAMROCKContextImpl::init_sched, METH_VARARGS, "doc str"},
    {"close_sched", (PyCFunction)PySHAMROCKContextImpl::close_sched, METH_NOARGS, "doc str"},
    {"collect_data", (PyCFunction)PySHAMROCKContextImpl::collect_data, METH_NOARGS, "doc str"},
    {"set_box_size", (PyCFunction)PySHAMROCKContextImpl::set_box_size, METH_VARARGS, "doc str"},
    {"get_world_size", (PyCFunction)PySHAMROCKContextImpl::get_world_size, METH_NOARGS, "doc str"},
    {"get_world_rank", (PyCFunction)PySHAMROCKContextImpl::get_world_rank, METH_NOARGS, "doc str"},
    {NULL} /* Sentinel */
};



static auto getctxtype_obj = []() -> PyTypeObject {
    PyTypeObject ptype = {
        PyVarObject_HEAD_INIT(NULL, 0)
    };

    ptype.tp_name = "shamrock.Context";
    ptype.tp_doc                                = PyDoc_STR("Shamrock context");
    ptype.tp_basicsize                          = sizeof(PySHAMROCKContext);
    ptype.tp_itemsize                           = 0;
    ptype.tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    ptype.tp_new                                = PySHAMROCKContextImpl::objnew;
    //ptype.tp_init = (initproc) PySHAMROCKContext_init;
    ptype.tp_dealloc = (destructor)PySHAMROCKContextImpl::dealloc;
    ptype.tp_methods = methods;

    return ptype;

};

static PyTypeObject PyShamCtxType = getctxtype_obj();

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
