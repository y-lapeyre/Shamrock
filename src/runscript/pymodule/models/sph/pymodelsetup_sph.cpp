#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/utils/string_utils.hpp"
#include "runscript/pymodule/pylib.hpp"
#include "runscript/shamrockapi.hpp"
#include "runscript/pymodule/pyshamrockcontext.hpp"
#include "models/sph/setup/sph_setup.hpp"
#include "models/sph/base/kernels.hpp"


using namespace models::sph;

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

template <class flt, class u_morton, class Kernel> 
MakeContainer(Container_Model_SetupSPH, SetupSPH<flt, u_morton, Kernel>);

template<class flt, class u_morton, class Kernel>
MakePyContainer(PySHAMROCK_Model_SetupSPH, Container_Model_SetupSPH<flt, u_morton, Kernel>)






template<class flt, class u_morton, class Kernel>
struct PySHAMROCK_Model_SetupSPHIMPL{

    using Type = PySHAMROCK_Model_SetupSPH<flt,u_morton,Kernel>;
    using IntType = SetupSPH<flt,u_morton,Kernel>;
    inline static const std::string descriptor = "SPH setup";
                                                                                           
                                                                                                                            
    AddPyContainer_methods(Type)




    static std::string get_name();


    inline static PyTypeObject* type_ptr = nullptr;



    static PyObject * init(Type * self, PyObject * arg){
        
        if (PyObject_IsInstance(arg, (PyObject *)PyShamCtxType_ptr)){

            PySHAMROCKContext* ctx = (PySHAMROCKContext*) arg;

            std::cout << "ctx ptr : " << ctx <<std::endl;

            self->container.ptr->init(* ctx->ctx->sched);
        }else {
            return NULL;
        }

        return Py_None;
    }

    static PyObject * add_cube_fcc(Type * self, PyObject * args){
        
        PySHAMROCKContext * pyctx;
        f64 dr;
        f64 xmin;
        f64 xmax;
        f64 ymin;
        f64 ymax;
        f64 zmin;
        f64 zmax;

        std::cout << PyShamCtxType_ptr << std::endl;

        if(!PyArg_ParseTuple(args, "O!d((dd)(dd)(dd))",PyShamCtxType_ptr,&pyctx,&dr,&xmin,&xmax,&ymin,&ymax,&zmin,&zmax)) {
            return NULL;
        }

        /*
        if (!PyObject_IsInstance(pyctx, (PyObject *)PyShamCtxType_ptr)){

            PyErr_SetString(PyExc_RuntimeError, "arg 1 is not of type context");  

            return NULL;
        }*/
        

        std::cout << "ctx ptr : " << pyctx <<std::endl;

        if(!self->container.ptr){
            PyErr_SetString(PyExc_RuntimeError, "setup is not initialized \n please run init(ctx)"); 
            return NULL; 
        }

        ShamrockCtx* sctx = pyctx->ctx;

        if (!sctx) {
            PyErr_SetString(PyExc_RuntimeError, "Please provide a valid context"); 
            return NULL; 
        }

        sctx->pdata_layout_print();

        if (!(sctx->sched)) {
            PyErr_SetString(PyExc_RuntimeError, "Please provide a valid context with an initialized scheduler"); 
            return NULL; 
        }

        IntType & set = * self->container.ptr;

        std::cout << "dr : " << dr << std::endl;

        std::cout << format("box = ((%f,%f)(%f,%f)(%f,%f))\n", xmin,xmax,ymin,ymax,zmin,zmax) << std::endl;

        PatchScheduler & sched = *(sctx->sched);

        set.add_particules_fcc(
            sched, 
            dr, 
            {{xmin,ymin,zmin},{xmax,ymax,zmax}});

        return Py_None;
    }




    static void add_object_pybind(PyObject * module){

        static PyMethodDef methods [] = {

            {"reset", (PyCFunction) reset, METH_NOARGS,"reset object"},
            {"clear", (PyCFunction) clear, METH_NOARGS,"clear memory for object"},

            {"init", (PyCFunction) init, METH_O,"init"},
            {"add_cube_fcc", (PyCFunction) add_cube_fcc, METH_VARARGS,"add_cube_fcc"},
            {NULL}  /* Sentinel */
        };

        static std::string name = get_name();
        static std::string tp_name_str = "shamrock." + name;                                                                
                                                                                                                            
        static PyTypeObject pytype = {                                                                                      
            PyVarObject_HEAD_INIT(NULL, 0).tp_name = tp_name_str.c_str(),                                                   
            .tp_doc                                = PyDoc_STR(descriptor.c_str()),                                         
            .tp_basicsize                          = sizeof(Type),                                                          
            .tp_itemsize                           = 0,                                                                     
            .tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                              
            .tp_new                                = objnew,                                                                
            .tp_dealloc                            = (destructor)dealloc,                                                   
            .tp_methods                            = methods,                                                               
        };  

        type_ptr = &pytype;

                                                                      
        static std::string notready_str = "[pybind] " + name + " not ready";                                                
        static std::string failedtype   = "[pybind] " + name + " failed to be added";                                       
        static std::string typeready    = "[pybind] " + name + " type ready";                                               
                                                                                                                            
        if (PyType_Ready(type_ptr) < 0)                                                                                      
            throw ShamAPIException(notready_str);                                                                           
                                                                                                                            
        Py_INCREF(type_ptr);                                                                                                 
        if (PyModule_AddObject(module, name.c_str(), (PyObject *)type_ptr) < 0) {                                            
            Py_DECREF(type_ptr);                                                                                             
            Py_DECREF(module);                                                                                              
            throw ShamAPIException(failedtype);                                                                             
        }                                                                                                                   
                                                                                                                            
        std::cout << typeready << std::endl;    
    }




};

template<> std::string PySHAMROCK_Model_SetupSPHIMPL<f32, u32, kernels::M4<f32>>::get_name(){
    return "SetupSPH_single_morton32_M4";
}





addpybinding(setupsph){
    PySHAMROCK_Model_SetupSPHIMPL<f32, u32, kernels::M4<f32>>::add_object_pybind(module);
    //PySHAMROCK_Model_SetupSPHIMPL<f64, u32, kernels::M4<f64>>::add_object_pybind(module);
    //PySHAMROCK_Model_SetupSPHIMPL<f32, u64, kernels::M4<f32>>::add_object_pybind(module);
    //PySHAMROCK_Model_SetupSPHIMPL<f64, u64, kernels::M4<f64>>::add_object_pybind(module);
}
