#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/utils/string_utils.hpp"
#include "runscript/pymodule/pylib.hpp"
#include "runscript/shamrockapi.hpp"
#include "runscript/pymodule/pyshamrockcontext.hpp"
#include "models/sph/setup/sph_setup.hpp"
#include "models/sph/base/kernels.hpp"


using namespace models::sph;


template <class flt, class Kernel> 
MakeContainer(Container_Model_SetupSPH, SetupSPH<flt, Kernel>);

template<class flt, class Kernel>
MakePyContainer(PySHAMROCK_Model_SetupSPH, Container_Model_SetupSPH<flt, Kernel>)


template<class flt, class Kernel>
struct PySHAMROCK_Model_SetupSPHIMPL{

    using Type = PySHAMROCK_Model_SetupSPH<flt,Kernel>;
    using IntType = SetupSPH<flt,Kernel>;
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
            PyVarObject_HEAD_INIT(NULL, 0)
            };
        pytype.tp_name = tp_name_str.c_str();
        pytype.tp_basicsize                          = sizeof(Type);                                              
        pytype.tp_itemsize                           = 0;                                                                                                         
        pytype.tp_dealloc                            = (destructor)dealloc;                           
        pytype.tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;         
        pytype.tp_doc                                = PyDoc_STR(descriptor.c_str());                                  
        pytype.tp_methods                            = methods;                                   
        pytype.tp_new                                = objnew;                                                                 
          

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

template<> std::string PySHAMROCK_Model_SetupSPHIMPL<f32, kernels::M4<f32>>::get_name(){
    return "SetupSPH_M4_single";
}





addpybinding(setupsph){
    PySHAMROCK_Model_SetupSPHIMPL<f32, kernels::M4<f32>>::add_object_pybind(module);
    //PySHAMROCK_Model_SetupSPHIMPL<f64, u32, kernels::M4<f64>>::add_object_pybind(module);
    //PySHAMROCK_Model_SetupSPHIMPL<f32, u64, kernels::M4<f32>>::add_object_pybind(module);
    //PySHAMROCK_Model_SetupSPHIMPL<f64, u64, kernels::M4<f64>>::add_object_pybind(module);
}
