#include "basic_sph_gas.hpp"
#include "aliases.hpp"
#include "models/sph/base/kernels.hpp"

#include <array>
#include <memory>

#include "runscript/pymodule/pylib.hpp"
#include "runscript/shamrockapi.hpp"
#include "runscript/pymodule/pyshamrockcontext.hpp"



using namespace models::sph;




template<class flt, class u_morton, class Kernel> 
void BasicSPHGas<flt,u_morton,Kernel>::init(){

}

template<class flt, class u_morton, class Kernel> 
void BasicSPHGas<flt,u_morton,Kernel>::evolve(PatchScheduler &sched, f64 &step_time){

}
template<class flt, class u_morton, class Kernel> 
void BasicSPHGas<flt,u_morton,Kernel>::dump(std::string prefix){

}
template<class flt, class u_morton, class Kernel> 
void BasicSPHGas<flt,u_morton,Kernel>::restart_dump(std::string prefix){

}
template<class flt, class u_morton, class Kernel> 
void BasicSPHGas<flt,u_morton,Kernel>::close(){

}






template<class flt, class u_morton, class Kernel>
struct PySHAMROCK_Model_BasicSPHGas{
    PyObject_HEAD
    /* Type-specific fields go here. */
    std::unique_ptr<BasicSPHGas<flt, u_morton, Kernel>> model;
    
};

template<class flt, class u_morton, class Kernel>
struct PySHAMROCK_Model_BasicSPHGasIMPL{

    using Type = PySHAMROCK_Model_BasicSPHGas<flt,u_morton,Kernel>;
    using IntType = BasicSPHGas<flt,u_morton,Kernel>;
    inline static const std::string descriptor = "SPH model for basic gas";

    __ADD_METHODS__


    static std::string get_name();


    inline static PyTypeObject* type_ptr = nullptr;



    static PyObject * init(Type * self, PyObject *Py_UNUSED(ignored)){
        self->model = std::make_unique<IntType>();
        self->model->init();
        return Py_None;
    }

    static PyObject * evolve(Type * self, PyObject * args){
        
        //*
        if (PyObject_IsInstance(args, (PyObject *)PyShamCtxType_ptr)){
            std::cout << "testing" << std::endl;
        }else {
            return NULL;
        }
        //*/

        return Py_None;
    }

    static void dump(std::string prefix){}
    static void restart_dump(std::string prefix){}

    static PyObject * close(Type * self, PyObject *Py_UNUSED(ignored)){
        self->model.reset();
        return Py_None;
    }

    static void set_cfl_cour(flt Ccour){}
    static void set_cfl_force(flt Cforce){}




    static void add_object_pybind(PyObject * module){

        static PyMethodDef methods [] = {
            {"init", (PyCFunction) init, METH_NOARGS,"init"},
            {"evolve", (PyCFunction) evolve, METH_O,"evolve"},
            {"close", (PyCFunction) close, METH_NOARGS,"close"},
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

template<> std::string PySHAMROCK_Model_BasicSPHGasIMPL<f32, u32, kernels::M4<f32>>::get_name(){
    return "BasicSPHGas_single_morton32_M4";
}





addpybinding(basicsphgas){
    PySHAMROCK_Model_BasicSPHGasIMPL<f32, u32, kernels::M4<f32>>::add_object_pybind(module);
    //PySHAMROCK_Model_BasicSPHGasIMPL<f64, u32, kernels::M4<f64>>::add_object_pybind(module);
    //PySHAMROCK_Model_BasicSPHGasIMPL<f32, u64, kernels::M4<f32>>::add_object_pybind(module);
    //PySHAMROCK_Model_BasicSPHGasIMPL<f64, u64, kernels::M4<f64>>::add_object_pybind(module);
}

template<> class BasicSPHGas<f32,u32,kernels::M4<f32>>;
