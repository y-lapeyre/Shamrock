// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#include "runscript/pymodule/pylib.hpp"
#include "runscript/shamrockapi.hpp"
#include "runscript/pymodule/pyshamrockcontext.hpp"
#include "shammodels/nbody/models/nbody_selfgrav.hpp"


using namespace models::nbody;




template<class flt>
MakeContainer(Container_Model_Nbody_SelfGrav, Nbody_SelfGrav<flt>);

template<class flt>
MakePyContainer(PySHAMROCK_Container_Model_Nbody_SelfGrav, Container_Model_Nbody_SelfGrav<flt>)



template<class flt>
struct PySHAMROCK_Container_Model_Nbody_SelfGravIMPL{

    using Type = PySHAMROCK_Container_Model_Nbody_SelfGrav<flt>;
    using IntType = Nbody_SelfGrav<flt>;
    inline static const std::string descriptor = "Nbody selfgravity model";

    AddPyContainer_methods(Type)


    static std::string get_name();


    inline static PyTypeObject* type_ptr = nullptr;



    static PyObject * init(Type * self, PyObject *Py_UNUSED(ignored)){
        self->container.ptr->init();
        return Py_None;
    }

    static PyObject * evolve(Type * self, PyObject * args){
        
        PySHAMROCKContext * pyctx;

        f64 current_t;
        f64 target_time;

        if(!PyArg_ParseTuple(args, "O!dd",PyShamCtxType_ptr,&pyctx,&current_t,&target_time)) {
            return NULL;
        }

        f64 new_t = self->container.ptr->evolve(
            *pyctx->ctx->sched, 
            current_t, 
            target_time);

        return PyFloat_FromDouble(new_t);
    }

    static PyObject * simulate_until(Type * self, PyObject * args){ 
        PySHAMROCKContext * pyctx;

        f64 start_time;
        f64 end_time;
        u32 freq_dump;
        u32 freq_restart_dump;

        const char* prefix_dump;


        if(!PyArg_ParseTuple(args, "O!ddIIs",PyShamCtxType_ptr,&pyctx,&start_time,&end_time,&freq_dump,&freq_restart_dump,&prefix_dump)) {
            return NULL;
        }

        std::string str_prefix_dump = std::string(prefix_dump);

        std::cout << "start_time :" << start_time << std::endl;
        std::cout << "end_time :" << end_time << std::endl;
        std::cout << "freq_dump :" << freq_dump << std::endl;
        std::cout << "freq_restart_dump :" << freq_restart_dump << std::endl;
        std::cout << "prefix_dump :" << str_prefix_dump << std::endl;

        f64 cur_t = self->container.ptr->simulate_until(*pyctx->ctx->sched, start_time, end_time, freq_dump, freq_restart_dump, str_prefix_dump);

        return PyFloat_FromDouble(cur_t);
    }


    static PyObject * close(Type * self, PyObject *Py_UNUSED(ignored)){
        return Py_None;
    }

    static PyObject * set_cfl_force(Type * self, PyObject * args){
        f64 c_val;
        if(!PyArg_ParseTuple(args, "d",&c_val)) {
            return NULL;
        }

        self->container.ptr->set_cfl_force(c_val);

        return Py_None;
    }

    static PyObject * set_particle_mass(Type * self, PyObject * args){
        f64 c_val;
        if(!PyArg_ParseTuple(args, "d",&c_val)) {
            return NULL;
        }

        self->container.ptr->set_particle_mass(c_val);

        return Py_None;
    }




    static void add_object_pybind(PyObject * module){

        static PyMethodDef methods [] = {


            {"reset", (PyCFunction) reset, METH_NOARGS,"reset object"},
            {"clear", (PyCFunction) clear, METH_NOARGS,"clear memory for object"},

            {"init", (PyCFunction) init, METH_NOARGS,"init"},

            {"evolve", (PyCFunction) evolve, METH_VARARGS,"evolve"},
            {"simulate_until", (PyCFunction) simulate_until, METH_VARARGS,"simulate_until"},


            {"set_cfl_force", (PyCFunction) set_cfl_force, METH_VARARGS,"set_cfl_force"},
            {"set_particle_mass", (PyCFunction) set_particle_mass, METH_VARARGS,"set_particle_mass"},

            {"close", (PyCFunction) close, METH_NOARGS,"close"},
            {NULL}  /* Sentinel */
        };

        static std::string name = get_name();
        static std::string tp_name_str = "shamrock." + name;                                                                
                                                                                                                            
        static PyTypeObject pytype = {                                                                                      
            PyVarObject_HEAD_INIT(NULL, 0)
        };
        pytype.tp_name = tp_name_str.c_str();                                                 
        pytype.tp_doc                                = PyDoc_STR(descriptor.c_str());                                        
        pytype.tp_basicsize                          = sizeof(Type);                                                         
        pytype.tp_itemsize                           = 0;                                                                    
        pytype.tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;                             
        pytype.tp_new                                = objnew;                                                               
        pytype.tp_dealloc                            = (destructor)dealloc;                                                  
        pytype.tp_methods                            = methods;                                                              
          

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

template<> std::string PySHAMROCK_Container_Model_Nbody_SelfGravIMPL<f32>::get_name(){
    return "NBody_selfgrav_f32";
}





addpybinding(nbodyselfgrav){
    PySHAMROCK_Container_Model_Nbody_SelfGravIMPL<f32>::add_object_pybind(module);
    //PySHAMROCK_Model_BasicSPHGasSelfGravIMPL<f64, u32, kernels::M4<f64>>::add_object_pybind(module);
    //PySHAMROCK_Model_BasicSPHGasSelfGravIMPL<f32, u64, kernels::M4<f32>>::add_object_pybind(module);
    //PySHAMROCK_Model_BasicSPHGasSelfGravIMPL<f64, u64, kernels::M4<f64>>::add_object_pybind(module);
}

