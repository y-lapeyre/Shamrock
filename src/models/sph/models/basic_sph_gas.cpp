#include "basic_sph_gas.hpp"
#include "aliases.hpp"
#include "models/sph/base/kernels.hpp"

#include <array>
#include <memory>

#include "runscript/pymodule/pylib.hpp"
#include "runscript/shamrockapi.hpp"



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

    static void init(){}
    static void evolve(PatchScheduler &sched, f64 &step_time){}
    static void dump(std::string prefix){}
    static void restart_dump(std::string prefix){}
    static void close(){}
    static void set_cfl_cour(flt Ccour){}
    static void set_cfl_force(flt Cforce){}

    static void add_object_pybind(PyObject * module,std::string name){

        static PyMethodDef methods [] = {
            {"reset", (PyCFunction) init, METH_NOARGS,"doc str"},
            {NULL}  /* Sentinel */
        };

        __ADD_PYBIND__
    }

};



addpybinding(basicsphgas){
    PySHAMROCK_Model_BasicSPHGasIMPL<f32, u32, kernels::M4<f32>>::add_object_pybind(module, "BasicSPHGas_single_morton32_M4");
    PySHAMROCK_Model_BasicSPHGasIMPL<f64, u32, kernels::M4<f64>>::add_object_pybind(module, "BasicSPHGas_double_morton32_M4");
    PySHAMROCK_Model_BasicSPHGasIMPL<f32, u64, kernels::M4<f32>>::add_object_pybind(module, "BasicSPHGas_single_morton64_M4");
    PySHAMROCK_Model_BasicSPHGasIMPL<f64, u64, kernels::M4<f64>>::add_object_pybind(module, "BasicSPHGas_double_morton64_M4");
}

template<> class BasicSPHGas<f32,u32,kernels::M4<f32>>;
