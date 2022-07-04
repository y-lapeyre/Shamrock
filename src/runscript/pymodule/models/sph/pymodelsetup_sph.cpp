#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/utils/string_utils.hpp"
#include "runscript/pymodule/pylib.hpp"
#include "runscript/shamrockapi.hpp"
#include "runscript/pymodule/pyshamrockcontext.hpp"
#include "models/sph/setup/sph_setup.hpp"
#include "models/sph/base/kernels.hpp"
#include <floatobject.h>
#include <methodobject.h>
#include <modsupport.h>
#include <object.h>
#include <tuple>


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

    static PyObject * get_box_dim_icnt(Type * self, PyObject * args){
        f64 dr;
        u32 ix,iy,iz;

        if(!PyArg_ParseTuple(args, "d(III)",&dr,&ix,&iy,&iz)) {
            return NULL;
        }

        auto dim = self->container.ptr->get_box_dim(dr, ix, iy, iz);

        return Py_BuildValue("ddd", f64(dim.x()), f64(dim.y()), f64(dim.z()));
    }

    static PyObject * get_ideal_box(Type * self, PyObject * args){
        f64 dr, xm,xM, ym,yM, zm,zM;

        if(!PyArg_ParseTuple(args, "d((dd)(dd)(dd))",&dr,&xm,&xM,&ym,&yM,&zm,&zM)) {
            return NULL;
        }

        auto new_dim = self->container.ptr->get_ideal_box(dr, {{flt(xm),flt(ym),flt(zm)},{flt(xM),flt(yM),flt(zM)}});

        xm = std::get<0>(new_dim).x();
        ym = std::get<0>(new_dim).y();
        zm = std::get<0>(new_dim).z();
        xM = std::get<1>(new_dim).x();
        yM = std::get<1>(new_dim).y();
        zM = std::get<1>(new_dim).z();

        return Py_BuildValue("((dd)(dd)(dd))", f64(xm),f64(ym),f64(zm),f64(xM),f64(yM),f64(zM));
    }

    static PyObject * add_cube_fcc(Type * self, PyObject * args){

        //TODO add check in the code to check if box size is set before running function that depend on it
        
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

    static PyObject* set_boundaries(Type * self, PyObject * args){
        i32 boundary_mode;
        if(!PyArg_ParseTuple(args, "p",&boundary_mode)) {
            return NULL;
        }

        std::cout << "setting boundary mode : " << boundary_mode << std::endl;

        self->container.ptr->set_boundaries(boundary_mode);

        return Py_None;

    }

    static PyObject* set_total_mass(Type * self, PyObject * args){
        f64 val;
        if(!PyArg_ParseTuple(args, "d",&val)) {
            return NULL;
        }

        self->container.ptr->set_total_mass(val);

        return Py_None;
    }

    static PyObject* get_part_mass(Type * self, PyObject *Py_UNUSED(ignored)){
        return PyFloat_FromDouble(f64(self->container.ptr->get_part_mass()));
    }

    static PyObject* update_smoothing_lenght(Type * self, PyObject * args){
        PySHAMROCKContext * pyctx;

        if(!PyArg_ParseTuple(args, "O!",PyShamCtxType_ptr,&pyctx)) {
            return NULL;
        }

        self->container.ptr->update_smoothing_lenght(*pyctx->ctx->sched);

        return Py_None;
    }


    static void add_object_pybind(PyObject * module){

        static PyMethodDef methods [] = {

            {"reset", (PyCFunction) reset, METH_NOARGS,"reset object"},
            {"clear", (PyCFunction) clear, METH_NOARGS,"clear memory for object"},

            {"init", (PyCFunction) init, METH_O,"init"},
            {"get_box_dim_icnt", (PyCFunction) get_box_dim_icnt, METH_VARARGS,"get_box_dim_icnt"},
            {"add_cube_fcc", (PyCFunction) add_cube_fcc, METH_VARARGS,"add_cube_fcc"},
            {"get_ideal_box", (PyCFunction) get_ideal_box, METH_VARARGS, "get_ideal_box"},
            {"set_boundaries", (PyCFunction) set_boundaries, METH_VARARGS, "set boundary conditions mode"},

            {"set_total_mass", (PyCFunction) set_total_mass, METH_VARARGS, "set total mass"},
            {"get_part_mass", (PyCFunction) get_part_mass, METH_NOARGS, "get particle mass"},

            {"update_smoothing_lenght", (PyCFunction) update_smoothing_lenght, METH_VARARGS, "update smoothing lenght"},

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
