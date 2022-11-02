#include "aliases.hpp"

#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/patch/utility/serialpatchtree.hpp"


#pragma once


//%Impl status : Clean unfinished

enum InterfacehandlerImpl {
    Tree_Send
};

//forward definition of the interface handler

template<InterfacehandlerImpl impl_type, class pos_prec, class Tree> class Interfacehandler{

    public:

    using flt = pos_prec;
    using vec = sycl::vec<flt, 3>;

    template<class Func_interactcrit>
    void compute_interface_list(
        PatchScheduler &sched, 
        SerialPatchTree<vec> sptree, 
        Func_interactcrit && interact_crit,
        bool periodic);

    //TODO
    void initial_fetch();

    void fetch_field();


    template<class Function> void for_each_interface(u64 patch_id, Function && fct);
};



