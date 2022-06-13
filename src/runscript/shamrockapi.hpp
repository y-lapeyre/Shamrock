#pragma once

#include "aliases.hpp"
#include "io/logs.hpp"
#include "patch/patchdata_layout.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include <memory>


class ShamAPIException : public std::exception
{
public:
    explicit ShamAPIException(const char* message)
        : msg_(message) {}


    explicit ShamAPIException(const std::string& message)
        : msg_(message) {}

    virtual ~ShamAPIException() noexcept {}

    virtual const char* what() const noexcept {
       return msg_.c_str();
    }

protected:
    std::string msg_;
};



class ShamrockAPI{

    std::unique_ptr<PatchDataLayout> pdl;
    std::unique_ptr<PatchScheduler> sched;

    

    inline void print_version(){
        std::cout << git_info_str << std::endl;
    }




    inline void pdata_layout_reset(){
        if(sched){
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl = std::make_unique<PatchDataLayout>();
    }

    inline void pdata_layout_do_double_prec_mode(){
        if(sched){
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl->xyz_mode = xyz64;
    }

    inline void pdata_layout_do_single_prec_mode(){
        if(sched){
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl->xyz_mode = xyz32;
    }

    template<class T>
    inline void pdata_layout_add_field(std::string fname, u32 nvar){
        if(sched){
            throw ShamAPIException("cannot modify patch data layout while the scheduler is on");
        }
        pdl->add_field<T>(fname, nvar);
    }

    inline void pdata_layout_print(){
        if(!pdl){
            throw ShamAPIException("patch data layout is not initialized");
        }
        std::cout << pdl->get_description_str() << std::endl;
    }



   

    inline void init_sched(u64 crit_split,u64 crit_merge){
        sched = std::make_unique<PatchScheduler>(*pdl,crit_split,crit_merge);
        sched->init_mpi_required_types();
    }

    inline void close_sched(){
        sched = std::unique_ptr<PatchScheduler>();
    }


    inline ShamrockAPI(){
        logfiles::open_log_files();
    }

    inline ~ShamrockAPI(){
        logfiles::close_log_files();
    }
};