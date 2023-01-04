// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "core/io/logs.hpp"
#include "core/patch/base/patchdata.hpp"
#include "core/patch/base/patchdata_layout.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "shamsys/mpi_handler.hpp"
#include <map>
#include <memory>
#include <tuple>
#include <vector>


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



class ShamrockCtx{public:

    std::unique_ptr<PatchDataLayout> pdl;
    std::unique_ptr<PatchScheduler> sched;

    inline void pdata_layout_new(){
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

        if(!pdl){
            throw ShamAPIException("patch data layout is not initialized");
        }

        sched = std::make_unique<PatchScheduler>(*pdl,crit_split,crit_merge);
        sched->init_mpi_required_types();
    }

    inline void close_sched(){
        sched.reset();
    }


    inline ShamrockCtx(){
        //logfiles::open_log_files();
    }

    inline ~ShamrockCtx(){
        //logfiles::close_log_files();
    }

    inline std::vector<std::unique_ptr<PatchData>> gather_data(u32 rank){
        return sched->gather_data(rank);
    }

    inline std::vector<std::unique_ptr<PatchData>> allgather_data(){
        std::vector<std::unique_ptr<PatchData>> recv_data;

        for(u32 i = 0; i < mpi_handler::uworld_size; i++){
            if (i == mpi_handler::uworld_rank) {
                recv_data = sched->gather_data(i);
            }else{
                sched->gather_data(i);
            }
        }

        return recv_data;
    }

    void set_box_size(std::tuple<f64_3, f64_3> box) {

        if (!pdl) {
            throw ShamAPIException("patch data layout is not initialized");
        }

        if (!sched) {
            throw ShamAPIException("scheduler is not initialized");
        }

        switch (pdl->xyz_mode) {
        case xyz32: {
            auto conv_vec = [](f64_3 v) -> f32_3 { return {v.x(), v.y(), v.z()}; };

            f32_3 vec0 = conv_vec(std::get<0>(box));
            f32_3 vec1 = conv_vec(std::get<1>(box));

            sched->set_box_volume<f32_3>({vec0, vec1});
        }; break;
        case xyz64: {
            sched->set_box_volume<f64_3>(box);
        }; break;
        }
    }
};