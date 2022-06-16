// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file sycl_handler.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief header file to manage sycl
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 */

#pragma once

#include <memory>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>
#include "aliases.hpp"


/**
 * @brief new implementation of the SYCL handler
 * 
 */
class SyCLHandler{public:
    private:
        SyCLHandler(){};
        SyCLHandler(const SyCLHandler&);
        SyCLHandler& operator=(const SyCLHandler&);

        std::unordered_map<u32,sycl::queue> queues;

        /**
         * @brief contain all sycl queues that will be used for compute (assume separated memory architecture)
         * 
         */
        std::unordered_map<u32, sycl::queue*> compute_queues;

        /**
         * @brief contain all sycl queues that for host parralelisation 
         */
        std::unordered_map<u32, sycl::queue*> alt_queues;

    public:
        
        inline sycl::queue & get_queue_alt(u32 id){
            return *alt_queues[id];
        }

        inline sycl::queue & get_queue_compute(u32 id){
            return *compute_queues[id];
        }

        inline std::unordered_map<u32, sycl::queue*> & get_alt_queues(){
            return alt_queues;
        }

        inline std::unordered_map<u32, sycl::queue*> & get_compute_queues(){
            return compute_queues;
        }

        /**
         * @brief init sycl handler
         * 
         */
        void init_sycl();

        /**
         * @brief Get the default sycl queue
         * 
         * @return sycl::queue& reference to the default sycl queue
         */
        sycl::queue & get_default();

        /**
         * @brief Get the unique instance of the sycl handler
         * 
         * @return SyCLHandler& sycl handler instance
         */
        static SyCLHandler& get_instance();
};