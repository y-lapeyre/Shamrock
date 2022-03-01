/**
 * @file sycl_handler.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief header file to manage sycl
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 */

#pragma once

#include <CL/sycl.hpp>
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

    public:
        
        /**
         * @brief contain all sycl queues that will be used for compute (assume separated memory architecture)
         * 
         */
        std::unordered_map<u32, sycl::queue> compute_queues;

        /**
         * @brief contain all sycl queues that for host parralelisation 
         */
        std::unordered_map<u32, sycl::queue> alt_queues;

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