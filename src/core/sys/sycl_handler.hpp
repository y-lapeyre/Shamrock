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


#include <sycl/sycl.hpp>
#include "aliases.hpp"




class ShamrockSyclException : public std::exception
{
public:
    explicit ShamrockSyclException(const char* message)
        : msg_(message) {}


    explicit ShamrockSyclException(const std::string& message)
        : msg_(message) {}

    virtual ~ShamrockSyclException() noexcept {}

    virtual const char* what() const noexcept {
       return msg_.c_str();
    }

protected:
    std::string msg_;
};


namespace sycl_handler {

    void init();

    sycl::queue & get_compute_queue();
    sycl::queue & get_alt_queue();


} // namespace sycl_handler