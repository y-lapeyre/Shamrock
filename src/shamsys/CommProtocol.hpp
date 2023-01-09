// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#pragma once

#include <exception>
#include <string>

namespace shamsys::comm {

    enum Protocol{
        /**
         * @brief copy data to the host and then perform the call
         */
        CopyToHost, 
        
        /**
         * @brief copy data straight from the GPU
         */
        DirectGPU, 
        
        /**
         * @brief  copy data straight from the GPU & flatten sycl vector to plain arrays
         */
        DirectGPUFlatten,
    };

    /**
     * @brief Exception type for the NodeInstance
     */
    class CommProtocolError : public std::exception {
      public:
        explicit CommProtocolError(const char *message) : msg_(message) {}

        explicit CommProtocolError(const std::string &message) : msg_(message) {}

        ~CommProtocolError() noexcept override = default;

        [[nodiscard]] const char *what() const noexcept override { return msg_.c_str(); }

      protected:
        std::string msg_;
    };

} // namespace shamsys::comm


namespace shamsys::comm::details {

    template<class T> class CommDetails;
    template<class T, Protocol comm_mode> class CommBuffer;
    template<class T, Protocol comm_mode> class CommRequest;
    
} // namespace shamsys::comm::details