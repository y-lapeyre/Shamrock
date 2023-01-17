// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#pragma once

#include "shamsys/MpiWrapper.hpp"

#include <vector>
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

    /**
     * @brief Class to handle a collection of MPI_Request(s)
     * 
     */
    class CommRequests{

        std::vector<MPI_Request> rqs;

        public:

        /**
         * @brief add a request to the collection
         * 
         * @param rq the mpi request
         */
        inline void push(MPI_Request rq){
            rqs.push_back(rq);
        }

        /**
         * @brief call a mpi_wait for the request held by the list
         * 
         */
        inline void wait_all(){

            std::vector<MPI_Status> st_lst(rqs.size());
            mpi::waitall(rqs.size(), rqs.data(), st_lst.data());

            rqs.clear();

            //TODO shall we return the status list in a CommStatus class ?
        }
    };

} // namespace shamsys::comm


namespace shamsys::comm::details {

    template<class T> class CommDetails;
    template<class T, Protocol comm_mode> class CommBuffer;
    
} // namespace shamsys::comm::details