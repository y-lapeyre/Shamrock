// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeInstance.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Header file describing a Node Instance
 * @copyright Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
 *
 * A NodeInstance is a wrapper to perform a combined initialisation of both SYCL & MPI.
 * Essentially it handle the MPI process and corresponding accelerators
 */

#pragma once

#include "aliases.hpp"

#include <vector>



namespace shamsys::instance {


    class ShamsysInstanceException : public std::exception {
    public:
        explicit ShamsysInstanceException(const char *message) : msg_(message) {}

        explicit ShamsysInstanceException(const std::string &message) : msg_(message) {}

        ~ShamsysInstanceException() noexcept override = default;

        [[nodiscard]] 
        const char *what() const noexcept override { return msg_.c_str(); }

    protected:
        std::string msg_;
    };

    struct SyclInitInfo{
        u32 alt_queue_id;
        u32 compute_queue_id;
    };

    struct MPIInitInfo{
        int argc;
        char ** argv;
    };

    bool initialized();


    void init(int argc, char *argv[]);
    void init(SyclInitInfo sycl_info, MPIInitInfo mpi_info);

    void close();

    ////////////////////////////
    // sycl related routines
    ////////////////////////////

    sycl::queue & get_compute_queue(u32 id = 0);
    sycl::queue & get_alt_queue(u32 id = 0);


    ////////////////////////////
    //MPI related routines
    ////////////////////////////

    // idea to handle multiple GPU with MPI : i32 get_mpi_tag(u32 tag);

    inline u32 world_rank;
    inline u32 world_size;

    /**
    * @brief Get the process name 
    * @return std::string process name
    */
    std::string get_process_name();

} // namespace shamsys::instance