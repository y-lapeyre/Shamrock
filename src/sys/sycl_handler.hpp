#pragma once

#include "mpi_handler.hpp"

#include <CL/sycl.hpp>


inline auto exception_handler = [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch(sycl::exception const& e) {
            printf("Caught synchronous SYCL exception: %s\n",e.what());
        }
    }
};

inline sycl::queue* queue;


inline void init_sycl(){

    if(world_rank == 0)
        printf("\x1B[36m >>> init SYCL instances <<< \033[0m\n");

    global_logger->log(">>> init SYCL instances <<<\n");

    queue = new sycl::queue(sycl::default_selector(),exception_handler);

    global_logger->log("Running on : %s\n", queue->get_device().get_info<sycl::info::device::name>().c_str());

}
