#pragma once

#include "CL/sycl/device_selector.hpp"
#include "CL/sycl/queue.hpp"
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
inline sycl::queue* host_queue;


inline void init_sycl(){

    //if(world_rank == 0)
        printf("\x1B[36m >>> init SYCL instances <<< \033[0m\n");

    //global_logger->log(">>> init SYCL instances <<<\n");

    queue = new sycl::queue(sycl::default_selector(),exception_handler);

    host_queue = new sycl::queue(sycl::host_selector(),exception_handler);

    printf("Running on : %s\n", queue->get_device().get_info<sycl::info::device::name>().c_str());
    //global_logger->log("Running on : %s\n", queue->get_device().get_info<sycl::info::device::name>().c_str());

    const auto &Platforms = sycl::platform::get_platforms();

    for (const auto &Platform : Platforms) {
        sycl::backend Backend = Platform.get_backend();
        auto PlatformName = Platform.get_info<sycl::info::platform::name>();

        std::cout << " -> platform name : " << PlatformName <<std::endl;
        const auto &Devices = Platform.get_devices();
        for (const auto &Device : Devices) {
            // std::cout << "[" << Backend << ":" << getDeviceTypeName(Device) << ":"
            //             << DeviceNums[Backend] << "] ";
            // ++DeviceNums[Backend];
            // // Verbose parameter is set to false to print regular devices output first
            // printDeviceInfo(Device, false, PlatformName);
        }
    }


}
