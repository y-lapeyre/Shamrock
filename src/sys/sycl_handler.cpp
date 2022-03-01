#include "sycl_handler.hpp"

#include <unordered_map>

#include "utils/string_utils.hpp"


auto exception_handler = [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch(sycl::exception const& e) {
            printf("Caught synchronous SYCL exception: %s\n",e.what());
        }
    }
};

std::string getDeviceTypeName(const sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    switch (DeviceType) {
    case sycl::info::device_type::cpu:
        return "CPU        ";
    case sycl::info::device_type::gpu:
        return "GPU        ";
    case sycl::info::device_type::host:
        return "HOST       ";
    case sycl::info::device_type::accelerator:
        return "ACCELERATOR";
    default:
        return "UNKNOWN    ";
    }
}

void print_device_info(const sycl::device &Device){
    std::cout 
        << "   - " 
        << Device.get_info<sycl::info::device::name>()
        << " " 
        << readable_sizeof(Device.get_info<sycl::info::device::global_mem_size>()) << "\n";
}




SyCLHandler& SyCLHandler::get_instance(){
    static SyCLHandler instance;
    return instance;
}


void SyCLHandler::init_sycl(){
    printf("\x1B[36m >>> init SYCL instances <<< \033[0m\n");

    u32 key_comp_qu = 0;
    u32 key_alt_qu = 0;

    const auto &Platforms = sycl::platform::get_platforms();

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "|          Device name            |       Platform name         |     Type     |\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    for (const auto &Platform : Platforms) {
        const auto &Devices = Platform.get_devices();

        auto PlatformName = Platform.get_info<sycl::info::platform::name>();
        for (const auto &Device : Devices) {
            auto DeviceName = Device.get_info<sycl::info::device::name>();

            std::cout << format("| %-31s | %-27s | %-12s |\n",
                trunc_str(DeviceName,30).c_str(),
                trunc_str(PlatformName,30).c_str(),
                trunc_str(getDeviceTypeName(Device),12).c_str());

            auto DeviceType = Device.get_info<sycl::info::device::device_type>();
            switch (DeviceType) {
            case sycl::info::device_type::cpu:
                ;break;
            case sycl::info::device_type::gpu:
                compute_queues[key_comp_qu] = sycl::queue(Device,exception_handler);key_comp_qu ++;break;
            case sycl::info::device_type::host:
                alt_queues[key_alt_qu] = sycl::queue(Device,exception_handler);key_alt_qu ++;break;
            case sycl::info::device_type::accelerator:
                ;break;
            default:
                ;break;
            }
        }

    }
    std::cout << "--------------------------------------------------------------------------------\n";

    std::cout << "alt queues : \n"; 
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "|  Key  |       Device name         |     Type    |    Memory    |    Cache    |\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    for (auto & [key,qu] : alt_queues) {
        std::cout << format("| %5d | %-25s | %-11s | %-12s | %-11s |\n",
            key,
            qu.get_device().get_info<sycl::info::device::name>().c_str(),
            getDeviceTypeName(qu.get_device()).c_str(),
            readable_sizeof(qu.get_device().get_info<sycl::info::device::global_mem_size>()).c_str(),
            readable_sizeof(qu.get_device().get_info<sycl::info::device::local_mem_size>()).c_str()
            );
    }
    std::cout << "--------------------------------------------------------------------------------\n";

    std::cout << "Compute queues : \n"; 
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "|  Key  |       Device name         |     Type    |    Memory    |    Cache    |\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    for (auto & [key,qu] : compute_queues) {
        std::cout << format("| %5d | %-25s | %-11s | %-12s | %-11s |\n",
            key,
            qu.get_device().get_info<sycl::info::device::name>().c_str(),
            getDeviceTypeName(qu.get_device()).c_str(),
            readable_sizeof(qu.get_device().get_info<sycl::info::device::global_mem_size>()).c_str(),
            readable_sizeof(qu.get_device().get_info<sycl::info::device::local_mem_size>()).c_str()
            );
    }
    std::cout << "--------------------------------------------------------------------------------\n";

}


sycl::queue & SyCLHandler::get_default(){
    if(compute_queues.empty()){
        return alt_queues[0];
    }else {
        return compute_queues[0];
    }
}





/*
void init_sycl(){

    //if(world_rank == 0)
        printf("\x1B[36m >>> init SYCL instances <<< \033[0m\n");

    //global_logger->log(">>> init SYCL instances <<<\n");

    const auto &Platforms = sycl::platform::get_platforms();


    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "|          Device name            |       Platform name         |     Type     |\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    for (const auto &Platform : Platforms) {
        const auto &Devices = Platform.get_devices();

        auto PlatformName = Platform.get_info<sycl::info::platform::name>();
        for (const auto &Device : Devices) {
            auto DeviceName = Device.get_info<sycl::info::device::name>();

            std::cout << format("| %-31s | %-27s | %-12s |\n",
                trunc_str(DeviceName,30).c_str(),
                trunc_str(PlatformName,30).c_str(),
                trunc_str(getDeviceTypeName(Device),12).c_str());
        }

    }
    std::cout << "--------------------------------------------------------------------------------\n";

    queue = new sycl::queue(sycl::default_selector(),exception_handler);
    host_queue = new sycl::queue(sycl::host_selector(),exception_handler);

    std::cout << " -> level 1 queues :\n";
    print_device_info(queue->get_device());

    std::cout << "\n -> level 2 queues :\n";
    print_device_info(host_queue->get_device());



    

    //printf("Running on : %s\n", queue->get_device().get_info<sycl::info::device::name>().c_str());
    //global_logger->log("Running on : %s\n", queue->get_device().get_info<sycl::info::device::name>().c_str());



}*/