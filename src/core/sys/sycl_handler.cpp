// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sycl_handler.hpp"

#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/utils/string_utils.hpp"

#include "cmdopt.hpp"


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
        return "CPU";
    case sycl::info::device_type::gpu:
        return "GPU";
    case sycl::info::device_type::host:
        return "HOST";
    case sycl::info::device_type::accelerator:
        return "ACCELERATOR";
    default:
        return "UNKNOWN";
    }
}

void print_device_info(const sycl::device &Device){
    std::cout 
        << "   - " 
        << Device.get_info<sycl::info::device::name>()
        << " " 
        << readable_sizeof(Device.get_info<sycl::info::device::global_mem_size>()) << "\n";
}


namespace sycl_handler {
    bool already_on = false;

    const auto &Platforms = sycl::platform::get_platforms();

    std::unique_ptr<sycl::queue> compute_queue;
    std::unique_ptr<sycl::queue> alt_queue;

    void init(){

        if(already_on) { 
            throw ShamrockSyclException("Sycl Handler is already on");
        }

        printf("\x1B[36m >>> init SYCL instances <<< \033[0m\n");

        Cmdopt & opt = Cmdopt::get_instance();

        if(opt.has_option("--sycl-cfg")){
            std::string sycl_cfg = std::string(opt.get_option("--sycl-cfg"));
            std::cout << "chosen sycl config : " << sycl_cfg << std::endl;

            size_t split_alt_comp = 0;
            split_alt_comp = sycl_cfg.find(":");

            if(split_alt_comp == std::string::npos){
                throw shamrock_exc("sycl-cfg layout should be x:x");
            }

            std::string alt_cfg = sycl_cfg.substr(0, split_alt_comp);
            std::string comp_cfg = sycl_cfg.substr(split_alt_comp+1, sycl_cfg.length());

        }else {
            std::cout << "[SYCL Handler] Please specify a sycl configuration (--sycl-cfg x:x)" << std::endl;
            throw ShamrockSyclException("Sycl Handler need configuration (--sycl-cfg x:x)");
        }





        already_on = true;
    }
} // namespace sycl_handler



SyCLHandler& SyCLHandler::get_instance(){
    static SyCLHandler instance;
    return instance;
}


void SyCLHandler::init_sycl(){

    if(compute_queues.size() + alt_queues.size() > 0){
        throw shamrock_exc("ERROR : sycl already initialized");
    }

    printf("\x1B[36m >>> init SYCL instances <<< \033[0m\n");






    Cmdopt & opt = Cmdopt::get_instance();

    bool custom_config = false;
    std::string queue_choices_alt;
    std::string queue_choices_compute;

    std::vector<u32> queue_choice_alt_ids;
    std::vector<u32> queue_choice_compute_ids;


    if(opt.has_option("--sycl-cfg")){
        std::string sycl_cfg = std::string(opt.get_option("--sycl-cfg"));
        std::cout << "chosen sycl config : " << sycl_cfg << std::endl;

        
        size_t split_alt_comp = 0;
        split_alt_comp = sycl_cfg.find(":");

        if(split_alt_comp == std::string::npos){
            throw shamrock_exc("wring layout should be ...:...");
        }

        std::string alt_cfg = sycl_cfg.substr(0, split_alt_comp);
        std::string comp_cfg = sycl_cfg.substr(split_alt_comp+1, sycl_cfg.length());

        {
            std::string s = alt_cfg;
            size_t pos = 0;
            std::string token;
            while ((pos = s.find(",")) != std::string::npos) {
                token = s.substr(0, pos);
                queue_choice_alt_ids.push_back(std::atoi(token.c_str()));
                s.erase(0, pos + 1);
            }
            queue_choice_alt_ids.push_back(std::atoi(s.c_str()));
        }


        {
            std::string s = comp_cfg;
            size_t pos = 0;
            std::string token;
            while ((pos = s.find(",")) != std::string::npos) {
                token = s.substr(0, pos);
                queue_choice_compute_ids.push_back(std::atoi(token.c_str()));
                s.erase(0, pos + 1);
            }
            queue_choice_compute_ids.push_back(std::atoi(s.c_str()));
        }

        std::cout << " alt  : " ;
        for(auto a : queue_choice_alt_ids){
            std::cout << a << " ";
        }std::cout << std::endl;

        std::cout << " comp : " ;
        for(auto a : queue_choice_compute_ids){
            std::cout << a << " ";
        }std::cout << std::endl;

        custom_config = true;
    }else{
        std::cout << "using default sycl config" << std::endl;
        custom_config = false;
    }







    u32 key_comp_qu = 0;
    u32 key_alt_qu = 0;

    const auto &Platforms = sycl::platform::get_platforms();

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "| id |       Device name            |       Platform name         |    Type    |\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    u32 key_global = 0;
    for (const auto &Platform : Platforms) {
        const auto &Devices = Platform.get_devices();

        auto PlatformName = Platform.get_info<sycl::info::platform::name>();
        for (const auto &Device : Devices) {
            auto DeviceName = Device.get_info<sycl::info::device::name>();

            std::cout << format("| %02d | %-29s | %-26s | %-10s |\n",key_global,
                trunc_str(DeviceName,28).c_str(),
                trunc_str(PlatformName,29).c_str(),
                trunc_str(getDeviceTypeName(Device),10).c_str());

            

            if(custom_config){

                bool is_dev_used = false;

                is_dev_used = is_dev_used || (std::count(queue_choice_alt_ids.begin(), queue_choice_alt_ids.end(), key_global) > 0);
                is_dev_used = is_dev_used || (std::count(queue_choice_compute_ids.begin(), queue_choice_compute_ids.end(), key_global) > 0);

                if(is_dev_used){
                    queues[key_global] = sycl::queue(Device,exception_handler);
                }
                

            }else{
                queues[key_global] = sycl::queue(Device,exception_handler);

                auto DeviceType = Device.get_info<sycl::info::device::device_type>();
                switch (DeviceType) {
                case sycl::info::device_type::host:
                    alt_queues[key_alt_qu] = &queues[key_global];key_alt_qu ++;break;
                case sycl::info::device_type::cpu:
                    alt_queues[key_alt_qu] = &queues[key_global];key_alt_qu ++;break;
                case sycl::info::device_type::gpu:
                    compute_queues[key_comp_qu] = &queues[key_global];key_comp_qu ++;break;
                case sycl::info::device_type::accelerator:
                    ;break;
                default:
                    ;break;
                }
            }

            key_global++;
        }

    }
    std::cout << "--------------------------------------------------------------------------------\n";


    if(custom_config){
        for (u32 id : queue_choice_alt_ids) {
            alt_queues[key_alt_qu] = &queues[id];
            key_alt_qu ++;
        }

        for (u32 id : queue_choice_compute_ids) {
            compute_queues[key_comp_qu] = &queues[id];
            key_comp_qu ++;
        }
    }

    std::cout << "alt queues : \n"; 
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "|  Key  |       Device name         |     Type    |    Memory    |    Cache    |\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    for (auto & [key,qu] : alt_queues) {
        std::cout << format("| %5d | %-25s | %-11s | %-12s | %-11s |\n",
            key,
            qu->get_device().get_info<sycl::info::device::name>().c_str(),
            getDeviceTypeName(qu->get_device()).c_str(),
            readable_sizeof(qu->get_device().get_info<sycl::info::device::global_mem_size>()).c_str(),
            readable_sizeof(qu->get_device().get_info<sycl::info::device::local_mem_size>()).c_str()
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
            qu->get_device().get_info<sycl::info::device::name>().c_str(),
            getDeviceTypeName(qu->get_device()).c_str(),
            readable_sizeof(qu->get_device().get_info<sycl::info::device::global_mem_size>()).c_str(),
            readable_sizeof(qu->get_device().get_info<sycl::info::device::local_mem_size>()).c_str()
            );
    }
    std::cout << "--------------------------------------------------------------------------------\n";

}


sycl::queue & SyCLHandler::get_default(){
    if(compute_queues.empty()){
        return *alt_queues[0];
    }else {
        return *compute_queues[0];
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