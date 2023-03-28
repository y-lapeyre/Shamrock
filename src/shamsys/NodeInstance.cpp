// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "NodeInstance.hpp"

#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"

#include "MpiDataTypeHandler.hpp"

namespace shamsys::instance {




    auto check_queue_is_valid(sycl::queue & q){

        auto test_kernel = [](sycl::queue & q){
            sycl::buffer<u32> b{10};

            q.submit([&](sycl::handler & cgh){
                sycl::accessor acc {b, cgh,sycl::write_only,sycl::no_init};

                cgh.parallel_for(sycl::range<1>{1},[=](sycl::item<1> i){
                    acc[i] = i.get_linear_id();
                });
            });

            q.wait();
        };


        std::exception_ptr eptr;
        try {
            test_kernel(q);
            logger::info_ln("NodeInstance", "selected queue :",q.get_device().get_info<sycl::info::device::name>()," working !");
        } catch(...) {
            eptr = std::current_exception(); // capture
        }

        if (eptr) {
            logger::err_ln("NodeInstance", "selected queue :",q.get_device().get_info<sycl::info::device::name>(),"does not function properly");
            std::rethrow_exception(eptr);
        }


    }

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
            << shambase::readable_sizeof(Device.get_info<sycl::info::device::global_mem_size>()) << "\n";
    }


    std::string get_process_name(){

        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;

        int err_code = mpi::get_processor_name(processor_name, &name_len);

        if(err_code != MPI_SUCCESS){
            throw ShamsysInstanceException("failed getting the process name");
        }
        
        return std::string(processor_name);
    }











    


    std::unique_ptr<sycl::queue> compute_queue;
    std::unique_ptr<sycl::queue> alt_queue;










    bool is_initialized(){
        int flag = false;
        mpi::initialized(&flag);
        return bool(compute_queue) && bool(alt_queue) && flag;
    };


    void init(int argc, char *argv[]){


        if(opts::has_option("--sycl-cfg")){
            std::string sycl_cfg = std::string(opts::get_option("--sycl-cfg"));

            logger::normal_ln("NodeInstance", "chosen sycl config :",sycl_cfg);

            size_t split_alt_comp = 0;
            split_alt_comp = sycl_cfg.find(":");

            if(split_alt_comp == std::string::npos){
                logger::err_ln("NodeInstance", "sycl-cfg layout should be x:x");
                throw ShamsysInstanceException("sycl-cfg layout should be x:x");
            }

            std::string alt_cfg = sycl_cfg.substr(0, split_alt_comp);
            std::string comp_cfg = sycl_cfg.substr(split_alt_comp+1, sycl_cfg.length());


            u32 ialt, icomp;
            try {
                try {
                    ialt = std::stoi(alt_cfg);
                } catch (const std::invalid_argument & a) {
                    logger::err_ln("NodeInstance", "alt config is not an int");
                    throw ShamsysInstanceException("alt config is not an int");
                }
            } catch (const std::out_of_range & a) {
                logger::err_ln("NodeInstance", "alt config is to big for an integer");
                throw ShamsysInstanceException("alt config is to big for an integer");
            }

            try {
                try {
                    icomp = std::stoi(comp_cfg);
                } catch (const std::invalid_argument & a) {
                    logger::err_ln("NodeInstance", "compute config is not an int");
                    throw ShamsysInstanceException("compute config is not an int");
                }
            } catch (const std::out_of_range & a) {
                logger::err_ln("NodeInstance", "compute config is to big for an integer");
                throw ShamsysInstanceException("compute config is to big for an integer");
            }

            init(SyclInitInfo{ialt,icomp}, MPIInitInfo{argc,argv});

        }else {

            logger::err_ln("NodeInstance", "Please specify a sycl configuration (--sycl-cfg x:x)");
            //std::cout << "[NodeInstance] Please specify a sycl configuration (--sycl-cfg x:x)" << std::endl;
            throw ShamsysInstanceException("Sycl Handler need configuration (--sycl-cfg x:x)");
        }

    }




    void start_sycl(SyclInitInfo sycl_info){


        //start sycl

        if(bool(compute_queue) && bool(alt_queue)){
            throw ShamsysInstanceException("Sycl is already initialized");
        }

        logger::raw_ln(terminal_effects::colors_foreground_8b::cyan + " >>> init SYCL instances <<< " + terminal_effects::reset);



        const auto &Platforms = sycl::platform::get_platforms();

        logger::raw_ln("--------------------------------------------------------------------------------");
        logger::raw_ln("| sel | id |        Device name            |       Platform name      |  Type  |");
        logger::raw_ln("--------------------------------------------------------------------------------");

        u32 key_global = 0;
        for (const auto &Platform : Platforms) {
            const auto &Devices = Platform.get_devices();

            auto PlatformName = Platform.get_info<sycl::info::platform::name>();
            for (const auto &Device : Devices) {
                auto DeviceName = Device.get_info<sycl::info::device::name>();


                auto selected_k = [&](i32 k){

                    std::string ret = "";

                    if (k == sycl_info.alt_queue_id) {
                        ret += "a";
                    }

                    if (k == sycl_info.compute_queue_id) {
                        ret += "c";
                    }

                    return ret;
                };

                std::string selected = selected_k(key_global);

                std::string devname = shambase::trunc_str(DeviceName,29);
                std::string platname = shambase::trunc_str(PlatformName,24);
                std::string devtype = shambase::trunc_str(getDeviceTypeName(Device),6);

                logger::raw_ln(shambase::format("| {:>3} | {:>2} | {:>29.29} | {:>24.24} | {:>6} |",
                    selected.c_str(),key_global,devname.c_str(),platname.c_str(),devtype.c_str()
                ));


                if(key_global == sycl_info.alt_queue_id){
                    alt_queue = std::make_unique<sycl::queue>(Device,exception_handler);
                }

                if(key_global == sycl_info.compute_queue_id){
                    compute_queue = std::make_unique<sycl::queue>(Device,exception_handler);
                }


                key_global ++;

            }

        }

        logger::raw_ln("--------------------------------------------------------------------------------");

        key_global = 0;
        for (const auto &Platform : Platforms) {
            auto PlatformName = Platform.get_info<sycl::info::platform::name>();
            for (const auto &Device : Platform.get_devices()) {
                auto DeviceName = Device.get_info<sycl::info::device::name>();

                if(key_global == sycl_info.alt_queue_id){
                    logger::info_ln("NodeInstance", "init alt queue  : ", "|",DeviceName, "|", PlatformName, "|" , getDeviceTypeName(Device), "|");
                    alt_queue = std::make_unique<sycl::queue>(Device,exception_handler);
                }

                if(key_global == sycl_info.compute_queue_id){
                    logger::info_ln("NodeInstance", "init comp queue : ", "|",DeviceName, "|", PlatformName, "|" , getDeviceTypeName(Device), "|");
                    compute_queue = std::make_unique<sycl::queue>(Device,exception_handler);
                }

                key_global++;
            }
        }

        check_queue_is_valid(*compute_queue);
        check_queue_is_valid(*alt_queue);


        logger::info_ln("NodeInstance", "init done");


        

        

    }

    void init(SyclInitInfo sycl_info, MPIInitInfo mpi_info){

        start_sycl(sycl_info);



        #ifdef MPI_LOGGER_ENABLED
        std::cout << "%MPI_DEFINE:MPI_COMM_WORLD="<<MPI_COMM_WORLD<<"\n";
        #endif

        
        mpi::init(&mpi_info.argc, &mpi_info.argv);

        i32 iworld_size, iworld_rank;

        mpi::comm_size(MPI_COMM_WORLD, &iworld_size);
        mpi::comm_rank(MPI_COMM_WORLD, &iworld_rank);

        world_rank = iworld_rank;
        world_size = iworld_size;

        #ifdef MPI_LOGGER_ENABLED
        std::cout << "%MPI_VALUE:world_size="<<world_size<<"\n";
        std::cout << "%MPI_VALUE:world_rank="<<world_rank<<"\n";
        #endif

        if(world_size < 1){
            throw ShamsysInstanceException("world size is < 1");
        }

        if(world_rank < 0){
            throw ShamsysInstanceException("world size is above i32_max");
        }

        int error ;
        //error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        error = mpi::comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
        
        
        if(error != MPI_SUCCESS){
            throw ShamsysInstanceException("failed setting the MPI error mode");
        }

        logger::raw_ln(shambase::format("[{:03}]: \x1B[32mMPI_Init : node n°{:03} | world size : {} | name = {}\033[0m\n",world_rank,world_rank,world_size,get_process_name().c_str()));

        mpi::barrier(MPI_COMM_WORLD);
        //if(world_rank == 0){
        logger::raw_ln("------------ MPI init ok ------------ \n");

        logger::info_ln("NodeInstance", "creating MPI type for interop");
        create_sycl_mpi_types();
        logger::info_ln("NodeInstance", "MPI type for interop created");

        logger::raw_ln("------------ MPI / SYCL init ok ------------ \n");

        mpidtypehandler::init_mpidtype();

    }

    void close(){

        mpidtypehandler::free_mpidtype();

        logger::raw_ln("------------ MPI_Finalize ------------\n");
        mpi::finalize(); 

        alt_queue.reset();
        compute_queue.reset();
    }


















    ////////////////////////////
    // sycl related routines
    ////////////////////////////




    sycl::queue & get_compute_queue(u32  /*id*/){
        if(!compute_queue){ throw ShamsysInstanceException("sycl handler is not initialized");}
        return * compute_queue;
    }

    sycl::queue & get_alt_queue(u32  /*id*/){
        if(!alt_queue){ throw ShamsysInstanceException("sycl handler is not initialized");}
        return * alt_queue;
    }






    ////////////////////////////
    //MPI related routines
    ////////////////////////////



}