// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "core/sys/mpi_handler.hpp"
#include "unittests/shamrocktest.hpp"

#include "core/sys/sycl_handler.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <ostream>
#include <queue>
#include <thread>
#include <vector>

Test_start("",intmult,1){
    int a = 3;
    a*=2;
    Test_assert("int multiplication", a==6);
    Test_assert_log("int multiplication log", a==6,"wesh");
}

Test_start("",intdiv,-1){
    int a = 6;
    a/=2;

    Test_assert("int division", a==3);
}

Test_start("",multiple_asserts,1){

    int t[]{0,1,2,3};
    for (int i = 0; i<4; i++) {
        Test_assert("for loop assert", t[i]==i);
    }

}



Test_start("",test_overload_new,2){

    int* a = new int(0);


    *a = 1;

    std::cout << *a << std::endl;

    Test_assert_log("int multiplication log", *a==6,"wesh");

    delete a;

}


class ForcePressure{public:
    static int f(int i){
        return 2*i;
    }
};

class ForcePressure2{public:
    static int f(int i){
        return 4*i;
    }
};





namespace integrator {

    template<class ForceFunc> class Leapfrog{public:

        virtual void step(std::vector<i32> & fres_arr){
            
            sycl::buffer<i32> fres(fres_arr.data(),fres_arr.size());

            sycl::range<1> range{10};

            sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
                auto ff = fres.get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<class Write_chosen_node>(range, [=](sycl::item<1> item) {
                    u64 i = (u64)item.get_id(0);
                    ff[i] = ForceFunc::f(ff[i]);
                });
            });
            
        }

    };
}



template<class Timestepper> class Simulation{public:

    void simu_main(){

        Timestepper t;

        std::vector<i32> fres_arr = {0,8,4,1,8,7,2,3,77,1};
        t.step(fres_arr);
        std::cout << fres_arr[0] << std::endl;
        std::cout << fres_arr[1] << std::endl;
        std::cout << fres_arr[2] << std::endl;
        std::cout << fres_arr[3] << std::endl;
        std::cout << fres_arr[4] << std::endl;
        std::cout << fres_arr[5] << std::endl;
        std::cout << fres_arr[6] << std::endl;
        std::cout << fres_arr[7] << std::endl;
        std::cout << fres_arr[8] << std::endl;
        std::cout << fres_arr[9] << std::endl;
        
    }

};


Test_start("",sycl_static_func,1){

    //SyCLHandler::get_instance().init_sycl();

    Simulation<integrator::Leapfrog<ForcePressure>> sim;
    Simulation<integrator::Leapfrog<ForcePressure2>> sim2;


    sim.simu_main();


}


Test_start("issue_mpi::", allgatherv, 4){

    
    

    int recv_int[1];
    int recv_count[4]{1,0,0,0};
    int displs[4]{0,1,1,1};


    if(mpi_handler::world_rank == 0){
        int send_int = mpi_handler::world_rank + 10;
        mpi::allgatherv(
        &send_int, 
        1, 
        MPI_INT, 
        recv_int, 
        recv_count, 
        displs, 
        MPI_INT, 
        MPI_COMM_WORLD);
    }else{
        int send_int = mpi_handler::world_rank + 10;
        mpi::allgatherv(
        &send_int, 
        0, 
        MPI_INT, 
        recv_int, 
        recv_count, 
        displs, 
        MPI_INT, 
        MPI_COMM_WORLD);
    }


    std::cout << recv_int[0]  << "\n";
    

}






Test_start("sycl::", custom_iterator, 1){

    std::vector<u32> test_vec(10);
    {
        sycl::buffer<u32> buf(test_vec.data(),test_vec.size());

        sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
            auto out = buf.get_access<sycl::access::mode::write>(cgh);


            class Tree_it{public:

                u32 max_val;

                Tree_it(u32 i){
                    max_val = i;
                }

                u32 curr_id;

                using value_type = const u32 ;
                using reference = const u32& ;
                using pointer = const u32* ;
                using difference_type = std::ptrdiff_t ;
                using iterator_category	= std::forward_iterator_tag ;

                reference operator* () { return curr_id ; }
                //pointer operator-> () { return &**this ; }

                Tree_it& operator++ () { ++curr_id ; return *this ; }
                Tree_it operator++ (int) { const auto temp(*this) ; ++*this ; return temp ; }

                bool operator== ( const Tree_it& that ) const { return curr_id == that.curr_id ; }
                bool operator!= ( const Tree_it& that ) const { return !(*this==that) ; }

                const Tree_it begin(){
                    return Tree_it{curr_id};
                }

                const Tree_it end(){
                    return Tree_it{curr_id};
                }
            };

            

            cgh.parallel_for<class TestIterator>(sycl::range<1>(10), [=](sycl::item<1> item) {
                u32 acc = 0;

                // for(const u32 & i : Tree_it(10)){
                //     acc++;
                // }

                for(u32 i = 0 ; i < 10; i ++){
                    acc++;
                }

                out[item] = acc;

            });
        });
    }

    for(u32 i : test_vec){
        std::cout << i << std::endl;
    }

}


#if false
Test_start("sycl::", parallel_sumbit, 1){

    std::vector<std::vector<u32>> vec_to_up;
    std::queue<u32> id_to_tread;

    u32 s_p = 100000;

    for (int i = 0; i < 100; i++) {
        id_to_tread.emplace(i);
        vec_to_up.push_back(
            std::vector<u32>(s_p)
        );

        for(u32 j = 0 ; j < s_p ; j++){
            vec_to_up[i][j] = j+1;
        }
        
    }

    std::mutex m;

    std::vector<std::thread> workers;
    u32 id_t = 0;
    for (std::pair<const u32, sycl::queue*> & a : SyCLHandler::get_instance().get_alt_queues()) {

        sycl::queue* queue = std::get<1>(a);

        workers.push_back(std::thread([&,id_t,s_p,queue]()  {
            
            while(true){

                u32 working_id = -1;
                {
                    std::lock_guard<std::mutex> lock(m);
                    
                    if(id_to_tread.empty()) break;
                    working_id = id_to_tread.front();
                    id_to_tread.pop();

                }

                sycl::buffer<u32> buf(vec_to_up[working_id].data(),vec_to_up[working_id].size());


                queue->submit([&](sycl::handler &cgh) {
                    auto ff = buf.get_access<sycl::access::mode::read_write>(cgh);

                    auto wmult = working_id;

                    cgh.parallel_for(sycl::range<1>(s_p), [=](sycl::item<1> item) {
                        u64 i = (u64)item.get_id(0);
                        ff[i] *= wmult;
                    });
                });

                std::cout << "thread " << id_t << " working on " << working_id << std::endl;

            }

            

        }));
        id_t++;
    }

    for (std::pair<const u32, sycl::queue*> & a : SyCLHandler::get_instance().get_compute_queues()) {

        sycl::queue* queue = std::get<1>(a);

        workers.push_back(std::thread([&,id_t,s_p,queue]()  {
            
            while(true){

                u32 working_id = -1;
                {
                    std::lock_guard<std::mutex> lock(m);
                    
                    if(id_to_tread.empty()) break;
                    working_id = id_to_tread.front();
                    id_to_tread.pop();

                }

                sycl::buffer<u32> buf(vec_to_up[working_id].data(),vec_to_up[working_id].size());


                queue->submit([&](sycl::handler &cgh) {
                    auto ff = buf.get_access<sycl::access::mode::read_write>(cgh);

                    auto wmult = working_id;

                    cgh.parallel_for(sycl::range<1>(s_p), [=](sycl::item<1> item) {
                        u64 i = (u64)item.get_id(0);
                        ff[i] *= wmult;
                    });
                });

                std::cout << "thread " << id_t << " working on " << working_id << std::endl;
                
            }

            

        }));
        id_t++;
    }
    std::cout << "main thread\n";

    std::for_each(workers.begin(), workers.end(), [](std::thread &t) 
    {
        t.join();
    });

    std::cout << "main thread synced\n";




    std::cout << "checking results\n";
    bool check = true;
    for (int i = 0; i < 100; i++) {
        for(u32 j = 0 ; j < s_p ; j++){
            check = check && (vec_to_up[i][j] == i*(j+1));
            //std::cout << i*(j+1) << " : " << vec_to_up[i][j] <<"\n";
        }
    }

    std::cout << "check : " << check << std::endl;

}

#endif

namespace walker{



    template<class Tpred>
    void walk(Tpred predicate){
        predicate(1,2);
    }

};

Test_start("testcpp::", test_lambda_walker, 1){
    walker::walk([&](int i, int j){

    });
}

Test_start("test_MPI_CUDA", test1, 2){

    sycl::queue & q = sycl_handler::get_compute_queue();

    u32 len_test = 10000;

    f32* ptr_send;
    f32* ptr_recv;
    
    if(mpi_handler::world_rank == 0){
        ptr_send = sycl::malloc_device<f32>(len_test,q);

        q.parallel_for(sycl::range<1>(len_test),[=] (sycl::item<1> item) {
            size_t id = item.get_linear_id();
            ptr_send[id] = id*0.2;
        });

        mpi::send(ptr_send, len_test,MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

        sycl::free(ptr_send,q);
    }

    if(mpi_handler::world_rank == 1){
        ptr_recv = sycl::malloc_device<f32>(len_test,q);

        MPI_Status st;
        mpi::recv(ptr_recv, len_test,MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &st);

        f32* res = new f32[len_test];


        q.memcpy(res,ptr_recv,sizeof(int)*len_test);
        q.wait();

        sycl::free(ptr_recv,q);

        for(u32 i = 0 ; i < len_test; i++){
            std::cout << res[i] << " ";
        }

        std::cout << std::endl;

        delete[] res;

    }



    
    

}


/*
#include <iostream>

template <typename ... Ts>
int Foo (int acc_in,Ts && ... multi_inputs)
{

    int acc = acc_in;

    ([&] (auto & input)
    {
        // Do things in your "loop" lambda

        acc += input(acc);

    } (multi_inputs), ...);

    return acc;
}

int main (int argc, char *argv[])
{
    int a = Foo(argc,[](int a){
        return a*2;
    },[](int a){
        return a*2;
    },[](int a){
        return a*2;
    },[](int a){
        return a*2;
    });

    std::cout << a << std::endl;
}
*/