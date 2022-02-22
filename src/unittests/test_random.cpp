#include "shamrocktest.hpp"

#include "../sys/sycl_handler.hpp"
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
            
            cl::sycl::buffer<i32> fres(fres_arr);

            cl::sycl::range<1> range{10};

            queue->submit([&](cl::sycl::handler &cgh) {
                auto ff = fres.get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<class Write_chosen_node>(range, [=](cl::sycl::item<1> item) {
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

    init_sycl();

    Simulation<integrator::Leapfrog<ForcePressure>> sim;
    Simulation<integrator::Leapfrog<ForcePressure2>> sim2;


    sim.simu_main();


}