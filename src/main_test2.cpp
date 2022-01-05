#include <CL/sycl.hpp>

#include "aliases.hpp"
#include "unittests/shamrocktest.hpp"






/*
Test_start(test_sycl,0){
    auto def_sel = sycl::default_selector();

    sycl::queue queue = sycl::queue(def_sel);

    float* rho = new float[10];
    sycl::buffer<float>* buf_rho = new sycl::buffer<float>(rho,10);

    queue.submit( [&](sycl::handler & cgh){
        auto rho = buf_rho->get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<class Compute_Flux>(
            sycl::range<1>(10), 
            [=](sycl::item<1> item){
                unsigned int i = item.get_linear_id();

                    rho[i] = i;

                }
        );

    });

    delete buf_rho;

    std::cout << rho[0] << " " << rho[1] << " " << rho[2] << " " << rho[3] << " " << rho[4] << std::endl;

    delete [] rho;
}
*/



Test_start(intmult,1){
    int a = 3;
    a*=2;
    Test_assert("int multiplication", a==6);
}

Test_start(intdiv,-1){
    int a = 6;
    a/=2;

    Test_assert("int division", a==3);
}

Test_start(multiple_asserts,1){

    int t[]{0,1,2,3};
    for (int i = 0; i<4; i++) {
        Test_assert("for loop assert", t[i]==i);
    }

}



Test_start(test_overload_new,2){

    int* a = new int(0);


    *a = 1;

    std::cout << *a << std::endl;



    delete a;

}

int main(int argc, char *argv[]){
    return run_all_tests(argc,argv);
}
