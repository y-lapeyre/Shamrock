#include "shamrocktest.hpp"


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