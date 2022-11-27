#include "core/algs/sycl/reduction/generic.hpp"
#include "core/sys/log.hpp"
#include "unittests/shamrocktest.hpp"

#include <random>
#include "core/algs/sycl/sycl_algs.hpp"
#include "core/utils/sycl_vector_utils.hpp"

template<class T,class Fct> void unit_test_reduc(std::string name, Fct && red_fct){
    std::vector<T> vals;

    constexpr u32 size_test = 2048;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f64> distf(0, 1);


    for(u32 i = 0; i < size_test; i++){
        vals.push_back(next_obj<T>(eng,distf));
    }

    T sycl_ret, check_val;

    {
        sycl::buffer<T> buf {vals.data(),vals.size()};

        sycl_ret = red_fct(sycl_handler::get_compute_queue(), buf, 0, size_test);

    }

    {
        check_val = T{0};
        for(auto & f : vals){
            check_val += f;
        }
    }

    shamrock::test::asserts().assert_float_equal(name, sycl::distance(sycl_ret ,check_val),0,1e-6);

}


template<class T,class Fct> f64 bench_reduction(Fct && red_fct, const u32 & size_test,std::vector<T> & vals){
    

    T sycl_ret;

    auto & q = sycl_handler::get_compute_queue();

    Timer t;

    {
        sycl::buffer<T> buf {vals.data(),size_test};

        q.submit([&](sycl::handler & cgh){
            sycl::accessor a {buf,sycl::read_only};
            cgh.single_task([](){});
        });

        q.wait();

        t.start();

        

        sycl_ret = red_fct(q, buf, 0, size_test);

        q.wait();

        t.end();

    }


    return t.nanosec*1e-9;

}







template<class T>
void unit_test_reduc(){
    struct Op{
        T operator()(const T & a, const T & b)
        {
            return a + b;
        }
    };

    unit_test_reduc<f64>("reduction : manual wg=32",syclalgs::reduction::impl::reduce_manual<f64,Op,32>);
    unit_test_reduc<f64>("reduction : sycl2020",syclalgs::reduction::impl::reduce_sycl_2020<f64,Op>);
    unit_test_reduc<f64>("reduction : main",syclalgs::reduction::reduce<f64,Op>);
}

TestStart(Unittest, "core/utils/sycl_algs:reduction", reduc_kernel_utest, 1){
    unit_test_reduc<f64>();
}










constexpr u32 lim_bench = 1e8;

template<u32 wg_size, class T>
void bench_mark_indiv_manual(std::string name,std::vector<T> & vals){

    struct Op{
            T operator()(const T & a, const T & b)
            {
                return a + b;
            }
        };

    logger::info_ln("ShamrockTest","testing :",name);
    
    std::vector<f64> test_sz;
    for(f64 i = 16; i < lim_bench; i*=1.3){
        test_sz.push_back(i);
    }

    auto & res = shamrock::test::test_data().new_dataset(name);

    std::vector<f64> results;

    for(const f64 & sz : test_sz){
        logger::debug_ln("ShamrockTest","N=",sz);
        results.push_back(bench_reduction<T>(syclalgs::reduction::impl::reduce_manual<T,Op,wg_size>,u32(sz),vals));
    }

    res.add_data("Nobj", test_sz);
    res.add_data("t_sort", results);
    
}

template<class T>
void bench_mark_indiv_sycl2020(std::string name,std::vector<T> & vals){

    struct Op{
            T operator()(const T & a, const T & b)
            {
                return a + b;
            }
        };

    logger::info_ln("ShamrockTest","testing :",name);
    
    std::vector<f64> test_sz;
    for(f64 i = 16; i < lim_bench; i*=1.3){
        test_sz.push_back(i);
    }

    auto & res = shamrock::test::test_data().new_dataset(name);

    std::vector<f64> results;

    for(const f64 & sz : test_sz){
        logger::debug_ln("ShamrockTest","N=",sz);
        results.push_back(bench_reduction<T>(syclalgs::reduction::impl::reduce_sycl_2020<T, Op>,u32(sz),vals));
    }

    res.add_data("Nobj", test_sz);
    res.add_data("t_sort", results);
    
}

template<class T> void bench_type(std::string Tname){

    std::vector<T> vals;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f64> distf(0, 1);

    for(u32 i = 0; i < lim_bench; i++){
        vals.push_back(next_obj<T>(eng,distf));
    }

    bench_mark_indiv_manual<2,T>(Tname + " manual : wg=2",vals);
    bench_mark_indiv_manual<4,T>(Tname + " manual : wg=4",vals);
    bench_mark_indiv_manual<8,T>(Tname + " manual : wg=8",vals);
    bench_mark_indiv_manual<16,T>(Tname + " manual : wg=16",vals);
    bench_mark_indiv_manual<32,T>(Tname + " manual : wg=32",vals);
    bench_mark_indiv_manual<64,T>(Tname + " manual : wg=64",vals);
    bench_mark_indiv_manual<128,T>(Tname + " manual : wg=128",vals);

    bench_mark_indiv_sycl2020<T>(Tname + " sycl2020 reduction",vals);
}

TestStart(Benchmark, "core/utils/sycl_algs:reduction", reduc_kernel_bench, 1){

    bench_type<f64>("f32");
    bench_type<f64>("f64");
}
