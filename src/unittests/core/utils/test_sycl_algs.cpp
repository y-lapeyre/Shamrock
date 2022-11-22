#include "core/sys/log.hpp"
#include "unittests/shamrocktest.hpp"

#include <random>
#include "core/utils/sycl_algs.hpp"
#include "core/utils/sycl_vector_utils.hpp"

template<class T> void unit_test_reduc(){
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

        struct Op{
            T operator()(const T & a, const T & b)
            {
                return a + b;
            }
        };

        sycl_ret = syclalgs::reduction::reduce<T,Op,32>(sycl_handler::get_compute_queue(), buf, 0, size_test);

    }

    {
        check_val = T{0};
        for(auto & f : vals){
            check_val += f;
        }
    }

    shamrock::test::asserts().assert_float_equal("check reduction", sycl::distance(sycl_ret ,check_val),0,1e-6);

}



template<class T> f64 bench_reduction(const u32 & size_test){
    std::vector<T> vals;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f64> distf(0, 1);

    for(u32 i = 0; i < size_test; i++){
        vals.push_back(next_obj<T>(eng,distf));
    }

    T sycl_ret;

    auto & q = sycl_handler::get_compute_queue();

    Timer t;

    {
        sycl::buffer<T> buf {vals.data(),vals.size()};

        q.submit([&](sycl::handler & cgh){
            sycl::accessor a {buf,sycl::read_only};
        });

        q.wait();

        t.start();

        struct Op{
            T operator()(const T & a, const T & b)
            {
                return a + b;
            }
        };

        sycl_ret = syclalgs::reduction::reduce<T,Op,32>(q, buf, 0, size_test);

        q.wait();

        t.end();

    }


    return t.nanosec*1e-9;

}





TestStart(Unittest, "core/utils/sycl_algs:reduction", reduc_kernel_utest, 1){
    unit_test_reduc<f64>();
}


constexpr u32 lim_bench = 1e8;

template<u32 wg_size, class T>
void bench_mark_indiv(std::string name){

    logger::info_ln("ShamrockTest","testing :",name);
    
    std::vector<f64> test_sz;
    for(f64 i = 16; i < lim_bench; i*=1.3){
        test_sz.push_back(i);
    }

    auto & res = shamrock::test::test_data().new_dataset(name);

    std::vector<f64> results;

    for(const f64 & sz : test_sz){
        logger::debug_ln("ShamrockTest","N=",sz);
        results.push_back(bench_reduction<f64>(u32(sz)));
    }

    res.add_data("Nobj", test_sz);
    res.add_data("t_sort", results);
    
}

TestStart(Benchmark, "core/utils/sycl_algs:reduction", reduc_kernel_bench, 1){

    bench_mark_indiv<2,f64>("f64 wg=2");
    bench_mark_indiv<4,f64>("f64 wg=4");
    bench_mark_indiv<8,f64>("f64 wg=8");
    bench_mark_indiv<16,f64>("f64 wg=16");
    bench_mark_indiv<32,f64>("f64 wg=32");
    bench_mark_indiv<64,f64>("f64 wg=64");
    bench_mark_indiv<128,f64>("f64 wg=128");

    bench_mark_indiv<2,f32>("f32 wg=2");
    bench_mark_indiv<4,f32>("f32 wg=4");
    bench_mark_indiv<8,f32>("f32 wg=8");
    bench_mark_indiv<16,f32>("f32 wg=16");
    bench_mark_indiv<32,f32>("f32 wg=32");
    bench_mark_indiv<64,f32>("f32 wg=64");
    bench_mark_indiv<128,f32>("f32 wg=128");
}
