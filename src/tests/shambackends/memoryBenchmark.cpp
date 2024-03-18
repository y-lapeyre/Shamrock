#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include "shambase/time.hpp"


template<class T, int ptrsep>
inline f64 get_bandwith(){

    constexpr u32 count = 1e9;
    constexpr u32 buf_len = count + ptrsep;

    sycl::queue & q = shamsys::instance::get_compute_queue();

    T* ptr_read = sycl::malloc_device<T>(buf_len,q);
    T* ptr_write = sycl::malloc_device<T>(buf_len,q);

    f64 duration_empty = shambase::timeitfor([&](){
        q.parallel_for(sycl::range<1>(count), [=](sycl::item<1> id){
            u64 gid = id.get_linear_id();
            u64 gidmul = (ptrsep+1)*gid % buf_len;

            u32 write_addr = gid;
            u32 read_addr = gidmul;

        }).wait();
    }, 2);

    f64 duration = shambase::timeitfor([&](){
        q.parallel_for(sycl::range<1>(count), [=](sycl::item<1> id){
            u64 gid = id.get_linear_id();

            u64 gidmul = (ptrsep+1)*gid % buf_len;

            u32 write_addr = gid;
            u32 read_addr = gidmul;
            
            ptr_write[write_addr] = ptr_read[read_addr];
        }).wait();
    }, 2);
    

    sycl::free(ptr_write, q);
    sycl::free(ptr_read, q);

    return 2*f64(count)*sizeof(T) / (duration - duration_empty);
}

TestStart(Benchmark, "memory-pointer-div-perf", memorypointerdivperf, 1) {

    using T = double;

    shamcomm::logs::raw_ln("sep =",1*sizeof(T),"   B = ", shambase::readable_sizeof(get_bandwith<T,1>()),"/s");
    shamcomm::logs::raw_ln("sep =",2*sizeof(T),"  B = ", shambase::readable_sizeof(get_bandwith<T,2>()),"/s");
    shamcomm::logs::raw_ln("sep =",4*sizeof(T),"  B = ", shambase::readable_sizeof(get_bandwith<T,4>()),"/s");
    shamcomm::logs::raw_ln("sep =",8*sizeof(T),"  B = ", shambase::readable_sizeof(get_bandwith<T,8>()),"/s");
    shamcomm::logs::raw_ln("sep =",16*sizeof(T)," B = ", shambase::readable_sizeof(get_bandwith<T,16>()),"/s");
    shamcomm::logs::raw_ln("sep =",32*sizeof(T)," B = ", shambase::readable_sizeof(get_bandwith<T,32>()),"/s");
    shamcomm::logs::raw_ln("sep =",64*sizeof(T)," B = ", shambase::readable_sizeof(get_bandwith<T,64>()),"/s");
    shamcomm::logs::raw_ln("sep =",128*sizeof(T),"B = ", shambase::readable_sizeof(get_bandwith<T,128>()),"/s");
    shamcomm::logs::raw_ln("sep =",256*sizeof(T),"B = ", shambase::readable_sizeof(get_bandwith<T,256>()),"/s");
    shamcomm::logs::raw_ln("sep =",512*sizeof(T),"B = ", shambase::readable_sizeof(get_bandwith<T,512>()),"/s");
    shamcomm::logs::raw_ln("sep =",1024*sizeof(T),"B = ", shambase::readable_sizeof(get_bandwith<T,1024>()),"/s");

}