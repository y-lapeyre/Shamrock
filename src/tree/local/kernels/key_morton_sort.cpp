#include "key_morton_sort.hpp"
#include "aliases.hpp"
#include "flags.hpp"


//modified from http://www.bealto.com/gpu-sorting.html

#define MAXORDER_SORT_KERNEL 16


#define ORDER(a,b,ida,idb) { \
    bool swap = reverse ^ (a<b); \
    u_morton auxa = a; \
    u_morton auxb = b; \
    uint auxida = ida; \
    uint auxidb = idb; \
    a = (swap)?auxb:auxa; \
    b = (swap)?auxa:auxb; \
    ida = (swap)?auxidb:auxida; \
    idb = (swap)?auxida:auxidb; \
}

#define ORDERV(x,idx,a,b) { \
    bool swap = reverse ^ (x[a]<x[b]); \
    u_morton auxa = x[a];\
    u_morton auxb = x[b]; \
    uint auxida = idx[a]; \
    uint auxidb = idx[b]; \
    x[a] = (swap)?auxb:auxa;\
    x[b] = (swap)?auxa:auxb; \
    idx[a] = (swap)?auxidb:auxida; \
    idx[b] = (swap)?auxida:auxidb; \
}
#define B2V(x,idx,a) { ORDERV(x,idx,a,a+1) }
#define B4V(x,idx,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,idx,a+i4,a+i4+2) } B2V(x,idx,a) B2V(x,idx,a+2) }
#define B8V(x,idx,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,idx,a+i8,a+i8+4) } B4V(x,idx,a) B4V(x,idx,a+4) }
#define B16V(x,idx,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,idx,a+i16,a+i16+8) } B8V(x,idx,a) B8V(x,idx,a+8) }
#define B32V(x,idx,a) { for (int i32=0;i32<16;i32++) { ORDERV(x,idx,a+i32,a+i32+16) } B16V(x,idx,a) B16V(x,idx,a+16) }

class Bitonic_sort_B32;
class Bitonic_sort_B16;
class Bitonic_sort_B8;
class Bitonic_sort_B4;
class Bitonic_sort_B2;

void sycl_sort_morton_key_pair(
    sycl::queue & queue,
    u32 morton_count_rounded_pow,
    sycl::buffer<u32>*      buf_index,
    sycl::buffer<u_morton>* buf_morton
    ){

    for (unsigned int length=1;length<morton_count_rounded_pow;length<<=1){
        unsigned int inc = length;
        while (inc > 0){
            //log("inc : %d\n",inc);
            //int ninc = 1;
            unsigned int ninc = 0;


//B32 sort kernel is less performant than the B16 because of cache size
#if MAXORDER_SORT_KERNEL >= 32
            if (inc >= 16 && ninc == 0){
                ninc = 5;
                unsigned int nThreads = morton_count_rounded_pow >> ninc;
                cl::sycl::range<1> range{nThreads};

                auto ker_sort_morton_b32 = [&](cl::sycl::handler &cgh) {

                    auto m =  buf_morton->get_access<sycl::access::mode::read_write>(cgh);
                    auto id = buf_index->get_access<sycl::access::mode::read_write>(cgh);

                    cgh.parallel_for<Bitonic_sort_B32>(
                        range, [=](cl::sycl::item<1> item) {
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            u32 _inc = inc;
                            u32 _dir = length<<1;

                            _inc >>= 4;
                            int t = item.get_id(); // thread index
                            int low = t & (_inc - 1); // low order bits (below INC)
                            int i = ((t - low) << 5) + low; // insert 000 at position INC
                            bool reverse = ((_dir & i) == 0); // asc/desc order


                            // Load
                            u_morton x[32];
                            for (int k=0;k<32;k++) x[k] = m[k*_inc + i];

                            uint idx[32];
                            for (int k=0;k<32;k++) idx[k] = id[k*_inc + i];

                            // Sort
                            B32V(x,idx,0)

                            // Store
                            for (int k=0;k<32;k++) m[k*_inc + i] = x[k];
                            for (int k=0;k<32;k++) id[k*_inc + i] = idx[k];

                        }
                    );
                };queue->submit(ker_sort_morton_b32);

            }
#endif



#if MAXORDER_SORT_KERNEL >= 16
            if (inc >= 8 && ninc == 0){
                ninc = 4;
                unsigned int nThreads = morton_count_rounded_pow >> ninc;
                cl::sycl::range<1> range{nThreads};

                auto ker_sort_morton_b16 = [&](cl::sycl::handler &cgh) {

                    auto m =  buf_morton->get_access<sycl::access::mode::read_write>(cgh);
                    auto id = buf_index->get_access<sycl::access::mode::read_write>(cgh);

                    cgh.parallel_for<Bitonic_sort_B16>(
                        range, [=](cl::sycl::item<1> item) {
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            u32 _inc = inc;
                            u32 _dir = length<<1;

                            _inc >>= 3;
                            int t = item.get_id(0); // thread index
                            int low = t & (_inc - 1); // low order bits (below INC)
                            int i = ((t - low) << 4) + low; // insert 000 at position INC
                            bool reverse = ((_dir & i) == 0); // asc/desc order


                            // Load
                            u_morton x[16];
                            for (int k=0;k<16;k++) x[k] = m[k*_inc + i];

                            uint idx[16];
                            for (int k=0;k<16;k++) idx[k] = id[k*_inc + i];

                            // Sort
                            B16V(x,idx,0)

                            // Store
                            for (int k=0;k<16;k++) m[k*_inc + i] = x[k];
                            for (int k=0;k<16;k++) id[k*_inc + i] = idx[k];

                        }
                    );
                };queue.submit(ker_sort_morton_b16);

                //sort_kernel_B8(arg_eq,* buf_morton->buf,* particles::buf_ids->buf,inc,length<<1);//.wait();
            }
#endif



#if MAXORDER_SORT_KERNEL >= 8
            //B8
            if (inc >= 4 && ninc == 0){
                ninc = 3;
                unsigned int nThreads = morton_count_rounded_pow >> ninc;
                cl::sycl::range<1> range{nThreads};

                auto ker_sort_morton_b8 = [&](cl::sycl::handler &cgh) {

                    auto m =  buf_morton->get_access<sycl::access::mode::read_write>(cgh);
                    auto id = buf_index->get_access<sycl::access::mode::read_write>(cgh);

                    cgh.parallel_for<Bitonic_sort_B8>(
                        range, [=](cl::sycl::item<1> item) {
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            u32 _inc = inc;
                            u32 _dir = length<<1;

                            _inc >>= 2;
                            int t = item.get_id(0); // thread index
                            int low = t & (_inc - 1); // low order bits (below INC)
                            int i = ((t - low) << 3) + low; // insert 000 at position INC
                            bool reverse = ((_dir & i) == 0); // asc/desc order


                            // Load
                            u_morton x[8];
                            for (int k=0;k<8;k++) x[k] = m[k*_inc + i];

                            uint idx[8];
                            for (int k=0;k<8;k++) idx[k] = id[k*_inc + i];

                            // Sort
                            B8V(x,idx,0)

                            // Store
                            for (int k=0;k<8;k++) m[k*_inc + i] = x[k];
                            for (int k=0;k<8;k++) id[k*_inc + i] = idx[k];

                        }
                    );
                };queue.submit(ker_sort_morton_b8);

                //sort_kernel_B8(arg_eq,* buf_morton->buf,* particles::buf_ids->buf,inc,length<<1);//.wait();
            }
#endif

#if MAXORDER_SORT_KERNEL >= 4
            //B4
            if (inc >= 2 && ninc == 0){
                ninc = 2;
                unsigned int nThreads = morton_count_rounded_pow >> ninc;
                cl::sycl::range<1> range{nThreads};
                //sort_kernel_B4(arg_eq,* buf_morton->buf,* particles::buf_ids->buf,inc,length<<1);
                auto ker_sort_morton_b4 = [&](cl::sycl::handler &cgh) {

                    auto m =  buf_morton->get_access<sycl::access::mode::read_write>(cgh);
                    auto id = buf_index->get_access<sycl::access::mode::read_write>(cgh);
                    cgh.parallel_for<Bitonic_sort_B4>(
                        range, [=](cl::sycl::item<1> item) {
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            u32 _inc = inc;
                            u32 _dir = length<<1;

                            _inc >>= 1;
                            int t = item.get_id(0); // thread index
                            int low = t & (_inc - 1); // low order bits (below INC)
                            int i = ((t - low) << 2) + low; // insert 00 at position INC
                            bool reverse = ((_dir & i) == 0); // asc/desc order
                            

                            // Load
                            u_morton x0 = m[    0  + i];
                            u_morton x1 = m[  _inc + i];
                            u_morton x2 = m[2*_inc + i];
                            u_morton x3 = m[3*_inc + i];

                            uint idx0 = id[    0  + i];
                            uint idx1 = id[  _inc + i];
                            uint idx2 = id[2*_inc + i];
                            uint idx3 = id[3*_inc + i];

                            // Sort
                            ORDER(x0,x2,idx0,idx2)
                            ORDER(x1,x3,idx1,idx3)
                            ORDER(x0,x1,idx0,idx1)
                            ORDER(x2,x3,idx2,idx3)

                            // Store
                            m[    0  + i] = x0;
                            m[  _inc + i] = x1;
                            m[2*_inc + i] = x2;
                            m[3*_inc + i] = x3;

                            id[    0  + i] = idx0;
                            id[  _inc + i] = idx1;
                            id[2*_inc + i] = idx2;
                            id[3*_inc + i] = idx3;

                        }
                    );
                };queue.submit(ker_sort_morton_b4);
            }
#endif
            

            //B2
            if (ninc == 0){
                ninc = 1;
                unsigned int nThreads = morton_count_rounded_pow >> ninc;
                cl::sycl::range<1> range{nThreads};
                //sort_kernel_B2(arg_eq,* buf_morton->buf,* particles::buf_ids->buf,inc,length<<1);
                auto ker_sort_morton_b2 = [&](cl::sycl::handler &cgh) {

                    auto m =  buf_morton->get_access<sycl::access::mode::read_write>(cgh);
                    auto id = buf_index->get_access<sycl::access::mode::read_write>(cgh);

                    cgh.parallel_for<Bitonic_sort_B2>(
                        range, [=](cl::sycl::item<1> item) {
                            //(__global data_t * data,__global uint * ids,int inc,int dir)

                            u32 _inc = inc;
                            u32 _dir = length<<1;

                            int t = item.get_id(0); // thread index
                            int low = t & (_inc - 1); // low order bits (below INC)
                            int i = (t<<1) - low; // insert 0 at position INC
                            bool reverse = ((_dir & i) == 0); // asc/desc order
                            
                            
                            // Load
                            u_morton x0 = m[  0  + i];
                            u_morton x1 = m[_inc + i];
                            uint idx0 = id[  0  + i];
                            uint idx1 = id[_inc + i];

                            // Sort
                            ORDER(x0,x1,idx0,idx1)

                            // Store
                            m[0     + i] = x0;
                            m[_inc  + i] = x1;
                            id[0    + i] = idx0;
                            id[_inc + i] = idx1;

                        }
                    );
                };queue.submit(ker_sort_morton_b2);
            }
            
            
            inc >>= ninc;
        }
    }


}