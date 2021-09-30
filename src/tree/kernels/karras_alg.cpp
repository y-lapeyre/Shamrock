#include "karras_alg.hpp"
#include "../../aliases.hpp"


#define SGN(x) (x==0) ? 0 : ( (x>0) ? 1 : -1 )
#define DELTA( x,  y) ((y>morton_lenght-1 || y < 0) ? -1 : int(sycl::clz(m[x] ^ m[y])))

class Kernel_Karras_alg;

void sycl_karras_alg(
    sycl::queue* queue,
    u32 internal_cell_count,
    sycl::buffer<u_morton>* in_morton,
    sycl::buffer<u32>* out_buf_lchild_id   ,
    sycl::buffer<u32>* out_buf_rchild_id   ,
    sycl::buffer<u8 >* out_buf_lchild_flag ,
    sycl::buffer<u8 >* out_buf_rchild_flag,
    sycl::buffer<u32>* out_buf_endrange    ){

    cl::sycl::range<1> range_radix_tree{internal_cell_count};

    if(in_morton == NULL)           throw_with_pos("in_morton isn't allocated");
    if(out_buf_lchild_id == NULL)   throw_with_pos("out_buf_lchild_id isn't allocated");
    if(out_buf_rchild_id == NULL)   throw_with_pos("out_buf_rchild_id isn't allocated");
    if(out_buf_lchild_flag == NULL) throw_with_pos("out_buf_lchild_flag isn't allocated");
    if(out_buf_rchild_flag == NULL) throw_with_pos("out_buf_rchild_flag isn't allocated");
    if(out_buf_endrange == NULL)    throw_with_pos("out_buf_endrange isn't allocated");

    queue->submit(
        [&](cl::sycl::handler &cgh) {

            //@TODO add check if split count above 2G
            i32 morton_lenght = (i32) internal_cell_count+1;

            auto m = in_morton->get_access<sycl::access::mode::read>(cgh);
            
            auto lchild_id      = out_buf_lchild_id  ->get_access<sycl::access::mode::discard_write>(cgh);
            auto rchild_id      = out_buf_rchild_id  ->get_access<sycl::access::mode::discard_write>(cgh);
            auto lchild_flag    = out_buf_lchild_flag->get_access<sycl::access::mode::discard_write>(cgh);
            auto rchild_flag    = out_buf_rchild_flag->get_access<sycl::access::mode::discard_write>(cgh);
            auto end_range_cell = out_buf_endrange   ->get_access<sycl::access::mode::discard_write>(cgh);
            

            cgh.parallel_for<Kernel_Karras_alg>(range_radix_tree, [=](cl::sycl::item<1> item) {

                int i = (int) item.get_id(0);
                
                int ddelta = DELTA(i,i+1) - DELTA(i,i-1);
                
                int d = SGN(ddelta);
                
                //Compute upper bound for the lenght of the range
                int delta_min = DELTA(i,i-d);
                int lmax = 2;
                while(DELTA(i,i + lmax*d) > delta_min){
                    lmax *= 2;
                }
                
                //Find the other end using 
                int l = 0;
                int t = lmax/2;
                while(t > 0){
                    if(DELTA(i, i + (l + t)*d) > delta_min){
                        l = l + t;
                    }
                    t = t / 2;
                }
                int j = i + l*d;
                
                
                end_range_cell[i] = j;
                
                
                //Find the split position using binary search
                int delta_node = DELTA(i,j);
                int s= 0;

                //@todo why float
                float div = 2;
                t = sycl::ceil(l/div);
                while(true){
                    int tmp_ = i + (s + t)*d;
                    if(DELTA(i, tmp_) > delta_node){
                        s = s + t;
                    }
                    if(t <= 1) break;
                    div *= 2;
                    t = sycl::ceil(l/div);
                    
                }
                int gamma = i + s*d + sycl::min(d,0);
                
                if(sycl::min(i,j) == gamma){
                    lchild_id[i] = gamma;
                    lchild_flag[i] = 1; // leaf
                }else{
                    lchild_id[i] = gamma;
                    lchild_flag[i] = 0; // leaf
                }
                
                if(sycl::max(i,j) == gamma + 1){
                    rchild_id[i] = gamma + 1;
                    rchild_flag[i] = 1; // leaf
                }else{
                    rchild_id[i] = gamma + 1;
                    rchild_flag[i] = 0; // leaf
                }

                
            });


        }

    );
}