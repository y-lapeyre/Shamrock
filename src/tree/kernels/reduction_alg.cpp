#include "aliases.hpp"
#include "flags.hpp"

#include <vector>

class Kernel_generate_split_table;

void sycl_generate_split_table(
    sycl::queue & queue,
    u32 morton_count,
    sycl::buffer<u_morton>* buf_morton,
    sycl::buffer<u8>* buf_split_table
    ){

    sycl::range<1> range_morton_count{morton_count};

    queue.submit([&](sycl::handler &cgh) {

        auto m = buf_morton->get_access<sycl::access::mode::read>(cgh);
        auto split_out = buf_split_table->get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<Kernel_generate_split_table>(range_morton_count, [=](sycl::item<1> item) {

                u32 i = (u32) item.get_id(0);
    
                if(i>0){
                    if(m[i-1] != m[i]){
                        split_out[i] = true;
                    }else{
                        split_out[i] = false;
                    }
                }else{
                    split_out[i] = true;
                }

            }
        );

    });

}



#define DELTA( x,  y) (y>_morton_cnt-1) ? -1 : sycl::clz(m[x] ^ m[y]);

#define DELTA_host( x,  y) (y>_morton_cnt-1) ? -1 : __builtin_clz(m[x] ^ m[y]);

class Kernel_iterate_reduction;

void sycl_reduction_iteration(
    sycl::queue & queue,
    u32 morton_count,
    sycl::buffer<u_morton>* buf_morton,
    sycl::buffer<u8>* buf_split_table_in,
    sycl::buffer<u8>* buf_split_table_out
    ){

    sycl::range<1> range_morton_count{morton_count};

    queue.submit([&](sycl::handler &cgh) {

        u32 _morton_cnt = morton_count;

        auto m = buf_morton->get_access<sycl::access::mode::read>(cgh);
        auto split_in = buf_split_table_in->get_access<sycl::access::mode::read>(cgh);
        auto split_out = buf_split_table_out->get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<Kernel_iterate_reduction>(range_morton_count, [=](sycl::item<1> item) {

            int i = item.get_id(0);

            //find index of preceding i-1 non duplicate morton code
            uint before1 = i-1;
            while(before1 <= _morton_cnt-1 && !split_in[before1+1]) before1 --;
            
            //find index of preceding i-2 non duplicate morton code
            //safe bc delta(before1,before2) return -1 if any of the 2 are -1 because of order
            uint before2 = before1-1;
            while(before2 <= _morton_cnt-1 && !split_in[before2+1]) before2 --;
            
            //find index of next i+1 non duplicate morton code
            uint next1 = i+1;
            while(next1 <= _morton_cnt-1 && !split_in[next1]) next1 ++;

            #ifdef SYCL_COMP_DPCPP
            int delt_0 =  DELTA(i,next1);
            int delt_m =  DELTA(i,before1);
            int delt_mm = DELTA(before1,before2);
            #endif

            #ifdef SYCL_COMP_HIPSYCL
            __hipsycl_if_target_host(
                int delt_0 =  DELTA_host(i,next1);
                int delt_m =  DELTA_host(i,before1);
                int delt_mm = DELTA_host(before1,before2);
            )
            #endif

            if(!(delt_0 < delt_m && delt_mm < delt_m) && split_in[i]){
                split_out[i] = true;
            }else{
                split_out[i] = false;
            }

            
            
        });


    });

}






void reduction_alg(
    //in
    sycl::queue & queue,
    u32 morton_count,
    sycl::buffer<u_morton>* buf_morton,
    u32 reduction_level,
    //out
    std::vector<u32> & reduc_index_map,
    u32 & morton_leaf_count){


    
    sycl::buffer<u8>* buf_split_table1 = new sycl::buffer<u8>(morton_count);
    sycl::buffer<u8>* buf_split_table2 = new sycl::buffer<u8>(morton_count);

    sycl_generate_split_table(queue,morton_count,buf_morton,buf_split_table1);

    for(unsigned int iter = 1; iter <= reduction_level; iter ++){
        sycl::buffer<u8>* buf_split_t_in;
        sycl::buffer<u8>* buf_split_t_out;

        if(iter%2 == 0){
            buf_split_t_in  = buf_split_table2;
            buf_split_t_out = buf_split_table1;
        }else{
            buf_split_t_in  = buf_split_table1;
            buf_split_t_out = buf_split_table2;
        }

        sycl_reduction_iteration(queue, morton_count, buf_morton, buf_split_t_in, buf_split_t_out);
    }

    sycl::buffer<u8>* buf_split_table;
    if((reduction_level)%2 == 0){
        buf_split_table = buf_split_table1;
    }else{
        buf_split_table = buf_split_table2;
    }


    {
        auto acc = buf_split_table->get_access<sycl::access::mode::read>();        
        
        morton_leaf_count = 0;

        //reduc_index_map.reserve(split_count);
        for(unsigned int i = 0;i < morton_count;i++){
            if(acc[i]) {reduc_index_map.push_back(i); morton_leaf_count ++;}
        }
        reduc_index_map.push_back(morton_count);
        
    }

    delete buf_split_table1;
    delete buf_split_table2;

}



class Kernel_remap_morton_code;

void sycl_morton_remap_reduction(
    //in
    sycl::queue & queue,
    u32 morton_leaf_count,
    sycl::buffer<u32>* buf_reduc_index_map,
    sycl::buffer<u_morton>* buf_morton,
    //out
    sycl::buffer<u_morton>* buf_leaf_morton){
    sycl::range<1> range_remap_morton{morton_leaf_count};


    queue.submit([&](sycl::handler &cgh) {

        auto id_remaped = buf_reduc_index_map->get_access<sycl::access::mode::read>(cgh);
        auto m = buf_morton->get_access<sycl::access::mode::read>(cgh);
        auto m_remaped = buf_leaf_morton->get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<Kernel_remap_morton_code>(range_remap_morton, [=](sycl::item<1> item) {

            int i = item.get_id(0);

            m_remaped[i] = m[id_remaped[i]];
            
        });


    });

}


/*

void sycl_reduction_alg2(
    //in
    sycl::queue* queue,
    u32 morton_count,
    sycl::buffer<u_morton>* buf_morton,
    u32 reduction_level,
    //out
    std::vector<u32> & reduc_index_map,
    u32 & morton_leaf_count,
    sycl::buffer<u32>* & buf_reduc_index_map,
    sycl::buffer<u_morton>* & buf_leaf_morton){



    sycl::buffer<u8>* buf_split_table1 = new sycl::buffer<u8>(morton_count);
    sycl::buffer<u8>* buf_split_table2 = new sycl::buffer<u8>(morton_count);

    sycl_generate_split_table(queue,morton_count,buf_morton,buf_split_table1);

    for(unsigned int iter = 1; iter <= reduction_level; iter ++){
        sycl::buffer<u8>* buf_split_t_in;
        sycl::buffer<u8>* buf_split_t_out;

        if(iter%2 == 0){
            buf_split_t_in  = buf_split_table2;
            buf_split_t_out = buf_split_table1;
        }else{
            buf_split_t_in  = buf_split_table1;
            buf_split_t_out = buf_split_table2;
        }

        sycl_reduction_iteration(queue, morton_count, buf_morton, buf_split_t_in, buf_split_t_out);
    }

    sycl::buffer<u8>* buf_split_table;
    if((reduction_level)%2 == 0){
        buf_split_table = buf_split_table1;
    }else{
        buf_split_table = buf_split_table2;
    }


    {
        auto acc = buf_split_table->get_access<sycl::access::mode::read>();        
        
        morton_leaf_count = 0;

        //reduc_index_map.reserve(split_count);
        for(unsigned int i = 0;i < morton_count;i++){
            if(acc[i]) {reduc_index_map.push_back(i); morton_leaf_count ++;}
        }
        reduc_index_map.push_back(morton_count);
        
    }

    delete buf_split_table1;
    delete buf_split_table2;



    buf_reduc_index_map = new sycl::buffer<u32     >( reduc_index_map );
    buf_leaf_morton     = new sycl::buffer<u_morton>(morton_leaf_count);

    sycl::range<1> range_remap_morton{morton_leaf_count};

    auto ker_remap_morton = [&](sycl::handler &cgh) {

        auto id_remaped = buf_reduction_index_map->get_access<sycl::access::mode::read>(cgh);
        auto m = buf_morton->get_access<sycl::access::mode::read>(cgh);
        auto m_remaped = buf_reduced_morton->get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<Remap_morton>(range_remap_morton, [=](sycl::item<1> item) {

            int i = item.get_id(0);

            m_remaped[i] = m[id_remaped[i]];
            
        });


    };

    queue->submit(ker_remap_morton);
    log_debug("queue->submit(remap_morton);\n");


    reduction_factor = float(particles::npart) / split_count;

    log_debug("reduction factor : %f\n",reduction_factor);

    if(split_count < 2){
        log_warn("too few cells to use tree (%d) -> using one_cell_mode\n",split_count);
        one_cell_mode = true;
    }

 

}   
*/