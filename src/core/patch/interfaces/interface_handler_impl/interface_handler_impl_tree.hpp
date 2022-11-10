#pragma once

#include "core/tree/radix_tree.hpp"
#include "interface_handler_impl_list.hpp"


//%Impl status : Clean unfinished





namespace impl{

    struct CommInd {
        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;
    };

    namespace pfield_convertion {

        template<class T>
        struct buffered_pfield{
            sycl::buffer<T> local_vals;
            sycl::buffer<T> global_vals;

            inline explicit buffered_pfield(PatchField<T> & pf) :
                local_vals(pf.local_nodes_value.data(),pf.local_nodes_value.size()),
                global_vals(pf.global_values.data(),pf.global_values.size())
            {}
        };

        template<class T>
        struct accessed_pfield{
            sycl::accessor<T ,1,sycl::access::mode::read,sycl::target::device> acc_loc;
            sycl::accessor<T ,1,sycl::access::mode::read,sycl::target::device> acc_glo;


            inline accessed_pfield(buffered_pfield<T> & pf, sycl::handler & cgh) :
                acc_loc(sycl::accessor{pf.local_vals,cgh,sycl::read_only}),
                acc_glo(sycl::accessor{pf.global_vals,cgh,sycl::read_only})
            {}

            inline T get_local(u32 i) const {
                return acc_loc[i];
            }

            inline T get_global(u32 i) const {
                return acc_glo[i];
            }
        };

    }

    namespace generator {
        
        template<class flt,class Func_interactcrit,class... Args>
        inline sycl::buffer<impl::CommInd, 2> compute_buf_interact(PatchScheduler &sched, SerialPatchTree<sycl::vec<flt, 3>> & sptree, sycl::vec<flt, 3> interf_offset, Func_interactcrit && interact_crit, Args ... args){

            using vec = sycl::vec<flt, 3>;

            const u64 local_pcount  = sched.patch_list.local.size();
            const u64 global_pcount = sched.patch_list.global.size();

            if (local_pcount == 0){
                throw shamrock_exc("local patch count is zero this function can not run");
            }
                
            sycl::buffer<u64> patch_ids_buf(local_pcount);
            sycl::buffer<vec> local_box_min_buf(local_pcount);
            sycl::buffer<vec> local_box_max_buf(local_pcount);

            sycl::buffer<vec> global_box_min_buf(global_pcount);
            sycl::buffer<vec> global_box_max_buf(global_pcount);

            

            sycl::buffer<impl::CommInd, 2> interface_list_buf({local_pcount, global_pcount});
            sycl::buffer<u64> global_ids_buf(global_pcount);

            //compute interfaces in float space
            {
                auto pid      = patch_ids_buf.get_access<sycl::access::mode::discard_write>();
                auto lbox_min = local_box_min_buf.template get_access<sycl::access::mode::discard_write>();
                auto lbox_max = local_box_max_buf.template get_access<sycl::access::mode::discard_write>();

                auto gbox_min = global_box_min_buf.template get_access<sycl::access::mode::discard_write>();
                auto gbox_max = global_box_max_buf.template get_access<sycl::access::mode::discard_write>();

                std::tuple<vec, vec> box_transform = sched.get_box_tranform<vec>();

                for (u64 i = 0; i < local_pcount; i++) {
                    pid[i] = sched.patch_list.local[i].id_patch;

                    lbox_min[i] = vec{sched.patch_list.local[i].x_min, sched.patch_list.local[i].y_min,
                                        sched.patch_list.local[i].z_min} *
                                    std::get<1>(box_transform) +
                                std::get<0>(box_transform);
                    lbox_max[i] = (vec{sched.patch_list.local[i].x_max, sched.patch_list.local[i].y_max,
                                        sched.patch_list.local[i].z_max} +
                                1) *
                                    std::get<1>(box_transform) +
                                std::get<0>(box_transform);
                }

                auto g_pid = global_ids_buf.get_access<sycl::access::mode::discard_write>();
                for (u64 i = 0; i < global_pcount; i++) {
                    g_pid[i] = sched.patch_list.global[i].id_patch;

                    gbox_min[i] = vec{sched.patch_list.global[i].x_min, sched.patch_list.global[i].y_min,
                                        sched.patch_list.global[i].z_min} *
                                    std::get<1>(box_transform) +
                                std::get<0>(box_transform); 
                    gbox_max[i] = (vec{sched.patch_list.global[i].x_max, sched.patch_list.global[i].y_max,
                                        sched.patch_list.global[i].z_max} +
                                1) *
                                    std::get<1>(box_transform) +
                                std::get<0>(box_transform); 
                }
            }

            

            //was used for smoothing lenght
            //sycl::buffer<flt> buf_local_field_val(pfield.local_nodes_value.data(),pfield.local_nodes_value.size());
            //sycl::buffer<flt> buf_global_field_val(pfield.global_values.data(),pfield.global_values.size());

            sycl_handler::get_alt_queue().submit([&](sycl::handler &cgh) {
                

                

                auto compute_interf = [&](auto && inter_crit, auto&& ... acc_fields){

                    auto pid  = sycl::accessor{patch_ids_buf,cgh,sycl::read_only};
                    auto gpid = sycl::accessor{global_ids_buf,cgh,sycl::read_only};

                    auto lbox_min = sycl::accessor{local_box_min_buf,cgh,sycl::read_only};
                    auto lbox_max = sycl::accessor{local_box_max_buf,cgh,sycl::read_only};

                    auto gbox_min = sycl::accessor{global_box_min_buf,cgh,sycl::read_only};
                    auto gbox_max = sycl::accessor{global_box_max_buf,cgh,sycl::read_only};
                    
                    auto interface_list = interface_list_buf.template get_access<sycl::access::mode::discard_write>(cgh);

                    u64 cnt_patch = global_pcount;

                    vec offset = -interf_offset;

                    bool is_off_not_bull = (offset.x() == 0) && (offset.y() == 0) && (offset.z() == 0);

                    cgh.parallel_for(sycl::range<1>(local_pcount), [=](sycl::item<1> item) {
                        u64 cur_patch_idx    = (u64)item.get_id(0);
                        u64 cur_patch_id     = pid[cur_patch_idx];
                        vec cur_lbox_min = lbox_min[cur_patch_idx];
                        vec cur_lbox_max = lbox_max[cur_patch_idx];

                        u64 interface_ptr = 0;

                        for (u64 test_patch_idx = 0; test_patch_idx < cnt_patch; test_patch_idx++) {

                            vec test_lbox_min = gbox_min[test_patch_idx];
                            vec test_lbox_max = gbox_max[test_patch_idx];
                            u64 test_patch_id     = gpid[test_patch_idx];

                            {


                                bool is_itself = ((!is_off_not_bull) || (test_patch_id != cur_patch_id));

                                bool int_crit = inter_crit(
                                    cur_lbox_min, cur_lbox_max, test_lbox_min, test_lbox_max,
                                    acc_fields.get_local(cur_patch_idx)...,acc_fields.get_global(test_patch_idx)...
                                    );

                                if (int_crit && is_itself) {
                                    interface_list[{cur_patch_idx, interface_ptr}] = impl::CommInd{
                                        cur_patch_idx, 
                                        test_patch_idx,
                                        cur_patch_id,
                                        test_patch_id
                                        };

                                    interface_ptr++;
                                }
                            }
                        }

                        if (interface_ptr < global_pcount) {
                            interface_list[{cur_patch_idx, interface_ptr}] = impl::CommInd{u64_max,u64_max,u64_max,u64_max};
                        }
                    });
                };
                

                compute_interf(interact_crit, impl::pfield_convertion::accessed_pfield{args, cgh}...);


            });

            return std::move(interface_list_buf);
        }

    }

}








template<class pos_prec, class u_morton> 
class Interfacehandler<Tree_Send,pos_prec,Radix_Tree<u_morton, sycl::vec<pos_prec, 3>>>{

    

    public:

    using flt = pos_prec;
    using vec = sycl::vec<flt, 3>;

    private : 

    using CutTree = typename Radix_Tree<u_morton, vec>::CuttedTree;

    //Store the result of a tree cut
    struct CommListing {
        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;

        vec applied_offset;
        u32_3 periodicity_vector;

        CutTree cutted_tree;
    };


    

    
    template<class Func_interactcrit,class... Args>
    inline void internal_compute_interf_list(PatchScheduler &sched, SerialPatchTree<vec> & sptree, SimulationDomain<flt> & bc, Func_interactcrit && interact_crit, Args ... args){

        vec per_vec = bc.get_periodicity_vector();


        logger::debug_ln("Interfacehandler", "computing interface list");

        

        auto append_interface = [&](u32_3 periodicity_vec) -> sycl::buffer<impl::CommInd, 2> {
            vec off {
                per_vec.x() * periodicity_vec.x(),
                per_vec.y() * periodicity_vec.y(),
                per_vec.z() * periodicity_vec.z(),
            };
            return impl::generator::compute_buf_interact(sched, sptree, off, interact_crit, args...);
        };

        //now convert to vectors and group them

        //then make trees
    }


    public:

    // for now interact crit has shape (vec,vec) -> bool 
    // in order to pass for exemple h max we need a full tree field (patch field + radix tree field) 
    template<class Func_interactcrit,class... Args>
    inline void compute_interface_list(PatchScheduler &sched, SerialPatchTree<vec> & sptree, SimulationDomain<flt> & bc, Func_interactcrit && interact_crit, Args & ... args){

        internal_compute_interf_list(sched, sptree, bc, interact_crit, impl::pfield_convertion::buffered_pfield{args}...);

    }

    //TODO
    void initial_fetch();

    void fetch_field();


    template<class Function> void for_each_interface(u64 patch_id, Function && fct);
};



