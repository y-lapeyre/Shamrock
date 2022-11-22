#pragma once

#include "core/tree/radix_tree.hpp"
#include "interface_handler_impl_list.hpp"


//%Impl status : Clean unfinished





namespace impl{

    template<class flt>
    struct CommInd {
        using vec = sycl::vec<flt, 3>;

        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;

        vec receiver_box_min;
        vec receiver_box_max;
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

        template<class flt>
        struct GeneratorBuffer{

            using vec = sycl::vec<flt, 3>;

            const u64 local_pcount  ;
            const u64 global_pcount ;

            sycl::buffer<u64> patch_ids_buf;
            sycl::buffer<u64> global_ids_buf;
            sycl::buffer<vec> local_box_min_buf;
            sycl::buffer<vec> local_box_max_buf;

            sycl::buffer<vec> global_box_min_buf;
            sycl::buffer<vec> global_box_max_buf;

            explicit GeneratorBuffer(PatchScheduler & sched) : 
                local_pcount ( sched.patch_list.local.size()),
                global_pcount( sched.patch_list.global.size()),                
                patch_ids_buf(local_pcount),
                global_ids_buf(global_pcount),
                local_box_min_buf(local_pcount),
                local_box_max_buf(local_pcount),
                global_box_min_buf(global_pcount),
                global_box_max_buf(global_pcount){


                if (local_pcount == 0){
                    throw shamrock_exc("local patch count is zero this function can not run");
                }


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



            };

        };
        
        template<class flt,class InteractCd,class... Args>
        inline sycl::buffer<impl::CommInd<flt>, 2> compute_buf_interact(
            PatchScheduler &sched, GeneratorBuffer<flt> & gen,
            SerialPatchTree<sycl::vec<flt, 3>> & sptree, 
            sycl::vec<flt, 3> test_patch_offset, 
            const InteractCd & interact_crit, 
            Args ... args){

            using vec = sycl::vec<flt, 3>;

            const u64 & local_pcount  = gen.local_pcount;
            const u64 & global_pcount = gen.global_pcount;
                
            sycl::buffer<u64> & patch_ids_buf      = gen.patch_ids_buf     ;
            sycl::buffer<u64> & global_ids_buf     = gen.global_ids_buf    ;
            sycl::buffer<vec> & local_box_min_buf  = gen.local_box_min_buf ;
            sycl::buffer<vec> & local_box_max_buf  = gen.local_box_max_buf ;
            sycl::buffer<vec> & global_box_min_buf = gen.global_box_min_buf;
            sycl::buffer<vec> & global_box_max_buf = gen.global_box_max_buf;

            

            sycl::buffer<impl::CommInd<flt>, 2> interface_list_buf({local_pcount, global_pcount});
            


            

            //was used for smoothing lenght
            //sycl::buffer<flt> buf_local_field_val(pfield.local_nodes_value.data(),pfield.local_nodes_value.size());
            //sycl::buffer<flt> buf_global_field_val(pfield.global_values.data(),pfield.global_values.size());

            sycl_handler::get_alt_queue().submit([&](sycl::handler &cgh) {
                

                

                auto compute_interf = [&](auto ... acc_fields){

                    auto pid  = sycl::accessor{patch_ids_buf,cgh,sycl::read_only};
                    auto gpid = sycl::accessor{global_ids_buf,cgh,sycl::read_only};

                    auto lbox_min = sycl::accessor{local_box_min_buf,cgh,sycl::read_only};
                    auto lbox_max = sycl::accessor{local_box_max_buf,cgh,sycl::read_only};

                    auto gbox_min = sycl::accessor{global_box_min_buf,cgh,sycl::read_only};
                    auto gbox_max = sycl::accessor{global_box_max_buf,cgh,sycl::read_only};
                    
                    auto interface_list = interface_list_buf.template get_access<sycl::access::mode::discard_write>(cgh);

                    u64 cnt_patch = global_pcount;

                    vec offset = test_patch_offset;

                    bool is_off_not_bull = (offset.x() == 0) && (offset.y() == 0) && (offset.z() == 0);

                    InteractCd cd = interact_crit;

                    cgh.parallel_for(sycl::range<1>(local_pcount), [=](sycl::item<1> item) {
                        u64 cur_patch_idx    = (u64)item.get_id(0);
                        u64 cur_patch_id     = pid[cur_patch_idx];
                        vec cur_lbox_min = lbox_min[cur_patch_idx];
                        vec cur_lbox_max = lbox_max[cur_patch_idx];

                        u64 interface_ptr = 0;

                        for (u64 test_patch_idx = 0; test_patch_idx < cnt_patch; test_patch_idx++) {

                            //keep in mind that we compute patch that we have to send
                            //so we apply this offset on the patch we test against rather than ours
                            vec test_lbox_min = gbox_min[test_patch_idx] + offset;
                            vec test_lbox_max = gbox_max[test_patch_idx] + offset;
                            u64 test_patch_id     = gpid[test_patch_idx];

                            {


                                bool is_itself = ((!is_off_not_bull) || (test_patch_id != cur_patch_id));



                                bool int_crit ;


                                // check if us (cur_patch_id) : (patch) interact with any of the leafs of the (other) traget patch (eg test_patch_id)
                                // so the relation is : R(Sender, U receiver leaf)  aka interact_cd_cell_patch
                                // TODO interact_cd_cell_patch is confusing : cell <=> root cell of the patch => unclear
                                if (is_off_not_bull) {
                                    int_crit = InteractCd::interact_cd_cell_patch_outdomain(cd,
                                        cur_lbox_min, cur_lbox_max, test_lbox_min, test_lbox_max,
                                        acc_fields.get_local(cur_patch_idx)...,acc_fields.get_global(test_patch_idx)...
                                    );
                                }else{
                                    int_crit = InteractCd::interact_cd_cell_patch(cd,
                                        cur_lbox_min, cur_lbox_max, test_lbox_min, test_lbox_max,
                                        acc_fields.get_local(cur_patch_idx)...,acc_fields.get_global(test_patch_idx)...
                                    );
                                }                                

                                if (int_crit && is_itself) {
                                    interface_list[{cur_patch_idx, interface_ptr}] = impl::CommInd<flt>{
                                        cur_patch_idx, 
                                        test_patch_idx,
                                        cur_patch_id,
                                        test_patch_id,
                                        test_lbox_min,
                                        test_lbox_max
                                        };

                                    interface_ptr++;
                                }
                            }
                        }

                        if (interface_ptr < global_pcount) {
                            interface_list[{cur_patch_idx, interface_ptr}] = impl::CommInd<flt>{u64_max,u64_max,u64_max,u64_max,vec{0,0,0},vec{0,0,0}};
                        }
                    });
                };
                

                compute_interf(impl::pfield_convertion::accessed_pfield{args, cgh}...);


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
    using RadixTree = std::unique_ptr<Radix_Tree<u_morton, vec>>;
    using CutTree = typename Radix_Tree<u_morton, vec>::CuttedTree;

    //Store the result of a tree cut
    struct CommListing {
        u64 local_patch_idx_send;
        u64 global_patch_idx_recv;
        u64 sender_patch_id;
        u64 receiver_patch_id;

        vec applied_offset;
        i32_3 periodicity_vector;

        vec receiver_box_min;
        vec receiver_box_max;

        std::unique_ptr<CutTree> cutted_tree;
    };


    //contain the list of interface that this node should send
    std::vector<CommListing> interf_send_map;
    
    template<class InteractCrit,class... Args>
    inline void internal_compute_interf_list(PatchScheduler &sched, SerialPatchTree<vec> & sptree, SimulationDomain<flt> & bc,std::unordered_map<u64, RadixTree> & rtrees, const InteractCrit & interact_crit, Args ... args){

        const vec per_vec = bc.get_periodicity_vector();

        const u64 local_pcount  = sched.patch_list.local.size();
        const u64 global_pcount = sched.patch_list.global.size();


        logger::debug_ln("Interfacehandler", "computing interface list");

        impl::generator::GeneratorBuffer<flt> gen {sched};




        auto append_interface = [&](i32_3 periodicity_vec) -> auto {

            using namespace impl;


            //meaning in the interface we look at r |-> r + off
            //equivalent to our patch being moved r |-> r - off
            //keep in mind that we compute patch that we have to send
            //so we apply this offset on the patch we test against rather than ours
            vec off {
                per_vec.x() * periodicity_vec.x(),
                per_vec.y() * periodicity_vec.y(),
                per_vec.z() * periodicity_vec.z(),
            };

            sycl::buffer<CommInd<flt>, 2> cbuf = generator::compute_buf_interact(sched,gen, sptree, -off, interact_crit,args...);

            {
                auto interface_list = sycl::host_accessor {cbuf, sycl::read_only};

                for (u64 i = 0; i < local_pcount; i++) {
                    for (u64 j = 0; j < global_pcount; j++) {

                        if (interface_list[{i, j}].sender_patch_id == u64_max){
                            break;
                        }
                        CommInd tmp = interface_list[{i, j}];

                        CommListing tmp_push;
                        tmp_push.applied_offset = off;
                        tmp_push.periodicity_vector = periodicity_vec;
                        tmp_push.local_patch_idx_send  = tmp.local_patch_idx_send ;
                        tmp_push.global_patch_idx_recv = tmp.global_patch_idx_recv;
                        tmp_push.sender_patch_id       = tmp.sender_patch_id      ;
                        tmp_push.receiver_patch_id     = tmp.receiver_patch_id    ;

                        tmp_push.receiver_box_min      = tmp.receiver_box_min      ;
                        tmp_push.receiver_box_max      = tmp.receiver_box_max    ;

                        interf_send_map.push_back(std::move(tmp_push));

                    }
                }

            }

        };

        interf_send_map.clear();

        //TODO rethink this part to be able to use fixed bc for grid
        //probably one implementation of the whole thing for each boundary condition and then move user through

        if(bc.has_outdomain_object()){
            if (bc.periodic_search_min_vec.has_value() && bc.periodic_search_max_vec.has_value() ) {

                u32_3 min = bc.periodic_search_min_vec.value();
                u32_3 max = bc.periodic_search_max_vec.value();

                for(u32 x = min.x();  x < max.x(); x++){
                    for(u32 y = min.y(); y < max.y(); y++){
                        for(u32 z = min.z(); z < max.z(); z++){
                            append_interface({x,y,z});
                        }
                    }
                }
            }else{
                throw "Periodic search range not set";
            }
        }else{
            append_interface({0,0,0});
        }




        //then cutted make trees

        //Before impl this we have to code fullTreeFields (the local version (aka without interfaces)) //Tree cutter class allow more granularity over data 

        for(CommListing & comm : interf_send_map){

            auto & rtree = rtrees[comm.sender_patch_id];

            u32 total_count             = rtree.tree_internal_count + rtree.tree_leaf_count;
            sycl::range<1> range_tree{total_count};

            logger::debug_sycl_ln("Radixtree", "computing valid node buf");

            

            auto init_valid_buf = [&]() -> sycl::buffer<u8> {
                
                sycl::buffer<u8> valid_node = sycl::buffer<u8>(total_count);

                sycl::range<1> range_tree{total_count};

                sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor acc_valid_node{valid_node, cgh, sycl::write_only, sycl::no_init};

                    sycl::accessor acc_pos_cell_min{*rtree.buf_pos_min_cell_flt, cgh, sycl::read_only};
                    sycl::accessor acc_pos_cell_max{*rtree.buf_pos_max_cell_flt, cgh, sycl::read_only};

                    
                    InteractCrit cd = interact_crit;

                    vec test_lbox_min = comm.box_receiver_min;
                    vec test_lbox_max = comm.box_receiver_max;

                    u32 cur_patch_idx = comm.local_patch_idx_send;

                    u32 test_patch_idx = comm.global_patch_idx_send;

                    bool is_off_not_bull = (comm.applied_offset.x() == 0) && (comm.applied_offset.y() == 0) && (comm.applied_offset.z() == 0);


                    auto kernel = [&](auto ... interact_args){
                        cgh.parallel_for(range_tree, [=](sycl::item<1> item) {

                            auto cur_lbox_min = acc_pos_cell_min[item];
                            auto cur_lbox_max = acc_pos_cell_max[item];



                            bool int_crit ;

                            // check if us (cur_patch_id) : (patch) interact with any of the leafs of the (other) traget patch (eg test_patch_id)
                            // so the relation is : R(Sender, U receiver leaf)  aka interact_cd_cell_patch
                            if (is_off_not_bull) {
                                int_crit = InteractCrit::interact_cd_cell_patch_outdomain(cd,
                                    cur_lbox_min, cur_lbox_max, test_lbox_min, test_lbox_max,
                                    //acc_fields.get_local(cur_patch_idx)...,
                                    interact_args...
                                );
                                //Ok for sph and nobyd but we should implement it with a general tree
                                //TODO reimplement with tree field
                                // TODO here it shouldnt be the field of the patch
                            }else{
                                int_crit = InteractCrit::interact_cd_cell_patch(cd,
                                    cur_lbox_min, cur_lbox_max, test_lbox_min, test_lbox_max,
                                    //acc_fields.get_local(cur_patch_idx)...,
                                    interact_args...
                                );
                            } 

                            
                            acc_valid_node[item] = int_crit;
                        });
                    };

                    auto convert_to_vals = [&](auto ... acc_fields){
                        kernel(acc_fields.get_global(test_patch_idx)...);
                    };

                    

                    convert_to_vals(impl::pfield_convertion::accessed_pfield{args, cgh}...);

                });
                

                return valid_node;
            };

            
            //sender_patch_id -> receiver_patch_id

            //better if done using class i thinks

        }
        //u32 total_count = rtree.tree_internal_count + rtree.tree_leaf_count;
        //sycl::range<1> range_tree{total_count};


    }


    template<class T> struct field_extract_type{
        using type = void;
    };
    template<class T> struct field_extract_type<PatchField<T>>{
        using type = T;
    };

    


    public:

    template<class InteractCrit,class... Args>
    struct check{
        static constexpr bool has_patch_special_case = (std::is_same<decltype(InteractCrit::interact_cd_cell_patch),bool(vec,vec,vec,vec,field_extract_type<Args>... , field_extract_type<Args>...)>::value);
        static_assert(has_patch_special_case, "malformed call type should be bool(vec,vec,vec,vec,(types of the inputs field)...(types of the inputs field)...)");
    };

    // for now interact crit has shape (vec,vec) -> bool 
    // in order to pass for exemple h max we need a full tree field (patch field + radix tree field) 
    template<class InteractCrit,class... Args>
    inline void compute_interface_list(PatchScheduler &sched, SerialPatchTree<vec> & sptree, SimulationDomain<flt> & bc,std::unordered_map<u64, RadixTree> & rtrees, InteractCrit && interact_crit, Args & ... args){

        //check<InteractCrit,Args...>{};
        //constexpr bool has_patch_special_case = (std::is_same<decltype(InteractCrit::interact_cd_cell_patch),bool(field_extract_type<decltype(args)>...)>::value);
        //static_assert(has_patch_special_case, "special case must be written for this");//TODO better err msg

        internal_compute_interf_list(sched, sptree, bc,rtrees, interact_crit, impl::pfield_convertion::buffered_pfield{args} ...);

    }

    //TODO
    void initial_fetch();

    void fetch_field();


    template<class Function> void for_each_interface(u64 patch_id, Function && fct);
};



