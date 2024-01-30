// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file HilbertLoadBalance.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief implementation of the hilbert curve load balancing
 *
 */

#include "HilbertLoadBalance.hpp"

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"



template<class Torder, class Tweight>
struct TileWithLoad{
    Torder ordering_val;
    Tweight load_value;
};

template<class Torder, class Tweight>
struct LoadBalancedTile{
    Torder ordering_val;
    Tweight load_value;
    Tweight accumulated_load_value;
    u64 index;
    i32 new_owner;

    LoadBalancedTile (TileWithLoad< Torder,  Tweight> in, u64 inindex): 
    ordering_val(in.ordering_val),
    load_value(in.load_value),
    index(inindex){

    }
};

/**
 * @brief load balance the input vector
 * 
 * @tparam Torder ordering value (hilbert, morton, ...)
 * @tparam Tweight weight type
 * @param lb_vector 
 * @return std::vector<i32> The new owner list
 */
template<class Torder, class Tweight>
inline std::vector<i32> load_balance(std::vector<TileWithLoad<Torder,Tweight>> && lb_vector){
    
    using LBTile = TileWithLoad<Torder,Tweight>;
    using LBTileResult = LoadBalancedTile<Torder,Tweight>;

    std::vector<LBTileResult> res;
    for (u64 i = 0; i < lb_vector.size(); i++) {
        res.push_back(LBTileResult{lb_vector[i],i});
    }

    // apply the ordering
    std::sort(res.begin(), res.end(), [](LBTileResult & left, LBTileResult & right){
        return left.ordering_val < right.ordering_val;
    });

    // compute increments for load
    u64 accum = 0;
    for (LBTileResult & tile : res) {
        u64 cur_val = tile.load_value;
        tile.accumulated_load_value = accum;
        accum += cur_val;
    }


    double target_datacnt = double(res[res.size() - 1].accumulated_load_value) /
                            shamcomm::world_size();

    i32 wsize = shamcomm::world_size();
    for (LBTileResult & tile : res) {
        tile.new_owner = sycl::clamp(i32(tile.accumulated_load_value / target_datacnt), 0, wsize - 1);
    }

    if (shamcomm::world_rank() == 0) {
        for (LBTileResult t : res) {
            logger::debug_ln("HilbertLoadBalance",
                                t.ordering_val,
                                t.accumulated_load_value,
                                t.index,
                                sycl::clamp(i32(t.accumulated_load_value / target_datacnt),
                                            0,
                                            i32(shamcomm::world_size()) - 1),
                                (t.accumulated_load_value / target_datacnt));
        }
    }

    std::vector<i32> new_owners(res.size());
    for (LBTileResult & tile : res) {
        new_owners[tile.index] = tile.new_owner;
    }
    
    return new_owners;

}


inline void apply_node_patch_packing(std::vector<shamrock::patch::Patch> &global_patch_list,std::vector<i32> &new_owner_table  ){
    using namespace shamrock::patch;
    sycl::buffer<i32> new_owner(new_owner_table.data(), new_owner_table.size());
    sycl::buffer<Patch> patch_buf(global_patch_list.data(), global_patch_list.size());

    sycl::range<1> range{global_patch_list.size()};

    // pack nodes
    shamsys::instance::get_alt_queue().submit([&](sycl::handler &cgh) {
        auto ptch = patch_buf.get_access<sycl::access::mode::read>(cgh);
        // auto pdt  = dt_buf.get_access<sycl::access::mode::read>(cgh);
        auto chosen_node = new_owner.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(range, [=](sycl::item<1> item) {
            u64 i = (u64)item.get_id(0);

            if (ptch[i].pack_node_index != u64_max) {
                chosen_node[i] = chosen_node[ptch[i].pack_node_index];
            }
        });
    }).wait();
}


namespace shamrock::scheduler {

    template<class T>
    class Compute_HilbLoad;
    template<class T>
    class Write_chosen_node;
    template<class T>
    class Edit_chosen_node;

    template<class hilbert_num>
    LoadBalancingChangeList HilbertLoadBalance<hilbert_num>::make_change_list(
        std::vector<shamrock::patch::Patch> &global_patch_list) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        //result
        LoadBalancingChangeList change_list;



        using Torder = hilbert_num;
        using Tweight = u64;
        using LBTile = TileWithLoad<Torder,Tweight>;
        using LBTileResult = LoadBalancedTile<Torder,Tweight>;

        // generate hilbert code, load value, and index before sort
        std::vector<LBTile> patch_dt(global_patch_list.size());

        {

            sycl::buffer<LBTile> dt_buf(patch_dt.data(),
                                                                   patch_dt.size());
            sycl::buffer<Patch> patch_buf(global_patch_list.data(), global_patch_list.size());

            for (u64 i = 0; i < global_patch_list.size(); i++) {

                Patch p = global_patch_list[i];

                patch_dt[i] = {
                    SFC::icoord_to_hilbert(p.coord_min[0], p.coord_min[1], p.coord_min[2]),
                    p.load_value};
            }
        }

        std::vector<i32> new_owner_table = load_balance(std::move(patch_dt));

        // apply patch packing in same node for merge
        apply_node_patch_packing(global_patch_list, new_owner_table);

        // make change list
        {
            std::vector<u64> load_per_node(shamcomm::world_size());

            std::vector<i32> tags_it_node(shamcomm::world_size());
            for (u64 i = 0; i < global_patch_list.size(); i++) {

                i32 old_owner = global_patch_list[i].node_owner_id;
                i32 new_owner = new_owner_table[i];

                // TODO add bool for optional print verbosity
                // std::cout << i << " : " << old_owner << " -> " << new_owner << std::endl;

                using ChangeOp = LoadBalancingChangeList::ChangeOp;

                ChangeOp op;
                op.patch_idx      = i;
                op.patch_id       = global_patch_list[i].id_patch;
                op.rank_owner_new = new_owner;
                op.rank_owner_old = old_owner;
                op.tag_comm       = tags_it_node[old_owner];

                if (new_owner != old_owner) {
                    change_list.change_ops.push_back(op);
                    tags_it_node[old_owner]++;
                }

                load_per_node[new_owner_table[i]] += global_patch_list[i].load_value;
            }

            //logger::debug_ln("HilbertLoadBalance", "loads after balancing");
            f64 min = shambase::VectorProperties<f64>::get_inf();
            f64 max = -shambase::VectorProperties<f64>::get_inf();
            f64 avg = 0;
            f64 var = 0;

            for (i32 nid = 0; nid < shamcomm::world_size(); nid++) {
                f64 val = load_per_node[nid];
                min = sycl::fmin(min, val);
                max = sycl::fmax(max, val);
                avg += val;
                
                logger::debug_ln("HilbertLoadBalance", nid, load_per_node[nid]);
            }
            avg /= shamcomm::world_size();
            for (i32 nid = 0; nid < shamcomm::world_size(); nid++) {
                f64 val = load_per_node[nid];
                var += (val - avg)*(val - avg);
            }
            var /= shamcomm::world_size();

            if(shamcomm::world_rank() == 0){
                std::string str = "Loadbalance stats : \n";
                str += shambase::format("    npatch = {}\n", global_patch_list.size() );
                str += shambase::format("    min = {}\n", min);
                str += shambase::format("    max = {}\n", max);
                str += shambase::format("    avg = {}\n", avg);
                str += shambase::format("    efficiency = {:.2f}%", 100 - (100*(max - min)/max));
                logger::info_ln("LoadBalance", str);
            }
        }

        return change_list;
    }

    template class HilbertLoadBalance<u64>;
    template class HilbertLoadBalance<shamrock::sfc::quad_hilbert_num>;

} // namespace shamrock::scheduler