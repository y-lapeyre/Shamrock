// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "BasicSPHGhosts.hpp"

using namespace shammodels::sph;





template<class vec>
auto BasicGasPeriodicGhostHandler<vec>::find_interfaces(
    SerialPatchTree<vec> &sptree,
    shamrock::patch::PatchtreeField<flt> &int_range_max_tree,
    shamrock::patch::PatchField<flt> &int_range_max) -> GeneratorMap {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shammath;

    i32 repetition_x = 1;
    i32 repetition_y = 1;
    i32 repetition_z = 1;

    PatchCoordTransform<vec> patch_coord_transf = sched.get_sim_box().get_patch_transform<vec>();
    vec bsize                                   = sched.get_sim_box().get_bounding_box_size<vec>();

    GeneratorMap interf_map;

    {
        sycl::host_accessor acc_tf{shambase::get_check_ref(int_range_max_tree.internal_buf),
                                   sycl::read_only};

        for (i32 xoff = -repetition_x; xoff <= repetition_x; xoff++) {
            for (i32 yoff = -repetition_y; yoff <= repetition_y; yoff++) {
                for (i32 zoff = -repetition_z; zoff <= repetition_z; zoff++) {

                    // sender translation
                    vec periodic_offset = vec{xoff * bsize.x(), yoff * bsize.y(), zoff * bsize.z()};

                    sched.for_each_local_patch([&](const Patch psender) {
                        CoordRange<vec> sender_bsize     = patch_coord_transf.to_obj_coord(psender);
                        CoordRange<vec> sender_bsize_off = sender_bsize.add_offset(periodic_offset);

                        flt sender_volume = sender_bsize.get_volume();

                        flt sender_h_max = int_range_max.get(psender.id_patch);

                        using PtNode = typename SerialPatchTree<vec>::PtNode;

                        sptree.host_for_each_leafs(
                            [&](u64 tree_id, PtNode n) {
                                flt receiv_h_max = acc_tf[tree_id];
                                CoordRange<vec> receiv_exp{n.box_min - receiv_h_max,
                                                           n.box_max + receiv_h_max};

                                return receiv_exp.get_intersect(sender_bsize_off).is_not_empty();
                            },
                            [&](u64 id_found, PtNode n) {
                                if ((id_found == psender.id_patch) && (xoff == 0) && (yoff == 0) &&
                                    (zoff == 0)) {
                                    return;
                                }

                                CoordRange<vec> receiv_exp =
                                    CoordRange<vec>{n.box_min, n.box_max}.expand_all(
                                        int_range_max.get(id_found));

                                CoordRange<vec> interf_volume = sender_bsize.get_intersect(
                                    receiv_exp.add_offset(-periodic_offset));

                                interf_map.add_obj(psender.id_patch,
                                                   id_found,
                                                   {periodic_offset,
                                                    interf_volume,
                                                    interf_volume.get_volume() / sender_volume});
                            });
                    });
                }
            }
        }
    }

    // interf_map.for_each([](u64 sender, u64 receiver, InterfaceBuildInfos build){
    //     logger::raw_ln("found interface
    //     :",sender,"->",receiver,"ratio:",build.volume_ratio,
    //     "volume:",build.cut_volume.lower,build.cut_volume.upper);
    // });

    return interf_map;
}






template class shammodels::sph::BasicGasPeriodicGhostHandler<f64_3>;