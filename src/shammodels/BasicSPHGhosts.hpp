// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/DistributedData.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/BasicGas.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"

namespace shammodels::sph {

    template<class vec>
    class BasicGasPeriodicGhostHandler {

        PatchScheduler &sched;

        public:
        using flt = shambase::VecComponent<vec>;

        struct InterfaceBuildInfos {
            vec offset;
            shammath::CoordRange<vec> cut_volume;
            flt volume_ratio;
        };

        using GeneratorMap = shambase::DistributedDataShared<InterfaceBuildInfos>;

        BasicGasPeriodicGhostHandler(PatchScheduler &sched) : sched(sched) {}

        GeneratorMap find_interfaces(SerialPatchTree<vec> &sptree,
                                     shamrock::patch::PatchtreeField<flt> &int_range_max_tree,
                                     shamrock::patch::PatchField<flt> &int_range_max);




        struct InterfaceIdTable {
            InterfaceBuildInfos build_infos;
            std::unique_ptr<sycl::buffer<u32>> ids_interf;
            f64 part_cnt_ratio;
        };

        shambase::DistributedDataShared<InterfaceIdTable>
        gen_id_table_interfaces(GeneratorMap &&gen) {
            StackEntry stack_loc{};
            using namespace shamrock::patch;

            shambase::DistributedDataShared<InterfaceIdTable> res;

            std::map<u64, f64> send_count_stats;

            gen.for_each([&](u64 sender, u64 receiver, InterfaceBuildInfos &build) {
                shamrock::patch::PatchData &src = sched.patch_data.get_pdat(sender);
                PatchDataField<vec> &xyz        = src.get_field<vec>(0);

                std::unique_ptr<sycl::buffer<u32>> idxs = xyz.get_elements_with_range_buf(
                    [&](vec val, vec vmin, vec vmax) {
                        return Patch::is_in_patch_converted(val, vmin, vmax);
                    },
                    build.cut_volume.lower,
                    build.cut_volume.upper);

                u32 pcnt = 0;
                if (bool(idxs)) {
                    pcnt = idxs->size();
                }

                // prevent sending empty patches
                if (pcnt == 0) {
                    return;
                }

                f64 ratio = f64(pcnt) / f64(src.get_obj_cnt());

                logger::debug_sycl_ln("InterfaceGen",
                                      "gen interface :",
                                      sender,
                                      "->",
                                      receiver,
                                      "volume ratio:",
                                      build.volume_ratio,
                                      "part_ratio:",
                                      ratio);

                res.add_obj(sender, receiver, InterfaceIdTable{build, std::move(idxs), ratio});

                send_count_stats[sender] += ratio;
            });

            bool has_warn = false;

            for (auto &[k, v] : send_count_stats) {
                if (v > 0.2) {
                    logger::warn_ln("InterfaceGen", "patch", k, " high ratio volume/interf:", v);
                    has_warn = true;
                }
            }

            if (has_warn) {
                logger::warn_ln("InterfaceGen",
                                "the ratio patch/interface is high, which can lead to high mpi "
                                "overhead, try incresing the patch split crit");
            }

            return res;
        }

        shambase::DistributedDataShared<InterfaceIdTable>
        make_interface_cache(SerialPatchTree<vec> &sptree,
                             shamrock::patch::PatchtreeField<flt> &int_range_max_tree,
                             shamrock::patch::PatchField<flt> &int_range_max) {
            StackEntry stack_loc{};

            return gen_id_table_interfaces(
                find_interfaces(sptree, int_range_max_tree, int_range_max));
        }

        struct PositionInterface {
            PatchDataField<vec> position_field;
            vec bmin;
            vec bmax;
        };

        inline shambase::DistributedDataShared<PositionInterface>
        build_position_interf_field(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            StackEntry stack_loc{};

            // clang-format off
            return builder.template map<PositionInterface>([&](u64 sender, u64 receiver, InterfaceIdTable &build_table) {
                if (!bool(build_table.ids_interf)) {
                    throw shambase::throw_with_loc<std::runtime_error>(
                        "their is an empty id table in the interface, it should have been removed");
                }

                PatchDataField<vec> pfield = sched
                    .patch_data
                    .get_pdat(sender)
                    .get_field<vec>(0)
                    .make_new_from_subset(*build_table.ids_interf, build_table.ids_interf->size());

                auto &binfo = build_table.build_infos;

                pfield.apply_offset(binfo.offset);

                vec bmin = binfo.cut_volume.lower + binfo.offset;
                vec bmax = binfo.cut_volume.upper + binfo.offset;

                return PositionInterface{std::move(pfield), bmin, bmax};
            });
            // clang-format on
        }

        inline shambase::DistributedDataShared<PositionInterface>
        communicate_positions(shambase::DistributedDataShared<PositionInterface> &interf) {
            StackEntry stack_loc{};

            shamalgs::collective::SerializedDDataComm dcomm_send =
                interf.template map<std::unique_ptr<sycl::buffer<u8>>>(
                    [](u64, u64, PositionInterface &pos_interf) {
                        shamalgs::SerializeHelper ser;

                        u64 size = pos_interf.position_field.serialize_buf_byte_size();
                        size += 2 * shamalgs::SerializeHelper::serialize_byte_size<vec>();

                        ser.allocate(size);

                        pos_interf.position_field.serialize_buf(ser);
                        ser.write(pos_interf.bmin);
                        ser.write(pos_interf.bmax);

                        return ser.finalize();
                    });

            shamalgs::collective::SerializedDDataComm dcomm_recv;

            // ISSUE : to much comm to itself
            shamalgs::collective::distributed_data_sparse_comm(
                dcomm_send, dcomm_recv, shamsys::DirectGPU, [&](u64 id) {
                    return sched.get_patch_rank_owner(id);
                });

            shambase::DistributedDataShared<PositionInterface> recv_dat;
            {StackEntry stack_loc{};
                recv_dat =
                    dcomm_recv.map<PositionInterface>(
                        [&](u64, u64, std::unique_ptr<sycl::buffer<u8>> &buf) {
                            // exchange the buffer held by the distrib data and give it to the
                            // serializer
                            shamalgs::SerializeHelper ser(
                                std::exchange(buf, std::unique_ptr<sycl::buffer<u8>>{}));

                            PatchDataField<vec> f = PatchDataField<vec>::deserialize_buf(ser, "xyz", 1);
                            vec bmin, bmax;
                            ser.load(bmin);
                            ser.load(bmax);

                            return PositionInterface{std::move(f), bmin, bmax};
                        });
            }
            return recv_dat;
        }

        inline shambase::DistributedDataShared<PositionInterface>
        build_communicate_positions(shambase::DistributedDataShared<InterfaceIdTable> &builder) {
            auto pos_interf = build_position_interf_field(builder);
            return communicate_positions(pos_interf);
        }

        inline shambase::DistributedData<shamrock::MergedPatchDataField<vec>> merge_position_buf(shambase::DistributedDataShared<PositionInterface> && positioninterfs){
            StackEntry stack_loc{};

            shambase::DistributedData<shamrock::MergedPatchDataField<vec>> pos_fields;

            sched.for_each_patchdata_nonempty([&] (const shamrock::patch::Patch p, shamrock::patch::PatchData & pdat){
                PatchDataField<vec> & pos = pdat.get_field<vec>(0);
                vec bmax = pos.compute_max();
                vec bmin = pos.compute_min();
                u32 or_elem = pos.get_obj_cnt();
                PatchDataField<vec> new_pos = pos.duplicate();


                u32 total_elements = or_elem;

                positioninterfs.for_each([&](u64 sender, u64 receiver, PositionInterface & pint){
                    if(receiver == p.id_patch){
                        total_elements += pint.position_field.get_obj_cnt();
                        new_pos.insert(pint.position_field);
                        bmax = shambase::sycl_utils::g_sycl_max(bmax, pint.bmax);
                        bmin = shambase::sycl_utils::g_sycl_min(bmin, pint.bmin);
                    }
                });

                logger::debug_ln("sph::Interface", "merging :",or_elem,"->",total_elements);


                pos_fields.add_obj(p.id_patch, shamrock::MergedPatchDataField<vec>{
                    shammath::CoordRange<vec>{bmin,bmax},
                    or_elem,
                    total_elements,
                    std::move(new_pos)
                });

            });

            return pos_fields;

        }
    };

} // namespace shammodels::sph