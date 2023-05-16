// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


#include "shamrock/legacy/patch/scheduler/scheduler_mpi.hpp"
#include "shamrock/legacy/patch/utility/serialpatchtree.hpp"
namespace shamrock {
    class ReattributeDataUtility{
        PatchScheduler &sched;

        public:
        ReattributeDataUtility(PatchScheduler &sched) : sched(sched) {}



        template<class T> shambase::DistributedData<sycl::buffer<u64>> compute_new_pid(SerialPatchTree<T> & sptree, u32 ipos){
            shambase::DistributedData<sycl::buffer<u64>> newid_buf_map;


            sched.patch_data.for_each_patchdata([&](u64 id, shamrock::patch::PatchData & pdat){
                if(! pdat.is_empty()){

                    PatchDataField<T> & pos_field =  pdat.get_field<T>(ipos);

                    newid_buf_map.add_obj(
                        id,
                        sptree.compute_patch_owner(
                            shamsys::instance::get_compute_queue(), 
                            shambase::get_check_ref(pos_field.get_buf()), pos_field.size()));

                    bool err_id_in_newid = false;
                    {
                        sycl::host_accessor nid {newid_buf_map.get(id), sycl::read_only};
                        for(u32 i = 0 ; i < pdat.get_obj_cnt() ; i++){
                            bool err = nid[i] == u64_max;
                            err_id_in_newid = err_id_in_newid || (err);
                        }
                    }

                    if(err_id_in_newid){
                        throw shambase::throw_with_loc<std::runtime_error>("a new id could not be computed");
                    }

                }
            });

            return newid_buf_map;
        }


        inline shambase::DistributedDataShared<shamrock::patch::PatchData> extract_elements(shambase::DistributedData<sycl::buffer<u64>> new_pid){
            shambase::DistributedDataShared<patch::PatchData> part_exchange;

            using namespace shamrock::patch;
            
            sched.patch_data.for_each_patchdata([&](u64 current_pid, shamrock::patch::PatchData & pdat){
                if(! pdat.is_empty()){
                    
                    sycl::host_accessor nid {new_pid.get(current_pid), sycl::read_only};
                    
                    const u32 cnt = pdat.get_obj_cnt();

                    for(u32 i = cnt-1 ; i < cnt ; i--){
                        u64 new_pid = nid[i];
                        if(current_pid != new_pid){
                            
                            if(! part_exchange.has_key(current_pid, new_pid)){
                                part_exchange.add_obj(current_pid, new_pid, PatchData(sched.pdl));
                            }

                            part_exchange.for_each([&](u64 _old_id, u64 _new_id, PatchData & pdat_int){
                                if(_old_id == current_pid && _new_id == new_pid){
                                    pdat.extract_element(i, pdat_int);
                                }
                            });

                        }
                            
                    }
                    
                }
            });

            return part_exchange;
        }

        template<class T>
        inline void reatribute_patch_objects(SerialPatchTree<T> & sptree, std::string position_field){StackEntry stack_loc{};

            using namespace shambase;
            using namespace shamrock::patch;
            
            u32 ipos = sched.pdl.get_field_idx<T>(position_field);

            shambase::DistributedData<sycl::buffer<u64>> new_pid = compute_new_pid(sptree,ipos);

            shambase::DistributedDataShared<patch::PatchData> part_exchange = extract_elements(new_pid);

            shamalgs::collective::SerializedDDataComm dcomm = part_exchange.map<std::unique_ptr<sycl::buffer<u8>>>([](u64 ,u64,PatchData & pdat){
                shamalgs::SerializeHelper ser;
                ser.allocate(pdat.serialize_buf_byte_size());
                pdat.serialize_buf(ser);
                return ser.finalize();
            });


        }

    };
}