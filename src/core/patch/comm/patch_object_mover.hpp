// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "core/patch/base/patchdata.hpp"
#include "patchdata_exchanger.hpp"
#include "core/patch/base/patchdata_field.hpp"
#include "core/patch/utility/serialpatchtree.hpp"
#include "patch_content_exchanger.hpp"
#include "core/patch/scheduler/scheduler_mpi.hpp"
#include "core/sys/sycl_handler.hpp"
#include "core/utils/sycl_vector_utils.hpp"
#include <unordered_map>


template<class vecprec> 
inline std::unordered_map<u64, sycl::buffer<u64>> get_new_id_map(PatchScheduler & sched, SerialPatchTree<vecprec> & sptree);

template<> 
inline std::unordered_map<u64, sycl::buffer<u64>> get_new_id_map<f32_3>(PatchScheduler & sched, SerialPatchTree<f32_3> & sptree){

    

    std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;

    for(auto & [id,pdat] : sched.patch_data.owned_data ){
        if(! pdat.is_empty()){


            u32 ixyz = sched.pdl.get_field_idx<f32_3>("xyz");
            PatchDataField<f32_3> xyz_field =  pdat.fields_f32_3[ixyz];

            std::unique_ptr<sycl::buffer<f32_3>> pos = std::make_unique<sycl::buffer<f32_3>>(xyz_field.usm_data(),xyz_field.size());

            newid_buf_map.insert({
                id,
                __compute_object_patch_owner<f32_3, class ComputeObejctPatchOwners_f32>(
                    sycl_handler::get_compute_queue(), 
                    *pos, 
                    sptree)});

            pos.reset();

        }
        
    }

    return newid_buf_map;

}



template<> 
inline std::unordered_map<u64, sycl::buffer<u64>> get_new_id_map<f64_3>(PatchScheduler & sched, SerialPatchTree<f64_3> & sptree){

    

    std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;

    for(auto & [id,pdat] : sched.patch_data.owned_data ){
        if(! pdat.is_empty()){


            u32 ixyz = sched.pdl.get_field_idx<f64_3>("xyz");
            PatchDataField<f64_3> xyz_field =  pdat.fields_f64_3[ixyz];

            std::unique_ptr<sycl::buffer<f64_3>> pos = std::make_unique<sycl::buffer<f64_3>>(xyz_field.usm_data(),xyz_field.size());

            newid_buf_map.insert({
                id,
                __compute_object_patch_owner<f64_3, class ComputeObejctPatchOwners_f64>(
                    sycl_handler::get_compute_queue(), 
                    *pos, 
                    sptree)});

            pos.reset();

        }
        
    }

    return newid_buf_map;

}






template <class vecprec>
inline void reatribute_particles(PatchScheduler & sched, SerialPatchTree<vecprec> & sptree,bool periodic);

template<>
inline void reatribute_particles<f32_3>(PatchScheduler & sched, SerialPatchTree<f32_3> & sptree,bool periodic){

    

    bool err_id_in_newid = false;
    std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;
    for(auto & [id,pdat] : sched.patch_data.owned_data ){
        if(! pdat.is_empty()){


            u32 ixyz = sched.pdl.get_field_idx<f32_3>("xyz");
            PatchDataField<f32_3> xyz_field =  pdat.fields_f32_3[ixyz];

            std::unique_ptr<sycl::buffer<f32_3>> pos = std::make_unique<sycl::buffer<f32_3>>(xyz_field.usm_data(),xyz_field.size());

            newid_buf_map.insert({
                id,
                __compute_object_patch_owner<f32_3, class ComputeObjectPatchOwners_f32_old>(
                    sycl_handler::get_compute_queue(), 
                    *pos, 
                    sptree)});

            pos.reset();

            
            {
                auto nid = newid_buf_map.at(id).get_access<sycl::access::mode::read>();
                for(u32 i = 0 ; i < pdat.get_obj_cnt() ; i++){
                    err_id_in_newid = err_id_in_newid || (nid[i] == u64_max);
                }
            }

        }
        
    }

    

    logger::debug_ln("Patch Object Mover", "err_id_in_newid :",err_id_in_newid);

    bool synced_should_res_box = sched.should_resize_box(err_id_in_newid);

    if (periodic && synced_should_res_box) {
        throw shamrock_exc("box cannot be resized in periodic mode");
    }

    if(synced_should_res_box){
        sched.patch_data.sim_box.reset_box_size();
        
        for(auto & [id,pdat] : sched.patch_data.owned_data ){

            u32 ixyz = sched.pdl.get_field_idx<f32_3>("xyz");
            PatchDataField<f32_3> xyz_field =  pdat.fields_f32_3[ixyz];

            for(u32 i = 0 ; i < pdat.get_obj_cnt(); i++){

                f32_3 r = xyz_field.usm_data()[i];
                sched.patch_data.sim_box.min_box_sim_s = sycl::min(sched.patch_data.sim_box.min_box_sim_s,r);
                sched.patch_data.sim_box.max_box_sim_s = sycl::max(sched.patch_data.sim_box.max_box_sim_s,r);
            }
        }
        f32_3 new_minbox = sched.patch_data.sim_box.min_box_sim_s;
        f32_3 new_maxbox = sched.patch_data.sim_box.max_box_sim_s;
        mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_s.x(), &new_minbox.x(), 1 , mpi_type_f32,MPI_MIN, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_s.y(), &new_minbox.y(), 1 , mpi_type_f32,MPI_MIN, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_s.z(), &new_minbox.z(), 1 , mpi_type_f32,MPI_MIN, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_s.x(), &new_maxbox.x(), 1 , mpi_type_f32,MPI_MAX, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_s.y(), &new_maxbox.y(), 1 , mpi_type_f32,MPI_MAX, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_s.z(), &new_maxbox.z(), 1 , mpi_type_f32,MPI_MAX, MPI_COMM_WORLD);

        sched.patch_data.sim_box.min_box_sim_s = new_minbox;
        sched.patch_data.sim_box.max_box_sim_s = new_maxbox;

        logger::debug_ln("Patch Object Mover", "resize box to  :",new_minbox,new_maxbox);
        sched.patch_data.sim_box.clean_box<f32>(1.2);

        new_minbox = sched.patch_data.sim_box.min_box_sim_s;
        new_maxbox = sched.patch_data.sim_box.max_box_sim_s;
        logger::debug_ln("Patch Object Mover", "resize box to  :",new_minbox,new_maxbox);
        


        sptree.detach_buf();
        sptree = SerialPatchTree<f32_3>(sched.patch_tree, sched.get_box_tranform<f32_3>());
        sptree.attach_buf();

        for(auto & [id,pdat] : sched.patch_data.owned_data ){
            if(! pdat.is_empty()){
                u32 ixyz = sched.pdl.get_field_idx<f32_3>("xyz");
                PatchDataField<f32_3> xyz_field =  pdat.fields_f32_3[ixyz];

                std::unique_ptr<sycl::buffer<f32_3>> pos = std::make_unique<sycl::buffer<f32_3>>(xyz_field.usm_data(),xyz_field.size());

                newid_buf_map.insert({
                    id,
                    __compute_object_patch_owner<f32_3, class ComputeObjectPatchOwners2_f32_old>(
                        sycl_handler::get_compute_queue(), 
                        *pos, 
                        sptree)});

                pos.reset();

            }
            
        }
        
    }

    



    std::vector<std::unique_ptr<PatchData>> comm_pdat;
    std::vector<u64_2> comm_vec;

    for(auto & [id,pdat] : sched.patch_data.owned_data){
        if(! pdat.is_empty()){

            sycl::buffer<u64> & newid = newid_buf_map.at(id);

            if(true){

                auto nid = newid.get_access<sycl::access::mode::read>();
                
                std::unordered_map<u64 , std::unique_ptr<PatchData>> send_map;

                const u32 cnt = pdat.get_obj_cnt();

                for(u32 i = cnt-1 ; i < cnt ; i--){
                    if(id != nid[i]){
                        //std::cout << id  << " " << i << " " << nid[i] << "\n";
                        std::unique_ptr<PatchData> & pdat_int = send_map[nid[i]];

                        if(! pdat_int){
                            pdat_int = std::make_unique<PatchData>(sched.pdl);
                        }

                        pdat.extract_element(i, *pdat_int);
                    }
                        
                }//std::cout << std::endl;

                for(auto & [receiver_pid, pdat_ptr] : send_map){
                    //std::cout << "send " << id << " -> " << receiver_pid <<  " len : " << pdat_ptr->pos_s.size()<<std::endl;


                    comm_vec.push_back(u64_2{sched.patch_list.id_patch_to_global_idx[id],sched.patch_list.id_patch_to_global_idx[receiver_pid]});
                    comm_pdat.push_back(std::move(pdat_ptr));

                }
            }
        }
    }

    
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> part_xchg_map;
    for(u32 i = 0; i < comm_pdat.size(); i++){
        
        std::cout << comm_vec[i].x() << " -> " << comm_vec[i].y() << " data : " << comm_pdat[i].get() << std::endl; 

        PatchData & pdat = *comm_pdat[i];

        u32 ixyz = pdat.pdl.get_field_idx<f32_3>("xyz");

        /*
        for (u32 i = 0; i < pdat.get_obj_cnt(); i++) {
            print_vec(std::cout, pdat.fields_f32_3[ixyz].data()[i]);
            std::cout << std::endl;
        }
        */


    }

    patch_data_exchange_object(sched.pdl,
        sched.patch_list.global, 
        comm_pdat, comm_vec, 
        part_xchg_map);

    for(auto & [recv_id, vec_r] : part_xchg_map){
        //std::cout << "patch " << recv_id << "\n";
        for(auto & [send_id, pdat] : vec_r){
            //std::cout << "    " << send_id << " len : " << pdat->pos_s.size() << "\n"; 

            //TODO if crash here it means that this was implicit init => bad
            PatchData & pdat_recv = sched.patch_data.owned_data.at(recv_id);


            //std::cout << send_id << " -> " << recv_id << " recv data : " << std::endl; 

            u32 ixyz = pdat->pdl.get_field_idx<f32_3>("xyz");

            /*
            for (u32 i = 0; i < pdat->get_obj_cnt(); i++) {
                print_vec(std::cout, pdat->fields_f32_3[ixyz].data()[i]);
                std::cout << std::endl;
            }
            */

            /*{
                std::cout << "recv : " << recv_id << " <- " << send_id << std::endl;

                std::cout << "cnt : " << pdat->pos_s.size() << std::endl;

                for(f32 a : pdat->U1_s){
                    std :: cout << a << " ";
                }std::cout << std::endl;

                for (u32 i = 0; i < pdat->pos_s.size(); i++) {

                    f32 val = pdat->U1_s[i*2 + 0];
                    if(val == 0){
                        std::cout << "----- fail id " << i  << " " << val << std::endl;
                        int a ;
                        std::cin >> a;
                    }
                }
            }*/

            //*
            pdat_recv.insert_elements( *pdat);
                //*/
        }
    }

    
}


#if false

template<>
inline void reatribute_particles<f64_3>(SchedulerMPI & sched, SerialPatchTree<f64_3> & sptree,bool periodic){

    

    bool err_id_in_newid = false;
    std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;
    for(auto & [id,pdat] : sched.patch_data.owned_data ){
        std::unique_ptr<sycl::buffer<f64_3>> pos = std::make_unique<sycl::buffer<f64_3>>(pdat.pos_d.data(),pdat.pos_d.size());

        newid_buf_map.insert({
            id,
            __compute_object_patch_owner<f64_3, class ComputeObejctPatchOwners_f64>(
                sycl_handler::get_compute_queue(), 
                *pos, 
                sptree)});

        pos.reset();

        
        {
            auto nid = newid_buf_map.at(id).get_access<sycl::access::mode::read>();
            for(u32 i = 0 ; i < pdat.pos_s.size() ; i++){
                err_id_in_newid = err_id_in_newid || (nid[i] == u64_max);
            }
        }
        
    }

    printf("err_id_in_newid : %d \n", err_id_in_newid);

    bool synced_should_res_box = sched.should_resize_box(err_id_in_newid);

    if (periodic && synced_should_res_box) {
        throw shamrock_exc("box cannot be resized in periodic mode");
    }

    if(synced_should_res_box){
        sched.patch_data.sim_box.reset_box_size();

        
        for(auto & [id,pdat] : sched.patch_data.owned_data ){
            for(f64_3 & r : pdat.pos_d){
                sched.patch_data.sim_box.min_box_sim_d = sycl::min(sched.patch_data.sim_box.min_box_sim_d,r);
                sched.patch_data.sim_box.max_box_sim_d = sycl::max(sched.patch_data.sim_box.max_box_sim_d,r);
            }
        }

        f64_3 new_minbox = sched.patch_data.sim_box.min_box_sim_d;
        f64_3 new_maxbox = sched.patch_data.sim_box.max_box_sim_d;

        mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_d.x(), &new_minbox.x(), 1 , mpi_type_f64,MPI_MIN, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_d.y(), &new_minbox.y(), 1 , mpi_type_f64,MPI_MIN, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_d.z(), &new_minbox.z(), 1 , mpi_type_f64,MPI_MIN, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_d.x(), &new_maxbox.x(), 1 , mpi_type_f64,MPI_MAX, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_d.y(), &new_maxbox.y(), 1 , mpi_type_f64,MPI_MAX, MPI_COMM_WORLD);
        mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_d.z(), &new_maxbox.z(), 1 , mpi_type_f64,MPI_MAX, MPI_COMM_WORLD);

        sched.patch_data.sim_box.min_box_sim_d = new_minbox;
        sched.patch_data.sim_box.max_box_sim_d = new_maxbox;

        printf("resize box to  : {%f,%f,%f,%f,%f,%f}\n",new_minbox.x(),new_minbox.y(),new_minbox.z(),new_maxbox.x(),new_maxbox.y(),new_maxbox.z());
        sched.patch_data.sim_box.clean_box<f64>(1.2);

        new_minbox = sched.patch_data.sim_box.min_box_sim_d;
        new_maxbox = sched.patch_data.sim_box.max_box_sim_d;
        printf("resize box to  : {%f,%f,%f,%f,%f,%f}\n",new_minbox.x(),new_minbox.y(),new_minbox.z(),new_maxbox.x(),new_maxbox.y(),new_maxbox.z());

        

        sptree.detach_buf();
        sptree = SerialPatchTree<f64_3>(sched.patch_tree, sched.get_box_tranform<f64_3>());
        sptree.attach_buf();

        for(auto & [id,pdat] : sched.patch_data.owned_data ){
            std::unique_ptr<sycl::buffer<f64_3>> pos = std::make_unique<sycl::buffer<f64_3>>(pdat.pos_d.data(),pdat.pos_d.size());

            newid_buf_map.at(id)=
                __compute_object_patch_owner<f64_3, class ComputeObejctPatchOwners2_f64>(
                    sycl_handler::get_compute_queue(), 
                    *pos, 
                    sptree);

            pos.reset();
            
        }
        
    }

    



    std::vector<std::unique_ptr<PatchData>> comm_pdat;
    std::vector<u64_2> comm_vec;

    for(auto & [id,pdat] : sched.patch_data.owned_data){
        if(pdat.pos_s.size() > 0){

            sycl::buffer<u64> & newid = newid_buf_map.at(id);

            if(true){

                auto nid = newid.get_access<sycl::access::mode::read>();
                
                std::unordered_map<u64 , std::unique_ptr<PatchData>> send_map;
                for(u32 i = pdat.pos_s.size()-1 ; i < pdat.pos_s.size() ; i--){
                    if(id != nid[i]){
                        //std::cout << id  << " " << i << " " << nid[i] << "\n";
                        std::unique_ptr<PatchData> & pdat_int = send_map[nid[i]];

                        if(! pdat_int){
                            pdat_int = std::make_unique<PatchData>();
                        }

                        pdat.extract_particle(i, pdat_int->pos_s, pdat_int->pos_d, pdat_int->U1_s, pdat_int->U1_d, pdat_int->U3_s, pdat_int->U3_d);
                    }
                        
                }//std::cout << std::endl;

                for(auto & [receiver_pid, pdat_ptr] : send_map){
                    //std::cout << "send " << id << " -> " << receiver_pid <<  " len : " << pdat_ptr->pos_s.size()<<std::endl;


                    comm_vec.push_back(u64_2{sched.patch_list.id_patch_to_global_idx[id],sched.patch_list.id_patch_to_global_idx[receiver_pid]});
                    comm_pdat.push_back(std::move(pdat_ptr));

                }
            }
        }
    }

    
    std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> part_xchg_map;
    for(u32 i = 0; i < comm_pdat.size(); i++){
        
        //std::cout << comm_vec[i].x() << " " << comm_vec[i].y() << " " << comm_pdat[i].get() << std::endl; 
    }

    patch_data_exchange_object(
        sched.patch_list.global, 
        comm_pdat, comm_vec, 
        part_xchg_map);

    for(auto & [recv_id, vec_r] : part_xchg_map){
        //std::cout << "patch " << recv_id << "\n";
        for(auto & [send_id, pdat] : vec_r){
            //std::cout << "    " << send_id << " len : " << pdat->pos_s.size() << "\n"; 

            PatchData & pdat_recv = sched.patch_data.owned_data[recv_id];

            //*
            pdat_recv.insert_particles(
                pdat->pos_s,
                pdat->pos_d,
                pdat->U1_s,
                pdat->U1_d,
                pdat->U3_s,
                pdat->U3_d);
                //*/
        }
    }
}

#endif