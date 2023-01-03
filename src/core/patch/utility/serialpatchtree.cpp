// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "serialpatchtree.hpp"
#include "shamrock/sys/log.hpp"

void get_serial_tree_fp32(u64 root_key, PatchTree &ptree, std::vector<PtNode<f32_3>> &result_tree,
                          std::vector<u64> &result_tree_linked_patch_id, u64 &counter, u32 &max_level,
                          const f32_3 &translate_factor, const f32_3 &scale_factor) {

    PatchTree::PTNode ptn = ptree.tree[root_key];

    PtNode<f32_3> n;
    n.box_min    = f32_3{ptn.x_min, ptn.y_min, ptn.z_min} * scale_factor + translate_factor;
    n.box_max    = (f32_3{ptn.x_max, ptn.y_max, ptn.z_max} + 1) * scale_factor + translate_factor;
    n.childs_id0 = ptn.childs_id[0];
    n.childs_id1 = ptn.childs_id[1];
    n.childs_id2 = ptn.childs_id[2];
    n.childs_id3 = ptn.childs_id[3];
    n.childs_id4 = ptn.childs_id[4];
    n.childs_id5 = ptn.childs_id[5];
    n.childs_id6 = ptn.childs_id[6];
    n.childs_id7 = ptn.childs_id[7];

    max_level = sycl::max((u32)ptn.level, max_level);

    u64 id_nvec = counter;
    result_tree.push_back(n);
    result_tree_linked_patch_id.push_back(ptn.linked_patchid);
    counter++;


    //std::cout << "id : " << counter-1 << " leaf : " << ptn.is_leaf << std::endl;


    if (!ptn.is_leaf) {

        u64 old_id0 = ptn.childs_id[0];
        u64 old_id1 = ptn.childs_id[1];
        u64 old_id2 = ptn.childs_id[2];
        u64 old_id3 = ptn.childs_id[3];
        u64 old_id4 = ptn.childs_id[4];
        u64 old_id5 = ptn.childs_id[5];
        u64 old_id6 = ptn.childs_id[6];
        u64 old_id7 = ptn.childs_id[7];

        result_tree.at(id_nvec).childs_id0 = counter;
        get_serial_tree_fp32(old_id0, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        result_tree.at(id_nvec).childs_id1 = counter;
        get_serial_tree_fp32(old_id1, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        result_tree.at(id_nvec).childs_id2 = counter;
        get_serial_tree_fp32(old_id2, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        result_tree.at(id_nvec).childs_id3 = counter;
        get_serial_tree_fp32(old_id3, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        result_tree.at(id_nvec).childs_id4 = counter;
        get_serial_tree_fp32(old_id4, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        result_tree.at(id_nvec).childs_id5 = counter;
        get_serial_tree_fp32(old_id5, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        result_tree.at(id_nvec).childs_id6 = counter;
        get_serial_tree_fp32(old_id6, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        result_tree.at(id_nvec).childs_id7 = counter;
        get_serial_tree_fp32(old_id7, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);
    }

}

void get_serial_tree_fp64(u64 root_key, PatchTree &ptree, std::vector<PtNode<f64_3>> &result_tree,
                          std::vector<u64> &result_tree_linked_patch_id, u64 &counter, u32 &max_level,
                          const f64_3 &translate_factor, const f64_3 &scale_factor) {

    PatchTree::PTNode ptn = ptree.tree[root_key];

    PtNode<f64_3> n;
    n.box_min    = f64_3{ptn.x_min, ptn.y_min, ptn.z_min} * scale_factor + translate_factor;
    n.box_max    = (f64_3{ptn.x_max, ptn.y_max, ptn.z_max} + 1) * scale_factor + translate_factor;
    n.childs_id0 = ptn.childs_id[0];
    n.childs_id1 = ptn.childs_id[1];
    n.childs_id2 = ptn.childs_id[2];
    n.childs_id3 = ptn.childs_id[3];
    n.childs_id4 = ptn.childs_id[4];
    n.childs_id5 = ptn.childs_id[5];
    n.childs_id6 = ptn.childs_id[6];
    n.childs_id7 = ptn.childs_id[7];

    max_level = sycl::max((u32)ptn.level, max_level);

    result_tree.push_back(n);
    result_tree_linked_patch_id.push_back(ptn.linked_patchid);
    counter++;

    if (!ptn.is_leaf) {

        u64 old_id0 = ptn.childs_id[0];
        u64 old_id1 = ptn.childs_id[1];
        u64 old_id2 = ptn.childs_id[2];
        u64 old_id3 = ptn.childs_id[3];
        u64 old_id4 = ptn.childs_id[4];
        u64 old_id5 = ptn.childs_id[5];
        u64 old_id6 = ptn.childs_id[6];
        u64 old_id7 = ptn.childs_id[7];

        ptn.childs_id[0] = counter;
        get_serial_tree_fp64(old_id0, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        ptn.childs_id[1] = counter;
        get_serial_tree_fp64(old_id1, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        ptn.childs_id[2] = counter;
        get_serial_tree_fp64(old_id2, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        ptn.childs_id[3] = counter;
        get_serial_tree_fp64(old_id3, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        ptn.childs_id[4] = counter;
        get_serial_tree_fp64(old_id4, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        ptn.childs_id[5] = counter;
        get_serial_tree_fp64(old_id5, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        ptn.childs_id[6] = counter;
        get_serial_tree_fp64(old_id6, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);

        ptn.childs_id[7] = counter;
        get_serial_tree_fp64(old_id7, ptree, result_tree, result_tree_linked_patch_id, counter, max_level, translate_factor,
                             scale_factor);
    }
}

template <>
void SerialPatchTree<f32_3>::build_from_patch_tree(PatchTree &ptree, f32_3 translate_factor, f32_3 scale_factor) {

    u64 cnt     = 0;
    level_count = 0;//TODO add root_id set
    for (u64 root_id : ptree.roots_id) {
        logger::debug_ln("Serial Patch Tree","get serial tree fp32 root id :" , root_id);
        get_serial_tree_fp32(root_id, ptree, serial_tree, linked_patch_ids, cnt, level_count, translate_factor,
                             scale_factor);
    }

    logger::debug_ln("Serial Patch Tree","tree internal cell count = " , serial_tree.size());
    logger::debug_ln("Serial Patch Tree","level_count =" , level_count);
}

template <>
void SerialPatchTree<f64_3>::build_from_patch_tree(PatchTree &ptree, f64_3 translate_factor, f64_3 scale_factor) {

    u64 cnt     = 0;
    level_count = 0;
    for (u64 root_id : ptree.roots_id) {
        get_serial_tree_fp64(root_id, ptree, serial_tree, linked_patch_ids, cnt, level_count, translate_factor,
                             scale_factor);
    }
}