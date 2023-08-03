// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/stacktrace.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/zeus/GhostZoneData.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeTaversalCache.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamrock/utils/SolverStorageComponent.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
namespace shammodels::zeus {

    template<class T>
    using Component = shamrock::StorageComponent<T>;

    template<class Tvec, class TgridVec, class Tmorton>
    class SolverStorage {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using RTree = RadixTree<Tmorton, TgridVec>;

        Component<SerialPatchTree<TgridVec>> serial_patch_tree;

        Component<GhostZonesData<Tvec, TgridVec>> ghost_zone_infos;

        Component<shamrock::patch::PatchDataLayout> ghost_layout;

        Component<shambase::DistributedData<shamrock::MergedPatchData>> merged_patchdata_ghost;
        Component<shambase::DistributedData<shammath::AABB<TgridVec>>> merge_patch_bounds;
        Component<shambase::DistributedData<RTree>> trees;

        Component<shamrock::tree::ObjectCacheHandler> neighbors_cache;

        Component<shamrock::ComputeField<Tscal>> pressure;

        /**
         * @brief for each face give a lookup table for the normal orientation
         * 0 = x-
         * 1 = x+
         * 2 = y-
         * 3 = y+
         * 4 = z-
         * 5 = z+
         * 
         */
        Component<shambase::DistributedData<sycl::buffer<u8>>> face_normals_lookup;
        
        struct {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io = 0;

            void reset(){
                *this = {};
            }
        } timings_details;
    };

} // namespace shammodels::zeus