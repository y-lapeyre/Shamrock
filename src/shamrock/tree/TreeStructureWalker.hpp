// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"

#include "TreeStructure.hpp"

namespace shamrock::tree {

    enum WalkPolicy { Recompute, Cache };

    namespace details {
        template<enum WalkPolicy, class InteractCrit>
        class TreeStructureWalkerPolicy;
    } // namespace details

    template<WalkPolicy policy, class InteractCrit>
    class TreeStructureWalker {
        public:
        details::TreeStructureWalkerPolicy<policy, InteractCrit> walker;

        using AccessedWalker =
            typename details::TreeStructureWalkerPolicy<policy, InteractCrit>::Accessed;

        TreeStructureWalker(TreeStructure &str, InteractCrit &&crit)
            : walker(str, std::forward<InteractCrit>(crit)) {}

        inline void generate() { walker.generate(); }

        inline AccessedWalker get_access(sycl::handler &device_handle) {
            return walker.get_access(device_handle);
        }
    };

    template<WalkPolicy policy, class InteractCrit>
    static TreeStructureWalker<policy, InteractCrit>
    generate_walk(TreeStructure &str,u32 walker_count , InteractCrit &&crit) {
        TreeStructureWalker<policy, InteractCrit> walk(str, std::forward<InteractCrit>(crit));
        walk.generate();
        return walk;
    }

} // namespace shamrock::tree

namespace shamrock::tree::details {

    template<class InteractCrit>
    class TreeStructureWalkerPolicy<Recompute, InteractCrit> {
        public:
        TreeStructure &tree_struct;
        InteractCrit crit;

        class Accessed {
            public:
            sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> rchild_id;
            sycl::accessor<u32, 1, sycl::access::mode::read, sycl::target::device> lchild_id;
            sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> rchild_flag;
            sycl::accessor<u8, 1, sycl::access::mode::read, sycl::target::device> lchild_flag;

            Accessed(TreeStructure &tree_struct, sycl::handler &device_handle)
                : rchild_id{*tree_struct.buf_rchild_id, device_handle, sycl::read_only},
                  lchild_id{*tree_struct.buf_lchild_id, device_handle, sycl::read_only},
                  rchild_flag{*tree_struct.buf_rchild_flag, device_handle, sycl::read_only},
                  lchild_flag{*tree_struct.buf_lchild_flag, device_handle, sycl::read_only} {}

            template<class FuncNodeFound, class FuncNodeReject>
            inline void
            for_each_node(FuncNodeFound &&found_case, FuncNodeReject &&reject_case) const {}
        };

        inline void generate() {}

        explicit TreeStructureWalkerPolicy(TreeStructure &str, InteractCrit &&crit)
            : tree_struct(str), crit(crit) {}

        inline Accessed get_access(sycl::handler &device_handle) {
            return Accessed(tree_struct, device_handle);
        }
    };
} // namespace shamrock::tree::details