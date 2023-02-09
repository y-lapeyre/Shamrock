// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamrock/math/CoordRange.hpp"
#include "shamalgs/syclManip.hpp"
#include "shamalgs/vectorManip.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"

#include <tuple>

namespace shamrock::patch {
    /**
     * @brief Store the information related to the size of the simulation box to convert patch
     * integer coordinates to floating point ones.
     */
    class SimulationBoxInfo {

        using var_t = std::variant<
            CoordRange<f32>,
            CoordRange<f32_2>,
            CoordRange<f32_3>,
            CoordRange<f32_4>,
            CoordRange<f32_8>,
            CoordRange<f32_16>,
            CoordRange<f64>,
            CoordRange<f64_2>,
            CoordRange<f64_3>,
            CoordRange<f64_4>,
            CoordRange<f64_8>,
            CoordRange<f64_16>,
            CoordRange<u32>,
            CoordRange<u64>,
            CoordRange<u32_3>,
            CoordRange<u64_3>>;

        PatchDataLayout &pdl;

        var_t bounding_box;
        CoordRange<u64_3> patch_coord_bounding_box;

        public:
        inline SimulationBoxInfo(PatchDataLayout &pdl, CoordRange<u64_3> patch_coord_bounding_box)
            : pdl(pdl), patch_coord_bounding_box(std::move(patch_coord_bounding_box)) {

            reset_box_size();
        }

        /**
         * @brief Get the stored bounding box of the domain
         *
         * @tparam T type of position vector
         * @return std::tuple<T, T> return [low bound, high bound[
         */
        template <class T> [[nodiscard]] std::tuple<T, T> get_bounding_box() const;

        /**
         * @brief Override the stored bounding box by the one given in new_box
         *
         * @tparam T type of position vector
         * @param new_box the new bounding box
         */
        template <class T> void set_bounding_box(CoordRange<T> new_box);

        /**
         * @brief get the patch coordinates on the domain
         *
         * @tparam T type of position vector
         * @param p the patch
         * @return std::tuple<T,T> the [low bound, high bound[ coordinate of the patch in the domain
         */
        template <class T> std::tuple<T, T> partch_coord_to_domain(const Patch &p) const;

        // TODO implement box size reduction here

        /**
         * @brief reset box simulation size
         */
        void reset_box_size();

        // TODO replace vectype primtype in the code by primtype and sycl::vec<primtype,3> for the
        // others
        template <class primtype> void clean_box(primtype tol);

        template <> inline void clean_box<f32>(f32 tol) {

            auto [bmin, bmax] = get_bounding_box<f32_3>();

            f32_3 center   = (bmin + bmax) / 2;
            f32_3 cur_delt = bmax - bmin;
            cur_delt /= 2;

            cur_delt *= tol;

            bmin = center - cur_delt;
            bmax = center + cur_delt;

            set_bounding_box<f32_3>({bmin, bmax});
        }

        template <> inline void clean_box<f64>(f64 tol) {
            auto [bmin, bmax] = get_bounding_box<f64_3>();

            f64_3 center   = (bmin + bmax) / 2;
            f64_3 cur_delt = bmax - bmin;
            cur_delt /= 2;

            cur_delt *= tol;

            bmin = center - cur_delt;
            bmax = center + cur_delt;

            set_bounding_box<f64_3>({bmin, bmax});
        }

        template <class primtype>
        inline std::tuple<sycl::vec<primtype, 3>, sycl::vec<primtype, 3>> get_box(Patch &p) {
            return partch_coord_to_domain<sycl::vec<primtype, 3>>(p);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // out of line implementation of the simbox
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <class T>
    [[nodiscard]] inline std::tuple<T, T> SimulationBoxInfo::get_bounding_box() const {

        if (!pdl.check_main_field_type<T>()) {

            throw std::invalid_argument(
                __LOC_PREFIX__ +
                "the chosen type for the main field does not match the required template type\n" +
                "call : " + __PRETTY_FUNCTION__
            );
        }

        const CoordRange<T> *pval = std::get_if<CoordRange<T>>(&bounding_box);

        if (!pval) {

            throw std::invalid_argument(
                __LOC_PREFIX__ +
                "the type in SimulationBoxInfo does not match the one in the layout\n" +
                "call : " + __PRETTY_FUNCTION__
            );
        }

        return {pval->low_bound, pval->high_bound};
    }

    template <class T> inline void SimulationBoxInfo::set_bounding_box(CoordRange<T> new_box) {
        if (pdl.check_main_field_type<T>()) {
            bounding_box = new_box;
        } else {
            throw std::runtime_error(
                __LOC_PREFIX__ + "The main field is not of the required type\n" +
                "call : " + __PRETTY_FUNCTION__
            );
        }
    }

    template <class T>
    inline std::tuple<T, T> SimulationBoxInfo::partch_coord_to_domain(const Patch &p) const {

        using ptype = typename shamalgs::vec_manip::VectorProperties<T>::component_type;

        auto [bmin, bmax] = get_bounding_box<T>();

        T translate_factor = bmin;

        using namespace shamalgs::sycl_manip;

        T patch_b_size = VecConvert<u64_3, T>::convert(patch_coord_bounding_box.delt());

        T div_factor = patch_b_size / (bmax - bmin);

        return p.convert_coord(patch_coord_bounding_box.low_bound, div_factor, translate_factor);
    }

    inline void SimulationBoxInfo::reset_box_size() {

        if (pdl.check_main_field_type<f32_3>()) {
            bounding_box = CoordRange<f32_3>::max_range();
        } else if (pdl.check_main_field_type<f64_3>()) {
            bounding_box = CoordRange<f64_3>::max_range();
        } else {
            throw std::runtime_error(
                __LOC_PREFIX__ + "the chosen type for the main field is not handled"
            );
        }
    }
} // namespace shamrock::patch