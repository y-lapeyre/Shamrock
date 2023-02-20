// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "PatchDataField.hpp"
#include "PatchDataLayout.hpp"
#include "aliases.hpp"

#include "Patch.hpp"
#include "shamutils/intervals.hpp"
#include "shamutils/sycl_utilities.hpp"
#include <variant>

namespace shamrock::patch {
    /**
     * @brief PatchData container class, the layout is described in patchdata_layout
     */
    class PatchData {

        void init_fields();

        using var_t = std::variant<
            PatchDataField<f32>,
            PatchDataField<f32_2>,
            PatchDataField<f32_3>,
            PatchDataField<f32_4>,
            PatchDataField<f32_8>,
            PatchDataField<f32_16>,
            PatchDataField<f64>,
            PatchDataField<f64_2>,
            PatchDataField<f64_3>,
            PatchDataField<f64_4>,
            PatchDataField<f64_8>,
            PatchDataField<f64_16>,
            PatchDataField<u32>,
            PatchDataField<u64>,
            PatchDataField<u32_3>,
            PatchDataField<u64_3>>;

        std::vector<var_t> fields;

        public:
        PatchDataLayout &pdl;

        inline PatchData(PatchDataLayout &pdl) : pdl(pdl) { init_fields(); }

        template<class Functor>
        inline void for_each_field_any(Functor &&func) {
            for (auto &f : fields) {
                std::visit([&](auto &arg) { func(arg); }, f);
            }
        }

        inline PatchData(const PatchData &other) : pdl(other.pdl) {

            for (auto &field_var : other.fields) {

                std::visit(
                    [&](auto &field) {
                        using base_t =
                            typename std::remove_reference<decltype(field)>::type::Field_type;
                        fields.emplace_back(PatchDataField<base_t>(field));
                    },
                    field_var
                );
            };
        }

        inline PatchData duplicate() {
            const PatchData &current = *this;
            return PatchData(current);
        }

        inline std::unique_ptr<PatchData> duplicate_to_ptr() {
            const PatchData &current = *this;
            return std::make_unique<PatchData>(current);
        }

        PatchData &operator=(const PatchData &other) = delete;

        /**
         * @brief extract particle at index pidx and insert it in the provided vectors
         *
         * @param pidx
         * @param out_pos_s
         * @param out_pos_d
         * @param out_U1_s
         * @param out_U1_d
         * @param out_U3_s
         * @param out_U3_d
         */
        void extract_element(u32 pidx, PatchData &out_pdat);

        void insert_elements(PatchData &pdat);

        /**
         * @brief insert elements of pdat only if they are within the range
         *
         * @tparam T
         * @param pdat
         * @param bmin
         * @param bmax
         */
        template<class T>
        void insert_elements_in_range(PatchData &pdat, T bmin, T bmax);

        void resize(u32 new_obj_cnt);

        void expand(u32 obj_cnt);

        /**
        * @brief this function remaps the patchdatafield like so
        *   val[id] = val[index_map[id]]
        * This function can be used to apply the result of a sort to the field
        * 
        * @param index_map 
        * @param len the lenght of the map (must match with the current count)
        */
        void index_remap(sycl::buffer<u32> index_map, u32 len);

        /**
        * @brief this function remaps the patchdatafield like so
        *   val[id] = val[index_map[id]]
        * This function can be used to apply the result of a sort to the field
        * 
        * @param index_map 
        * @param len the lenght of the map
        */
        void index_remap_resize(sycl::buffer<u32> index_map, u32 len);

        // template<class Tvecbox>
        // void split_patchdata(PatchData & pd0,PatchData & pd1,PatchData & pd2,PatchData &
        // pd3,PatchData & pd4,PatchData & pd5,PatchData & pd6,PatchData & pd7,
        //     Tvecbox bmin_p0,Tvecbox bmin_p1,Tvecbox bmin_p2,Tvecbox bmin_p3,Tvecbox
        //     bmin_p4,Tvecbox bmin_p5,Tvecbox bmin_p6,Tvecbox bmin_p7, Tvecbox bmax_p0,Tvecbox
        //     bmax_p1,Tvecbox bmax_p2,Tvecbox bmax_p3,Tvecbox bmax_p4,Tvecbox bmax_p5,Tvecbox
        //     bmax_p6,Tvecbox bmax_p7);

        template<class Tvecbox>
        void split_patchdata(
            std::array<std::reference_wrapper<PatchData>, 8> pdats,
            std::array<Tvecbox, 8> min_box,
            std::array<Tvecbox, 8> max_box
        );

        void append_subset_to(std::vector<u32> &idxs, PatchData &pdat) const;
        void append_subset_to(sycl::buffer<u32> &idxs, u32 sz, PatchData &pdat) const;

        inline u32 get_obj_cnt() {

            bool is_empty = fields.empty();

            if (!is_empty) {
                return std::visit([](auto &field) { return field.get_obj_cnt(); }, fields[0]);
            }

            throw std::runtime_error("this patchdata does not contains any fields");
        }

        inline u64 memsize() {
            u64 sum = 0;

            for (auto &field_var : fields) {

                std::visit([&](auto &field) { sum += field.memsize(); }, field_var);
            }

            return sum;
        }

        inline bool is_empty() { return get_obj_cnt() == 0; }

        void overwrite(PatchData &pdat, u32 obj_cnt);

        template<class T>
        bool check_field_type(u32 idx) {
            var_t &tmp = fields[idx];

            PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&tmp);

            if (pval) {
                return true;
            } else {
                return false;
            }
        }

        template<class T>
        PatchDataField<T> &get_field(u32 idx) {

            var_t &tmp = fields[idx];

            PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&tmp);

            if (pval) {
                return *pval;
            }

            throw std::runtime_error(
                "the request id is not of correct type\n"
                "   current map is : \n" +
                pdl.get_description_str() + " this call : " + std::string(__PRETTY_FUNCTION__) +
                "\n"
                "    arg : idx = " +
                std::to_string(idx)
            );
        }

        // template<class T> inline std::vector<PatchDataField<T> & > get_field_list(){
        //     std::vector<PatchDataField<T> & > ret;
        //
        //
        //}

        template<class T, class Functor>
        inline void for_each_field(Functor &&func) {
            for (auto &f : fields) {
                PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&f);

                if (pval) {
                    func(*pval);
                }
            }
        }

        inline friend bool operator==(const PatchData &p1, const PatchData &p2) {
            bool check = true;

            if (p1.fields.size() != p2.fields.size()) {
                return false;
            }

            for (u32 idx = 0; idx < p1.fields.size(); idx++) {

                bool ret = std::visit(
                    [&](auto &pf1, auto &pf2) -> bool {
                        using t1 = typename std::remove_reference<decltype(pf1)>::type::Field_type;
                        using t2 = typename std::remove_reference<decltype(pf2)>::type::Field_type;

                        if constexpr (std::is_same<t1, t2>::value) {
                            return pf1.check_field_match(pf2);
                        } else {
                            return false;
                        }
                    },
                    p1.fields[idx],
                    p2.fields[idx]
                );

                check = check && ret;
            }

            return check;
        }
    };

    template<class T>
    inline void PatchData::insert_elements_in_range(PatchData &pdat, T bmin, T bmax) {

        if (!pdl.check_main_field_type<T>()) {

            throw std::invalid_argument(
                __LOC_PREFIX__ +
                "the chosen type for the main field does not match the required template type\n" +
                "call : " + __PRETTY_FUNCTION__
            );
        }

        PatchDataField<T> & main_field = pdat.get_field<T>(0);

        auto get_vec_idx = [&](T vmin, T vmax) -> std::vector<u32> {
            return main_field.get_elements_with_range(
                [&](T val,T vmin, T vmax){

                    if(shamutils::sycl_utils::VectorProperties<T>::dimension == 3){
                        return shamutils::is_in_half_open(val, vmin,vmax);
                    }else{
                        throw std::runtime_error("dimension != 3 is not handled");
                    }

                },
                vmin,vmax
            );
        };

        std::vector<u32> idx_lst = get_vec_idx(bmin,bmax);

        logger::debug_sycl_ln("PatchData", "inserting element cnt =", idx_lst.size());

        pdat.append_subset_to(idx_lst,*this);

    }

} // namespace shamrock::patch