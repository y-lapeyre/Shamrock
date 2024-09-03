// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchData.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "Patch.hpp"
#include "PatchDataField.hpp"
#include "PatchDataLayout.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammath/intervals.hpp"
#include <variant>
#include <vector>

namespace shamrock::patch {
    /**
     * @brief PatchData container class, the layout is described in patchdata_layout
     */
    class PatchData {

        void init_fields();

        using var_t = FieldVariant<PatchDataField>;

        std::vector<var_t> fields;

        public:
        using field_variant_t = var_t;

        PatchDataLayout &pdl;

        inline PatchData(PatchDataLayout &pdl) : pdl(pdl) { init_fields(); }

        inline PatchData(const PatchData &other) : pdl(other.pdl) {

            NamedStackEntry stack_loc{"PatchData::copy_constructor", true};

            for (auto &field_var : other.fields) {

                field_var.visit([&](auto &field) {
                    using base_t =
                        typename std::remove_reference<decltype(field)>::type::Field_type;
                    fields.emplace_back(PatchDataField<base_t>(field));
                });
            };
        }

        /**
         * @brief PatchData move constructor
         *
         * @param other
         */
        inline PatchData(PatchData &&other) noexcept
            : fields(std::move(other.fields)), pdl(other.pdl) {}

        /**
         * @brief PatchData move assignment
         *
         * @param other
         */
        inline PatchData &operator=(PatchData &&other) noexcept {
            fields = std::move(other.fields);
            pdl    = std::move(other.pdl);

            return *this;
        }

        PatchData &operator=(const PatchData &other) = delete;

        static PatchData mock_patchdata(u64 seed, u32 obj_cnt, PatchDataLayout &pdl);

        template<class Functor>
        inline void for_each_field_any(Functor &&func) {
            for (auto &f : fields) {
                f.visit([&](auto &arg) {
                    func(arg);
                });
            }
        }

        template<class Func>
        inline PatchData(PatchDataLayout &pdl, Func &&fct_init) : pdl(pdl) {

            u32 cnt = 0;

            fct_init(fields);
        }

        inline PatchData duplicate() {
            const PatchData &current = *this;
            return PatchData(current);
        }

        inline std::unique_ptr<PatchData> duplicate_to_ptr() {
            const PatchData &current = *this;
            return std::make_unique<PatchData>(current);
        }

        /**
         * @brief extract particle at index pidx and insert it in the provided vectors
         *
         * @param pidx
         * @param out_pdat
         */
        void extract_element(u32 pidx, PatchData &out_pdat);

        void keep_ids(sycl::buffer<u32> &index_map, u32 len);

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

        void reserve(u32 new_obj_cnt);

        void expand(u32 obj_cnt);

        /**
         * @brief this function remaps the patchdatafield like so
         *   val[id] = val[index_map[id]]
         * This function can be used to apply the result of a sort to the field
         *
         * @param index_map
         * @param len the length of the map (must match with the current count)
         */
        void index_remap(sycl::buffer<u32> &index_map, u32 len);

        /**
         * @brief this function remaps the patchdatafield like so
         *   val[id] = val[index_map[id]]
         * This function can be used to apply the result of a sort to the field
         *
         * @param index_map
         * @param len the length of the map
         */
        void index_remap_resize(sycl::buffer<u32> &index_map, u32 len);

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
            std::array<Tvecbox, 8> max_box);

        void append_subset_to(std::vector<u32> &idxs, PatchData &pdat);
        void append_subset_to(sycl::buffer<u32> &idxs, u32 sz, PatchData &pdat);

        inline u32 get_obj_cnt() {

            bool is_empty = fields.empty();

            if (!is_empty) {
                return fields[0].visit_return([](auto &field) {
                    return field.get_obj_cnt();
                });
            }

            throw shambase::make_except_with_loc<std::runtime_error>(
                "this patchdata does not contains any fields");
        }

        inline u64 memsize() {
            u64 sum = 0;

            for (auto &field_var : fields) {

                field_var.visit([&](auto &field) {
                    sum += field.memsize();
                });
            }

            return sum;
        }

        inline bool is_empty() { return get_obj_cnt() == 0; }

        void overwrite(PatchData &pdat, u32 obj_cnt);

        template<class T>
        bool check_field_type(u32 idx) {
            var_t &tmp = fields.at(idx);

            PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&tmp.value);

            if (pval) {
                return true;
            } else {
                return false;
            }
        }

        template<class T>
        PatchDataField<T> &get_field(u32 idx) {

            var_t &tmp = fields.at(idx);

            PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&tmp.value);

            if (pval) {
                return *pval;
            }

            throw shambase::make_except_with_loc<std::runtime_error>(
                "the request id is not of correct type\n"
                "   current map is : \n"
                + pdl.get_description_str()
                + "\n"
                  "    arg : idx = "
                + std::to_string(idx));
        }

        template<class T>
        sycl::buffer<T> &get_field_buf_ref(u32 idx) {

            var_t &tmp = fields.at(idx);

            PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&tmp.value);

            if (pval) {
                return shambase::get_check_ref(pval->get_buf());
            }

            throw shambase::make_except_with_loc<std::runtime_error>(
                "the request id is not of correct type\n"
                "   current map is : \n"
                + pdl.get_description_str()
                + "\n"
                  "    arg : idx = "
                + std::to_string(idx));
        }

        /**
         * @brief check that all contained field have the same obj cnt
         *
         */
        inline void check_field_obj_cnt_match() {
            u32 cnt = get_obj_cnt();
            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    if (field.get_obj_cnt() != cnt) {
                        throw shambase::make_except_with_loc<std::runtime_error>(
                            "mismatch in obj cnt");
                    }
                });
            }
        }

        // template<class T> inline std::vector<PatchDataField<T> & > get_field_list(){
        //     std::vector<PatchDataField<T> & > ret;
        //
        //
        //}

        template<class T, class Functor>
        inline void for_each_field(Functor &&func) {
            for (auto &f : fields) {
                PatchDataField<T> *pval = std::get_if<PatchDataField<T>>(&f.value);

                if (pval) {
                    func(*pval);
                }
            }
        }

        inline friend bool operator==(PatchData &p1, PatchData &p2) {
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
                    p1.fields[idx].value,
                    p2.fields[idx].value);

                check = check && ret;
            }

            return check;
        }

        void serialize_buf(shamalgs::SerializeHelper &serializer);

        shamalgs::SerializeSize serialize_buf_byte_size();

        static PatchData
        deserialize_buf(shamalgs::SerializeHelper &serializer, PatchDataLayout &pdl);

        void fields_raz();

        bool has_nan() {
            StackEntry stack_loc{};

            bool ret = false;

            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    if (field.has_nan()) {
                        ret = true;
                    }
                });
            }
            return ret;
        }
        bool has_inf() {
            StackEntry stack_loc{};

            bool ret = false;

            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    if (field.has_inf()) {
                        ret = true;
                    }
                });
            }
            return ret;
        }
        bool has_nan_or_inf() {
            StackEntry stack_loc{};

            bool ret = false;

            for (auto &field_var : fields) {
                field_var.visit([&](auto &field) {
                    if (field.has_nan_or_inf()) {
                        ret = true;
                    }
                });
            }
            return ret;
        }

        /**
         * @brief
         * \todo should add a check in patch data to check that
         * size in ovveride match with the one in the input vec
         * @tparam T
         * @param field_name
         * @param vec
         */
        template<class T>
        void override_patch_field(std::string field_name, std::vector<T> &vec) {
            u32 len              = vec.size();
            PatchDataField<T> &f = get_field<T>(pdl.get_field_idx<T>(field_name));
            sycl::buffer<T> buf(vec.data(), len);
            f.override(buf, len);
        }

        /**
         * @brief Fetch data of a patchdata field into a std::vector
         * @tparam T
         * @param key
         * @param pdat
         * @return std::vector<T>
         */
        template<class T>
        inline std::vector<T> fetch_data(std::string key) {

            std::vector<T> vec;

            auto appender = [&](auto &field) {
                if (field.get_name() == key) {

                    logger::debug_ln("PyShamrockCTX", "appending field", key);

                    if (!field.is_empty()) {
                        sycl::host_accessor acc{shambase::get_check_ref(field.get_buf())};
                        u32 len = field.size();

                        for (u32 i = 0; i < len; i++) {
                            vec.push_back(acc[i]);
                        }
                    }
                }
            };

            for_each_field<T>([&](auto &field) {
                appender(field);
            });

            return vec;
        }
    };

    template<class T>
    inline void PatchData::insert_elements_in_range(PatchData &pdat, T bmin, T bmax) {

        StackEntry stack_loc{};

        if (!pdl.check_main_field_type<T>()) {

            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the chosen type for the main field does not match the required template type");
        }

        PatchDataField<T> &main_field = pdat.get_field<T>(0);

        auto get_vec_idx = [&](T vmin, T vmax) -> std::vector<u32> {
            return main_field.get_elements_with_range(
                [&](T val, T vmin, T vmax) {
                    if (shambase::VectorProperties<T>::dimension == 3) {
                        return shammath::is_in_half_open(val, vmin, vmax);
                    } else {
                        throw shambase::make_except_with_loc<std::runtime_error>(
                            "dimension != 3 is not handled");
                    }
                },
                vmin,
                vmax);
        };

        std::vector<u32> idx_lst = get_vec_idx(bmin, bmax);

        logger::debug_sycl_ln("PatchData", "inserting element cnt =", idx_lst.size());

        pdat.append_subset_to(idx_lst, *this);
    }

} // namespace shamrock::patch
