// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shamsys/legacy/log.hpp"
#include <sstream>
#include <variant>
#include <vector>

namespace shamrock::patch {
    class PatchDataLayout {

        template<class T>
        class FieldDescriptor {
            public:
            using field_T = T;

            std::string name;
            u32 nvar;

            inline FieldDescriptor() : name(""), nvar(1){};
            inline FieldDescriptor(std::string name, u32 nvar) : nvar(nvar), name(name) {}
        };

        using var_t = std::variant<
            FieldDescriptor<f32>,
            FieldDescriptor<f32_2>,
            FieldDescriptor<f32_3>,
            FieldDescriptor<f32_4>,
            FieldDescriptor<f32_8>,
            FieldDescriptor<f32_16>,
            FieldDescriptor<f64>,
            FieldDescriptor<f64_2>,
            FieldDescriptor<f64_3>,
            FieldDescriptor<f64_4>,
            FieldDescriptor<f64_8>,
            FieldDescriptor<f64_16>,
            FieldDescriptor<u32>,
            FieldDescriptor<u64>,
            FieldDescriptor<u32_3>,
            FieldDescriptor<u64_3>>;

        std::vector<var_t> fields;

        public:
        /**
         * @brief add a field of type T to the layout
         *
         * @tparam T type of the field
         * @param field_name field name
         * @param nvar number of varaible per object
         */
        template<class T>
        void add_field(std::string field_name, u32 nvar);

        /**
         * @brief Get the field description id if matching name & type
         *
         * @tparam T
         * @param field_name
         * @return FieldDescriptor<T>
         */
        template<class T>
        FieldDescriptor<T> get_field(std::string field_name);

        /**
         * @brief Get the field description at given index
         *
         * @tparam T
         * @param idx
         * @return FieldDescriptor<T>
         */
        template<class T>
        FieldDescriptor<T> get_field(u32 idx);

        /**
         * @brief Get the field id if matching name & type
         *
         * @tparam T
         * @param field_name
         * @return u32
         */
        template<class T>
        u32 get_field_idx(std::string field_name);

        /**
         * @brief check that field of id @idx is of type T
         *
         * @tparam T
         * @param idx
         * @return true
         * @return false
         */
        template<class T>
        bool check_field_type(u32 idx);

        /**
         * @brief check that main field (id=0)is of type T
         *
         * @tparam T
         * @return true
         * @return false
         */
        template<class T>
        inline bool check_main_field_type() {
            return check_field_type<T>(0);
        }

        /**
         * @brief Get the main field description as a variant object
         *
         * @return const var_t& the variant field description
         */
        [[nodiscard]] inline const var_t &get_main_field_any() const { return fields[0]; }

        /**
         * @brief Get the description of the layout
         *
         * @return std::string
         */
        std::string get_description_str();

        /**
         * @brief Get the list of field names
         *
         * @return std::vector<std::string>
         */
        std::vector<std::string> get_field_names();

        /**
         * @brief for each visit of each field
         * 
         * @tparam Functor the signature of the lambda must be : [?](auto & arg){...}
         * @param func 
         */
        template<class Functor>
        inline void for_each_field_any(Functor &&func) {
            for (auto &f : fields) {
                std::visit([&](auto &arg) { func(arg); }, f);
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // out of line implementation of the PatchDataLayout
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline void PatchDataLayout::add_field(std::string field_name, u32 nvar) {
        bool found = false;

        for (var_t &fvar : fields) {
            std::visit(
                [&](auto &arg) {
                    if (field_name == arg.name) {
                        found = true;
                    }
                },
                fvar
            );
        }

        if (found) {
            throw std::invalid_argument("add_field -> the name already exists");
        }

        logger::info_ln("PatchDataLayout", "adding field :", field_name, nvar);

        fields.push_back(FieldDescriptor<T>(field_name, nvar));
    }

    template<class T>
    inline PatchDataLayout::FieldDescriptor<T> PatchDataLayout::get_field(std::string field_name) {

        for (var_t &fvar : fields) {
            if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fvar)) {
                if (pval->name == field_name) {
                    return *pval;
                }
            }
        }

        throw std::invalid_argument(
            "the requested field does not exists\n    current table : " + get_description_str()
        );
    }

    template<class T>
    inline PatchDataLayout::FieldDescriptor<T> PatchDataLayout::get_field(u32 idx) {

        if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fields[idx])) {
            return *pval;
        }

        throw std::invalid_argument(
            "the required type does no match at index "+std::to_string(idx)+"\n    current table : " + get_description_str()
        );
    }

    template<class T>
    inline u32 PatchDataLayout::get_field_idx(std::string field_name) {
        for (u32 i = 0; i < fields.size(); i++) {
            if (FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&fields[i])) {
                if (pval->name == field_name) {
                    return i;
                }
            }
        }

        throw std::invalid_argument(
            "the requested field does not exists\n    current table : " + get_description_str()
        );
    }

    template<class T>
    inline bool PatchDataLayout::check_field_type(u32 idx) {
        var_t &tmp = fields[idx];

        FieldDescriptor<T> *pval = std::get_if<FieldDescriptor<T>>(&tmp);

        if (pval) {
            return true;
        } else {
            return false;
        }
    }

} // namespace shamrock::patch