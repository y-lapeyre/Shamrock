
#include "aliases.hpp"
#include "core/sys/log.hpp"
#include "core/sys/sycl_handler.hpp"
#include "core/tree/radix_tree.hpp"
#include "models/generic/math/tensors/collections.hpp"
#include "models/generic/physics/fmm.hpp"


#include "unittests/shamrockbench.hpp"
#include "unittests/shamrocktest.hpp"
#include <cstdio>
#include <memory>
#include <vector>

#include <random>
#include <iostream>
#include <fstream>

template<class T,u32 order>
class FMM_prec_eval{public:
    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb);
    static T eval_prec_fmm_force(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb);
};

template<class T>
class FMM_prec_eval<T,5>{public:
    static constexpr u32 order = 5;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = SymTensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;
            f64 phi_2 = M_k.t2*a_k.t2;
            f64 phi_3 = M_k.t3*a_k.t3;
            f64 phi_4 = M_k.t4*a_k.t4;
            f64 phi_5 = M_k.t5*a_k.t5;

            f64 phi_val = phi_0+ phi_1+ phi_2+ phi_3+ phi_4+ phi_5;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }



    static T eval_prec_fmm_force(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        using moment_types = SymTensorCollection<f64,0,order-1>;

        constexpr u32 number_elem_multip = moment_types::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = moment_types::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = moment_types::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 1, order>::get_der_tensors(r_fmm);

            auto dM_k = get_dM_mat(D_n,Q_n);

            auto tensor_to_sycl = [](SymTensor3d_1<T> a){
                return sycl::vec<T, 3>{a.v_0,a.v_1,a.v_2};
            };

            auto dphi_0 = tensor_to_sycl(dM_k.t1*a_k.t0);
            auto dphi_1 = tensor_to_sycl(dM_k.t2*a_k.t1);
            auto dphi_2 = tensor_to_sycl(dM_k.t3*a_k.t2);
            auto dphi_3 = tensor_to_sycl(dM_k.t4*a_k.t3);
            auto dphi_4 = tensor_to_sycl(dM_k.t5*a_k.t4);

            f64_3 force_val = dphi_0+ dphi_1+ dphi_2+ dphi_3+ dphi_4;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_n = sycl::sqrt(sycl::dot(real_r,real_r));
            f64_3 f_th = real_r/(r_n*r_n*r_n);


            f64_3 delta = force_val-f_th;

            return sycl::distance(force_val,f_th)/sycl::length(f_th);

        }
    }
};



template<class T>
class FMM_prec_eval<T,4>{public:
    static constexpr u32 order = 4;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = SymTensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;
            f64 phi_2 = M_k.t2*a_k.t2;
            f64 phi_3 = M_k.t3*a_k.t3;
            f64 phi_4 = M_k.t4*a_k.t4;

            f64 phi_val = phi_0+ phi_1+ phi_2+ phi_3+ phi_4;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }


    static T eval_prec_fmm_force(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        using moment_types = SymTensorCollection<f64,0,order-1>;

        constexpr u32 number_elem_multip = moment_types::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = moment_types::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = moment_types::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 1, order>::get_der_tensors(r_fmm);

            auto dM_k = get_dM_mat(D_n,Q_n);

            auto tensor_to_sycl = [](SymTensor3d_1<T> a){
                return sycl::vec<T, 3>{a.v_0,a.v_1,a.v_2};
            };

            auto dphi_0 = tensor_to_sycl(dM_k.t1*a_k.t0);
            auto dphi_1 = tensor_to_sycl(dM_k.t2*a_k.t1);
            auto dphi_2 = tensor_to_sycl(dM_k.t3*a_k.t2);
            auto dphi_3 = tensor_to_sycl(dM_k.t4*a_k.t3);

            f64_3 force_val = dphi_0+ dphi_1+ dphi_2+ dphi_3;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_n = sycl::sqrt(sycl::dot(real_r,real_r));
            f64_3 f_th = real_r/(r_n*r_n*r_n);


            f64_3 delta = force_val-f_th;

            return sycl::distance(force_val,f_th)/sycl::length(f_th);

        }
    }
};




template<class T>
class FMM_prec_eval<T,3>{public:
    static constexpr u32 order = 3;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = SymTensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;
            f64 phi_2 = M_k.t2*a_k.t2;
            f64 phi_3 = M_k.t3*a_k.t3;

            f64 phi_val = phi_0+ phi_1+ phi_2+ phi_3;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }


    static T eval_prec_fmm_force(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        using moment_types = SymTensorCollection<f64,0,order-1>;

        constexpr u32 number_elem_multip = moment_types::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = moment_types::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = moment_types::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 1, order>::get_der_tensors(r_fmm);

            auto dM_k = get_dM_mat(D_n,Q_n);

            auto tensor_to_sycl = [](SymTensor3d_1<T> a){
                return sycl::vec<T, 3>{a.v_0,a.v_1,a.v_2};
            };

            auto dphi_0 = tensor_to_sycl(dM_k.t1*a_k.t0);
            auto dphi_1 = tensor_to_sycl(dM_k.t2*a_k.t1);
            auto dphi_2 = tensor_to_sycl(dM_k.t3*a_k.t2);

            f64_3 force_val = dphi_0+ dphi_1+ dphi_2;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_n = sycl::sqrt(sycl::dot(real_r,real_r));
            f64_3 f_th = real_r/(r_n*r_n*r_n);


            f64_3 delta = force_val-f_th;

            return sycl::distance(force_val,f_th)/sycl::length(f_th);

        }
    }
};



template<class T>
class FMM_prec_eval<T,2>{public:
    static constexpr u32 order = 2;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = SymTensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;
            f64 phi_2 = M_k.t2*a_k.t2;

            f64 phi_val = phi_0+ phi_1+ phi_2;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }

    static T eval_prec_fmm_force(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        using moment_types = SymTensorCollection<f64,0,order-1>;

        constexpr u32 number_elem_multip = moment_types::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = moment_types::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = moment_types::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 1, order>::get_der_tensors(r_fmm);

            auto dM_k = get_dM_mat(D_n,Q_n);

            auto tensor_to_sycl = [](SymTensor3d_1<T> a){
                return sycl::vec<T, 3>{a.v_0,a.v_1,a.v_2};
            };

            auto dphi_0 = tensor_to_sycl(dM_k.t1*a_k.t0);
            auto dphi_1 = tensor_to_sycl(dM_k.t2*a_k.t1);

            f64_3 force_val = dphi_0+ dphi_1;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_n = sycl::sqrt(sycl::dot(real_r,real_r));
            f64_3 f_th = real_r/(r_n*r_n*r_n);


            f64_3 delta = force_val-f_th;

            return sycl::distance(force_val,f_th)/sycl::length(f_th);

        }
    }
};






template<class T>
class FMM_prec_eval<T,1>{public:
    static constexpr u32 order = 1;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = SymTensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;
            f64 phi_1 = M_k.t1*a_k.t1;

            f64 phi_val = phi_0+ phi_1;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }


    static T eval_prec_fmm_force(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        using moment_types = SymTensorCollection<f64,0,order-1>;

        constexpr u32 number_elem_multip = moment_types::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = moment_types::from_vec(bj);

                B_n *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = moment_types::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 1, order>::get_der_tensors(r_fmm);

            auto dM_k = get_dM_mat(D_n,Q_n);

            auto tensor_to_sycl = [](SymTensor3d_1<T> a){
                return sycl::vec<T, 3>{a.v_0,a.v_1,a.v_2};
            };

            auto dphi_0 = tensor_to_sycl(dM_k.t1*a_k.t0);

            f64_3 force_val = dphi_0;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_n = sycl::sqrt(sycl::dot(real_r,real_r));
            f64_3 f_th = real_r/(r_n*r_n*r_n);


            f64_3 delta = force_val-f_th;

            return sycl::distance(force_val,f_th)/sycl::length(f_th);

        }
    }
};



template<class T>
class FMM_prec_eval<T,0>{public:
    static constexpr u32 order = 0;

    static T eval_prec_fmm_pot(sycl::vec<T, 3> xi,sycl::vec<T, 3> xj,sycl::vec<T, 3> sa,sycl::vec<T, 3> sb){

        f64 m_j = 1;

        std::vector<f64_3> pos_table = {xj};

        sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

        constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
        sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


        //compute multipoles
        {
            sycl::host_accessor<f64_3> pos {buf_pos};
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            for (u32 j = 0; j < pos_table.size(); j ++) {

                f64_3 xj = pos[j];

                f64_3 bj = xj - sb;

                auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);

                B_n.t0 *= m_j;

                B_n.store(multipoles, 0);
                
            }
        }


        //compute fmm
        {
            
            sycl::host_accessor<f64> multipoles {buf_multipoles};

            f64_3 r_fmm = sb-sa;

            f64_3 a_i = xi-sa;

            auto a_k = SymTensorCollection<f64, 0, order>::from_vec(a_i);

            auto Q_n = SymTensorCollection<f64,0,order>::load(multipoles, 0);

            auto D_n = GreenFuncGravCartesian<f64, 0, order>::get_der_tensors(r_fmm);

            auto M_k = get_M_mat(D_n,Q_n);

            f64 phi_0 = M_k.t0*a_k.t0;

            f64 phi_val = phi_0;

            //printf("contrib phi : %e %e %e %e %e %e\n",phi_0,phi_1,phi_2,phi_3,phi_4,phi_5);


            f64_3 real_r = xi-xj;

            f64 r_sq = sycl::sqrt(sycl::dot(real_r,real_r));
            f64 phi_th = 1/r_sq;


            //printf("phi (fmm = %e) (th  = %e) , delta = %e\n",phi_val,phi_th,phi_val-phi_th);

            return (phi_val-phi_th)/phi_th;

        }
    }
};





const std::string fmm_plot_script_pot = R"==(

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('GTK3Agg')


plt.style.use('custom_style.mplstyle')

res = np.loadtxt("fmm_pot_prec.txt")

res = res[np.argsort(res[:, 0])]


def plot_curve(X,Y,lab):

    ratio = 40

    cnt = len(X)//ratio 

    X_m = X.reshape((cnt,ratio))
    X_m = np.max(X_m,axis=1)

    Y_m = Y.reshape((cnt,ratio))
    Y_m = np.max(Y_m,axis=1)


    plt.plot(X_m,Y_m, label = lab)


plot_curve(res[:,0],np.abs(res[:,6]), "order 0")
plot_curve(res[:,0],np.abs(res[:,5]), "order 1")
plot_curve(res[:,0],np.abs(res[:,4]), "order 2")
plot_curve(res[:,0],np.abs(res[:,3]), "order 3")
plot_curve(res[:,0],np.abs(res[:,2]), "order 4")
plot_curve(res[:,0],np.abs(res[:,1]), "order 5")

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r"$\theta$")

plt.ylabel(r"$\vert \phi_{\rm fmm} - \phi_{\rm th} \vert /\vert \phi_{\rm th}\vert$")

plt.legend()
plt.grid()
plt.savefig("fmm_precision_pot.pdf")

)==";





const std::string fmm_plot_script_force = R"==(

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('GTK3Agg')


plt.style.use('custom_style.mplstyle')

res = np.loadtxt("fmm_force_prec.txt")

res = res[np.argsort(res[:, 0])]


def plot_curve(X,Y,lab):

    ratio = 40

    cnt = len(X)//ratio 

    X_m = X.reshape((cnt,ratio))
    X_m = np.max(X_m,axis=1)

    Y_m = Y.reshape((cnt,ratio))
    Y_m = np.max(Y_m,axis=1)


    plt.plot(X_m,Y_m, label = lab)


plot_curve(res[:,0],np.abs(res[:,5]), "order 1")
plot_curve(res[:,0],np.abs(res[:,4]), "order 2")
plot_curve(res[:,0],np.abs(res[:,3]), "order 3")
plot_curve(res[:,0],np.abs(res[:,2]), "order 4")
plot_curve(res[:,0],np.abs(res[:,1]), "order 5")

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r"$\theta$")

plt.ylabel(r"$\vert \mathbf{f}_{\rm fmm} - \mathbf{f}_{\rm th} \vert /\vert \mathbf{f}_{\rm th}\vert$")

plt.legend()
plt.grid()
plt.savefig("fmm_precision_force.pdf")

)==";






Test_start("fmm", tmp, 1){


    std::mt19937 eng(0x1111);


    std::uniform_real_distribution<f64> distf64(-1, 1);

    f64 avg_spa = 3e-3;
    std::uniform_real_distribution<f64> distf64_red(-avg_spa, avg_spa);


    std::ofstream prec_fmm_pot_out_file("fmm_pot_prec.txt");
    std::ofstream prec_fmm_force_out_file("fmm_force_prec.txt");
    for(u32 i = 0; i < 2e4; i++){


        f64_3 s_a = f64_3{distf64(eng), distf64(eng), distf64(eng)};
        f64_3 s_b = f64_3{distf64(eng), distf64(eng), distf64(eng)};

        f64_3 x_i = s_a + f64_3{distf64_red(eng), distf64_red(eng), distf64_red(eng)};
        f64_3 x_j = s_b + f64_3{distf64_red(eng), distf64_red(eng), distf64_red(eng)};

        auto dist_func = [](f64_3 a, f64_3 b){
            f64_3 d = a - b;

            f64_3 dabs = sycl::fabs(d);

            return sycl::max(sycl::max(dabs.x(),dabs.y()),dabs.z());
        };

        f64 angle = 2*(dist_func(x_i, s_a) + dist_func(x_j, s_b)) / dist_func(s_a, s_b);

        prec_fmm_pot_out_file << format(" %e %e %e %e %e %e %e\n"
            ,angle
            ,FMM_prec_eval<f64, 5>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 4>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 3>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 2>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 1>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 0>::eval_prec_fmm_pot(x_i, x_j, s_a, s_b));

        prec_fmm_force_out_file << format(" %e %e %e %e %e %e\n"
            ,angle
            ,FMM_prec_eval<f64, 5>::eval_prec_fmm_force(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 4>::eval_prec_fmm_force(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 3>::eval_prec_fmm_force(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 2>::eval_prec_fmm_force(x_i, x_j, s_a, s_b)
            ,FMM_prec_eval<f64, 1>::eval_prec_fmm_force(x_i, x_j, s_a, s_b));
    
    }
    prec_fmm_pot_out_file.close();
    prec_fmm_force_out_file.close();

    run_py_script(fmm_plot_script_pot);

    run_py_script(fmm_plot_script_force);

}


Test_start("fmm", multipole_moment_offset, 1){

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<f64> distf64(-1, 1);



    //f64_3 s_bp = f64_3{distf64(eng), distf64(eng), distf64(eng)};
    //f64_3 s_b  = f64_3{distf64(eng), distf64(eng), distf64(eng)};

    f64_3 s_bp = f64_3{1, 0, 0};
    f64_3 s_b  = f64_3{0, 0, 0};


    auto B_nB = SymTensorCollection<f64,0,5>::zeros();
    auto B_nBp = SymTensorCollection<f64,0,5>::zeros();
    
    for (u32 i = 0; i < 100; i++) {
        
        f64_3 x_1 = f64_3{distf64(eng), distf64(eng), distf64(eng)};

        f64_3 bj = x_1 - s_b;
        f64_3 bpj = x_1 - s_bp;

        auto tB_nB = SymTensorCollection<f64,0,5>::from_vec(bj);

        

        auto tB_nBp = SymTensorCollection<f64,0,5>::from_vec(bpj);

        B_nB.t0 += tB_nB.t0;
        B_nB.t1 += tB_nB.t1;
        B_nB.t2 += tB_nB.t2;
        B_nB.t3 += tB_nB.t3;
        B_nB.t4 += tB_nB.t4;
        B_nB.t5 += tB_nB.t5;

        B_nBp.t0 += tB_nBp.t0;
        B_nBp.t1 += tB_nBp.t1;
        B_nBp.t2 += tB_nBp.t2;
        B_nBp.t3 += tB_nBp.t3;
        B_nBp.t4 += tB_nBp.t4;
        B_nBp.t5 += tB_nBp.t5;

    }

    auto d = s_b-s_bp;


    SymTensorCollection<f64, 0, 5> d_ = SymTensorCollection<f64, 0, 5>::from_vec(d);

    auto B_nb_offseted = offset_multipole(B_nB,d);

    
    auto diff_0 = B_nBp.t0 - B_nb_offseted.t0;
    auto diff_1 = B_nBp.t1 - B_nb_offseted.t1;
    auto diff_2 = B_nBp.t2 - B_nb_offseted.t2;
    auto diff_3 = B_nBp.t3 - B_nb_offseted.t3;
    auto diff_4 = B_nBp.t4 - B_nb_offseted.t4;
    auto diff_5 = B_nBp.t5 - B_nb_offseted.t5;

    f64 diff = 
        (diff_0*diff_0) + 
        (diff_1*diff_1) + 
        (diff_2*diff_2) + 
        (diff_3*diff_3) + 
        (diff_4*diff_4) + 
        (diff_5*diff_5);

    printf("order 0 %f\n",B_nB.t0);




    printf("diff %e\n",diff);

    printf("dvec   %f %f %f\n",d.x(),d.y(),d.z());

    printf("np     %f %f %f\n",B_nBp.t1.v_0,B_nBp.t1.v_1,B_nBp.t1.v_2);
    printf("B      %f %f %f\n",B_nB.t1.v_0,B_nB.t1.v_1,B_nB.t1.v_2);
    printf("offset %f %f %f\n",B_nb_offseted.t1.v_0,B_nb_offseted.t1.v_1,B_nb_offseted.t1.v_2);
    
    printf("------ order 2 ---------\n");
    printf("B      %f %f %f %f %f %f\n",B_nB.t2.v_00,B_nB.t2.v_01,B_nB.t2.v_02,B_nB.t2.v_11,B_nB.t2.v_12,B_nB.t2.v_22);
    printf("np     %f %f %f %f %f %f\n",B_nBp.t2.v_00,B_nBp.t2.v_01,B_nBp.t2.v_02,B_nBp.t2.v_11,B_nBp.t2.v_12,B_nBp.t2.v_22);
    printf("offset %f %f %f %f %f %f\n",B_nb_offseted.t2.v_00,B_nb_offseted.t2.v_01,B_nb_offseted.t2.v_02,B_nb_offseted.t2.v_11,B_nb_offseted.t2.v_12,B_nb_offseted.t2.v_22);
    printf("d      %f %f %f %f %f %f\n",d_.t2.v_00,d_.t2.v_01,d_.t2.v_02,d_.t2.v_11,d_.t2.v_12,d_.t2.v_22);
    

    auto DG = GreenFuncGravCartesian<f64, 0, 5>::get_der_tensors(s_b - f64_3{5,1,2});


    printf("%e %e %e %e %e %e\n",(diff_0*diff_0) , 
        (diff_1*diff_1) , 
        (diff_2*diff_2) , 
        (diff_3*diff_3) , 
        (diff_4*diff_4) , 
        (diff_5*diff_5));
}






Test_start("fmm", radix_tree_fmm, 1){

    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 1e3;

    auto pos_part = std::make_unique<sycl::buffer<vec>>(npart);
    auto buf_force = std::make_unique<sycl::buffer<vec>>(npart);

    {
        sycl::host_accessor<vec> pos {*pos_part};
        sycl::host_accessor<vec> force {*buf_force};

        for (u32 i = 0; i < npart; i ++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
            force[i] = vec{0,0,0};
        }
    }

    auto rtree = Radix_Tree<morton_mode, vec>(
            sycl_handler::get_compute_queue(), 
            {vec{-1,-1,-1},vec{1,1,1}},
            pos_part, 
            npart 
        );

    rtree.compute_cellvolume(sycl_handler::get_compute_queue());

    constexpr u32 fmm_order = 4;

    
    
    u32 num_component_multipoles_fmm = (rtree.tree_internal_count + rtree.tree_leaf_count)*SymTensorCollection<flt,0,fmm_order>::num_component;
    logger::debug_sycl_ln("RTreeFMM", "allocating",num_component_multipoles_fmm,"component for multipoles");
    auto grav_multipoles = std::make_unique< sycl::buffer<flt>>( num_component_multipoles_fmm  );

    logger::debug_sycl_ln("RTreeFMM", "computing leaf moments (",rtree.tree_leaf_count,")");
    sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

        u32 offset_leaf = rtree.tree_internal_count;

        auto xyz = sycl::accessor {*pos_part, cgh,sycl::read_only};
        auto cell_particle_ids =sycl::accessor {*rtree.buf_reduc_index_map, cgh,sycl::read_only};
        auto particle_index_map = sycl::accessor {*rtree.buf_particle_index_map, cgh,sycl::read_only};
        auto cell_max = sycl::accessor{*rtree.buf_pos_max_cell_flt,cgh,sycl::read_only};
        auto cell_min = sycl::accessor{*rtree.buf_pos_min_cell_flt,cgh,sycl::read_only};
        auto multipoles = sycl::accessor {*grav_multipoles, cgh,sycl::write_only,sycl::no_init};


        sycl::range<1> range_leaf_cell{rtree.tree_leaf_count};

        cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
                u32 gid = (u32) item.get_id(0);

                u32 min_ids = cell_particle_ids[gid];
                u32 max_ids = cell_particle_ids[gid+1];

                vec cell_pmax = cell_max[offset_leaf + gid];
                vec cell_pmin = cell_min[offset_leaf + gid];

                vec s_b = (cell_pmax + cell_pmin)/2;

                auto B_n = SymTensorCollection<flt,0,fmm_order>::zeros();

                for(u32 id_s = min_ids; id_s < max_ids;id_s ++){
                    u32 idx_j = particle_index_map[id_s];
                    vec bj = xyz[idx_j] - s_b;

                    auto tB_n = SymTensorCollection<flt,0,fmm_order>::from_vec(bj);
    
                    constexpr flt m_j = 1;

                    tB_n *= m_j;
                    B_n += tB_n;
                }
                
                B_n.store(multipoles, (gid+offset_leaf)*SymTensorCollection<flt,0,fmm_order>::num_component);

            }
        );

    });



    auto buf_is_computed = std::make_unique< sycl::buffer<u8>>( (rtree.tree_internal_count + rtree.tree_leaf_count)  );

    sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
        auto is_computed = sycl::accessor {*buf_is_computed, cgh , sycl::write_only, sycl::no_init};
        sycl::range<1> range_internal_count{rtree.tree_internal_count + rtree.tree_leaf_count};

        u32 int_cnt = rtree.tree_internal_count;

        cgh.parallel_for(range_internal_count, [=](sycl::item<1> item) {
            is_computed[item] = item.get_linear_id() >= int_cnt;
        });
    });



    for (u32 iter = 0; iter < 64 ; iter ++) {
    
        sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

            u32 leaf_offset = rtree.tree_internal_count;

            auto cell_max = sycl::accessor{*rtree.buf_pos_max_cell_flt,cgh,sycl::read_only};
            auto cell_min = sycl::accessor{*rtree.buf_pos_min_cell_flt,cgh,sycl::read_only};
            auto multipoles = sycl::accessor {*grav_multipoles, cgh,sycl::read_write};
            auto is_computed = sycl::accessor {*buf_is_computed, cgh , sycl::read_write};

            sycl::range<1> range_internal_count{rtree.tree_internal_count};

            auto rchild_id   = sycl::accessor{*rtree.buf_rchild_id  ,cgh,sycl::read_only};
            auto lchild_id   = sycl::accessor{*rtree.buf_lchild_id  ,cgh,sycl::read_only};
            auto rchild_flag = sycl::accessor{*rtree.buf_rchild_flag,cgh,sycl::read_only};
            auto lchild_flag = sycl::accessor{*rtree.buf_lchild_flag,cgh,sycl::read_only};

            cgh.parallel_for(range_internal_count, [=](sycl::item<1> item) {

                u32 cid = item.get_linear_id();

                u32 lid = lchild_id[cid] + leaf_offset * lchild_flag[cid];
                u32 rid = rchild_id[cid] + leaf_offset * rchild_flag[cid];

                bool should_compute = (!is_computed[cid]) && (is_computed[lid] && is_computed[rid]);

                if(should_compute){

                    vec cell_pmax = cell_max[cid];
                    vec cell_pmin = cell_min[cid];

                    vec sbp = (cell_pmax + cell_pmin)/2;

                    auto B_n = SymTensorCollection<flt,0,fmm_order>::zeros();



                    auto add_multipole_offset = [&](u32 s_cid){
                        vec s_cell_pmax = cell_max[s_cid];
                        vec s_cell_pmin = cell_min[s_cid];

                        vec sb = (s_cell_pmax + s_cell_pmin)/2;

                        auto d = sb-sbp;

                        auto B_ns = SymTensorCollection<flt, 0, fmm_order>::load(multipoles, s_cid*SymTensorCollection<flt,0,fmm_order>::num_component);

                        auto B_ns_offseted = offset_multipole(B_ns,d);

                        B_n += B_ns_offseted;

                    };


                    add_multipole_offset(lid);
                    add_multipole_offset(rid);

                    is_computed[cid] = true;
                    B_n.store(multipoles,cid*SymTensorCollection<flt,0,fmm_order>::num_component);

                }

            });

        });

    }



    

    sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

        using Rta = walker::Radix_tree_accessor<morton_mode, vec>;
        Rta tree_acc(rtree, cgh);

        sycl::range<1> range_leaf = sycl::range<1>{rtree.tree_leaf_count};

        u32 leaf_offset = rtree.tree_internal_count;

        auto xyz = sycl::accessor {*pos_part, cgh,sycl::read_only};
        auto fxyz = sycl::accessor {*buf_force, cgh,sycl::read_write};

        auto multipoles = sycl::accessor {*grav_multipoles, cgh,sycl::read_only};


        cgh.parallel_for(range_leaf, [=](sycl::item<1> item) {

            u32 id_cell_a = (u32)item.get_id(0) + leaf_offset;

            vec cur_pos_min_cell_a = tree_acc.pos_min_cell[id_cell_a];
            vec cur_pos_max_cell_a = tree_acc.pos_max_cell[id_cell_a];

            vec sa = (cur_pos_min_cell_a + cur_pos_max_cell_a)/2;

            vec dc_a = (cur_pos_max_cell_a - cur_pos_min_cell_a);

            flt l_cell_a = sycl::max(sycl::max(dc_a.x(),dc_a.y()),dc_a.z());

            auto dM_k = SymTensorCollection<flt, 1, 5>::zeros();

            constexpr flt open_crit = 0.1;
            constexpr flt open_crit_sq = open_crit*open_crit;

            walker::rtree_for_cell(
                tree_acc,
                [&tree_acc,&cur_pos_min_cell_a,&cur_pos_max_cell_a,&sa,&l_cell_a](u32 id_cell_b){
                    vec cur_pos_min_cell_b = tree_acc.pos_min_cell[id_cell_b];
                    vec cur_pos_max_cell_b = tree_acc.pos_max_cell[id_cell_b];

                    vec dc_b = (cur_pos_max_cell_b - cur_pos_min_cell_b);

                    vec sb = (cur_pos_min_cell_b + cur_pos_max_cell_b)/2;
                    vec r_fmm = sb-sa;

                    flt l_cell_b = sycl::max(sycl::max(dc_b.x(),dc_b.y()),dc_b.z());

                    flt opening_angle_sq = (l_cell_a + l_cell_b)*(l_cell_a + l_cell_b)/sycl::dot(r_fmm,r_fmm);

                    using namespace walker::interaction_crit;

                    return sph_cell_cell_crit(
                        cur_pos_min_cell_a, 
                        cur_pos_max_cell_a, 
                        cur_pos_min_cell_b,
                        cur_pos_max_cell_b, 
                        0, 
                        0) || (opening_angle_sq > open_crit_sq);
                },
                [&](u32 node_b) {

                    vec cur_pos_min_cell_b = tree_acc.pos_min_cell[node_b];
                    vec cur_pos_max_cell_b = tree_acc.pos_max_cell[node_b];

                    vec dc_b = (cur_pos_max_cell_b - cur_pos_min_cell_b);

                    vec sb = (cur_pos_min_cell_b + cur_pos_max_cell_b)/2;
                    vec r_fmm = sb-sa;

                    flt l_cell_b = sycl::max(sycl::max(dc_b.x(),dc_b.y()),dc_b.z());

                    flt opening_angle_sq = (l_cell_a + l_cell_b)*(l_cell_a + l_cell_b)/sycl::dot(r_fmm,r_fmm);

                    if(opening_angle_sq < open_crit_sq){
                        //this is useless this lambda is already executed only if cd above true
                        auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(multipoles,node_b*SymTensorCollection<flt,0,fmm_order>::num_component);
                        auto D_n = GreenFuncGravCartesian<flt, 1, fmm_order+1>::get_der_tensors(r_fmm);
                        
                        dM_k += get_dM_mat(D_n,Q_n);
                    }else{
                        walker::iter_object_in_cell(tree_acc, id_cell_a, [&](u32 id_a){

                            vec x_i = xyz[id_a];
                            vec sum_fi{0,0,0};

                            walker::iter_object_in_cell(tree_acc, node_b, [&](u32 id_b){

                                if(id_a != id_b){
                                    vec x_j = xyz[id_b];

                                    vec real_r = x_i-x_j;

                                    flt r_n = sycl::sqrt(sycl::dot(real_r,real_r));
                                    sum_fi += real_r/(r_n*r_n*r_n);
                                }

                            });

                            fxyz[id_a] += sum_fi;

                        });
                    }

                    

                },
                [&](u32 node_b){

                    vec sb = (tree_acc.pos_min_cell[node_b] + tree_acc.pos_max_cell[node_b])/2;
                    vec r_fmm = sb-sa;

                    auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(multipoles,node_b*SymTensorCollection<flt,0,fmm_order>::num_component);
                    auto D_n = GreenFuncGravCartesian<flt, 1, fmm_order+1>::get_der_tensors(r_fmm);
                    
                    dM_k += get_dM_mat(D_n,Q_n);

                }
            );



            walker::iter_object_in_cell(tree_acc, id_cell_a, [&](u32 id_a){

                auto ai = SymTensorCollection<flt, 0, fmm_order>::from_vec(xyz[id_a]);

                auto tensor_to_sycl = [](SymTensor3d_1<flt> a){
                    return vec{a.v_0,a.v_1,a.v_2};
                };

                auto dphi_0 = tensor_to_sycl(dM_k.t1*ai.t0);
                auto dphi_1 = tensor_to_sycl(dM_k.t2*ai.t1);
                auto dphi_2 = tensor_to_sycl(dM_k.t3*ai.t2);
                auto dphi_3 = tensor_to_sycl(dM_k.t4*ai.t3);
                auto dphi_4 = tensor_to_sycl(dM_k.t5*ai.t4);

                fxyz[id_a]  += dphi_0+ dphi_1+ dphi_2+ dphi_3+ dphi_4;

            });

        });


    });





    sycl_handler::get_compute_queue().wait();



    {

        sycl::host_accessor<vec> pos {*pos_part};
        sycl::host_accessor<vec> force {*buf_force};

        flt err_sum = 0;
        flt err_max = 0;

        for (u32 i = 0; i < npart; i++) {
            vec sum_fi{0,0,0};

            vec x_i = pos[i];

            for(u32 j = 0; j < npart; j++){

                if (i!=j) {

                    vec x_j = pos[j];

                    vec real_r = x_i-x_j;

                    flt r_n = sycl::sqrt(sycl::dot(real_r,real_r));
                    sum_fi += real_r/(r_n*r_n*r_n);

                }

            }

            flt err = sycl::distance(force[i],sum_fi)/sycl::length(sum_fi);

            if(i<10){
                logger::raw_ln("local relative error : ",format("%e (%e %e %e) (%e %e %e)",err, force[i].x(),force[i].y(),force[i].z(), sum_fi.x(),sum_fi.y(),sum_fi.z()));
            }
            err_sum += err;

            err_max = sycl::max(err_max,err);

        }

        logger::raw_ln("global relative error :",format("avg = %e max = %e",err_sum/npart,err_max));
    }



}



Bench_start("fmm shit", "multipole_compute", fmm_perf_multipole, 1){
    
        
    std::vector<f64_3> pos_table = {{1,1,1}};

    sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

    constexpr u32 order = 5;

    constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
    sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


    //compute multipoles
    auto l = [&]{
        sycl::host_accessor<f64_3> pos {buf_pos};
        sycl::host_accessor<f64> multipoles {buf_multipoles};

        f64_3 sa = {0,0,0};
        f64_3 sb = {0,0,0};

        for (u32 j = 0; j < pos_table.size(); j ++) {

            f64_3 xj = pos[j];

            f64_3 bj = xj - sb;

            auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);


            
        }

    };

    TimeitFor(100,l())

}


Bench_start("fmm shit", "multipole_offset", fmm_perf_offset, 1){
    
        
    std::vector<f64_3> pos_table = {{1,1,1}};

    sycl::buffer<f64_3> buf_pos = sycl::buffer<f64_3>(pos_table.data(), pos_table.size());

    constexpr u32 order = 5;

    constexpr u32 number_elem_multip = SymTensorCollection<f64,0,order>::num_component;
    sycl::buffer<f64> buf_multipoles = sycl::buffer<f64>(number_elem_multip);


    //compute multipoles
    auto l = [&]{
        sycl::host_accessor<f64_3> pos {buf_pos};
        sycl::host_accessor<f64> multipoles {buf_multipoles};

        f64_3 sa = {0,0,0};
        f64_3 sb = {0,0,0};

        for (u32 j = 0; j < pos_table.size(); j ++) {

            f64_3 xj = pos[j];

            f64_3 bj = xj - sb;

            auto B_n = SymTensorCollection<f64,0,order>::from_vec(bj);


            
        }

    };

    l();



    sycl::buffer<f64> buf_multipoles2 = sycl::buffer<f64>(number_elem_multip);


    auto l2 = [&]{
        sycl::host_accessor<f64> multipoles_old {buf_multipoles};
        sycl::host_accessor<f64> multipoles_new {buf_multipoles2};

        f64_3 d_vec = {1,2,3};

        for (u32 j = 0; j < pos_table.size(); j ++) {

            auto B_n = SymTensorCollection<f64,0,order>::load(multipoles_old,0);

            offset_multipole(B_n,d_vec).store(multipoles_new, 0);
            
        }

    };


    TimeitFor(1000,l2())

}










Bench_start("fmm", "compute_tree_multipoles", compute_tree_multipoles, 1){

    using flt = f32;
    using vec = sycl::vec<flt,3>;
    using morton_mode = u32;

    std::mt19937 eng(0x1111);
    std::uniform_real_distribution<flt> distf(-1, 1);

    constexpr u32 npart = 1e7;

    auto pos_part = std::make_unique<sycl::buffer<vec>>(npart);
    auto buf_force = std::make_unique<sycl::buffer<vec>>(npart);

    {
        sycl::host_accessor<vec> pos {*pos_part};
        sycl::host_accessor<vec> force {*buf_force};

        for (u32 i = 0; i < npart; i ++) {
            pos[i] = vec{distf(eng), distf(eng), distf(eng)};
            force[i] = vec{0,0,0};
        }
    }



    auto bench_lambda = [&pos_part,&buf_force](){

    auto rtree = Radix_Tree<morton_mode, vec>(
            sycl_handler::get_compute_queue(), 
            {vec{-1,-1,-1},vec{1,1,1}},
            pos_part, 
            npart 
        );

    rtree.compute_cellvolume(sycl_handler::get_compute_queue());

    constexpr u32 fmm_order = 4;

    
    
    u32 num_component_multipoles_fmm = (rtree.tree_internal_count + rtree.tree_leaf_count)*SymTensorCollection<flt,0,fmm_order>::num_component;
    logger::debug_sycl_ln("RTreeFMM", "allocating",num_component_multipoles_fmm,"component for multipoles");
    auto grav_multipoles = std::make_unique< sycl::buffer<flt>>( num_component_multipoles_fmm  );

    logger::debug_sycl_ln("RTreeFMM", "computing leaf moments (",rtree.tree_leaf_count,")");
    sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

        u32 offset_leaf = rtree.tree_internal_count;

        auto xyz = sycl::accessor {*pos_part, cgh,sycl::read_only};
        auto cell_particle_ids =sycl::accessor {*rtree.buf_reduc_index_map, cgh,sycl::read_only};
        auto particle_index_map = sycl::accessor {*rtree.buf_particle_index_map, cgh,sycl::read_only};
        auto cell_max = sycl::accessor{*rtree.buf_pos_max_cell_flt,cgh,sycl::read_only};
        auto cell_min = sycl::accessor{*rtree.buf_pos_min_cell_flt,cgh,sycl::read_only};
        auto multipoles = sycl::accessor {*grav_multipoles, cgh,sycl::write_only,sycl::no_init};


        sycl::range<1> range_leaf_cell{rtree.tree_leaf_count};

        cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
                u32 gid = (u32) item.get_id(0);

                u32 min_ids = cell_particle_ids[gid];
                u32 max_ids = cell_particle_ids[gid+1];

                vec cell_pmax = cell_max[offset_leaf + gid];
                vec cell_pmin = cell_min[offset_leaf + gid];

                vec s_b = (cell_pmax + cell_pmin)/2;

                auto B_n = SymTensorCollection<flt,0,fmm_order>::zeros();

                for(u32 id_s = min_ids; id_s < max_ids;id_s ++){
                    u32 idx_j = particle_index_map[id_s];
                    vec bj = xyz[idx_j] - s_b;

                    auto tB_n = SymTensorCollection<flt,0,fmm_order>::from_vec(bj);
    
                    constexpr flt m_j = 1;

                    tB_n *= m_j;
                    B_n += tB_n;
                }
                
                B_n.store(multipoles, (gid+offset_leaf)*SymTensorCollection<flt,0,fmm_order>::num_component);

            }
        );

    });



    auto buf_is_computed = std::make_unique< sycl::buffer<u8>>( (rtree.tree_internal_count + rtree.tree_leaf_count)  );

    sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {
        auto is_computed = sycl::accessor {*buf_is_computed, cgh , sycl::write_only, sycl::no_init};
        sycl::range<1> range_internal_count{rtree.tree_internal_count + rtree.tree_leaf_count};

        u32 int_cnt = rtree.tree_internal_count;

        cgh.parallel_for(range_internal_count, [=](sycl::item<1> item) {
            is_computed[item] = item.get_linear_id() >= int_cnt;
        });
    });



    for (u32 iter = 0; iter < 64 ; iter ++) {
    
        sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

            u32 leaf_offset = rtree.tree_internal_count;

            auto cell_max = sycl::accessor{*rtree.buf_pos_max_cell_flt,cgh,sycl::read_only};
            auto cell_min = sycl::accessor{*rtree.buf_pos_min_cell_flt,cgh,sycl::read_only};
            auto multipoles = sycl::accessor {*grav_multipoles, cgh,sycl::read_write};
            auto is_computed = sycl::accessor {*buf_is_computed, cgh , sycl::read_write};

            sycl::range<1> range_internal_count{rtree.tree_internal_count};

            auto rchild_id   = sycl::accessor{*rtree.buf_rchild_id  ,cgh,sycl::read_only};
            auto lchild_id   = sycl::accessor{*rtree.buf_lchild_id  ,cgh,sycl::read_only};
            auto rchild_flag = sycl::accessor{*rtree.buf_rchild_flag,cgh,sycl::read_only};
            auto lchild_flag = sycl::accessor{*rtree.buf_lchild_flag,cgh,sycl::read_only};

            cgh.parallel_for(range_internal_count, [=](sycl::item<1> item) {

                u32 cid = item.get_linear_id();

                u32 lid = lchild_id[cid] + leaf_offset * lchild_flag[cid];
                u32 rid = rchild_id[cid] + leaf_offset * rchild_flag[cid];

                bool should_compute = (!is_computed[cid]) && (is_computed[lid] && is_computed[rid]);

                if(should_compute){

                    vec cell_pmax = cell_max[cid];
                    vec cell_pmin = cell_min[cid];

                    vec sbp = (cell_pmax + cell_pmin)/2;

                    auto B_n = SymTensorCollection<flt,0,fmm_order>::zeros();



                    auto add_multipole_offset = [&](u32 s_cid){
                        vec s_cell_pmax = cell_max[s_cid];
                        vec s_cell_pmin = cell_min[s_cid];

                        vec sb = (s_cell_pmax + s_cell_pmin)/2;

                        auto d = sb-sbp;

                        auto B_ns = SymTensorCollection<flt, 0, fmm_order>::load(multipoles, s_cid*SymTensorCollection<flt,0,fmm_order>::num_component);

                        auto B_ns_offseted = offset_multipole(B_ns,d);

                        B_n += B_ns_offseted;

                    };


                    add_multipole_offset(lid);
                    add_multipole_offset(rid);

                    is_computed[cid] = true;
                    B_n.store(multipoles,cid*SymTensorCollection<flt,0,fmm_order>::num_component);

                }

            });

        });

    }



    

    sycl_handler::get_compute_queue().submit([&](sycl::handler &cgh) {

        using Rta = walker::Radix_tree_accessor<morton_mode, vec>;
        Rta tree_acc(rtree, cgh);

        sycl::range<1> range_leaf = sycl::range<1>{rtree.tree_leaf_count};

        u32 leaf_offset = rtree.tree_internal_count;

        auto xyz = sycl::accessor {*pos_part, cgh,sycl::read_only};
        auto fxyz = sycl::accessor {*buf_force, cgh,sycl::read_write};

        auto multipoles = sycl::accessor {*grav_multipoles, cgh,sycl::read_only};


        cgh.parallel_for(range_leaf, [=](sycl::item<1> item) {

            u32 id_cell_a = (u32)item.get_id(0) + leaf_offset;

            vec cur_pos_min_cell_a = tree_acc.pos_min_cell[id_cell_a];
            vec cur_pos_max_cell_a = tree_acc.pos_max_cell[id_cell_a];

            vec sa = (cur_pos_min_cell_a + cur_pos_max_cell_a)/2;

            auto dM_k = SymTensorCollection<flt, 1, fmm_order+1>::zeros();

            walker::rtree_for_cell(
                tree_acc,
                [&tree_acc,&cur_pos_min_cell_a,&cur_pos_max_cell_a](u32 id_cell_b){
                    vec cur_pos_min_cell_b = tree_acc.pos_min_cell[id_cell_b];
                    vec cur_pos_max_cell_b = tree_acc.pos_max_cell[id_cell_b];

                    using namespace walker::interaction_crit;

                    return sph_cell_cell_crit(
                        cur_pos_min_cell_a, 
                        cur_pos_max_cell_a, 
                        cur_pos_min_cell_b,
                        cur_pos_max_cell_b, 
                        0, 
                        0);
                },
                [&](u32 node_b) {

                    vec sb = (tree_acc.pos_min_cell[node_b] + tree_acc.pos_max_cell[node_b])/2;
                    vec r_fmm = sb-sa;

                    auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(multipoles,node_b*SymTensorCollection<flt,0,fmm_order>::num_component);
                    auto D_n = GreenFuncGravCartesian<flt, 1, fmm_order+1>::get_der_tensors(r_fmm);
                    
                    dM_k += get_dM_mat(D_n,Q_n);

                },
                [&](u32 node_b){

                    vec sb = (tree_acc.pos_min_cell[node_b] + tree_acc.pos_max_cell[node_b])/2;
                    vec r_fmm = sb-sa;

                    auto Q_n = SymTensorCollection<flt, 0, fmm_order>::load(multipoles,node_b*SymTensorCollection<flt,0,fmm_order>::num_component);
                    auto D_n = GreenFuncGravCartesian<flt, 1, fmm_order+1>::get_der_tensors(r_fmm);
                    
                    dM_k += get_dM_mat(D_n,Q_n);

                }
            );



            walker::iter_object_in_cell(tree_acc, id_cell_a, [&](u32 id_a){

                auto ai = SymTensorCollection<flt, 0, fmm_order>::from_vec(xyz[id_a]);

                auto tensor_to_sycl = [](SymTensor3d_1<flt> a){
                    return vec{a.v_0,a.v_1,a.v_2};
                };

                auto dphi_0 = tensor_to_sycl(dM_k.t1*ai.t0);
                auto dphi_1 = tensor_to_sycl(dM_k.t2*ai.t1);
                auto dphi_2 = tensor_to_sycl(dM_k.t3*ai.t2);
                auto dphi_3 = tensor_to_sycl(dM_k.t4*ai.t3);
                auto dphi_4 = tensor_to_sycl(dM_k.t5*ai.t4);

                fxyz[id_a]  += dphi_0+ dphi_1+ dphi_2+ dphi_3+ dphi_4;

            });

        });


    });



    sycl_handler::get_compute_queue().wait();


    };

    TimeitFor(10, bench_lambda());



}