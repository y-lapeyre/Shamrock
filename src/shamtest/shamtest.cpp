// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

//%Impl status : Good

#include "shamtest.hpp"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <pybind11/embed.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "shambase/bytestream.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/time.hpp"
#include "shamsys/legacy/cmdopt.hpp"
#include "shamsys/legacy/log.hpp"

#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/sycl_handler.hpp"

#include "shambase/exception.hpp"

#include "shambindings/pybindaliases.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/details/reporters/texTestReport.hpp"

namespace shamtest {

    bool is_run_only         = false;
    bool is_full_output_mode = false;

    /**
     * @brief print the line in terminal when a test start
     *
     * @param test
     * @param test_num
     * @param test_count
     */
    void _start_test_print(details::Test &test, u32 test_num, u32 test_count) {

        if (is_run_only) {
            printf("- : ");
        } else {
            printf("- [%d/%d] :", test_num + 1, test_count);
        }

        bool any_node_cnt = test.node_count == -1;
        if (any_node_cnt) {
            printf(" [any] ");
        } else {
            printf(" [%03d] ", test.node_count);
        }

        std::cout << "\033[;34m" << test.name << "\033[0m " << std::endl;
    }

    /**
     * @brief print the line in terminal when a test ends
     *
     * @param res
     * @param timer
     */
    void _end_test_print(details::TestResult &res, shambase::Timer &timer) {

        for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {

            if (is_full_output_mode || (!res.asserts.asserts[j].value)) {
                printf("        [%d/%zu] : ", j + 1, res.asserts.asserts.size());
                printf("%-20s", res.asserts.asserts[j].name.c_str());

                if (res.asserts.asserts[j].value) {
                    std::cout << "  (\033[;32mSucces\033[0m)\n";
                } else {
                    std::cout << "  (\033[1;31m Fail \033[0m)\n";
                    if (!res.asserts.asserts[j].comment.empty()) {
                        std::cout << "----- logs : \n"
                                  << res.asserts.asserts[j].comment << "\n-----" << std::endl;
                    }
                }
            }
        }

        u32 succes_cnt = 0;
        for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {
            if (res.asserts.asserts[j].value) {
                succes_cnt++;
            }
        }

        if (succes_cnt == res.asserts.asserts.size()) {
            std::cout << "   -> Result : \033[;32mSucces\033[0m";
        } else {
            std::cout << "   -> Result : \033[1;31m Fail \033[0m";
        }

        std::string s_assert =
            shambase::format(" [{}/{}] ", succes_cnt, res.asserts.asserts.size());
        printf("%-15s", s_assert.c_str());
        std::cout << " (" << timer.get_time_str() << ")" << std::endl;

        std::cout << std::endl;
    }

    /**
     * @brief check is a test is failed
     *
     * @param res
     * @return true
     * @return false
     */
    bool is_test_failed(details::TestResult &res) {
        for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {

            if (!res.asserts.asserts[j].value) {
                return true;
            }
        }

        return false;
    }

    std::string gen_fail_log(details::TestResult &res) {
        std::string out = "";

        std::string sep = "\n-------------------------------------\n";

        out += " - Test : \033[;34m" + res.name +
               "\033[0m world_rank = " + std::to_string(res.world_rank);
        out += "\n    Assertion list :\n";

        for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {

            out += shambase::format_printf("     - [%d/%zu] ", j + 1, res.asserts.asserts.size());
            out += shambase::format_printf("%-20s", res.asserts.asserts[j].name.c_str());

            if (res.asserts.asserts[j].value) {
                out += "  (\033[;32mSucces\033[0m)\n";
            } else {
                out += "  (\033[1;31m Fail \033[0m)\n";
            }

            if ((!res.asserts.asserts[j].value) && !res.asserts.asserts[j].comment.empty()) {
                out += " -> failed assert logs : " + sep + res.asserts.asserts[j].comment + sep;
            }
        }

        out += "\n\n";

        return out;
    }

    void _print_summary(std::vector<details::TestResult> &results) {
        if (shamsys::instance::world_rank > 0) {
            return;
        }

        logger::print_faint_row();
        logger::print_faint_row();
        logger::print_faint_row();
        logger::raw_ln(terminal_effects::bold + "Test Report :" + terminal_effects::reset);
        logger::raw_ln();

        u32 test_count  = results.size();
        u32 succ_count  = 0;
        u32 fail_count  = 0;
        std::string log = "";
        for (details::TestResult &res : results) {
            if (!is_test_failed(res)) {
                succ_count++;
            } else {
                fail_count++;
                log += gen_fail_log(res);
            }
        }

        std::cout << "Test suite status : ";
        if (fail_count == 0) {
            std::cout << "  (\033[;32mSucces\033[0m)";
            printf(" [%d/%d] \n", succ_count, test_count);
        } else {
            std::cout << "  (\033[1;31m Fail \033[0m)";
            printf(" [%d/%d] \n", succ_count, test_count);
            std::cout << "\nFailed tests : \n\n" << log;
        }

        logger::print_faint_row();
        logger::print_faint_row();
        logger::print_faint_row();
    }

    std::basic_string<u8> gather_basic_string(std::basic_string<u8> in) {
        using namespace shamsys;

        std::basic_string<u8> out_res_string;

        if (instance::world_size == 1) {
            out_res_string = in;
        } else {
            std::basic_string<u8> loc_string = in;

            int *counts   = new int[instance::world_size];
            int nelements = (int)loc_string.size();
            // Each process tells the root how many elements it holds
            mpi::gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Displacements in the receive buffer for MPI_GATHERV
            int *disps = new int[instance::world_size];
            // Displacement for the first chunk of data - 0
            for (int i = 0; i < instance::world_size; i++)
                disps[i] = (i > 0) ? (disps[i - 1] + counts[i - 1]) : 0;

            // Place to hold the gathered data
            // Allocate at root only
            u8 *gather_data = NULL;
            if (instance::world_rank == 0)
                // disps[size-1]+counts[size-1] == total number of elements
                gather_data =
                    new u8[disps[instance::world_size - 1] + counts[instance::world_size - 1]];

            // Collect everything into the root
            mpi::gatherv(
                loc_string.c_str(),
                nelements,
                MPI_CHAR,
                gather_data,
                counts,
                disps,
                MPI_CHAR,
                0,
                MPI_COMM_WORLD);

            if (instance::world_rank == 0) {
                out_res_string = std::basic_string<u8>(
                    gather_data,
                    disps[instance::world_size - 1] + counts[instance::world_size - 1]);
            }

            delete[] counts;
            delete[] disps;
        }

        return out_res_string;
    }

    std::vector<details::TestResult> gather_tests(std::vector<details::TestResult> rank_result) {
        if (shamsys::instance::world_size == 1) {
            return rank_result;
        }

        // generate payload
        std::basic_stringstream<u8> outrank;

        shambase::stream_write_vector(outrank, rank_result);

        std::basic_string<u8> gathered = gather_basic_string(outrank.str());

        if (shamsys::instance::world_rank != 0) {
            return {};
        }

        logger::print_faint_row();

        logger::raw_ln("Test result gathered :", gathered.size(), "bytes");

        std::basic_stringstream<u8> reader(gathered);

        std::vector<details::TestResult> out;

        for (u32 i = 0; i < shamsys::instance::world_size; i++) {
            shambase::stream_read_vector(reader, out);
        }

        return out;
    }

    /**
     * @brief print all the available tests
     *
     */
    void print_test_list() {

        if (shamsys::instance::world_rank > 0) {
            return;
        }

        using namespace shamtest::details;

        auto print_list = [&](TestType t) {
            for (auto test : static_init_vec_tests) {
                if (test.type == t) {
                    if (test.node_count == -1) {
                        printf("- [any] %-15s\n", test.name.c_str());
                    } else {
                        printf("- [%03d] %-15s\n", test.node_count, test.name.c_str());
                    }
                }
            }
        };

        printf("--- Benchmark ---\n");

        print_list(Benchmark);

        printf("--- Analysis  ---\n");

        print_list(Analysis);

        printf("--- Unittest  ---\n");

        print_list(Unittest);
    }

    void write_json_report(std::vector<details::TestResult> &results) {
        if (shamsys::instance::world_rank > 0) {
            return;
        }

        std::stringstream rank_test_res_out;
        for (details::TestResult &res : results) {
            rank_test_res_out << res.serialize_json() << ",";
        }

        std::string out_res_string = rank_test_res_out.str();

        // generate json output and write it into the specified file

        if (out_res_string.back() == ',') {
            out_res_string = out_res_string.substr(0, out_res_string.size() - 1);
        }

        std::string s_out;

        s_out = "{\n";

        s_out += R"(    "commit_hash" : ")" + git_commit_hash + "\",\n";
        s_out +=
            R"(    "world_size" : ")" + std::to_string(shamsys::instance::world_size) + "\",\n";

#if defined(SYCL_COMP_INTEL_LLVM)
        s_out += R"(    "compiler" : "DPCPP",)"
                 "\n";
#elif defined(SYCL_COMP_HIPSYCL)
        s_out += R"(    "compiler" : "HipSYCL",)"
                 "\n";
#else
        s_out += R"(    "compiler" : "Unknown",)"
                 "\n";
#endif

        s_out += R"(    "comp_args" : ")" + compile_arg + "\",\n";

        s_out += R"(    "results" : )"
                 "[\n\n";
        s_out += shambase::increase_indent(out_res_string);
        s_out += "\n    ]\n}";

        // printf("%s\n",s_out.c_str());

        shambase::write_string_to_file(std::string(opts::get_option("-o")), s_out);
    }

    void write_tex_report(std::vector<details::TestResult> &results, bool mark_fail) {
        if (shamsys::instance::world_rank > 0) {
            return;
        }

        logger::raw("write report Tex : ");

        shambase::write_string_to_file("tests/report.tex", details::make_test_report_tex(results,mark_fail));

        logger::raw_ln("Done (tests/report.tex)");
    }

    int
    run_all_tests(int argc, char *argv[], bool run_bench, bool run_analysis, bool run_unittest) {
        StackEntry stack{};

        using namespace shamtest::details;

        if (opts::has_option("--test-list")) {
            print_test_list();
            return 0;
        }

        std::string run_only_name = "";
        if (opts::has_option("--run-only")) {
            run_only_name = opts::get_option("--run-only");
            is_run_only   = true;
        }

        is_full_output_mode = opts::has_option("--full-output");

        bool out_to_file = opts::has_option("-o");

        if (out_to_file) {
            if (opts::get_option("-o").size() == 0) {
                opts::print_help();
            }
        }

        using namespace shamsys;

        if (!is_run_only) {
            printf("\n------------ Tests list --------------\n");
        }

        std::vector<u32> selected_tests = {};

        auto can_run = [&](shamtest::details::Test &t) -> bool {
            bool any_node_cnt  = (t.node_count == -1);
            bool world_size_ok = t.node_count == instance::world_size;

            bool can_run_type = false;

            auto test_type = t.type;
            can_run_type   = can_run_type || (run_unittest && (Unittest == test_type));
            can_run_type   = can_run_type || (run_analysis && (Analysis == test_type));
            can_run_type   = can_run_type || (run_bench && (Benchmark == test_type));

            return can_run_type && (any_node_cnt || world_size_ok);
        };

        auto print_test = [&](shamtest::details::Test &t, bool enabled) {
            bool any_node_cnt = (t.node_count == -1);

            if (enabled) {

                if (any_node_cnt) {
                    printf(" - [\033[;32many\033[0m] ");
                } else {
                    printf(" - [\033[;32m%03d\033[0m] ", t.node_count);
                }
                std::cout << "\033[;32m" << t.name << "\033[0m " << std::endl;

            } else {
                if (any_node_cnt) {
                    printf(" - [\033[;31many\033[0m] ");
                } else {
                    printf(" - [\033[;31m%03d\033[0m] ", t.node_count);
                }
                std::cout << "\033[;31m" << t.name << "\033[0m " << std::endl;
            }
        };

        if (is_run_only) {

            for (u32 i = 0; i < static_init_vec_tests.size(); i++) {

                bool run_test = can_run(static_init_vec_tests[i]);
                if (run_only_name.compare(static_init_vec_tests[i].name) == 0) {
                    if (run_test) {
                        selected_tests.push_back(i);
                    } else {
                        logger::err_ln(
                            "TEST", "test can not run under the following configuration");
                    }
                }
            }

        } else {

            auto test_loop = [&](TestType t) {
                for (u32 i = 0; i < static_init_vec_tests.size(); i++) {

                    if (static_init_vec_tests[i].type == t) {
                        bool run_test = can_run(static_init_vec_tests[i]);
                        print_test(static_init_vec_tests[i], run_test);
                        if (run_test) {
                            selected_tests.push_back(i);
                        }
                    }
                }
            };

            if (run_bench) {
                printf("--- Benchmark ---\n");
                test_loop(Benchmark);
            }

            if (run_analysis) {
                printf("--- Analysis  ---\n");
                test_loop(Analysis);
            }

            if (run_unittest) {
                printf("--- Unittest  ---\n");
                test_loop(Unittest);
            }
        }

        if (!is_run_only) {
            printf("--------------------------------------\n\n");
        }

        u32 test_loc_cnt = 0;

        bool has_error = false;

        logger::info_ln("Test", "start python interpreter");
        py::initialize_interpreter();

        std::filesystem::create_directories("tests/figures");

        std::vector<TestResult> results;
        for (u32 i : selected_tests) {

            shamtest::details::Test &test = static_init_vec_tests[i];

            _start_test_print(test, test_loc_cnt, selected_tests.size());

            mpi::barrier(MPI_COMM_WORLD);
            shambase::Timer timer;
            timer.start();
            TestResult res = test.run();
            timer.end();
            mpi::barrier(MPI_COMM_WORLD);

            _end_test_print(res, timer);

            results.push_back(std::move(res));

            test_loc_cnt++;
        }

        logger::info_ln("Test", "close python interpreter");
        py::finalize_interpreter();

        results = gather_tests(std::move(results));

        for (TestResult &res : results) {
            has_error = has_error || is_test_failed(res);
        }

        _print_summary(results);

        if (out_to_file) {
            write_json_report(results);
        }

        write_tex_report(results,has_error);

        i32 errcode;
        if (has_error) {
            errcode = 255;
        } else {
            errcode = 0;
        }

        mpi::barrier(MPI_COMM_WORLD);

        if (instance::world_rank == 0) {
            logger::raw_ln("Tests done exiting ... exitcode =", errcode);
        }
        mpi::barrier(MPI_COMM_WORLD);
        instance::close();

        return errcode;
    }
} // namespace shamtest