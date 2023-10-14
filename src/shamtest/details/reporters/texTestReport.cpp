// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "texTestReport.hpp"
#include "shambase/string.hpp"
#include "shamtest/details/TestResult.hpp"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

std::string tex_template = R"==(

\documentclass[10pt]{article}

\usepackage[a4paper,total={170mm,260mm},left=20mm,top=20mm,]{geometry}


\usepackage{fancyhdr} % entêtes et pieds de pages personnalisés

\pagestyle{fancy}
\fancyhead[L]{\scriptsize \textsc{Test suite report}} % À changer
\fancyhead[R]{\scriptsize \textsc{\textsc{SHAMROCK}}} % À changer
\fancyfoot[C]{ \thepage}


\usepackage{titling}

\setlength{\droptitle}{-4\baselineskip} % Move the title up

\pretitle{\begin{center}\Huge\bfseries} % Article title formatting
\posttitle{\end{center}} % Article title closing formatting
\title{\textsc{SHAMROCK} test suite report} % Article title
\author{%
\textsc{Timothée David--Cléris}\thanks{timothee.david--cleris@ens-lyon.fr} \\[1ex] % Your name
\normalsize CRAL ENS de Lyon \\ % Your institution
}
\date{\today}

\usepackage{xcolor}
\definecolor{linkcolor}{rgb}{0,0,0.6}


\usepackage[ pdftex,colorlinks=true,
pdfstartview=ajustementV,
linkcolor= linkcolor,
citecolor= linkcolor,
urlcolor= linkcolor,
hyperindex=true,
hyperfigures=false]
{hyperref}



\usepackage{color}


\definecolor{GREEN}{rgb}{0,.7,0}
\definecolor{RED}{rgb}{.8,0,0}


\def\OK{\textcolor{GREEN}{OK}}
\def\FAIL{\textcolor{RED}{FAIL}}

\begin{document}
\maketitle



)==";

std::string tex_template_end = R"(
    \end{document}
)";

namespace shamtest::details {

    void add_unittest_section(std::stringstream &output, std::vector<TestResult> &results) {

        output << R"(\section{Unittests})"
               << "\n\n";

        using namespace details;

        std::unordered_map<std::string, u32> assert_test_count;
        std::unordered_map<std::string, u32> assert_test_count_success;

        for (TestResult &res : results) {
            if (res.type == Unittest) {
                assert_test_count[res.name] = 0;
            }
        }

        for (TestResult &res : results) {
            if (res.type == Unittest) {
                assert_test_count[res.name] += res.asserts.get_assert_count();
                assert_test_count_success[res.name] += res.asserts.get_assert_success_count();
            }
        }

        std::vector<std::string> strings;
        for (auto &[key, value] : assert_test_count) {
            strings.push_back(key);
        }
        std::sort(strings.begin(), strings.end());

        std::string table_header = R"==(
            \begin{center}
            \begin{tabular}{|l|c|c|}
            \hline
            Test name & Status & Asserts \\  \hline \hline
        )==";

        std::string table_footer = R"==(
            \hline
            \end{tabular}\end{center}
        )==";

        output << table_header;

        u32 table_cnt = 0;
        for (std::string key : strings) {
            u32 cnt     = assert_test_count[key];
            u32 cnt_suc = assert_test_count_success[key];

            if (table_cnt > 40) {
                output << table_footer << "\n\n" << table_header;
                table_cnt = 0;
            }

            output << R"(\verb|)" << key << "| & ";

            if (cnt == cnt_suc) {
                output << R"(\OK & )";
            } else {
                output << R"(\FAIL & )";
            }

            output << shambase::format(R"({}/{} \\)", cnt_suc, cnt);
            output << "\n";

            table_cnt++;
        }

        output << table_footer;

        output << "\n";

        for (TestResult &res : results) {
            if(res.asserts.get_assert_success_count() != res.asserts.get_assert_count()){
                output << "Test : " << res.name << "\n\n";

                for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {

                    output << shambase::format_printf("     - Assert [%d/%zu] \n\n", j + 1, res.asserts.asserts.size());


                    output << R"(\verb|)" << res.asserts.asserts[j].name << "| : ";

                    if (res.asserts.asserts[j].value) {
                        output << R"(\OK)";
                    } else {
                        output << R"(\FAIL)";
                    }

                    if ((!res.asserts.asserts[j].value) && !res.asserts.asserts[j].comment.empty()) {
                        output << "\n\n $\\rightarrow$ failed assert logs :\n\n" ;
                        output << R"(\begin{verbatim})" << res.asserts.asserts[j].comment << R"(\end{verbatim})";
                        output <<  "\n";
                    }

                    output <<  "\n\n";
                }

            }
        }

    }

    std::string make_test_report_tex(std::vector<TestResult> &results, bool mark_fail) {

        std::stringstream output;

        output << tex_template;

        output << "Global status : ";

        if (mark_fail) {
            output << R"(\FAIL)";
        } else {
            output << R"(\OK)";
        }

        output << "\n\n";
        output << R"(\tableofcontents)";
        output << "\n\n";

        add_unittest_section(output, results);

        output << tex_template_end;

        return output.str();
    }
} // namespace shamtest::details