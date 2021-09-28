#include "unit_test_handler.hpp"

#include "../sys/mpi_handler.hpp"
#include "unit_test_handler.hpp"
#include <cstring>
#include <string>
#include <vector>

namespace unit_test{
    

    //result output and log buffers
    std::string current_test_name;
    bool is_current_test_mpi = false;
    std::string log_buffer;
    std::vector<Assert_report> assert_buffer;

    bool test_start(std::string test_name,bool is_mpi){
        current_test_name = test_name;
        is_current_test_mpi = is_mpi;

        if(! is_mpi){
            return world_rank == 0;
        }else{
            return true;
        }
    }


    
    void test_log(std::string str){

        log_buffer += (str);

    }

    bool test_assert(const char name[DECRIPTION_ASSERT_MAX_LEN], bool cdt){
        assert_buffer.push_back(Assert_report(name, cdt));
        return cdt;
    }

    void test_end(){

        mpi_barrier();

        Test_report report;
        report.name = current_test_name;
        report.is_mpi = is_current_test_mpi;

        if(!is_current_test_mpi){
            
            report.assert_report_list = assert_buffer;
            report.logs = {log_buffer};
            
        }else{

            //recover asserts reports from each process
            {
                //https://stackoverflow.com/questions/12080845/mpi-receive-gather-dynamic-vector-length

                int *counts = new int[world_size];
                int nelements = (int)assert_buffer.size()*sizeof(Assert_report);
                // Each process tells the root how many elements it holds
                MPI_Gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

                // Displacements in the receive buffer for MPI_GATHERV
                int *disps = new int[world_size];
                // Displacement for the first chunk of data - 0
                for (int i = 0; i < world_size; i++)
                    disps[i] = (i > 0) ? (disps[i-1] + counts[i-1]) : 0;

                // Place to hold the gathered data
                // Allocate at root only
                char *gather_data = NULL;
                if (world_rank == 0)
                    // disps[size-1]+counts[size-1] == total number of elements
                    gather_data = new char[disps[world_size-1]+counts[world_size-1]];

                // Collect everything into the root
                MPI_Gatherv(&assert_buffer[0], nelements, MPI_CHAR,
                            gather_data, counts, disps, MPI_CHAR, 0, MPI_COMM_WORLD);

                if (world_rank == 0){
                    Assert_report* rep_dat = (Assert_report*) gather_data;
                    unsigned int len_rep_dat = (disps[world_size-1]+counts[world_size-1])/sizeof(Assert_report);

                    //printf("len_rep_dat : %d\n",len_rep_dat);


                    for(unsigned int i = 0; i < len_rep_dat; i++){
                        report.assert_report_list.push_back(rep_dat[i]);
                    }

                }

                delete[] counts;
            }



            mpi_barrier();
            //recover asserts reports from each process
            {
                //https://stackoverflow.com/questions/12080845/mpi-receive-gather-dynamic-vector-length

                //printf("rank %d : %s (%lu)\n",world_rank,log_buffer.c_str(),log_buffer.size());

                int *counts = new int[world_size];
                int nelements = (int)(log_buffer.size()+1)*sizeof(char);
                // Each process tells the root how many elements it holds
                MPI_Gather(&nelements, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

                // Displacements in the receive buffer for MPI_GATHERV
                int *disps = new int[world_size];
                // Displacement for the first chunk of data - 0
                for (int i = 0; i < world_size; i++)
                    disps[i] = (i > 0) ? (disps[i-1] + counts[i-1]) : 0;

                // Place to hold the gathered data
                // Allocate at root only
                char *gather_data = NULL;
                if (world_rank == 0)
                    // disps[size-1]+counts[size-1] == total number of elements
                    gather_data = new char[disps[world_size-1]+counts[world_size-1]];

                //printf("%s\n",gather_data);

                // Collect everything into the root
                MPI_Gatherv(log_buffer.c_str(), nelements, MPI_CHAR,
                            gather_data, counts, disps, MPI_CHAR, 0, MPI_COMM_WORLD);

                if (world_rank == 0){
                    char* rep_dat = (char*) gather_data;
                    unsigned int len_rep_dat = (disps[world_size-1]+counts[world_size-1])/sizeof(char);

                    //printf("len_rep_dat : %d\n",len_rep_dat);

                    unsigned int offset = 0;
                    for(unsigned int i = 0 ; i < world_size; i++){
                        //printf("counts : %d\n",counts[i]);
                        //printf("len str res : %lu\n",std::string(rep_dat).size());
                        report.logs.push_back(std::string(rep_dat+offset));
                        offset += counts[i];
                    }

                    
                    

                }
            }

        }

        if(world_rank == 0){
            report.assert_count = report.assert_report_list.size();
            
            unsigned int count_assert_succes = 0;
            for(Assert_report a : report.assert_report_list){
                if(a.condition) count_assert_succes ++;
            }

            report.succes_count = count_assert_succes;
            report.succes_rate = float(report.succes_count)/float(report.assert_count);

            tests_results.push_back(report);
        }

        log_buffer.clear();
        assert_buffer.clear();

    }

    void print_test_results(){

        for(Test_report report : tests_results){
            printf("%s\n",report.name.c_str());
            for(Assert_report a_report : report.assert_report_list){
                printf("| Assert : %s %d %d\n",a_report.description,a_report.condition,a_report.rank_emited);
            }

            printf("| Logs : \n");
            for(unsigned int i = 0; i < report.logs.size();i++){
                std::string to_print = report.logs[i];

                if(to_print[to_print.size()-1] == '\n'){
                    printf("| [%03d] \n %s",i,report.logs[i].c_str());
                }else{
                    printf("| [%03d] \n %s\n",i,report.logs[i].c_str());
                }

                
            }
        }

    }


    


}