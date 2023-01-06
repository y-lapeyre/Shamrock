
from .sections import unitests, configuration, sycl_algs, self_grav_fmm, sparse_comm, periodix_box_perf, nbody_selfgrav_perf, bandwith_test

def convert(fileprefix : str, json_input) -> str:

    buf = ""

    buf += configuration.make_configuration_report(json_input)

    buf += unitests.make_unittest_report(json_input)

    buf += sycl_algs.make_syclalgs_report(fileprefix,json_input)

    buf += self_grav_fmm.make_fmm_report(fileprefix,json_input)

    buf += sparse_comm.make_sparse_comm_report(fileprefix,json_input)

    buf += periodix_box_perf.make_bench_periodic_box_report(fileprefix, json_input)

    buf += nbody_selfgrav_perf.make_bench_nbody_selfgrav_report(fileprefix, json_input)

    buf += bandwith_test.make_bandwith_matrix_report(fileprefix, json_input)
    
    return buf