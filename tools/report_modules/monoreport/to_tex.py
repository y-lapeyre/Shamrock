
from .sections import unitests, configuration, sycl_algs, self_grav_fmm, sparse_comm

def convert(fileprefix : str, json_input) -> str:

    buf = ""

    buf += configuration.make_configuration_report(json_input)

    buf += unitests.make_unittest_report(json_input)

    buf += sycl_algs.make_syclalgs_report(fileprefix,json_input)

    buf += self_grav_fmm.make_fmm_report(fileprefix,json_input)

    buf += sparse_comm.make_sparse_comm_report(fileprefix,json_input)
    
    return buf