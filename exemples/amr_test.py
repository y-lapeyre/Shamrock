import shamrock
import numpy as np
import matplotlib.pyplot as plt 
import os



ctx = shamrock.Context()
ctx.pdata_layout_new()

ctx.pdata_layout_add_field("cell_min",1,"u64_3")
ctx.pdata_layout_add_field("cell_max",1,"u64_3")

ctx.init_sched(100,400)

grd = shamrock.AMRGrid(ctx)

grd.make_base_grid([0,0,0],[1,1,1],[5,5,5])

#ctx.dump_status()

print("recovered :",len(ctx.collect_data()["cell_min"][::]))

model = shamrock.AMRTestModel(grd)

model.step()