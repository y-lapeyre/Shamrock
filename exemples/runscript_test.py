import shamrock

import numpy
import matplotlib.pyplot as plt 


ctx = shamrock.Context()
ctx.pdata_layout_new()
ctx.pdata_layout_add_field("xyz",1,"f32_3")
ctx.pdata_layout_add_field("hpart",1,"f32")
ctx.init_sched(int(1e5),1)


def sim_setup(ctx : shamrock.Context):

    setup = shamrock.SetupSPH_M4_single()
    setup.init(ctx)

    bdim = (256,24,24)

    (xs,ys,zs) = setup.get_box_dim_icnt(1,bdim)

    #todo set box size to otherwise split patchdata won't work in the setup

    dr = 1/xs 

    (xs,ys,zs) = setup.get_box_dim_icnt(dr,bdim)


    ctx.set_box_size(((-xs,xs),(-ys/2,ys/2),(-zs/2,zs/2)))

    setup.add_cube_fcc(ctx,dr, ((-xs,0),(-ys/2,ys/2),(-zs/2,zs/2)))
    setup.add_cube_fcc(ctx,dr*2, ((0,xs),(-ys/2,ys/2),(-zs/2,zs/2)))

    

sim_setup(ctx)

dic = ctx.collect_data()

xyz = numpy.array(dic["xyz"])

plt.scatter(xyz[:,0], xyz[:,2])

plt.show()
