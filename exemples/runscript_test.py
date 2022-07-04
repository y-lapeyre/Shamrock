import shamrock

import numpy
import matplotlib.pyplot as plt 


ctx = shamrock.Context()
ctx.pdata_layout_new()

ctx.pdata_layout_add_field("xyz",1,"f32_3")
ctx.pdata_layout_add_field("hpart",1,"f32")

#field for leapfrog integrator
ctx.pdata_layout_add_field("vxyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz_old",1,"f32_3")


#start the scheduler
ctx.init_sched(int(1e5),1)



rho_g = 1
rho_d = 0.125

def sim_setup(ctx : shamrock.Context):

    setup = shamrock.SetupSPH_M4_single()
    setup.init(ctx)

    bdim = (256,24,24)

    (xs,ys,zs) = setup.get_box_dim_icnt(1,bdim)

    #todo set box size to otherwise split patchdata won't work in the setup

    dr = 1/xs 

    (xs,ys,zs) = setup.get_box_dim_icnt(dr,bdim)


    ctx.set_box_size(((-xs,xs),(-ys/2,ys/2),(-zs/2,zs/2)))
    setup.set_boundaries(True)

    setup.add_cube_fcc(ctx,dr, ((-xs,0),(-ys/2,ys/2),(-zs/2,zs/2)))
    setup.add_cube_fcc(ctx,dr*2, ((0,xs),(-ys/2,ys/2),(-zs/2,zs/2)))

    vol_b = xs*ys*zs

    totmass = (rho_d*vol_b) + (rho_g*vol_b)

    print("Total mass :", totmass)

    setup.set_total_mass(totmass)

    print("Current part mass :", setup.get_part_mass())

    setup.update_smoothing_lenght(ctx)



def print_dist(ctx : shamrock.Context):

    dic = ctx.collect_data()

    print(dic)

    xyz = numpy.array(dic["xyz"])

    hpart = numpy.array(dic["hpart"])

    plt.plot(xyz[:,0], hpart,".")

    plt.show()    



sim_setup(ctx)
print_dist(ctx)



