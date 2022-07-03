import shamrock



ctx = shamrock.Context()
ctx.pdata_layout_new()
ctx.pdata_layout_add_field("xyz",1,"f32_3")
ctx.pdata_layout_add_field("hpart",1,"f32")
ctx.init_sched(int(1e6),int(2e5))


def sim_setup(ctx : shamrock.Context):

    setup = shamrock.SetupSPH_M4_single()
    setup.init(ctx)


    dr = 0.04
    box = ((-1,1),(-1,1),(-1,1))
    box = setup.get_ideal_box(dr,box)

    print("Corrected box size : ", box)

    ((xm,xM),(ym,yM),(zm,zM)) = box

    nmode = 2
    phase = 0
    kmode = nmode*2*3.141612/(zM - zm)


sim_setup(ctx)

