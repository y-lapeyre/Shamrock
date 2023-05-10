import shamrock
import matplotlib.pyplot as plt



ctx = shamrock.Context()
ctx.pdata_layout_new()

ctx.pdata_layout_add_field("xyz",1,"f32_3")
ctx.pdata_layout_add_field("hpart",1,"f32")

#field for leapfrog integrator
ctx.pdata_layout_add_field("vxyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz_old",1,"f32_3")


ctx.pdata_layout_add_field("uint",1,"f32")
ctx.pdata_layout_add_field("duint",1,"f32")
ctx.pdata_layout_add_field("duint_old",1,"f32")


#start the scheduler
ctx.init_sched(int(1e5),1)




gamma = 5./3.

rho_g = 1
rho_d = 0.125


P_g = 1
P_d = 0.1

u_g = P_g/((gamma - 1)*rho_g)
u_d = P_d/((gamma - 1)*rho_d)


pmass = -1






setup = shamrock.SetupSPH(kernel = "M4", precision = "single")
setup.init(ctx)

(xs,ys,zs) = setup.get_box_dim(1,256,24,24)
dr = 1/xs
(xs,ys,zs) = setup.get_box_dim(dr,256,24,24)

ctx.set_coord_domain_bound((-xs,-ys/2,-zs/2),(xs,ys/2,zs/2))

setup.set_boundaries("periodic")


fact = (rho_g/rho_d)**(1./3.)

setup.add_particules_fcc(ctx,dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
setup.add_particules_fcc(ctx,dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))

setup.set_value_in_box(ctx, "f32", u_g, "uint",(-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
setup.set_value_in_box(ctx, "f32", u_d, "uint",(0,-ys/2,-zs/2),(xs,ys/2,zs/2))



vol_b = xs*ys*zs

totmass = (rho_d*vol_b) + (rho_g*vol_b)

print("Total mass :", totmass)

setup.set_total_mass(totmass)

pmass = setup.get_part_mass()

print("Current part mass :", pmass)

for it in range(5):
    setup.update_smoothing_lenght(ctx)

del setup


