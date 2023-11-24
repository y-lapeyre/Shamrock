import shamrock
import matplotlib.pyplot as plt


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")



gamma = 5./3.
rho = 1
uint = 1

dr = 0.01
bmin = (-0.6,-0.6,-0.1)
bmax = ( 0.6, 0.6, 0.1)
pmass = -1

bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax

eta = 0.01
kappa = 0.01

shear_speed = -(3/2)*(xM - xm) - eta/(1)


cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_shearing_periodic((1,0,0),(0,0,1),shear_speed)
cfg.set_eos_adiabatic(gamma)
cfg.add_ext_force_shearing_box(
    shear_speed         = shear_speed,
    pressure_background = eta,
    s                   = 3./2.,
)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)



model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho*vol_b)
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint","f64", 1 , bmin,bmax)
#model.set_value_in_a_box("vxyz","f64_3", (-10,0,0) , bmin,bmax)

pen_sz = 0.1


def vel_func(r):
    x,y,z = r

    s = (x - xm)/(xM - xm)
    vel = (shear_speed)*s

    return (2*eta/kappa,float(vel),0.)

model.set_field_value_lambda_f64_3("vxyz", vel_func)
#print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")
#print("total u :",tot_u)

a = input("continue ?")



model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)





#for i in range(9):
#    model.evolve(5e-4, False, False, "", False)



t_sum = 0
t_target = 10
current_dt = 1e-7

i_dump = 0
dt_dump = 1./100

do_dump = False
next_dt_target = t_sum + dt_dump
while t_sum < t_target:


    while t_sum < next_dt_target:

        do_dump = (t_sum + current_dt) == next_dt_target

        

        next_dt = model.evolve(t_sum,current_dt, do_dump, "dump_{:04}.vtk".format(i_dump), do_dump)
        print("--> do dump",do_dump)
        
        if do_dump:
            i_dump += 1

        t_sum += current_dt
        current_dt = next_dt

        if do_dump:
            break

        if (next_dt_target - t_sum) < next_dt:
            current_dt = next_dt_target - t_sum



    next_dt_target += dt_dump

    if (next_dt_target - t_sum) < next_dt:
        current_dt = next_dt_target - t_sum

