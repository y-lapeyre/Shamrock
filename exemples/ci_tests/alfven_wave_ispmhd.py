import shamrock
import matplotlib.pyplot as plt


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_None()
cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8),1)


dr = 0.02
bmin = (-2, -2, -0.5)#(0, 0, 0)
bmax = ( 2,  2,  0.5)

bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax

model.resize_simulation_box(bmin,bmax)
model.add_cube_fcc_3d(dr, bmin,bmax)

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho0*vol_b)

pmass = model.total_mass_to_part_mass(totmass)

print("Total mass :", totmass)
pmass = model.total_mass_to_part_mass(totmass)

model.set_particle_mass(pmass)
print("Current part mass :", pmass)

def B_func(r):
    x,y,z = r
    Bx = B0
    By = 0
    Bz = 0
    return (Bx, By, Bz)

model.set_field_value_lambda_f64_3("B/rho", B_func)

def v_func(r):
    x,y,z = r
    h = 0.05
    vx = 0.01 * np.exp(- (x / (3*h))**2)
    vy = 0.
    vz = 0
    return (vx, vy, vz)

model.set_field_value_lambda_f64_3("vxyz", v_func)

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

t_target = 0.245

model.evolve_until(t_target)

sod = shamrock.phys.SodTube(gamma = gamma, rho_1 = 1,P_1 = 1,rho_5 = 0.125,P_5 = 0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1,0,0), t_target, 0.0, -0.5,0.5)

#################
### Test CD
#################
rho, v, P = sodanalysis.compute_L2_dist()
vx,vy,vz = v

# normally :
# rho 0.0001615491818848632
# v (0.0011627047434807855, 2.9881306160215856e-05, 1.7413547093275864e-07)
# P0.0001248364612976704
# B ()

test_pass = True
pass_rho = 0.0001615491818848697
pass_vx = 0.0011627047434809158
pass_vy = 2.9881306160215856e-05
pass_vz = 1.7413547093275864e-07
pass_P = 0.0001248364612976704

err_log = ""

if rho > pass_rho:
    err_log += ("error on rho is too high "+str(rho) +">"+str(pass_rho) ) + "\n"
    test_pass = False
if vx > pass_vx:
    err_log += ("error on vx is too high "+str(vx) +">"+str(pass_vx) )+ "\n"
    test_pass = False
if vy > pass_vy:
    err_log += ("error on vy is too high "+str(vy) +">"+str(pass_vy) )+ "\n"
    test_pass = False
if vz > pass_vz:
    err_log += ("error on vz is too high "+str(vz) +">"+str(pass_vz) )+ "\n"
    test_pass = False
if P > pass_P:
    err_log += ("error on P is too high "+str(P) +">"+str(pass_P) )+ "\n"
    test_pass = False
if Bx > pass_Bx:
    err_log += ("error on Bx is too high "+str(Bx) +">"+str(pass_Bx) )+ "\n"
    test_pass = False
if By > pass_By:
    err_log += ("error on By is too high "+str(By) +">"+str(pass_By) )+ "\n"
    test_pass = False
if Bz > pass_Bz:
    err_log += ("error on Bz is too high "+str(Bz) +">"+str(pass_Bz) )+ "\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n"+err_log)
