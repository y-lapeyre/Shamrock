import os

import numpy as np

import shamrock

x0 = 0
y0 = 0
z0 = 0

wall_length = 0.4
wall_width = 0.4
wall_thickness = 0.4


def ghost_map(r):
    x, y, z = r
    in_wall = (
        x > 0
    )  # (x-x0  < wall_length) & (x - x0 > 0) & (y - y0 < wall_width)& (y - y0 > 0) & (z - z0 < wall_thickness) & (z - z0 > 0)
    maskvalue = 0
    if in_wall:
        maskvalue = 1
    return maskvalue
    # return 0


# %% # Configure the solver
outputdir = ""
sim_name = outputdir + "wall/" + "wall"
ctx = shamrock.Context()
ctx.pdata_layout_new()

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=1.0,
    unit_length=1.0,
    unit_mass=1.0,
)
ucte = shamrock.Constants(codeu)
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
cfg = model.gen_default_config()
cfg.set_units(codeu)

cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)

cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(5.0 / 3.0)
model.set_solver_config(cfg)

# Print the solver config
cfg.print_status()

# Set scheduler criteria to effectively disable patch splitting and merging.
crit_split = int(1e7)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)

# %% # Setup the simulation
nx = 8  # 512
ny = 8  # 590
nz = 8
xymin = -0.5
xmin = xymin
ymin = xymin

uuzero = 1.0
rhozero = 1.0
# Compute box size
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, nx, ny, nz)
print(f"Initial dim: x: {xs}\ty: {ys}\tz: {zs}")
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, nx, ny, nz)
print(f"Final dim: x: {xs}\ty: {ys}\tz: {zs}")

model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))

model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))
model.set_value_in_a_box(
    "uint", "f64", uuzero, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2)
)


# -------------------------------------------------------
# the velocity function to trigger Orstang Vortex
def vel_func(r):
    x, y, z = r
    # in_wall = (x-x0  < wall_length) & (x - x0 > 0) & (y - y0 < wall_width)& (y - y0 > 0) & (z - z0 < wall_thickness) & (z - z0 > 0)
    # if (in_wall):
    #    return (0., 0., 0.)
    # else:
    #    return (1., 1., 0.)
    return (1.0, 1.0, 0.0)


# the magnetic field (B/rho)function to trigger Orstang Vortex
model.set_field_value_lambda_f64_3("vxyz", vel_func)

model.set_field_value_lambda_u32("ghost_mask", ghost_map)
vol_b = xs * ys * zs
totmass = rhozero * vol_b

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

print("Total mass :", totmass)
print("Current part mass :", pmass)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

# model.add_wall((0,0,0), 0.1, 0.1, 0.1)
# model.apply_ghost_particles()

model.timestep()
cfg.print_status()

# %%
# Running the simulation

t_sum = 0
t_target = 0.1

model.do_vtk_dump(f"{sim_name}_{0:05}.vtk", True)
# model.dump(f"{sim_name}_{0:05}.sham")

i_dump = 1
dt_dump = 0.001

while t_sum < t_target:
    model.evolve_until(t_sum + dt_dump)
    model.do_vtk_dump(f"{sim_name}_{i_dump:05}.vtk", True)
    # model.dump(f"{sim_name}_{i_dump:05}.sham")
    t_sum += dt_dump
    i_dump += 1
