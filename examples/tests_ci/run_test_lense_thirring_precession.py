"""
CI test: Lense-Thirring nodal precession
=========================================

Sets up a thin, tilted ring of SPH particles orbiting a spinning point mass
deactivating ass dissipative terms. Checks that the numerically measured nodal precession angle of the ring's
orbital angular momentum vector matches the analytic (orbit-averaged)
Lense-Thirring precession rate

    omega_p(r) = 2 * a_spin * (G*M)^2 / (c^3 * r^3)

(e.g. Nixon & King 2012), independent of the ring inclination.

"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# %%
# Units chosen so that G = 1 exactly (unit_length = au * (4*pi^2)^(1/3)),
# matching examples/sph/run_circular_disc_lense_thirring.py

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au() * (39.42410494106729 ** (1 / 3)),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

G = ucte.G()
c = ucte.c()

print("G =", G)
print("c =", c)


# %%
# Setup parameters
#
# The ring radius is chosen far enough from the central mass (15 Rg) that the
# Lense-Thirring acceleration stays a modest perturbation to the Newtonian
# gravity (~3% here)

Npart = 4000
center_mass = c * c  # so that Rg = G*M/c^2 = 1 exactly
disc_mass = 1e-6

a_spin = 0.9
dir_spin = (0.0, 0.0, 1.0)
inclination = 30.0  # deg

Rg = G * center_mass / (c * c)
rin = 15.0 * Rg
rout = 1.02 * rin  # narrow ring: keeps the differential precession across its width small
r0 = rin
center_racc = 0.1 * rin

H_r = 0.05

omega_p_analytic = 2.0 * a_spin * (G * center_mass) ** 2 / (c**3 * rin**3)

t_target = (np.pi / 2) / omega_p_analytic
dphi_analytic = omega_p_analytic * t_target

DPHI_REL_TOL = 0.35

pmass = disc_mass / Npart
bsize = rout * 3
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)


def sigma_profile(r):
    return 1.0


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def rot_profile(r):
    return kep_profile(r)


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    return 0.0


def H_profile(r):
    return (H_r * r0) / omega_k(r0)


# %%

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_ConstantDisc(alpha_u=0, alpha_AV=0, beta_AV=0)
cfg.set_eos_isothermal(0)
cfg.set_units(codeu)
cfg.add_ext_force_lense_thirring(
    central_mass=center_mass, Racc=center_racc, a_spin=a_spin, dir_spin=dir_spin
)
cfg.set_particle_mass(pmass)
cfg.set_cfl_cour(0.3)
cfg.set_cfl_force(0.25)
model.set_solver_config(cfg)

model.init_scheduler(int(1e6), 1)
model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen_ring = setup.make_generator_disc_mc(
    part_mass=pmass,
    disc_mass=disc_mass,
    r_in=rin,
    r_out=rout,
    sigma_profile=sigma_profile,
    H_profile=H_profile,
    rot_profile=rot_profile,
    cs_profile=cs_profile,
    random_seed=42,
)

warp = setup.make_modifier_warp_disc(
    parent=gen_ring,
    Rwarp=0.1 * rin,
    Hwarp=0.05 * rin,
    inclination=inclination,
    posangle=0.0,
)
setup.apply_setup(warp)


def total_angular_momentum(ctx, pmass):
    dic = ctx.collect_data()
    if shamrock.sys.world_rank() > 0:
        return None
    xyz = dic["xyz"]
    vxyz = dic["vxyz"]
    return np.sum(np.cross(xyz, vxyz), axis=0) * pmass


L0 = total_angular_momentum(ctx, pmass)

model.evolve_until(t_target)

L1 = total_angular_momentum(ctx, pmass)

if shamrock.sys.world_rank() == 0:
    print("run_test_lense_thirring_precession: OK")


def check_precession(L0, L1):
    if shamrock.sys.world_rank() > 0:
        return

    # %%
    # Measure the angle swept by the transverse (xy) component of the ring's
    # angular momentum about dir_spin = z, and compare it to the analytic
    # nodal precession angle.

    Lx0, Ly0 = L0[0], L0[1]
    Lx1, Ly1 = L1[0], L1[1]
    cross_z = Lx0 * Ly1 - Ly0 * Lx1
    dot = Lx0 * Lx1 + Ly0 * Ly1
    dphi_measured = np.arctan2(cross_z, dot)

    rel_error = abs(dphi_measured - dphi_analytic) / dphi_analytic

    print(f"L0 = {L0}")
    print(f"L1 = {L1}")
    print(f"dphi_measured = {dphi_measured:.6f} rad ({np.degrees(dphi_measured):.3f} deg)")
    print(f"dphi_analytic = {dphi_analytic:.6f} rad ({np.degrees(dphi_analytic):.3f} deg)")
    print(f"relative error = {rel_error:.3e} (tolerance = {DPHI_REL_TOL:.3e})")

    if rel_error > DPHI_REL_TOL:
        raise ValueError(
            f"Measured Lense-Thirring nodal precession angle is out of tolerance: "
            f"{dphi_measured:.6f} rad vs analytic {dphi_analytic:.6f} rad "
            f"(relative error {rel_error:.3e} > {DPHI_REL_TOL:.3e})"
        )


check_precession(L0, L1)

# %%

# Build a full inviscid, pressureless disc starting at Rin = 4 Rg, evolve it
# while taking many snapshots, and for each radial bin fit the (unwrapped)
# precession angle of the bin's angular momentum vector vs time to get a
# measured precession timescale.

# This uses a much smaller Rin than the tight check above, where the
# Lense-Thirring acceleration is only a ~3% perturbation to the Newtonian
# gravity. At Rin = 4 Rg it is ~20% of the Newtonian gravity, so the
# leading-order (orbit-averaged) analytic formula is expected to be
# noticeably less accurate close to the inner edge -- as in the reference
# figure, this is informative rather than a strict pass/fail check.

def omega_p_of_r(r):
    return 2.0 * a_spin * (G * center_mass) ** 2 / (c**3 * r**3)


def t_orb_of_r(r):
    return 2 * np.pi * np.sqrt(r**3 / (G * center_mass))


Npart_disc = 12000
disc_mass_disc = 1e-6

rin_disc = 4.0 * Rg
rout_disc = 10.0 * rin_disc
nbins = 50

pmass_disc = disc_mass_disc / Npart_disc
bsize_disc = rout_disc * 3
bmin_disc = (-bsize_disc, -bsize_disc, -bsize_disc)
bmax_disc = (bsize_disc, bsize_disc, bsize_disc)


def sigma_profile_disc(r):
    return rin_disc / r


def H_profile_disc(r):
    q = 0.75
    cs_in = (H_r * rin_disc) * omega_k(rin_disc)
    cs_r = cs_in * (r / rin_disc) ** (-q)
    return cs_r / omega_k(r)


ctx_disc = shamrock.Context()
ctx_disc.pdata_layout_new()

model_disc = shamrock.get_Model_SPH(context=ctx_disc, vector_type="f64_3", sph_kernel="M4")

cfg_disc = model_disc.gen_default_config()
cfg_disc.set_artif_viscosity_ConstantDisc(alpha_u=0, alpha_AV=0, beta_AV=0)
cfg_disc.set_eos_isothermal(0)
cfg_disc.set_units(codeu)
cfg_disc.add_ext_force_lense_thirring(
    central_mass=center_mass, Racc=0.1 * rin_disc, a_spin=a_spin, dir_spin=dir_spin
)
cfg_disc.set_particle_mass(pmass_disc)
cfg_disc.set_cfl_cour(0.3)
cfg_disc.set_cfl_force(0.25)
model_disc.set_solver_config(cfg_disc)

model_disc.init_scheduler(int(1e6), 1)
model_disc.resize_simulation_box(bmin_disc, bmax_disc)

setup_disc = model_disc.get_setup()
gen_disc = setup_disc.make_generator_disc_mc(
    part_mass=pmass_disc,
    disc_mass=disc_mass_disc,
    r_in=rin_disc,
    r_out=rout_disc,
    sigma_profile=sigma_profile_disc,
    H_profile=H_profile_disc,
    rot_profile=rot_profile,
    cs_profile=cs_profile,
    random_seed=7,
)
# The whole disc is tilted rigidly by `inclination` (no warp propagation is
# expected/needed since there is no viscosity to communicate torques between
# radii): each annulus then precesses independently at its own local rate.
warp_disc = setup_disc.make_modifier_warp_disc(
    parent=gen_disc,
    Rwarp=0.1 * rin_disc,
    Hwarp=0.05 * rin_disc,
    inclination=inclination,
    posangle=0.0,
)
setup_disc.apply_setup(warp_disc)

n_snap = 30
t_prec_rin_disc = 2.0 * np.pi / omega_p_of_r(rin_disc)
dt_snap = t_prec_rin_disc / 12
snap_times = dt_snap * np.arange(n_snap)


def collect_positions_velocities(ctx):
    dic = ctx.collect_data()
    if shamrock.sys.world_rank() > 0:
        return None, None
    return dic["xyz"], dic["vxyz"]


r_edges_disc = np.linspace(rin_disc, rout_disc, nbins + 1)
r_centers_disc = 0.5 * (r_edges_disc[:-1] + r_edges_disc[1:])

bin_index_disc = None  #
phi_raw = np.full((n_snap, nbins), np.nan)  # atan2 angle of each bin's L_perp per snapshot
n_in_bin_disc = np.zeros(nbins, dtype=int)

for isnap, t_snap in enumerate(snap_times):
    if t_snap > 0:
        model_disc.evolve_until(t_snap)

    xyz, vxyz = collect_positions_velocities(ctx_disc)

    if shamrock.sys.world_rank() > 0:
        continue

    if bin_index_disc is None:
        r_at_t0 = np.linalg.norm(xyz, axis=1)
        bin_index_disc = np.digitize(r_at_t0, r_edges_disc) - 1  # -1: outside [rin, rout]
        for i in range(nbins):
            n_in_bin_disc[i] = np.count_nonzero(bin_index_disc == i)

    L = np.cross(xyz, vxyz)
    for i in range(nbins):
        if n_in_bin_disc[i] < 10:
            continue
        mask = bin_index_disc == i
        L_bin = np.sum(L[mask], axis=0) * pmass_disc
        phi_raw[isnap, i] = np.arctan2(L_bin[1], L_bin[0])

if shamrock.sys.world_rank() == 0:
    print("run_test_lense_thirring_precession: disc precession-vs-radius diagnostic done")


def plot_precession_vs_radius():
    if shamrock.sys.world_rank() > 0:
        return

    t_p_measured = np.full(nbins, np.nan)
    t_p_sigma = np.full(nbins, np.nan)

    for i in range(nbins):
        if n_in_bin_disc[i] < 10:
            continue

        phi_unwrapped = np.unwrap(phi_raw[:, i] - phi_raw[0, i])
        slope, intercept = np.polyfit(snap_times, phi_unwrapped, 1)
        residuals = phi_unwrapped - (slope * snap_times + intercept)
        dof = n_snap - 2
        residual_std = np.sqrt(np.sum(residuals**2) / dof)
        slope_sigma = residual_std / np.sqrt(np.sum((snap_times - np.mean(snap_times)) ** 2))

        if slope > 0:
            t_p_measured[i] = 2 * np.pi / slope
            t_p_sigma[i] = t_p_measured[i] * (slope_sigma / slope)

    print("radius bins [R/Rin] =", r_centers_disc / rin_disc)
    print("particles per bin =", n_in_bin_disc)
    print("measured precession timescale [orbits] =", t_p_measured / t_orb_of_r(r_centers_disc))

    t_orb_centers = t_orb_of_r(r_centers_disc)
    y = np.log10(t_p_measured / t_orb_centers)
    
    yerr = t_p_sigma / (t_p_measured * np.log(10))

    r_fine = np.linspace(rin_disc, rout_disc, 200)
    y_analytic = np.log10((2 * np.pi / omega_p_of_r(r_fine)) / t_orb_of_r(r_fine))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_fine / rin_disc, y_analytic, "-", color="red", label="Lense-Thirring (analytic)")
    ax.errorbar(
        r_centers_disc / rin_disc,
        y,
        yerr=yerr,
        fmt="x",
        color="black",
        capsize=3,
        label="measured (SPH disc)",
    )
    ax.set_xlabel(r"$R / R_{in}$")
    ax.set_ylabel("log Timescale (orbits)")
    ax.set_title(r"Lense-Thirring precession timescale vs radius ($R_{in} = 4 R_g$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("lense_thirring_precession_timescale.png", dpi=150)

    np.savetxt("LTtest.txt", (r_centers_disc / rin_disc, y, yerr))


plot_precession_vs_radius()
