# Shearing box in SPH

$$
\mathbf{f} = \Omega_0^2 (2 q x \hat{i} - z \hat{k} ) - 2 \Omega_0 \hat{k} \times \mathbf{v} 
$$

Shear speed :

$$
\omega = q \Omega_0 L_x 
$$

To add it : 

```py 
xm,ym,zm = bmin
xM,yM,zM = bmax

Omega_0 = 1
eta = 0.00
q = 3./2.

shear_speed = -q*Omega_0*(xM - xm)

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_shearing_periodic((1,0,0),(0,1,0),shear_speed)
cfg.set_eos_adiabatic(gamma)
cfg.add_ext_force_shearing_box(
    Omega_0  = Omega_0,
    eta      = eta,
    q        = q
)
cfg.set_units(shamrock.UnitSystem())
cfg.print_status()
```