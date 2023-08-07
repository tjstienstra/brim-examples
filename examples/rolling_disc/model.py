"""Defines the model of the rolling disc."""

import cloudpickle
import sympy as sm
import sympy.physics.mechanics as me
from brim import FlatGround, KnifeEdgeWheel, NonHolonomicTyre
from brim.other import RollingDisc

from utilities import DataStorage

g = sm.symbols("g")
t = me.dynamicsymbols._t
T_drive, T_steer, T_roll = me.dynamicsymbols("T_drive, T_steer, T_roll")

disc = RollingDisc("disc")
disc.disc = KnifeEdgeWheel("wheel")
disc.ground = FlatGround("ground")
disc.tyre = NonHolonomicTyre("tyre")
disc.define_all()

system = disc.to_system()
normal = disc.ground.get_normal(disc.ground.origin)
system.apply_gravity(-g * normal)
r_up = disc.disc.body.masscenter.pos_from(disc.tyre.contact_point).normalize()
r_long = me.cross(normal, disc.disc.rotation_axis).normalize()
system.add_loads(
    me.Torque(
        disc.disc.frame,
        T_drive * disc.disc.rotation_axis + T_steer * r_up + T_roll * r_long))
system.u_ind = disc.u[2:]
system.u_dep = disc.u[:2]
system.form_eoms()

controllable_loads = [T_steer, T_roll, T_drive]
constants = {
    disc.disc.body.mass: 1.0,
    disc.disc.radius: 0.3,
    g: 9.81,
}
mr2 = float((disc.disc.body.mass * disc.disc.radius ** 2).subs(constants))
constants[disc.disc.body.central_inertia.to_matrix(disc.disc.frame)[0, 0]] = mr2 / 4
constants[disc.disc.body.central_inertia.to_matrix(disc.disc.frame)[1, 1]] = mr2 / 2


data = DataStorage(**{
    "model": disc,
    "system": system,
    "controllable_loads": controllable_loads,
    "constants": constants,
})

with open("data.pkl", "wb") as f:
    cloudpickle.dump(data, f)
