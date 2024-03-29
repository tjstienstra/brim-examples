"""Set the initial conditions for the Whipple bicycle with an upper body rider."""
import os
import sys

import bicycleparameters as bp
import cloudpickle
import numpy as np
from brim import TwoPinStickLeftLeg
from scipy.optimize import fsolve
from sympy import S, lambdify, solve, symbols
from sympy.physics.mechanics import Dyadic, Vector, dynamicsymbols, inertia

data_dir = [os.path.join(pt, "data") for pt in sys.path if pt[-7:] == "brimopt"][0]

with open("data.pkl", "rb") as f:
    data = cloudpickle.load(f)
system = data.system
br = data.model
bicycle = br.bicycle
rider = br.rider
left_shoulder_torque = rider.left_shoulder.load_groups[0]
right_shoulder_torque = rider.right_shoulder.load_groups[0]
left_arm_torque = rider.left_arm.load_groups[0]
right_arm_torque = rider.right_arm.load_groups[0]
lean_torque = br.seat.load_groups[0]

bike_parametrization = "Browser"
bike_params = bp.Bicycle(bike_parametrization, pathToData=data_dir)
if bike_parametrization == "Fisher":
    bike_params.parameters["Measured"]["hbb"] = 0.3
    bike_params.parameters["Measured"]["lcs"] = 0.44
    bike_params.parameters["Measured"]["lsp"] = 0.22
    bike_params.parameters["Measured"]["lst"] = 0.53
    bike_params.parameters["Measured"]["lamst"] = 1.29
    bike_params.parameters["Measured"]["whb"] = 0.6
    bike_params.parameters["Measured"]["LhbR"] = 1.11
    bike_params.parameters["Measured"]["LhbF"] = 0.65
bike_params.add_rider("Jason", reCalc=True)
constants = br.get_param_values(bike_params)
k_shoulder_flexion, c_shoulder_flexion = 40, 3
k_shoulder_adduction, c_shoulder_adduction = 30, 3
k_shoulder_rotation, c_shoulder_rotation = 30, 3
k_elbow, c_elbow = 20, 2
# Lean stiffness and damping from:
# D. S. de Lorenzo and Mont Hubbard. Dynamic bicycle stability of a flexibly coupled
# rider. Internal report UC Davis, 1996.
constants.update({
    br.seat.symbols["alpha"]: -(
            bike_params.parameters["Benchmark"]["lam"] +
            bike_params.human.CFG["somersault"]).nominal_value - 0.1,
    symbols("g"): 9.81,
    bicycle.cranks.symbols["radius"]: 0.15,
    bicycle.symbols["gear_ratio"]: 2.0,
    lean_torque.symbols["k"]: 128,
    lean_torque.symbols["c"]: 50,
    lean_torque.symbols["q_ref"]: 0,
    left_shoulder_torque.symbols["k_flexion"]: k_shoulder_flexion,
    left_shoulder_torque.symbols["c_flexion"]: c_shoulder_flexion,
    left_shoulder_torque.symbols["k_adduction"]: k_shoulder_adduction,
    left_shoulder_torque.symbols["c_adduction"]: c_shoulder_adduction,
    left_shoulder_torque.symbols["k_rotation"]: k_shoulder_rotation,
    left_shoulder_torque.symbols["c_rotation"]: c_shoulder_rotation,
    right_shoulder_torque.symbols["k_flexion"]: k_shoulder_flexion,
    right_shoulder_torque.symbols["c_flexion"]: c_shoulder_flexion,
    right_shoulder_torque.symbols["k_adduction"]: k_shoulder_adduction,
    right_shoulder_torque.symbols["c_adduction"]: c_shoulder_adduction,
    right_shoulder_torque.symbols["k_rotation"]: k_shoulder_rotation,
    right_shoulder_torque.symbols["c_rotation"]: c_shoulder_rotation,
    left_arm_torque.symbols["k"]: k_elbow,
    left_arm_torque.symbols["c"]: c_elbow,
    right_arm_torque.symbols["k"]: k_elbow,
    right_arm_torque.symbols["c"]: c_elbow,
})

zero_conditions = {
    bicycle.q[0]: 0,
    bicycle.q[1]: 0,
    bicycle.q[2]: 0,
    bicycle.q[3]: 0,
    bicycle.q[4]: 0.314,
    bicycle.q[5]: 0,
    bicycle.q[6]: 0,
    bicycle.q[7]: 0,
    rider.left_arm.q[0]: 0.5,
    rider.right_arm.q[0]: 0.5,
    br.seat.q[0]: 0,
    rider.left_shoulder.q[0]: 0.6,
    rider.left_shoulder.q[1]: -0.2,
    rider.left_shoulder.q[2]: -0.3,
    rider.right_shoulder.q[0]: 0.6,
    rider.right_shoulder.q[1]: -0.2,
    rider.right_shoulder.q[2]: -0.3,
}
p, p_vals = zip(*constants.items())
eval_fh = lambdify((system.q_dep, system.q_ind, p), system.holonomic_constraints[:],
                   cse=True)
q0_ind = np.array([zero_conditions[qi] for qi in system.q_ind])
q0_dep = np.array([zero_conditions[qi] for qi in system.q_dep])
q0_dep = fsolve(eval_fh, q0_dep, args=(q0_ind, p_vals))
zero_conditions.update(dict(zip(system.q_dep, q0_dep)))
constants.update({
    left_shoulder_torque.symbols["q_ref_flexion"]:
        zero_conditions[rider.left_shoulder.q[0]],
    left_shoulder_torque.symbols["q_ref_adduction"]:
        zero_conditions[rider.left_shoulder.q[1]],
    left_shoulder_torque.symbols["q_ref_rotation"]:
        zero_conditions[rider.left_shoulder.q[2]],
    right_shoulder_torque.symbols["q_ref_flexion"]:
        zero_conditions[rider.right_shoulder.q[0]],
    right_shoulder_torque.symbols["q_ref_adduction"]:
        zero_conditions[rider.right_shoulder.q[1]],
    right_shoulder_torque.symbols["q_ref_rotation"]:
        zero_conditions[rider.right_shoulder.q[2]],
    left_arm_torque.symbols["q_ref"]: zero_conditions[rider.left_arm.q[0]],
    right_arm_torque.symbols["q_ref"]: zero_conditions[rider.right_arm.q[0]],
})

# Add the inertia of the legs to the rear frame
leg = TwoPinStickLeftLeg("left_leg")
q_hip = dynamicsymbols("q_hip")
leg.define_all()
leg.hip_interframe.orient_axis(bicycle.rear_frame.saddle.frame, q_hip,
                               bicycle.rear_frame.wheel_hub.axis)
saddle_point = br.seat.system.joints[0].parent_point
offset = Vector({bicycle.rear_frame.saddle.frame: rider.pelvis.left_hip_point.pos_from(
    saddle_point).to_matrix(rider.pelvis.frame)})
leg.hip_interpoint.set_pos(saddle_point, 0)
val_dict = {leg.q[1]: 0, **leg.get_param_values(bike_params), **constants}
v = leg.foot_interpoint.pos_from(bicycle.cranks.center_point).to_matrix(
    bicycle.rear_frame.wheel_hub.frame).xreplace(val_dict).simplify()
val_dict[q_hip], val_dict[leg.q[0]] = fsolve(
    lambdify([(q_hip, leg.q[0])], [v[0], v[2]]), (0.6, 1.5))
additional_inertia = Dyadic(0)
additional_mass = S.Zero
for body in leg.system.bodies:
    additional_inertia += 2 * body.parallel_axis(bicycle.rear_frame.body.masscenter)
    additional_inertia += 2 * body.mass * (inertia(
        bicycle.rear_frame.body.frame, 1, 1, 1
    ) * offset.dot(offset) - offset.outer(offset))
    additional_mass += 2 * body.mass
extra_i_vals = lambdify(val_dict.keys(),
                        additional_inertia.to_matrix(bicycle.rear_frame.body.frame),
                        cse=True)(*val_dict.values())
# Note: ixy and iyz are in this array not zero as I used the asymetric leg two times
i_rear = bicycle.rear_frame.body.central_inertia.to_matrix(
    bicycle.rear_frame.body.frame)
constants[bicycle.rear_frame.body.mass] += additional_mass.xreplace(val_dict)
for idx in [(0, 0), (1, 1), (2, 2), (2, 0)]:
    constants[i_rear[idx]] += extra_i_vals[idx]

initial_conditions = {
    **zero_conditions,

    bicycle.u[0]: 3,
    bicycle.u[1]: 0,
    bicycle.u[2]: 0,
    bicycle.u[3]: 0,
    bicycle.u[4]: 0,
    bicycle.u[5]: -3 / constants[bicycle.rear_wheel.radius],
    bicycle.u[6]: 0,
    bicycle.u[7]: 0,
    rider.left_arm.u[0]: 0,
    rider.right_arm.u[0]: 0,
    br.seat.u[0]: 0,
    rider.left_shoulder.u[0]: 0,
    rider.left_shoulder.u[1]: 0,
    rider.left_shoulder.u[2]: 0,
    rider.right_shoulder.u[0]: 0,
    rider.right_shoulder.u[1]: 0,
    rider.right_shoulder.u[2]: 0,
}
# Introduce small numerical error on zero values to avoid possible numerical problems
initial_conditions = {xi: np.random.random() * 1E-14 if xval == 0 else xval for xi, xval
                      in initial_conditions.items()}

p, p_vals = zip(*constants.items())
eval_fh = lambdify((system.q_dep, system.q_ind, p), system.holonomic_constraints[:],
                   cse=True)
vel_constrs = system.holonomic_constraints.diff(dynamicsymbols._t).col_join(
    system.nonholonomic_constraints).xreplace(
    solve(system.kdes, system.q.diff(dynamicsymbols._t)))
eval_fnh = lambdify((system.u_dep, system.q, system.u_ind, p), vel_constrs[:], cse=True)

q0_ind = np.array([initial_conditions[qi] for qi in system.q_ind])
q0_dep = np.array([initial_conditions[qi] for qi in system.q_dep])
q0_dep = fsolve(eval_fh, q0_dep, args=(q0_ind, p_vals))
q0 = np.concatenate((q0_ind, q0_dep))

u_bike_ind = [ui for ui in system.u_ind if ui in bicycle.u]
u_bike_dep = [ui for ui in system.u_dep if ui in bicycle.u]

fnh_ground = bicycle.rear_tire.system.nonholonomic_constraints.col_join(
    bicycle.front_tire.system.nonholonomic_constraints).col_join(
    bicycle.front_tire.system.holonomic_constraints.diff(dynamicsymbols._t)).xreplace(
    solve(system.kdes, system.q.diff(dynamicsymbols._t)))[:]

eval_fnh_ground = lambdify((u_bike_dep, system.q, u_bike_ind, p), fnh_ground, cse=True)
u_bike_dep0 = fsolve(eval_fnh_ground, [initial_conditions[ui] for ui in u_bike_dep],
                     args=(q0, [initial_conditions[ui] for ui in u_bike_ind], p_vals))
initial_conditions.update(dict(zip(u_bike_dep, u_bike_dep0)))

u0_ind = np.array([initial_conditions[ui] for ui in system.u_ind])
u0_dep = np.array([initial_conditions[ui] for ui in system.u_dep])
u0_dep = fsolve(eval_fnh, u0_dep, args=(q0, u0_ind, p_vals))
u0 = np.concatenate((u0_ind, u0_dep))

initial_conditions.update(dict(zip(system.q, q0)))
initial_conditions.update(dict(zip(system.u, u0)))

roll_idx = system.q[:].index(bicycle.q[3])
roll_rate_idx = len(system.q) + system.u[:].index(bicycle.u[3])
left_shoulder_idx = system.q[:].index(rider.left_shoulder.q[0])
right_shoulder_idx = system.q[:].index(rider.right_shoulder.q[0])
left_elbow_idx = system.q[:].index(rider.left_arm.q[0])
right_elbow_idx = system.q[:].index(rider.right_arm.q[0])
max_roll_rate = 0.2
excitation = (lambda t, x:  # noqa: E731
              max(-1, min(x[roll_rate_idx] / max_roll_rate, 1)))

controls = {
    lean_torque.symbols["q_ref"]: lambda t, x: -x[roll_idx],
    left_shoulder_torque.symbols["q_ref_flexion"]:
        lambda t, x: initial_conditions[rider.left_shoulder.q[0]] + excitation(t, x),
    left_shoulder_torque.symbols["q_ref_adduction"]:
        lambda t, x: initial_conditions[rider.left_shoulder.q[1]],
    left_shoulder_torque.symbols["q_ref_rotation"]:
        lambda t, x: initial_conditions[rider.left_shoulder.q[2]],
    right_shoulder_torque.symbols["q_ref_flexion"]:
        lambda t, x: initial_conditions[rider.right_shoulder.q[0]] - excitation(t, x),
    right_shoulder_torque.symbols["q_ref_adduction"]:
        lambda t, x: initial_conditions[rider.right_shoulder.q[1]],
    right_shoulder_torque.symbols["q_ref_rotation"]:
        lambda t, x: initial_conditions[rider.right_shoulder.q[2]],
    left_arm_torque.symbols["q_ref"]:
        lambda t, x: initial_conditions[rider.left_arm.q[0]] - excitation(t, x),
    right_arm_torque.symbols["q_ref"]:
        lambda t, x: initial_conditions[rider.right_arm.q[0]] + excitation(t, x),
    data.disturbance: lambda t, x: (100 + 100 * t) * np.sin(t * 2 * np.pi),
}
for control in controls:
    if control in constants:
        constants.pop(control)
assert set(constants.keys()).isdisjoint(controls.keys())

data.update({
    "x": system.q.col_join(system.u),
    "constants": constants,
    "initial_conditions": initial_conditions,
    "controllable_loads": controls.keys(),
    "controls": controls,
})

with open("data.pkl", "wb") as f:
    cloudpickle.dump(data, f)
