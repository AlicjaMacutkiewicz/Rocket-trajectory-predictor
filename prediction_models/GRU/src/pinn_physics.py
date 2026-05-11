import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CLASSICAL_MODEL_SRC = Path(__file__).resolve().parents[2] / "classical_model" / "src"
if str(CLASSICAL_MODEL_SRC) not in sys.path:
    sys.path.append(str(CLASSICAL_MODEL_SRC))

from RK4Sim import rk4_t  # noqa: E402, I001


# helo here im trying to explain the physics part of the code which honestly i do not understand myself
# so prepare for something crazy and proobably not entirely correct but hopefully it will be helpful for someone who is also confused like me
# Constant gravity vector
# Acceleration vector has three components:
# [X acceleration, Y acceleration, Z acceleration]
#
# Z is the vertical axis
# Gravity points down so only Z  is negative.
GRAVITY = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
R_EARTH = 6371000.0
DEFAULT_FUEL_MASS = 13.04
DEFAULT_ISP = 204.26
_RK4_CACHE = {}


def load_parameters(parameters_path):
    # We need those values to calculate the known deterministic part
    # of acceleration called x_b
    with open(parameters_path, encoding="utf-8") as file:
        return json.load(file)


def load_thrust_curve(thrust_path):

    # The file does not contain a thrust value for every possible time.
    # Later calculate_x_b() will interpolate between these known points hi hi.
    curve = np.loadtxt(thrust_path, delimiter=",", dtype=np.float32)

    # Defensive check Pozdro
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("Thrust curve must have two columns: time, thrust.")
    return curve


def get_launch_direction(parameters):
    # Thrust is a force with a direction (informacja dla bezrobotnych)
    # The thrust curve tells us only how strongg the engine is at time t
    # It does not tell us where the rocket points :(
    #
    # This function converts launch angles from parameters.json into a 3D direction vector
    # That vector says how much of the thrust goes into X, Y and Z.
    flight = parameters.get("flight", {})

    # JSON stores angles in degrees. Numpy trigonometric functions expect radians
    # so we convert degrees -> radians here
    inclination = np.deg2rad(float(flight.get("inclination", 90.0)))
    heading = np.deg2rad(float(flight.get("heading", 0.0)))

    # w sumie to nie wiem czy bedziemy puszczac rakiete inaczej niz pionowo ale moze sie przyda hihi
    # inclination = 90 degrees means fully vertical launch
    # cos(90 deg) is 0 (zdałam Elementarną to wiem tak), so horizontal part is 0.
    # sin(90 deg) is 1 so vertical Z part is 1.
    # For less vertical launches horizontal becomes larger and the thrust is split between horizontal axes and vertical Z.
    horizontal = np.cos(inclination)
    return np.array(
        [
            # X
            horizontal * np.cos(heading),
            # Y
            horizontal * np.sin(heading),
            # Z
            np.sin(inclination),
        ],
        dtype=np.float32,
    )


def _thrust_curve_to_dict(thrust_curve):
    return dict(zip(thrust_curve[:, 0], thrust_curve[:, 1], strict=False))


def _rk4_cache_key(parameters, thrust_curve):
    rocket = parameters.get("rocket", {})
    motors = parameters.get("motors", {})
    stored_results = parameters.get("stored_results", {})

    return (
        float(rocket.get("mass", 50.876)),
        float(motors.get("fuel_mass", DEFAULT_FUEL_MASS)),
        float(motors.get("isp", DEFAULT_ISP)),
        float(
            stored_results.get(
                "flight_time", np.max(thrust_curve[:, 0]) if len(thrust_curve) else 0.0
            )
        ),
        tuple(get_launch_direction(parameters).tolist()),
        thrust_curve.shape,
        float(thrust_curve[0, 0]),
        float(thrust_curve[-1, 0]),
        float(thrust_curve[:, 1].sum()),
    )


def _get_rk4_tables(parameters, thrust_curve):
    cache_key = _rk4_cache_key(parameters, thrust_curve)
    cached = _RK4_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rocket = parameters.get("rocket", {})
    motors = parameters.get("motors", {})
    stored_results = parameters.get("stored_results", {})

    rocket_mass = float(rocket.get("mass", 50.876))
    fuel_mass = float(motors.get("fuel_mass", DEFAULT_FUEL_MASS))
    isp = float(motors.get("isp", DEFAULT_ISP))
    flight_time = float(stored_results.get("flight_time", np.max(thrust_curve[:, 0])))

    trajectory = rk4_t(
        start_position=np.array([0.0, 0.0, R_EARTH], dtype=np.float32),
        rocket_mass=rocket_mass,
        fuel_mass=fuel_mass,
        angle=get_launch_direction(parameters),
        time=flight_time,
        thrust=_thrust_curve_to_dict(thrust_curve),
        isp=isp,
    )

    rk4_times = np.array([row[0] for row in trajectory], dtype=np.float32)
    rk4_positions = np.array([row[1] for row in trajectory], dtype=np.float32)
    rk4_velocities = np.array([row[2] for row in trajectory], dtype=np.float32)
    rk4_accelerations = np.array([row[3] for row in trajectory], dtype=np.float32)

    _RK4_CACHE[cache_key] = (rk4_times, rk4_positions, rk4_velocities, rk4_accelerations)
    return _RK4_CACHE[cache_key]


def calculate_x_b(times, parameters, thrust_curve):
    """Return base acceleration x_b(t) from the RK4 baseline model.

    x_b - base acceleration

    In this project we split total acceleration into two parts:

        x_total = x_b + x_s

    where:
        x_b = known/base physics calculated by classical_model/src/RK4Sim.py

        x_s = unknown /nonlinear /messy part for example wind type shit

    The neural network should not waste capacity learning the deterministic RK4 baseline.
    So this function gets x_b from RK4 and the GRU learns only the residual x_s.
    """

    # We get tensors but for fun i added that you can also get a list or NumPy array it converts to tensor
    if not torch.is_tensor(times):
        times = torch.as_tensor(times, dtype=torch.float32)

    # Keep all tensors on the same device and with the same dtype as times.

    device = times.device
    dtype = times.dtype

    rk4_times, _, _, rk4_accelerations = _get_rk4_tables(parameters, thrust_curve)
    flat_times = times.detach().cpu().numpy().reshape(-1)

    x_b = np.stack(
        [np.interp(flat_times, rk4_times, rk4_accelerations[:, axis]) for axis in range(3)],
        axis=-1,
    )
    x_b = x_b.reshape((*times.shape, 3))

    return torch.as_tensor(x_b, dtype=dtype, device=device)


def calculate_position(
    times,
    parameters,
    thrust_curve,
):

    if not torch.is_tensor(times):
        times = torch.as_tensor(
            times,
            dtype=torch.float32,
        )

    device = times.device
    dtype = times.dtype

    rk4_times, rk4_positions, _, _ = _get_rk4_tables(parameters, thrust_curve)
    flat_times = times.detach().cpu().numpy().reshape(-1)

    positions = np.stack(
        [np.interp(flat_times, rk4_times, rk4_positions[:, axis]) for axis in range(3)],
        axis=-1,
    )
    positions = positions.reshape((*times.shape, 3))

    return torch.as_tensor(positions, dtype=dtype, device=device)


class BaseAccelerationMSELoss(nn.Module):
    # Here everything comes togheter
    #
    # It is still a normal MSE loss but it compares the network output
    # against the nonlinear acceleration x_s not against the full x_total

    # Formula:
    #   true_x_s = true_x_total - x_b
    #   MSE_acc = MSE(predicted_x_s, true_x_s)
    #
    # The network output is interpreted as predicted_x_s
    # That means the network is learning only what the simple physics model
    # does not explain
    def __init__(self, parameters, thrust_curve):
        super().__init__()

        # Store physical data needed to calculate x_b every time loss is called
        self.parameters = parameters
        self.thrust_curve = thrust_curve

        # Standard PyTorch mean squared error:
        #   mean((prediction - target) ** 2)
        self.mse = nn.MSELoss()

    def forward(self, predicted_x_s, true_x_total, times):
        # Step 1:
        # Calculate known/base acceleration at the exact target times
        # x_b shape should match true_x_total:
        #   (batch_size, pred_len, 3)
        x_b = calculate_x_b(times, self.parameters, self.thrust_curve)

        # Step 2:
        # The simulator gives us total acceleration as the target for training
        # But the network is supposed to output only the nonlinear part x_s
        # So we subtract the known base part from the total:
        #
        #   true_x_s = true_x_total - x_b
        #
        # This is the target for the GRUUUUU
        # we only use the first 3 columns of the true_x_total as we only want
        # to take the acceleration data into consideration
        true_x_s = true_x_total[:, :, :3] - x_b

        # Step 3:
        # Compare what the network predicted with the residual target
        #
        # FYI: if this value is small we good
        return self.mse(predicted_x_s, true_x_s)


def integrate_acceleration(acceleration, times, initial_position, initial_velocity, initial_time):
    """
    Here we go again with the love for physics and newton kinematics :)
    Integrate acceleration into velocity and position with Newton kinematics

    acceleration shape: (batch_size, pred_len, 3)
    times shape: (batch_size, pred_len)
    initial_position / initial_velocity shape: (batch_size, 3)
    initial_time shape: (batch_size,)
    """
    if not torch.is_tensor(times):
        times = torch.as_tensor(times, dtype=torch.float32)

    # Keep all tensors on the same device and with the same dtype as times.

    positions = []
    position = initial_position
    velocity = initial_velocity
    previous_time = initial_time

    for step in range(acceleration.shape[1]):
        current_time = times[:, step]
        dt = (current_time - previous_time).clamp_min(0.0).unsqueeze(-1)
        current_acceleration = acceleration[:, step, :]

        position = position + (velocity * dt) + (0.5 * current_acceleration * dt * dt)
        velocity = velocity + (current_acceleration * dt)

        positions.append(position)
        previous_time = current_time

    return torch.stack(positions, dim=1), velocity


class PINNPositionMSELoss(nn.Module):
    # PINN loss for the GRU residual model:
    #   1. rebuild total acceleration: predicted_x_total = predicted_x_s + x_b
    #   2. integrate that acceleration into velocity and position
    #   3. compare integrated position with simulator position
    def __init__(self, parameters, thrust_curve):
        super().__init__()
        self.parameters = parameters
        self.thrust_curve = thrust_curve
        self.mse = nn.MSELoss()

    def forward(
        self,
        predicted_x_s,
        true_position,
        times,
        initial_position,
        initial_velocity,
        initial_time,
    ):
        x_b = calculate_x_b(times, self.parameters, self.thrust_curve)
        predicted_x_total = predicted_x_s + x_b

        integrated_position, _ = integrate_acceleration(
            predicted_x_total,
            times,
            initial_position,
            initial_velocity,
            initial_time,
        )

        return self.mse(integrated_position, true_position)


# Integrate the base mse function with calculated pinn_loss
class total_loss(nn.Module):
    def __init__(
        self, parameters, thrust_curve, mean_acc, std_acc, mean_pos, std_pos, lambda_h=1e-6
    ):
        super().__init__()
        self.acc_loss = BaseAccelerationMSELoss(parameters, thrust_curve)
        self.pinn_loss = PINNPositionMSELoss(parameters, thrust_curve)

        self.mean_acc = torch.tensor(mean_acc, dtype=torch.float32)
        self.std_acc = torch.tensor(std_acc, dtype=torch.float32)
        self.mean_pos = torch.tensor(mean_pos, dtype=torch.float32)
        self.std_pos = torch.tensor(std_pos, dtype=torch.float32)

        self.lambda_h = lambda_h

    def forward(
        self,
        preds,
        acc_batch,
        pos_batch,
        t_batch,
        initial_pos_batch,
        initial_vel_batch,
        initial_time_batch,
    ):

        # Denormalise accelerations and positions as values used to calculate pinn loss
        # are not normalized

        # everything now is nicely denormalized  :)

        device = preds.device
        denormalized_preds = preds * self.std_acc.to(device) + self.mean_acc.to(device)
        denormalized_acc_target = acc_batch[:, :, :3] * self.std_acc.to(device) + self.mean_acc.to(
            device
        )

        denormalized_pos_target = pos_batch[:, :, :3] * self.std_pos.to(device) + self.mean_pos.to(
            device
        )

        denorm_initial_pos = initial_pos_batch * self.std_pos.to(device) + self.mean_pos.to(device)
        denorm_initial_vel = initial_vel_batch * self.std_pos.to(device)

        pinn = self.pinn_loss(
            denormalized_preds,
            denormalized_pos_target,
            t_batch,
            denorm_initial_pos,
            denorm_initial_vel,
            initial_time_batch,
        )

        mse_acc = self.acc_loss(denormalized_preds, denormalized_acc_target, t_batch)
        return mse_acc + self.lambda_h * pinn


def default_physics_paths():

    root = Path(__file__).resolve().parents[3]
    model_root = root / "source_model" / "R7_SIMLE" / "R7_OUTPUT"

    return model_root / "parameters.json", model_root / "thrust_source.csv"
