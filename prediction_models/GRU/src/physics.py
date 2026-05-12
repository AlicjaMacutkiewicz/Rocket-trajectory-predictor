import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# adjust path to find the classical simulator module
CLASSICAL_MODEL_SRC = Path(__file__).resolve().parents[2] / "classical_model" / "src"
if str(CLASSICAL_MODEL_SRC) not in sys.path:
    sys.path.append(str(CLASSICAL_MODEL_SRC))

from RK4Sim import rk4_t  # noqa: E402

# ---> physical constants <---
GRAVITY = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
R_EARTH = 6371000.0
DEFAULT_FUEL_MASS = 13.04
DEFAULT_ISP = 204.26

# cache for RK4 baseline calculations to prevent redundant numerical integration
_RK4_CACHE = {}


def load_parameters(parameters_path):
    """
    Loads rocket parameters required to calculate the deterministic 
    baseline acceleration (x_b).
    """
    with open(parameters_path, encoding="utf-8") as file:
        return json.load(file)


def load_thrust_curve(thrust_path):
    """
    Loads the engine's thrust curve (Time vs. Thrust).
    The calculate_x_b() function will later interpolate between these known points.
    """
    curve = np.loadtxt(thrust_path, delimiter=",", dtype=np.float32)

    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("thrust curve must have two columns: time, thrust")
    return curve


def get_launch_direction(parameters):
    """
    Converts launch angles (inclination, heading) from parameters into a 3D direction vector.
    This vector dictates how the engine thrust is distributed across the X, Y, and Z axes.
    """
    flight = parameters.get("flight", {})

    # JSON stores angles in degrees, numpy trigonometric functions expect radians
    inclination = np.deg2rad(float(flight.get("inclination", 90.0)))
    heading = np.deg2rad(float(flight.get("heading", 0.0)))

   # inclination = 90 degrees implies a fully vertical launch
    horizontal = np.cos(inclination)
    return np.array(
        [
            horizontal * np.cos(heading),   # X
            horizontal * np.sin(heading),   # Y
            np.sin(inclination),            # Z
        ],
        dtype=np.float32,
    )


def _thrust_curve_to_dict(thrust_curve):
    return dict(zip(thrust_curve[:, 0], thrust_curve[:, 1], strict=False))


def _rk4_cache_key(parameters, thrust_curve):
    """Generates a unique hash key for the RK4 cache based on initial conditions."""
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


def _get_rk4_tables(parameters, thrust_curve, sampling_rate):
    """Executes the classical RK4 simulator and caches the resulting baseline trajectory."""
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
        sampling_rate=sampling_rate,
    )

    rk4_times = np.array([row[0] for row in trajectory], dtype=np.float32)
    rk4_positions = np.array([row[1] for row in trajectory], dtype=np.float32)
    rk4_velocities = np.array([row[2] for row in trajectory], dtype=np.float32)
    rk4_accelerations = np.array([row[3] for row in trajectory], dtype=np.float32)

    _RK4_CACHE[cache_key] = (rk4_times, rk4_positions, rk4_velocities, rk4_accelerations)
    return _RK4_CACHE[cache_key]


def calculate_x_b(times, parameters, thrust_curve, sampling_rate):
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

    rk4_times, _, _, rk4_accelerations = _get_rk4_tables(parameters, thrust_curve, sampling_rate)
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
    sampling_rate,
):
    """
    Returns base acceleration (x_b) calculated by the classical RK4 model.

    In this hybrid architecture, total acceleration is split: x_total = x_b + x_s.
    The deterministic physics (gravity, engine thrust) are handled here, allowing 
    the neural network to focus entirely on predicting the nonlinear residuals (x_s).
    """
    if not torch.is_tensor(times):
        times = torch.as_tensor(
            times,
            dtype=torch.float32,
        )

    device = times.device
    dtype = times.dtype

    rk4_times, rk4_positions, _, _ = _get_rk4_tables(parameters, thrust_curve, sampling_rate)
    flat_times = times.detach().cpu().numpy().reshape(-1)

    positions = np.stack(
        [np.interp(flat_times, rk4_times, rk4_positions[:, axis]) for axis in range(3)],
        axis=-1,
    )
    positions = positions.reshape((*times.shape, 3))

    return torch.as_tensor(positions, dtype=dtype, device=device)


class BaseAccelerationMSELoss(nn.Module):
    """
    Computes the Mean Squared Error against the nonlinear acceleration residual.

    Since the GRU is tasked with predicting only the residual (x_s), the target is 
    derived by subtracting the classical baseline (x_b) from the simulator's total output.
    """
    def __init__(self, parameters, thrust_curve, sampling_rate):
        super().__init__()
        self.parameters = parameters
        self.thrust_curve = thrust_curve
        self.mse = nn.MSELoss()
        self.sampling_rate = sampling_rate

    def forward(self, predicted_x_s, true_x_total, times):
        # calculate known/base acceleration at the exact target times
        x_b = calculate_x_b(times, self.parameters, self.thrust_curve, self.sampling_rate)

       # isolate the target residual: true_x_s = total_acceleration - base_physics
        true_x_s = true_x_total[:, :, :3] - x_b

       # compare network prediction with the residual target
        return self.mse(predicted_x_s, true_x_s)


def integrate_acceleration(acceleration, times, initial_position, initial_velocity, initial_time):
    """
    Numerically integrates acceleration into velocity and position using Newtonian kinematics.
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

        # standard kinematic equations
        position = position + (velocity * dt) + (0.5 * current_acceleration * dt * dt)
        velocity = velocity + (current_acceleration * dt)

        positions.append(position)
        previous_time = current_time

    return torch.stack(positions, dim=1), velocity


class PINNPositionMSELoss(nn.Module):
    """
    Physics-Informed Neural Network (PINN) Loss component.

    Reconstructs the total acceleration (GRU output + RK4 baseline) and integrates 
    it via Newtonian kinematics. Penalizes the network if the resulting simulated 
    trajectory drifts from the true simulator position.
    """
    def __init__(self, parameters, thrust_curve, sampling_rate):
        super().__init__()
        self.parameters = parameters
        self.thrust_curve = thrust_curve
        self.mse = nn.MSELoss()
        self.sampling_rate = sampling_rate

    def forward(
        self,
        predicted_x_s,
        true_position,
        times,
        initial_position,
        initial_velocity,
        initial_time,
    ):
        x_b = calculate_x_b(times, self.parameters, self.thrust_curve, self.sampling_rate)
        predicted_x_total = predicted_x_s + x_b

        integrated_position, _ = integrate_acceleration(
            predicted_x_total,
            times,
            initial_position,
            initial_velocity,
            initial_time,
        )

        return self.mse(integrated_position, true_position)


class TotalLoss(nn.Module):
    """
    Composite loss function balancing basic residual MSE and PINN kinematic constraints.

    Args:
        lambda_h (float): Hyperparameter controlling the strictness of physical constraints.
    """
    def __init__(
        self, parameters, thrust_curve, mean_acc, std_acc, mean_pos, std_pos, sampling_rate, lambda_h=1e-6,
    ):
        super().__init__()
        self.acc_loss = BaseAccelerationMSELoss(parameters, thrust_curve, sampling_rate)
        self.pinn_loss = PINNPositionMSELoss(parameters, thrust_curve, sampling_rate)

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

        # calculate isolated losses
        pinn = self.pinn_loss(
            denormalized_preds,
            denormalized_pos_target,
            t_batch,
            denorm_initial_pos,
            denorm_initial_vel,
            initial_time_batch,
        )

        mse_acc = self.acc_loss(denormalized_preds, denormalized_acc_target, t_batch)

        # Return weighted sum
        return mse_acc + self.lambda_h * pinn


def default_physics_paths():
    """Retrieves absolute paths for default physics configurations."""
    
    root = Path(__file__).resolve().parents[3]
    model_root = root / "source_model" / "R7_SIMLE" / "R7_OUTPUT"

    return model_root / "parameters.json", model_root / "thrust_source.csv"
