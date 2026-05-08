import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# helo here im trying to explain the physics part of the code which honestly i do not understand myself
# so prepare for something crazy and proobably not entirely correct but hopefully it will be helpful for someone who is also confused like me
# Constant gravity vector 
# Acceleration vector has three components:
#[X acceleration, Y acceleration, Z acceleration]
#
# Z is the vertical axis
# Gravity points down so only Z  is negative.
GRAVITY = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)


def load_parameters(parameters_path):
    # We need those values to calculate the known deterministic part
    # of acceleration called x_b
    with open(parameters_path, "r", encoding="utf-8") as file:
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


def calculate_x_b(times, parameters, thrust_curve):
    """Return base acceleration x_b(t): gravity and thrust only.

    x_b - base acceleration

    In this project we split total acceleration into two parts:

        x_total = x_b + x_s

    where:
        x_b = known physics that we caan caldulate directly
              currently: gravity + engine thrust

        x_s = unknown /nonlinear /messy part for example wind type shit 

    The neural network should not waste capacity learning gravity and thrust thats what I think
    So this function calculates them directly.
    """

    # We get tensors but for fun i added that you can also get a list or NumPy array it converts to tensor
    if not torch.is_tensor(times):
        times = torch.as_tensor(times, dtype=torch.float32)

    # Keep all tensors on the same device and with the same dtype as times.
    
    device = times.device
    dtype = times.dtype

    # Convert the thrust curve from numpy to torch
    # curve shape:(number_of_points, 2)
    # curve[:, 0] -> all times from the thrust table
    # curve[:, 1] -> all thrust values from the thrust table
    curve = torch.as_tensor(thrust_curve, dtype=dtype, device=device)

    # .contiguous() makes the memory layout simple for torch.bucketize().
    # Without this PyTorch may still work i think so
    curve_t = curve[:, 0].contiguous()
    curve_f = curve[:, 1].contiguous()

    # We need thrust(t) for every time in the training batch
    # The CSV contains thrust only at selected points so we do linear
    # interpolation :)
    #   known point A: (t0, f0)
    #   known point B: (t1, f1)
    #   requested time: t, somewhere between t0 and t1
    
    # We estimate:
    #   thrust(t) = f0 + alpha * (f1 - f0)
    #alpha says how far t isbetween t0 and t1.
    #
    # torch.bucketize(times, curve_t) finds the index of the first curve_t value >= each requested time
 
    idx = torch.bucketize(times, curve_t)

    # idx1 point on the right side of requested time
    # idx0 point on the left side.
    # clamp prevents indexing outside the thrust table for times at the beginning or end of the flight

    idx0 = torch.clamp(idx - 1, min=0, max=len(curve_t) - 1)
    idx1 = torch.clamp(idx, min=0, max=len(curve_t) - 1)

    # Gather the two neighboring time values and thrust values
    # These tensors have the same shape as times 
    # If times has shape (batch_size, pred_len), then t0, t1, f0 and f1
    # also have shape (batch_size, pred_len)
    t0 = curve_t[idx0]
    t1 = curve_t[idx1]
    f0 = curve_f[idx0]
    f1 = curve_f[idx1]

    # alpha is the interpolation ratio
    # alpha = 0 exactly at the left point
    # alpha = 1 exactly at the right point
    # alpha = 0.5  halfway between
    # If t0 == t1, division would be impossible
    # Happens at the edges after clamping, so then we use alpha = 0
    alpha = torch.where(t1 == t0, torch.zeros_like(times), (times - t0) / (t1 - t0))

    # Interpolated thrust force for each requested time
    # If time is after the last point in the thrust curve the motor is
    # considered burned out so thruust becomes 0
    thrust = torch.where(times <= curve_t[-1], f0 + alpha * (f1 - f0), torch.zeros_like(times))

    # z tego miejsca chcialabym pozdrowic newtona i jego drugie prawo ruchu bo bez tego to bysmy tu nie byli
   
    mass = float(parameters["rocket"]["mass"])

    # Direction is a 3D vector [dx, dy, dz].
    # It converts scalar thrust force into a 3D force/acceleration vector
    direction = torch.as_tensor(get_launch_direction(parameters), dtype=dtype, device=device)

    # Move gravity to the same device and dtype as the rest of the computation
    gravity = GRAVITY.to(device=device, dtype=dtype)

    # thrust currently has shape:
    #   (batch_size, pred_len)
    # direction has shape:
    #   (3,)
    # To multiply them thrust needs an extra last dimension:
    #   thrust.unsqueeze(-1) -> (batch_size, pred_len, 1)
    # Then PyTorch broadcasts it with direction:
    #   (batch_size, pred_len, 1) * (3,)
    #   -> (batch_size, pred_len, 3)
    # Result:
    #   thrust acceeleration vector + gravity vector
    #
    # max(mass, 1e-8) prevents division by zero if the config is broken :) becouse shit happens you know
    return gravity + (thrust.unsqueeze(-1) * direction / max(mass, 1e-8))


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
    def __init__(self, parameters, thrust_curve, lambda_h= 0.001):
        super().__init__()

        self.acc_loss = BaseAccelerationMSELoss(parameters, thrust_curve)
        self.pinn_loss = PINNPositionMSELoss(parameters, thrust_curve)
        
        # lambda_h will be determined experimantally. It determines how strongly
        # the physical constraints influence the training compared to the data loss

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
        
        mse_acc = self.acc_loss(preds, acc_batch, t_batch)

        pinn = self.pinn_loss(
            preds,
            pos_batch,
            t_batch,
            initial_pos_batch,
            initial_vel_batch,
            initial_time_batch, )
        return mse_acc + self.lambda_h * pinn

def default_physics_paths():

    root = Path(__file__).resolve().parents[3]
    model_root = root / "source_model" / "R7_SIMLE" / "R7_OUTPUT"

    return model_root / "parameters.json", model_root / "thrust_source.csv"
