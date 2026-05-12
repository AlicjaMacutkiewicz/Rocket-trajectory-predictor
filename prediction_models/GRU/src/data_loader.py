
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def read_flight_data(
    start_flight,
    num_of_flights,
    output_dir="../../../1955-1959",
):
    flights_inputs = []
    flights_targets = []
    flight_positions = []
    flight_times = []

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    flight_files = sorted(output_path.glob("flight_*.parquet"))

    if len(flight_files) < start_flight + num_of_flights:
        raise ValueError(
            f"Expected at least {start_flight + num_of_flights} flight files, found {len(flight_files)}."
        )

    for file_path in flight_files[start_flight : start_flight + num_of_flights]:
        flight_data = pd.read_parquet(file_path)

        sensor_columns = [
            "Best_Acc_X",
            "Best_Acc_Y",
            "Best_Acc_Z",
            "Best_AngVel_X",
            "Best_AngVel_Y",
            "Best_AngVel_Z",
            "Barometer_Value",
            "Sensor_Value",
        ]
        if not set(sensor_columns).issubset(flight_data.columns):
            raise ValueError(f"{file_path} is missing required sensor columns: {sensor_columns}")
        input_data = flight_data[sensor_columns].values.astype(np.float32)

        target_columns = ["Acceleration_X", "Acceleration_Y", "Acceleration_Z"]
        target_data = flight_data[target_columns].values.astype(np.float32)

        position_columns = ["Position_X", "Position_Y", "Position_Z"]
        position_data = flight_data[position_columns].values.astype(np.float32)

        if "Time" in flight_data.columns:
            time_data = flight_data["Time"].to_numpy(dtype=np.float32)
        else:
            time_data = flight_data.index.to_numpy(dtype=np.float32)

        flights_inputs.append(input_data)
        flights_targets.append(target_data)
        flight_positions.append(position_data)
        flight_times.append(time_data)

    return flights_inputs, flights_targets, flight_positions, flight_times


# Split list of flights into training and testing sets for the model
# where split_ratio is the fraction of flights used for training
def split_flights(flights, split_ratio=0.8):
    split_idx = int(len(flights) * split_ratio)

    train_flights = flights[:split_idx]
    test_flights = flights[split_idx:]

    return train_flights, test_flights


# Apply normalization to each flight independently using only the statistical
# values (mean, std) for the training dataset passed as a 1D array
def normalize_flights(flights, array):
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    return [(flight - mean) / std for flight in flights]


# X shape: (num_samples, seq_len, 3)
# y_acc shape: (num_samples, pred_len, 3)
# y_pos shape: (num_samples, pred_len, 3)
def estimate_velocity(positions, times):
    velocities = np.zeros_like(positions, dtype=np.float32)

    if len(positions) < 2:
        return velocities

    dt = np.diff(times).astype(np.float32)
    dt = np.where(dt == 0.0, 1e-6, dt)
    segment_velocity = np.diff(positions, axis=0) / dt[:, None]

    velocities[0] = segment_velocity[0]
    velocities[-1] = segment_velocity[-1]
    if len(positions) > 2:
        velocities[1:-1] = 0.5 * (segment_velocity[:-1] + segment_velocity[1:])

    return velocities



def make_sequences(flights_inputs, flights_targets, flight_positions, flight_times, seq_len, pred_len):
    # This converts long full-flight time series into many shorter training examples

    X, y_acc, y_pos, t_y, initial_pos, initial_vel, initial_time = [], [], [], [], [], [], []

    # ZMIANA: iterujemy jednocześnie po f_in (wejścia) i f_tar (targety)
    for f_in, f_tar, positions, times in zip(flights_inputs, flights_targets, flight_positions, flight_times, strict=False):
        velocities = estimate_velocity(positions, times)

        for i in range(len(f_in) - seq_len - pred_len):
            start_idx = i + seq_len - 1
            target_start_idx = i + seq_len
            target_end_idx = target_start_idx + pred_len

            # take seq_len values from past observations (z czujników - 8 kolumn)
            X.append(f_in[i : i + seq_len])
            
            # take pred_len future values to be predicted (z przyspieszenia - 3 kolumny)
            y_acc.append(f_tar[target_start_idx:target_end_idx])
            
            y_pos.append(positions[target_start_idx:target_end_idx])
            # Store exact times for the target part
            t_y.append(times[target_start_idx:target_end_idx])
            initial_pos.append(positions[start_idx])
            initial_vel.append(velocities[start_idx])
            initial_time.append(times[start_idx])

    # convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y_acc = np.array(y_acc, dtype=np.float32)
    y_pos = np.array(y_pos, dtype=np.float32)
    t_y = np.array(t_y, dtype=np.float32)
    initial_pos = np.array(initial_pos, dtype=np.float32)
    initial_vel = np.array(initial_vel, dtype=np.float32)
    initial_time = np.array(initial_time, dtype=np.float32)

    # convert to tensors (required for model training)
    return (
        torch.from_numpy(X),
        torch.from_numpy(y_acc),
        torch.from_numpy(y_pos),
        torch.from_numpy(t_y),
        torch.from_numpy(initial_pos),
        torch.from_numpy(initial_vel),
        torch.from_numpy(initial_time),
    )
