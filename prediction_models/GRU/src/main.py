import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from GRU_model import GRU
from pinn_physics import (
    PINNPositionMSELoss,
    calculate_x_b,
    default_physics_paths,
    load_parameters,
    load_thrust_curve,
)

# Check for current best hardware options:
#   mps - apple gpu,
#   cuda - nvidia gpu,
#   xpu - intel gpu
#   cpu - other/cpu options


def get_best_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        try:
            import intel_extension_for_pytorch  # type: ignore # noqa: F401

            dev = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
        except ImportError:
            dev = torch.device("cpu")

    print(f"running on {dev}")
    return dev


# Load data from num_of_flights parquet files into a single numpy array
# containing X, Y, Z acceleration data

# Structure of file paths from which the function reads:
# base_path - directory containing the flight parquet files + base file name
# flight_start_flight - a number added at the end of base_path to identify a single flight
# .parquet - all the data is read from parquet files


def read_flight_data(
    start_flight,
    num_of_flights,
    output_dir="../../../1955-1959",
):
   
    # We need flight_times because the physics baseline x_b depends on time
    flights = []
    flight_positions = []
    flight_times = []

   
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    flight_files = sorted(output_path.glob("flight_*.parquet"))

    if len(flight_files) < start_flight + num_of_flights:
        raise ValueError(
            f"Expected at least {start_flight + num_of_flights} flight files in {output_path}, "
            f"found {len(flight_files)}."
        )

    for file_path in flight_files[start_flight : start_flight + num_of_flights]:
        flight_data = pd.read_parquet(file_path)  # read parquet file into a dataframe

        
        if {"Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z"}.issubset(flight_data.columns):
            acc_columns = ["Best_Acc_X", "Best_Acc_Y", "Best_Acc_Z"]
        else:
            acc_columns = ["Acceleration_X", "Acceleration_Y", "Acceleration_Z"]

     
        acc_data = flight_data[acc_columns].values.astype(np.float32)

        position_columns = ["Position_X", "Position_Y", "Position_Z"]
        if not set(position_columns).issubset(flight_data.columns):
            raise ValueError(
                f"{file_path} is missing required simulator position columns: "
                f"{position_columns}."
            )
        position_data = flight_data[position_columns].values.astype(np.float32)


        if "Time" in flight_data.columns:
            time_data = flight_data["Time"].to_numpy(dtype=np.float32)
        else:
            time_data = flight_data.index.to_numpy(dtype=np.float32)

        flights.append(acc_data)
        flight_positions.append(position_data)
        flight_times.append(time_data)

    # "flights" is now a list of numpy arrays where each element contains
    # acceleration data (from X, Y, Z axis) from one flight
    # flights shape: (timesteps, 3)
    return flights, flight_positions, flight_times


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


# Convert flight time-series data into shorter learning
# samples for the GRU model
# Each sample consists of:

# X: past seq_len time steps - model input sequence
# y: next pred_len time steps - future values
# that the model is trained to predict


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


def make_sequences(flights, flight_positions, flight_times, seq_len, pred_len):
    # This converts long full-flight time series into many shorter training examples

    #   X     = past seq_len acceleration samples
    #   y     = next pred_len acceleration samples
    #   t_y   = times for those next pred_len samples


    X, y_acc, y_pos, t_y, initial_pos, initial_vel, initial_time = [], [], [], [], [], [], []

    # for every flight take a sequence of seq_len next time steps
    # so that the model can predcit pred_len values
    for flight, positions, times in zip(flights, flight_positions, flight_times, strict=False):
        velocities = estimate_velocity(positions, times)

        # flight[k] = acceleration at sample k
        # times[k]  = time of sample k
        # zip(flights, flight_times) keeps those two arrays paired
        for i in range(len(flight) - seq_len - pred_len):
            start_idx = i + seq_len - 1
            target_start_idx = i + seq_len
            target_end_idx = target_start_idx + pred_len

            # take seq_len values from past observations
            X.append(flight[i : i + seq_len])
            # take pred_len future values to be predicted
            y_acc.append(flight[target_start_idx:target_end_idx])
            y_pos.append(positions[target_start_idx:target_end_idx])
            # Store exact times for the target part
            t_y.append(times[target_start_idx:target_end_idx])
            initial_pos.append(positions[start_idx])
            initial_vel.append(velocities[start_idx])
            initial_time.append(times[start_idx])

    # convert to numpy arrays bcs apparently creating a tensor from
    # a normal list of numpy arrays is slow af
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


# Train the GRU model using small-batch gradient descent
# one training_round = one full pass over dataset


def train_model(
    model,
    X_train,
    pos_train,
    t_train,
    initial_pos_train,
    initial_vel_train,
    initial_time_train,
    X_test,
    pos_test,
    t_test,
    initial_pos_test,
    initial_vel_test,
    initial_time_test,
    loss,
    optimizer,
    pred_len,
    batch_size=64,
    training_rounds=10,
):

    train_losses = []
    test_losses = []
    last_pred = torch.zeros(
            (batch_size, pred_len, 3),
            dtype=torch.float32,
            device=device,
        )

    print("started training")

    for training_round in range(training_rounds):
        round_loss = 0.0  # cumulated prediction error for the whole round
        total_samples = 0

        model.train()  # set the mode to train (some layers behave
        # differently during training and evaluation)

        num_batches = len(X_train) // batch_size
        cutoff_start_index = random.randint(1, num_batches)
        cutoff_len = min(random.randint(0, num_batches-cutoff_start_index), num_batches//2)

        cutoff_end_index = cutoff_start_index + cutoff_len

        # iterate over the training dataset in smaller batches
        for batch_idx, i in enumerate(range(0, len(X_train), batch_size)):
            if batch_idx == cutoff_start_index:
                print("cutoff start at", i)
                model.change_mode()
            # take a slice of successive input sequences
            # starting from the current time stamp (i)

            # X_batch: (batch_size, seq_len, 3)
            # where 3 = [Acc_x, Acc_y, Acc_z]
            if model.get_mode() == "SpinUp":
                X_batch = X_train[i : i + batch_size].to(device)  # batch input
            else:
                X_batch = last_pred.to(device).to(device)
            # pos_batch shape: (batch_size, pred_len, 3)
            # correct future simulator positions used by the PINN loss
            pos_batch = pos_train[i : i + batch_size].to(device)

            # t_batch contains the target times matching y_batch
            # Shape:(batch_size, pred_len)

            # The PINN loss uses these times to calculate x_b(t), rebuild total
            # acceleration, and integrate it into position.
            t_batch = t_train[i : i + batch_size].to(device)
            initial_pos_batch = initial_pos_train[i : i + batch_size].to(device)
            initial_vel_batch = initial_vel_train[i : i + batch_size].to(device)
            initial_time_batch = initial_time_train[i : i + batch_size].to(device)

            # pass input sequence through GRU to get predictions for each time step
            # outputs shape: (batch_size, seq_len, 3)
            outputs, _ = model(X_batch)
            preds = outputs[:, -pred_len:, :]  # take only the part of the sequence
            last_pred = preds.detach()
            # that was predicted in the current iteration (so the last pred_len values)

            # The GRU output is treated as predicted_x_s. The PINN loss adds x_b
            # back, integrates total acceleration, and compares position.
            batch_loss = loss(
                preds,
                pos_batch,
                t_batch,
                initial_pos_batch,
                initial_vel_batch,
                initial_time_batch,
            )

            optimizer.zero_grad()  # reset gradients from previous step

            # calculate how should weights change to reduce loss (backpropagation)
            batch_loss.backward()
            # update weights using calculated gradients
            optimizer.step()

            round_loss += batch_loss.item()  # accumulate loss for this round
            total_samples += len(X_batch)

            if batch_idx == cutoff_end_index:
                model.change_mode()
                print("cutoff end at", i)

        # calcutate average loss over all training batches in this round
        avg_loss = round_loss / total_samples

        train_losses.append(avg_loss)

        # calcutate average loss over all test batches in this round
        avg_test_loss = evaluate_model(
            model,
            X_test,
            pos_test,
            t_test,
            initial_pos_test,
            initial_vel_test,
            initial_time_test,
            loss,
            batch_size,
            pred_len,
        )
        test_losses.append(avg_test_loss)

        print(
            f"Iteration {training_round + 1}, train loss: {avg_loss:.8e}, test loss: {avg_test_loss:.8e}"
        )

    return train_losses, test_losses


# Evaluate model using test data
def evaluate_model(
    model,
    X_test,
    pos_test,
    t_test,
    initial_pos_test,
    initial_vel_test,
    initial_time_test,
    loss,
    batch_size=64,
    pred_len=3,
):

    model.eval()  # set the mode to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        # iterate over the test dataset in smaller batches
        for i in range(0, len(X_test), batch_size):
            # select one batch of test inputs
            X_batch = X_test[i : i + batch_size].to(device)
            pos_batch = pos_test[i : i + batch_size].to(device)

            t_batch = t_test[i : i + batch_size].to(device)
            initial_pos_batch = initial_pos_test[i : i + batch_size].to(device)
            initial_vel_batch = initial_vel_test[i : i + batch_size].to(device)
            initial_time_batch = initial_time_test[i : i + batch_size].to(device)

            # pass input sequence through GRU to get predictions for each time step
            outputs, _ = model(X_batch)
            # take only the last pred_len timesteps from the output sequence
            preds = outputs[:, -pred_len:, :]
            # compute loss between predictions and true values
            curr_loss = loss(
                preds,
                pos_batch,
                t_batch,
                initial_pos_batch,
                initial_vel_batch,
                initial_time_batch,
            )

            # calculate total loss value
            test_loss += curr_loss.item()

    # return average loss over all test batches
    return test_loss / (len(X_test) / batch_size)


def plot_prediction(model, X_test, y_test, t_test, pred_len, parameters, thrust_curve, sample_idx=0, axis=0):
    model.eval()  # set the mode to evaluation mode

    with torch.no_grad():
        # X_test shape: (num_samples, seq_len, 3)
        # y_test shape: (num_samples, pred_len, 3)

        # select a single test sample (batch size = 1)
        # input_seq shape = (1, seq_len, 3)
        input_seq = X_test[sample_idx : sample_idx + 1].to(device)

        # select corresponding correct future values to be predicted (targets)
        # target shape: (pred_len, 3)
        target = y_test[sample_idx]


        target_times = t_test[sample_idx : sample_idx + 1].to(device)

        # pass the input sequence through the GRU model
        # output shape: (1, seq_len, 3)
        # hidden state (_) is ignored
        output, _ = model(input_seq)

        predicted_x_s = output[:, -pred_len:, :]

        # Calculate the known physics part for the same future times
        base_acc = calculate_x_b(target_times, parameters, thrust_curve)

        # Rebuild full acceleration for plotting:
        #   predicted_x_total = predicted_x_s + x_b
        # This is only for human-readable visualization
        # During training the loss compares predicted_x_s with true_x_s
        # 
        # prediction shape after [0]:
        #   (pred_len, 3)
        prediction = (predicted_x_s + base_acc)[0].cpu().numpy()

        # define time axes for past (input) and future (prediction)
        seq_len = input_seq.shape[1]  # length of input sequence

        # past_time: numpy array [0, 1, ..., seq_len-1]
        # represents time indices of the input sequence
        past_time = np.arange(seq_len)
        # future_time: numpy array [seq_len, ..., seq_len+pred_len-1]
        # Represents time indices of the future (prediction)
        future_time = np.arange(seq_len, seq_len + pred_len)

        plt.figure(figsize=(10, 5))
        axes_labels = ["X", "Y", "Z"]
        # plot historical data used as input
        plt.plot(
            past_time,
            X_test[sample_idx, :, axis],
            label="Historia (Input)",
            color="blue",
            marker="o",
        )
        # plot the actual future values
        plt.plot(future_time, target[:, axis], label="Prawda (Target)", color="green", marker="s")
        # plot the values predicted by the GRU model
        plt.plot(
            future_time,
            prediction[:, axis],
            label="Predykcja",
            color="red",
            linestyle="--",
            marker="x",
        )

        # add a vertical line seperating past and future
        plt.axvline(x=seq_len - 0.5, color="gray", linestyle="--")

        plt.title(f"Predykcja Przyspieszenia {axes_labels[axis]} (Próbka {sample_idx})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        #       plt.show()
        filename = f"prediction_sample_{sample_idx}.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()  # Free up memory
        print(f"Saved prediction plot to {filename}")


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(test_losses, label="Testing Loss", color="orange")
    plt.xlabel("Round")
    plt.ylabel("Loss (MSE)")
    plt.title("Model Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    plt.savefig("loss_progress.png", bbox_inches="tight")
    plt.close()
    print("Saved loss plot to loss_progress.png")


def parse_args():
    # zrobilam ladny parser argumentow z opisami i domyslnymi wartosciami
    # robione o 3 w nocy pozdrawiam 
    # UWAGA WAZNE JAK TO ODPALAC NAJLEPIEJ pare przykladow:
    # quick smoke test:
    #     python main.py --num-flights 2 --training-rounds 1 --seq-len 5
    #
    # larger training run:
    #       python main.py --num-flights 24 --training-rounds 20
    #
    parser = argparse.ArgumentParser()

  # TO ROBOCZE I TYLKO U MNIE ZAMIENCIE SE TO
    parser.add_argument("--output-dir", default="../../../1955-1959")

    # Allows skipping the first N sorted flight files
    # Useful if we want to test another slice of the same dataset
    parser.add_argument("--start-flight", type=int, default=0)

    # How many flight files to read
    parser.add_argument("--num-flights", type=int, default=24)

    # Batch size controls how many short sequences are processed at once
    parser.add_argument("--batch-size", type=int, default=64)

    # Number of full passes over the training sequences
    parser.add_argument("--training-rounds", type=int, default=10)

    # How many past samples the GRU sees
    # In this script pred_len is set equal to seq_len so the model predicts
    # the same number of future samples as it receives from the past.
    parser.add_argument("--seq-len", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()

    # For now predict as many future steps as the input history length
    #   seq_len = 40 -> use 40 past samples and predict 40 future samples.
    pred_len = args.seq_len

    parameters_path, thrust_curve_path = default_physics_paths()
    parameters = load_parameters(parameters_path)
    thrust_curve = load_thrust_curve(thrust_curve_path)

    loss = PINNPositionMSELoss(parameters, thrust_curve)

    flights, flight_positions, flight_times = read_flight_data(
        args.start_flight, args.num_flights, output_dir=args.output_dir
    )
    print("data loaded")

    train_flights, test_flights = split_flights(flights)
    train_positions, test_positions = split_flights(flight_positions)
    train_times, test_times = split_flights(flight_times)

    (
        X_train,
        y_train,
        pos_train,
        t_train,
        initial_pos_train,
        initial_vel_train,
        initial_time_train,
    ) = make_sequences(train_flights, train_positions, train_times, args.seq_len, pred_len)
    (
        X_test,
        y_test,
        pos_test,
        t_test,
        initial_pos_test,
        initial_vel_test,
        initial_time_test,
    ) = make_sequences(test_flights, test_positions, test_times, args.seq_len, pred_len)

    print("data preprocessing done")

    # device is global because train_model(), evaluate_model() and plot_prediction()already use it directly
    # idk if that's good practice but it is what it is
    global device
    device = get_best_device()
    model = GRU(input_size=3, hidden_size=64, output_size=3, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_losses, test_losses = train_model(
        model,
        X_train,
        pos_train,
        t_train,
        initial_pos_train,
        initial_vel_train,
        initial_time_train,
        X_test,
        pos_test,
        t_test,
        initial_pos_test,
        initial_vel_test,
        initial_time_test,
        loss,
        optimizer,
        batch_size=args.batch_size,
        training_rounds=args.training_rounds,
        pred_len=pred_len,
    )

    plot_losses(train_losses, test_losses)

    plot_prediction(
        model,
        X_test,
        y_test,
        t_test,
        pred_len,
        parameters,
        thrust_curve,
        sample_idx=np.random.randint(0, len(X_test)),
    )


if __name__ == "__main__":
    main()