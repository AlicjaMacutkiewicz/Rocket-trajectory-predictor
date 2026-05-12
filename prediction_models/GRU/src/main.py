import argparse

import numpy as np
import torch
import torch.optim as optim
from GRU_model import GRU
from pinn_physics import (
    default_physics_paths,
    load_parameters,
    load_thrust_curve,
    total_loss,
)

from prediction_models.GRU.src.data_loader import make_sequences, read_flight_data, split_flights
from prediction_models.GRU.src.train import train_model
from prediction_models.GRU.src.visualize import plot_losses, plot_prediction

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



# Convert flight time-series data into shorter learning
# samples for the GRU model
# Each sample consists of:

# X: past seq_len time steps - model input sequence
# y: next pred_len time steps - future values
# that the model is trained to predict


# Train the GRU model using small-batch gradient descent
# one training_round = one full pass over dataset


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

    parser.add_argument("--year", type=str, default="2025")

    return parser.parse_args()


def main():
    args = parse_args()
    global device
    device = get_best_device()
    # For now predict as many future steps as the input history length
    #   seq_len = 40 -> use 40 past samples and predict 40 future samples.
    pred_len = args.seq_len

    parameters_path, thrust_curve_path = default_physics_paths()
    parameters = load_parameters(parameters_path)
    thrust_curve = load_thrust_curve(thrust_curve_path)

    flights_inputs, flights_targets, flight_positions, flight_times = read_flight_data(
        args.start_flight, args.num_flights, output_dir=args.output_dir
    )
    print("data loaded")

    train_inputs, test_inputs = split_flights(flights_inputs)
    train_targets, test_targets = split_flights(flights_targets)
    train_positions, test_positions = split_flights(flight_positions)
    train_times, test_times = split_flights(flight_times)

    all_train_inputs = np.concatenate(train_inputs, axis=0)
    mean_in = all_train_inputs.mean(axis=0)
    std_in = all_train_inputs.std(axis=0)
    std_in = np.where(std_in == 0, 1e-6, std_in) # div by zero safeguard

    all_train_targets = np.concatenate(train_targets, axis=0)
    mean_acc = all_train_targets.mean(axis=0)
    std_acc = all_train_targets.std(axis=0)
    std_acc = np.where(std_acc == 0, 1e-6, std_acc)

    all_train_pos = np.concatenate(train_positions, axis=0)
    mean_pos = all_train_pos.mean(axis=0)
    std_pos = all_train_pos.std(axis=0)
    std_pos = np.where(std_pos == 0, 1e-6, std_pos)

    train_inputs = [(f - mean_in) / std_in for f in train_inputs]
    test_inputs = [(f - mean_in) / std_in for f in test_inputs]
    
    train_targets = [(f - mean_acc) / std_acc for f in train_targets]
    test_targets = [(f - mean_acc) / std_acc for f in test_targets]

    train_positions = [(p - mean_pos) / std_pos for p in train_positions]
    test_positions = [(p - mean_pos) / std_pos for p in test_positions]

    loss = total_loss(parameters, thrust_curve, mean_acc, std_acc, mean_pos, std_pos).to(device)
    (
        X_train,
        y_train,
        pos_train,
        t_train,
        initial_pos_train,
        initial_vel_train,
        initial_time_train,
    ) = make_sequences(train_inputs, train_targets, train_positions, train_times, args.seq_len, pred_len)
    (
        X_test,
        y_test,
        pos_test,
        t_test,
        initial_pos_test,
        initial_vel_test,
        initial_time_test,
    ) = make_sequences(test_inputs, test_targets, test_positions, test_times, args.seq_len, pred_len)

    print("data preprocessing done")

    # device is global because train_model(), evaluate_model() and plot_prediction()already use it directly
    # idk if that's good practice but it is what it is

    model = GRU(input_size=8, hidden_size=64, output_size=3, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses, test_losses = train_model(
        model,
        X_train,
        y_train,
        pos_train,
        t_train,
        initial_pos_train,
        initial_vel_train,
        initial_time_train,
        X_test,
        y_test,
        pos_test,
        t_test,
        initial_pos_test,
        initial_vel_test,
        initial_time_test,
        loss,
        optimizer,
        device=device,
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
        mean_acc,
        std_acc,
        mean_in,
        std_in,
        device,
        sample_idx=np.random.randint(0, len(X_test)),
    )

    model_filename = f"gru_model_rounds{args.training_rounds}_seq{args.seq_len}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"zapisano wagi modelu do pliku: {model_filename}")

    with open("stan_uczenia.txt", "a") as log_file:
        log_file.write(f"Trenowano model: {model_filename}\n")
        log_file.write(
            f"parametry: Epoki={args.training_rounds}, Batch={args.batch_size}, Rok={args.year}\n"
        )
        log_file.write(f"Wykorzystano lotów: {len(flights_inputs)}\n")
        log_file.write("-" * 40 + "\n")
    print("zaktualizowano plik stan_uczenia.txt")


if __name__ == "__main__":
    main()
