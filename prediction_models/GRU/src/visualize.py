
import numpy as np
import torch
from matplotlib import pyplot as plt

from prediction_models.GRU.src.pinn_physics import calculate_x_b


def plot_prediction(
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
    sample_idx=0,
    axis=0,
):
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
        # output, _ = model(input_seq)

        # predicted_x_s = output[:, -pred_len:, :]
        predicted_x_s, _ = model(input_seq, pred_len=pred_len)
        # Calculate the known physics part for the same future times
        base_acc = calculate_x_b(target_times, parameters, thrust_curve)

        # Rebuild full acceleration for plotting:
        #   predicted_x_total = predicted_x_s + x_b
        # This is only for human-readable visualization
        # During training the loss compares predicted_x_s with true_x_s
        #
        # prediction shape after [0]:
        #   (pred_len, 3)

        predicted_x_s_denorm = predicted_x_s[0].cpu().numpy() * std_acc + mean_acc
        target_denorm = target.cpu().numpy() * std_acc + mean_acc
        history_denorm = X_test[sample_idx, :, :3].cpu().numpy() * std_in[:3] + mean_in[:3]

        base_acc = calculate_x_b(target_times, parameters, thrust_curve)[0].cpu().numpy()
        prediction = predicted_x_s_denorm + base_acc
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
        # plot the actual future values
    plt.plot(past_time, history_denorm[:, axis], label="Historia (Input)", color="blue", marker="o")
    plt.plot(
        future_time, target_denorm[:, axis], label="Prawda (Target)", color="green", marker="s"
    )
    plt.plot(
        future_time, prediction[:, axis], label="Predykcja", color="red", linestyle="--", marker="x"
    )
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
