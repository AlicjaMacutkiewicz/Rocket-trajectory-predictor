
import numpy as np
import torch
from matplotlib import pyplot as plt
from physics import calculate_x_b


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
    """
    Generates and saves a plot comparing the model's trajectory prediction against the ground truth.

    This function extracts a single sequence, passes it through the GRU to predict the 
    nonlinear acceleration residual, adds back the deterministic RK4 baseline (x_b), 
    and plots the fully reconstructed physical acceleration.

    Args:
        model (nn.Module): The trained GRU model.
        X_test (torch.Tensor): Tensor of historical input sequences.
        y_test (torch.Tensor): Tensor of target future sequences.
        t_test (torch.Tensor): Tensor of timestamps corresponding to the targets.
        pred_len (int): Number of future steps the model predicts.
        parameters (dict): Physical rocket parameters for RK4 integration.
        thrust_curve (np.ndarray): Engine thrust data.
        mean_acc (np.ndarray): Mean values for acceleration denormalization.
        std_acc (np.ndarray): Standard deviation for acceleration denormalization.
        mean_in (np.ndarray): Mean values for sensor input denormalization.
        std_in (np.ndarray): Standard deviation for sensor input denormalization.
        device (torch.device): The hardware accelerator to use.
        sample_idx (int, optional): Index of the test sample to visualize. Defaults to 0.
        axis (int, optional): Spatial axis to plot (0=X, 1=Y, 2=Z). Defaults to 0.
    """

    model.eval()

    with torch.no_grad():
        # select a single test sample (batch size = 1)
        input_seq = X_test[sample_idx : sample_idx + 1].to(device)
        target = y_test[sample_idx]
        target_times = t_test[sample_idx : sample_idx + 1].to(device)

       # forward pass through the GRU to get the predicted residual (x_s)
        predicted_x_s, _ = model(input_seq, pred_len=pred_len)

        # denormalize network predictions and targets
        predicted_x_s_denorm = predicted_x_s[0].cpu().numpy() * std_acc + mean_acc
        target_denorm = target.cpu().numpy() * std_acc + mean_acc

        # denormalize historical inputs (using only the first 3 columns for acceleration plotting)
        history_denorm = X_test[sample_idx, :, :3].cpu().numpy() * std_in[:3] + mean_in[:3]

        # calculate and add the known physics baseline (x_b)
        base_acc = calculate_x_b(target_times, parameters, thrust_curve)[0].cpu().numpy()
        prediction = predicted_x_s_denorm + base_acc

    # define time axes for past (input) and future (prediction)
    seq_len = input_seq.shape[1]
    past_time = np.arange(seq_len)
    future_time = np.arange(seq_len, seq_len + pred_len)

    # plotting
    plt.figure(figsize=(10, 5))
    axes_labels = ["X", "Y", "Z"]

    plt.plot(past_time, history_denorm[:, axis], label="History (Input)", color="blue", marker="o")
    plt.plot(
        future_time, target_denorm[:, axis], label="Ground Truth (Target)", color="green", marker="s"
    )
    plt.plot(
        future_time, prediction[:, axis], label="Prediction", color="red", linestyle="--", marker="x"
    )
    plt.title(f"{axes_labels[axis]}-Axis Acceleration Prediction (Sample {sample_idx})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"prediction_sample_{sample_idx}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot to {filename}")


def plot_losses(train_losses, test_losses):
    """
    Generates and saves a plot of the training and testing loss over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(test_losses, label="Testing Loss", color="orange")
    plt.xlabel("Round")
    plt.ylabel("Loss (MSE)")
    plt.title("Model Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("loss_progress.png", bbox_inches="tight")
    plt.close()

    print("Saved loss plot to loss_progress.png")
