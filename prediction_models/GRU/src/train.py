import torch
import wandb


def train_model(
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
    pred_len,
    device,
    batch_size=64,
    training_rounds=10,
    year="all"
):

    train_losses = []
    test_losses = []

    wandb.init(
        project="rocket-trajectory", 
        name=f"gru_seq{pred_len}_yr{year}",
        config={"batch_size": batch_size, "epochs": training_rounds, "seq_len": pred_len}
    )
    # Automatyczne śledzenie wag i gradientów z modelu!
    wandb.watch(model, log="all")

    for training_round in range(training_rounds):
        round_loss = 0.0  # cumulated prediction error for the whole round
        total_samples = 0

        model.train()

        for i in range(0, len(X_train), batch_size):
            # if batch_idx == cutoff_start_index:
            #     print("cutoff start at", i)
            #     model.change_mode()
            # take a slice of successive input sequences
            # starting from the current time stamp (i)

            # X_batch: (batch_size, seq_len, 3)
            # where 3 = [Acc_x, Acc_y, Acc_z]
            # if model.get_mode() == "SpinUp":
            X_batch = X_train[i : i + batch_size].to(device)  # batch input
            # else:
            # X_batch = last_pred.to(device).to(device)
            # pos_batch shape: (batch_size, pred_len, 3)
            # correct future simulator positions used by the PINN loss
            y_batch = y_train[i : i + batch_size].to(device)
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
            # outputs, _ = model(X_batch)
            # preds = outputs[:, -pred_len:, :]  # take only the part of the sequence
            preds, _ = model(X_batch, pred_len=pred_len)
            # last_pred = preds.detach()
            # that was predicted in the current iteration (so the last pred_len values)

            # The GRU output is treated as predicted_x_s. The PINN loss adds x_b
            # back, integrates total acceleration, and compares position.
            batch_loss = loss(
                preds,
                y_batch,
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

            # if batch_idx == cutoff_end_index:
            #     model.change_mode()
            #     print("cutoff end at", i)
            pass

        # calcutate average loss over all training batches in this round
        avg_loss = round_loss / total_samples

        train_losses.append(avg_loss)

        # calcutate average loss over all test batches in this round
        avg_test_loss = evaluate_model(
            model,
            X_test,
            y_test,
            pos_test,
            t_test,
            initial_pos_test,
            initial_vel_test,
            initial_time_test,
            loss,
            device,
            batch_size,
            pred_len,
        )
        test_losses.append(avg_test_loss)

        print(
            f"Iteration {training_round + 1}, train loss: {avg_loss:.8e}, test loss: {avg_test_loss:.8e}"
        )

        wandb.log({
            "Train Loss": avg_loss,
            "Test Loss": avg_test_loss,
            "Epoch": training_round + 1
        })
    
        if (training_round + 1) % 5 == 0:
            checkpoint_filename = f"gru_checkpoint_round_{training_round + 1}_seq{pred_len}.pth"
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"checkpoint: {checkpoint_filename}")

    wandb.finish()
    return train_losses, test_losses


# Evaluate model using test data
def evaluate_model(
    model,
    X_test,
    y_test,
    pos_test,
    t_test,
    initial_pos_test,
    initial_vel_test,
    initial_time_test,
    loss,
    device,
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

            y_batch = y_test[i : i + batch_size].to(device)
            pos_batch = pos_test[i : i + batch_size].to(device)

            t_batch = t_test[i : i + batch_size].to(device)
            initial_pos_batch = initial_pos_test[i : i + batch_size].to(device)
            initial_vel_batch = initial_vel_test[i : i + batch_size].to(device)
            initial_time_batch = initial_time_test[i : i + batch_size].to(device)

            # # pass input sequence through GRU to get predictions for each time step
            # outputs, _ = model(X_batch)
            # # take only the last pred_len timesteps from the output sequence
            # preds = outputs[:, -pred_len:, :]

            # Model bezpośrednio zwraca tensor z predykcjami o długości pred_len
            preds, _ = model(X_batch, pred_len=pred_len)
            # compute loss between predictions and true values
            curr_loss = loss(
                preds,
                y_batch,
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


