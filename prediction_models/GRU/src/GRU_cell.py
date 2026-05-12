import math

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """
    A custom implementation of a Gated Recurrent Unit (GRU) cell.

    This module manually defines the weight matrices and bias vectors 
    required to compute the reset gate, update gate, and candidate hidden state.

    Args:
        input_size (int): The number of expected features in the input `x`.
        hidden_size (int): The number of features in the hidden state `h`.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier initialization bound for weight generation
        std = 1.0 / math.sqrt(hidden_size)

        # Reset gate (r) parameters: W_r (input weights), U_r (hidden weights), b_r (bias)
        self.W_r = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(-std, std))
        self.U_r = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-std, std))
        self.b_r = nn.Parameter(torch.zeros(hidden_size))

        # Update gate (z) parameters: W_z (input weights), U_z (hidden weights), b_z (bias)
        self.W_z = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(-std, std))
        self.U_z = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-std, std))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))

        # Candidate state (h') parameters: W (input weights), U (hidden weights), b (bias)
        self.W = nn.Parameter(torch.empty(input_size, hidden_size).uniform_(-std, std))
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-std, std))
        self.b = nn.Parameter(torch.zeros(hidden_size))

    def get_reset_gate(self, x, h_prev):
        """ Computes the reset gate: r(t) = sigmoid(x(t)@W_r + h(t-1)@U_r + b_r) """
        return torch.sigmoid((x @ self.W_r) + (h_prev @ self.U_r) + self.b_r)

    def get_update_gate(self, x, h_prev):
        """ Computes the update gate: z(t) = sigmoid(x(t)@W_z + h(t-1)@U_z + b_z) """
        return torch.sigmoid((x @ self.W_z) + (h_prev @ self.U_z) + self.b_z)

    def get_candidate_gate(self, x, h_prev):
        """ Computes the candidate hidden state: h'(t) = tanh(x(t)@W + (r(t) * h(t-1))@U + b) """
        r = self.get_reset_gate(x, h_prev)
        return torch.tanh(((r * h_prev) @ self.U) + (x @ self.W) + self.b)

    def forward(self, x, h_prev):
        """
        Computes the final hidden state for the current time step.
        h(t) = (1 - z(t)) * h'(t) + z(t) * h(t-1)
        """
        z = self.get_update_gate(x, h_prev)
        h_c = self.get_candidate_gate(x, h_prev)
        return ((1 - z) * h_c) + (h_prev) * z
