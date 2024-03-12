# convgru_cell.py
import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.reset_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, input_tensor, hidden_state):
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))
        combined_out = torch.cat([input_tensor, reset * hidden_state], dim=1)
        new_hidden = (1 - update) * hidden_state + update * torch.tanh(self.out_gate(combined_out))
        return new_hidden
