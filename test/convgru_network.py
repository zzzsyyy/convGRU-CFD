# convgru_network.py
import torch
import torch.nn as nn
from convgru_cell import ConvGRUCell

class ConvGRUNetwork(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvGRUNetwork, self).__init__()
        self.convgru_cell = ConvGRUCell(input_channels, hidden_channels, kernel_size)
        
    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        hidden = torch.zeros(batch_size, self.convgru_cell.hidden_channels, height, width, device=x.device)
        outputs = []
        for t in range(seq_len):
            hidden = self.convgru_cell(x[:, t, :, :, :], hidden)
            outputs.append(hidden.unsqueeze(1))
        return torch.cat(outputs, dim=1)
