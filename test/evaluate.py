# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CustomDataset  # Make sure this is implemented
from convgru_network import ConvGRUNetwork
from model_utils import load_model
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from plot import plot_true_pred_diff_plot, plot_true_pred_diff_gray, plot_save
# cmap = ListedColormap(['white', 'black'])
cmap='RdBu'

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        # for inputs, targets in dataloader:
        for batch_idx, (i, targets, inputs) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # save output to file
            # np.save(f'output_{batch_idx}.npy', outputs.cpu().numpy())
            # np.save(f'target_{batch_idx}.npy', targets.cpu().numpy())
            # print("in",inputs.shape, "out", outputs.shape)
            # loss = criterion(outputs, targets)
            # total_loss += loss.item()
            # predictions = model(inputs)
            # plot_comparison(inputs, predictions, targets)

            #======#
            # plot_true_pred_diff_plot(outputs, targets, batch_idx, cutline=('y', 100))
            # plot_true_pred_diff_gray(outputs, targets, batch_idx)
            plot_save(outputs, targets, batch_idx, cutline=('x', 60), break_=True)
            #======#
            break
    return total_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming the test dataset is similar to the training dataset
    test_dataset = CustomDataset(is_train=False,
                        dir='../',
                        n_frames_input=5,
                        n_frames_output=5)  # Implement this with your test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    model_path = "./models/convgru_model200.pt"  # Update this path
    model = ConvGRUNetwork(input_channels=1, hidden_channels=8, kernel_size=3)
    model = load_model(model, model_path).to(device)

    test_loss = evaluate_model(model, test_dataloader, device)
    print(f"Test Loss: {test_loss}")

    # Here, you could also load and evaluate other models for comparison
