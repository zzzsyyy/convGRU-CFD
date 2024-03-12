# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CustomDataset  # Make sure this is implemented
from convgru_network import ConvGRUNetwork
from model_utils import load_model
from plot import plot_comparison
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        # for inputs, targets in dataloader:
        for batch_idx, (i, targets, inputs) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # loss = criterion(outputs, targets)
            # total_loss += loss.item()
            # predictions = model(inputs)
            # plot_comparison(inputs, predictions, targets)
            for j in range(5):
                print(f"Plotting Sample {j} in batch {batch_idx}...")
                _, axarr = plt.subplots(3, 5, figsize=(6*6, 3*5), constrained_layout=True)
                plt.subplots_adjust(wspace=0.2, hspace=-0.8)
                for n in range(5):
                    if n == 0:  # Add labels only for the first column
                        axarr[0][n].set_title("Target")
                        axarr[1][n].set_title("Prediction")
                        axarr[2][n].set_title("Diff")
                    x1, y1, x2, y2 = 40, 50, 240, 150
                    # x1, y1, x2, y2 = 0,0,500,200
                    levels = np.linspace(0,1,30)
                    tar_img = targets[j, n, 0].cpu().numpy()
                    tar_img=(tar_img-tar_img.min())/(tar_img.max() - tar_img.min())
                    im0=axarr[0][n].imshow(tar_img, origin='lower')
                    axarr[0][n].set_xlim(x1, x2)
                    axarr[0][n].set_ylim(y2, y1)
                    axarr[0][n].axis("off")
                    out_img = outputs[j, n, 0].cpu().detach().numpy()
                    out_img=(out_img-out_img.min())/(out_img.max() - out_img.min())
                    im1=axarr[1][n].imshow(out_img, origin='lower')
                    axarr[1][n].set_xlim(x1, x2)
                    axarr[1][n].set_ylim(y2, y1)
                    axarr[1][n].axis("off")
                    diff = (tar_img - out_img)
                    # print(tar_img.max(), out_img.max())
                    im2=axarr[2][n].imshow(diff, origin="lower")
                    axarr[2][n].set_xlim(x1, x2)
                    axarr[2][n].set_ylim(y2, y1)
                    axarr[2][n].axis("off")
                    if n == 4:  # For the last column, adjust the layout to accommodate the colorbar
                        axarr[2][n].axis('on')
                        axarr[2][n].get_xaxis().set_visible(False)
                        axarr[2][n].get_yaxis().set_visible(False)
                        axarr[2][n].set_frame_on(False)
                plt.colorbar(im1, ax=axarr[1, :], location='right', shrink=0.6)
                plt.colorbar(im0, ax=axarr[0, :], location='right', shrink=0.6)
                plt.colorbar(im2, ax=axarr[2, :], location='right', shrink=0.6)
                plt.show()
                # break
            break  # Remove this to visualize more batches
    return total_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming the test dataset is similar to the training dataset
    test_dataset = CustomDataset(is_train=False,
                        dir='../',
                        n_frames_input=5,
                        n_frames_output=5)  # Implement this with your test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    model_path = "convgru_model40.pth"  # Update this path
    model = ConvGRUNetwork(input_channels=1, hidden_channels=8, kernel_size=3)
    model = load_model(model, model_path).to(device)

    test_loss = evaluate_model(model, test_dataloader, device)
    print(f"Test Loss: {test_loss}")

    # Here, you could also load and evaluate other models for comparison
