# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from convgru_network import ConvGRUNetwork
from dataset import create_dataloader
from model_utils import save_model
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

def train(model, dataloader, num_epochs=200, model_save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(num_epochs):
        for batch_idx, (i, targets, inputs) in enumerate(dataloader):
        # for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs[:,:,0,:,:].unsqueeze(2).shape, targets.shape)
            loss = criterion(outputs[:,:,0,:,:].unsqueeze(2), targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss vs. Step')
    plt.legend()
    # plt.show()
    plt.savefig("loss12.png")
    save_model(model, model_save_path)
    print(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    dataloader = create_dataloader()
    # print("loader:", dataloader.)
    # for i in dataloader:
    #     print(i[0].shape)
    model = ConvGRUNetwork(input_channels=1, hidden_channels=8, kernel_size=3)
    train(model, dataloader,model_save_path='models/convgru_model200.pt')
