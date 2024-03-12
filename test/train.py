# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from convgru_network import ConvGRUNetwork
from dataset import create_dataloader
from model_utils import save_model

def train(model, dataloader, num_epochs=40, model_save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_idx, (i, targets, inputs) in enumerate(dataloader):
        # for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    save_model(model, model_save_path)
    print(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    dataloader = create_dataloader()
    # print("loader:", dataloader.)
    # for i in dataloader:
    #     print(i[0].shape)
    model = ConvGRUNetwork(input_channels=1, hidden_channels=8, kernel_size=3)
    train(model, dataloader,model_save_path='convgru_model40.pth')
