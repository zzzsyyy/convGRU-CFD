import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(inputs, predictions, targets, num_samples=5, channel_to_visualize=0):
    """
    Plots a comparison between predictions and targets for a number of samples and displays the prediction error.
    
    Parameters:
    - inputs, predictions, targets: Tensors of shape (batch, seq_len, channels, height, width).
    - num_samples: Number of samples to plot.
    - channel_to_visualize: For predictions with multiple channels, specify which channel to visualize.
    """
    inputs = inputs.cpu().numpy()
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    print(inputs.shape, predictions.shape, targets.shape)

    for i in range(min(num_samples, inputs.shape[0])):
        for seq in range(inputs.shape[1]):
            plt.figure(figsize=(20, 5))

            # Input Frame
            plt.subplot(2, 2, 1)
            plt.imshow(np.squeeze(inputs[i, seq]), origin='lower')
            plt.title(f'Input Seq {seq+1}')
            plt.axis('off')

            # Predicted Frame - Visualizing the specified channel
            plt.subplot(2, 2, 2)
            plt.imshow(np.squeeze(predictions[i, seq, channel_to_visualize]), origin='lower')
            plt.title(f'Prediction Seq {seq+1}, Ch {channel_to_visualize+1}')
            plt.axis('off')

            # Ground Truth Frame
            plt.subplot(2, 2, 3)
            plt.imshow(np.squeeze(targets[i, seq]), origin='lower')
            plt.title(f'Ground Truth Seq {seq+1}')
            plt.axis('off')

            # Error Frame
            error = np.abs(np.squeeze(targets[i, seq]) - np.squeeze(predictions[i, seq, channel_to_visualize]))
            plt.subplot(2, 2, 4)
            plt.imshow(error, origin='lower')
            plt.title(f'Error Seq {seq+1}')
            plt.colorbar(label='Error')
            plt.axis('off')

            plt.show()
            break
        break
