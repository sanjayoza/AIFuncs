import numpy as np
import matplotlib.pyplot as plt
import torch

def plotLossCurve_PyTorch(history, train_loss_name = 'train_loss', 
                     val_loss_name = 'val_loss', 
                     figuresize = (5,4), 
                     ylim = None):
    """
    Plots the loss curves. This function is compatible with both a manual history 
    dictionary (containing Python lists) and a PyTorch Lightning logger history
    (containing Tensors). It automatically handles the case where the validation
    history has one more data point than the training history.

    Args:
        history (dict): A dictionary containing training history.
        train_loss_name (str): Key for training loss in the history dictionary.
        val_loss_name (str): Key for validation loss in the history dictionary.
        figuresize (tuple): Figure size for matplotlib - a tuple (5,4).
        ylim (tuple): y-axis limits.
    """
    
    # Check if the history values are PyTorch Tensors and convert if necessary
    if isinstance(history[train_loss_name], torch.Tensor):
        train_loss = history[train_loss_name].cpu().numpy()
        val_loss = history[val_loss_name].cpu().numpy()
    else:
        train_loss = history[train_loss_name]
        val_loss = history[val_loss_name]
        
    # Handle the length mismatch (validation history has 1 more element)
    # The first element of val_loss corresponds to the pre-training check.
    if len(val_loss) > len(train_loss):
        val_loss = val_loss[1:]

    epochs = np.arange(0, len(train_loss))
    plt.figure(figsize = figuresize)
    plt.style.use('ggplot')
    plt.plot(epochs, train_loss, label = 'Training Loss')
    plt.plot(epochs, val_loss, label = 'Validation Loss')
    
    if ylim:
        plt.ylim(ylim) # Corrected function call
        
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

# I also recommend a similar update for your plotTrainingCurves function
def plotTrainingCurves(train_metric, val_metric, title='Metric', figuresize=(5, 4), ylim=None):
    """
    Plots training and validation metrics.
    
    Args:
        train_metric (list or Tensor): Training metric history.
        val_metric (list or Tensor): Validation metric history.
        title (str): Title of the plot.
        figuresize (tuple): Figure size for matplotlib.
        ylim (tuple): y-axis limits.
    """
    # Check if the values are tensors and convert them
    if isinstance(train_metric, torch.Tensor):
        train_data = train_metric.cpu().numpy()
        val_data = val_metric.cpu().numpy()
    else:
        train_data = train_metric
        val_data = val_metric

    # Handle the case where validation history has an extra epoch
    if len(val_data) > len(train_data):
        val_data = val_data[1:]

    epochs = np.arange(0, len(train_data))
    
    plt.figure(figsize=figuresize)
    plt.style.use('ggplot')
    
    plt.plot(epochs, train_data, label=f'Training {title}')
    plt.plot(epochs, val_data, label=f'Validation {title}')
    
    if ylim:
        plt.ylim(ylim)
        
    plt.title(f'Training and Validation {title}')
    plt.ylabel(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
