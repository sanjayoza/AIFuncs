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

def plot_training_history(history, 
                          train_loss_name='train_loss', 
                          val_loss_name='val_loss', 
                          train_acc_name='train_acc',
                          val_acc_name='val_acc',
                          figuresize=(12, 5)):
    """
    Plots the loss and accuracy curves side-by-side.
    Handles mismatched lengths and tensor/list formats.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list) and isinstance(data[0], torch.Tensor):
            return np.array([x.item() for x in data])
        return np.array(data)

    train_loss = to_numpy(history[train_loss_name])
    val_loss = to_numpy(history[val_loss_name])
    train_acc = to_numpy(history[train_acc_name])
    val_acc = to_numpy(history[val_acc_name])

    # Trim validation metrics if they’re longer than training
    min_len = min(len(train_loss), len(val_loss))
    train_loss = train_loss[:min_len]
    val_loss = val_loss[:min_len]
    train_acc = train_acc[:min_len]
    val_acc = val_acc[:min_len]

    epochs = np.arange(min_len)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figuresize)
    fig.suptitle('Model Performance', fontsize=16)

    ax1.plot(epochs, train_loss, label='Training Loss')
    ax1.plot(epochs, val_loss, label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, label='Training Accuracy')
    ax2.plot(epochs, val_acc, label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_training_history_early_stopping(history, 
                          train_loss_name='train_loss', 
                          val_loss_name='val_loss', 
                          train_acc_name='train_acc',
                          val_acc_name='val_acc',
                          monitor='val_loss',
                          mode='min',
                          figuresize=(12, 5)):
    """
    Plots training and validation loss/accuracy with early stopping marker.
    Based on the early stopping used, please the appropriate metric
    e.g. plot_training_history_early_stopping(history, monitor='val_loss', mode='min')
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list) and isinstance(data[0], torch.Tensor):
            return np.array([x.item() for x in data])
        return np.array(data)

    train_loss = to_numpy(history[train_loss_name])
    val_loss = to_numpy(history[val_loss_name])
    train_acc = to_numpy(history[train_acc_name])
    val_acc = to_numpy(history[val_acc_name])

    # Trim to shortest length
    min_len = min(len(train_loss), len(val_loss))
    train_loss = train_loss[:min_len]
    val_loss = val_loss[:min_len]
    train_acc = train_acc[:min_len]
    val_acc = val_acc[:min_len]
    epochs = np.arange(min_len)

    # Determine early stopping epoch
    monitor_values = to_numpy(history[monitor])[:min_len]
    if mode == 'min':
        best_epoch = np.argmin(monitor_values)
    elif mode == 'max':
        best_epoch = np.argmax(monitor_values)
    else:
        raise ValueError("mode must be 'min' or 'max'")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figuresize)
    fig.suptitle('Model Performance', fontsize=16)

    # Loss plot
    ax1.plot(epochs, train_loss, label='Training Loss')
    ax1.plot(epochs, val_loss, label='Validation Loss')
    ax1.axvline(best_epoch, color='black', linestyle='--', label=f'Early Stop @ {best_epoch}')
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, train_acc, label='Training Accuracy')
    ax2.plot(epochs, val_acc, label='Validation Accuracy')
    ax2.axvline(best_epoch, color='red', linestyle='--', label=f'Early Stop @ {best_epoch}')
    ax2.set_title('Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
