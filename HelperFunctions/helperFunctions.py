"""
Helper functions that will be used in various notebooks
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import subprocess

import pandas as pd


# ==========================================================================
# Dataset Downloading Functions:
# ==========================================================================

def walk_through_dir(dir_path, printfnames = False):
    """
    Walks through dir_path returning its contents.
    Args:
        dir_path (str): target directory
    
    Returns:
    prints:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
        if printfnames is True:
            for f in filenames:
                print(f)


def getFromGCS(fname, toloc):
    """
    Get file from google cloud storage
    """
    cloc = "https://storage.googleapis.com/courses-datasets/AI-ML-Toolkit/"
    #dnfile = "IMDBDataset.csv"
    dnfile = cloc + fname
    print(f'Cloud file location: {dnfile}')
    print(f'to location: {toloc}')
    result = subprocess.run(['wget',  str(dnfile),  '-NP',  str(toloc)])
    print(result)
    print(f'Got file: {dnfile} copied to {toloc}')



def mountGoogleDrive():
    """
    Mount Google Drive
    """
    from google.colab import drive
    mydrive = '/content/gdrive/'
    drive.mount(mydrive)
    os.system("ls $mydrive")
    return mydrive

def downloadFromCloud(cloudfile, subfolder = None, COLAB = False, isZipped = False):
    """
    Download file from Google Cloud
    Args:
    cloudfile: File name to download
    subfolder: subfolder name if the files are in it
    COLAB: Flag to indicate if working in COLAB environment
    isZipped: Flag to indicate if the file is zipped
    """
    currLoc = os.getcwd()
    print(f'Current directory location {currLoc}')
    if COLAB is True:
        dest = './'
        if subfolder:
            upath = dest + subfolder + '/' + cloudfile
            srcpath = subfolder + '/' + cloudfile
            destpath = dest
        else:
            destpath = dest + '/' + cloudfile
            srcpath = cloudfile
    else:
        dest = '../datasets'
        if subfolder:
            upath = dest + subfolder + '/' + cloudfile
            srcpath = subfolder + '/' + cloudfile
            destpath = dest + '/' + subfolder
        else:
            destpath = dest
            srcpath = cloudfile

    print(f'src: {srcpath} dest: {destpath}')

    # download the file
    getFromGCS(srcpath, destpath)

    # unzip the file if zipped
    if isZipped is True:
        import zipfile
        if subfolder:
            fullpath = dest + '/' + subfolder + '/'
            if COLAB == False:
                os.chdir(fullpath)
                print(f'current dir {os.getcwd()}')
            fullpath = cloudfile
        else:
            fullpath = dest + '/' + cloudfile
            if COLAB == False:
                os.chdir(dest)
        print(f'fullpath: {fullpath}')
        zref = zipfile.ZipFile(fullpath, 'r')
        zref.extractall()
        zref.close()
        os.chdir(currLoc)
        print(f'current dir {os.getcwd()}')

# ==========================================================================
# Miscellaeous Utility Functions:
# ==========================================================================
def set_seed(seed: int = 2345):
    """
    Sets a random seed
    Args:
        seed (int, optional): Random seed to set - default 2345
    """

    np.random.seed(seed)

def convert(seconds):
    import time
    """
    Convert the time data into hour minutes and seconds
    """
    return time.strftime("%H:%M:%s", time.gmtime(seconds))

from IPython.display import Markdown, display
def printmd(string, color=None):
    """
    Function to print string as markdown
    """
    pd.options.display.float_format = '{:.4f}'.format
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))

# ==========================================================================
# Classfication Related Utility Functions:
# ==========================================================================
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def plotConfusionMatrix(y_test, y_pred, class_names):
    """
    Plot the confusiton matrix
    Args:
        y_test: test labels
        y_pred: predicted labels
        class_names: label values
    """
    cm = confusion_matrix(y_test, y_pred)
    cseg = class_names
    cm_df = pd.DataFrame(cm, index = cseg, columns = cseg)
    plt.figure(figsize = (5,4))
    sns.heatmap(cm_df, annot=True, cmap=plt.cm.Blues, fmt = 'g', annot_kws={"size": 16})
    sns.set(font_scale=0.5)
    plt.title('Confusion Matrix\n', fontsize = 18)
    plt.ylabel('True label', fontsize = 16)
    plt.xlabel('Predicted label', fontsize = 16);

def printSummary(y_test, y_pred):
    """
    Print summary - number of correct and incorrect predictions for multi-class classification.
    
    Args:
        y_test: Actual test labels
        y_pred: Predicted labels
    """
    cm = confusion_matrix(y_test, y_pred)
    
    # Correct predictions: Sum of diagonal elements in confusion matrix
    correct = cm.diagonal().sum()
    
    # Errors: Sum of all off-diagonal elements
    error = cm.sum() - correct
    total = correct + error

    print(f'✅ Correct predictions: {correct} of {total}')
    print(f'❌ Errored predictions: {error} of {total}')
    #print(f'Confusion Matrix:\n{cm}')


def plotRoC(classifier, Xtest, ytest, title = " "):
    """
    Function to plot Receiver Operating Curve (ROC)
    Args:
        classifier: classification model used
        Xtest: Testing data
        ytest: Testing labels
        title: Title for the plot
    """
    fig,ax = plt.subplots(figsize = (6,4))
    classAUC = roc_auc_score(ytest, classifier.predict(Xtest))
    fpr, tpr, thresholds = roc_curve(ytest, classifier.predict_proba(Xtest)[:,1])
    auc = str(np.round(classAUC, 4))
    # disable axes
    ax.grid(False)
    # set background color to white
    ax.set_facecolor('white')
    # set the border around the axes
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess', color = 'red')
    plt.plot(fpr, tpr, label = "Train AUC " + auc)
    plt.ylabel('True Positive Rate', fontsize = 16)
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.title('RoC ' + title, fontsize = 16)
    plt.legend(loc = 4, fontsize = 16, facecolor = 'white');

def extract_classification_report_df(y_true, y_pred):
    """
    Function will return classfication report as dataframe
    Args:
        y_true: testing labels from dataset
        y_pred: predicted labels
    """
    # Extract the classification report as a dictionary
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert the dictionary to a pandas DataFrame for easy inspection
    df = pd.DataFrame(report).transpose()
    return df


def extract_classification_report_info(y_true, y_pred):
    """
    Function that will return information from the classification report as
    key value pairs
    Args:
        y_true: testing labels from dataset
        y_pred: predicted labels
    """

    # Generate the classification report as a string
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract information from the report
    accuracy = report['accuracy']  # Accuracy is available in the report as a key
    labels = list(report.keys())[:-3]  # The last three entries are for accuracy, macro avg, and weighted avg
    labels_info = {label: report[label] for label in labels}
    
    # Prepare the extracted information in a structured format (e.g., a dictionary)
    extracted_info = {
        'accuracy': accuracy,
        'labels': labels,
        'precision': {label: labels_info[label]['precision'] for label in labels},
        'recall': {label: labels_info[label]['recall'] for label in labels},
        'f1-score': {label: labels_info[label]['f1-score'] for label in labels},
        'support': {label: labels_info[label]['support'] for label in labels}
    }

    # Return the information as a dictionary
    return extracted_info

# Example usage:
# Assuming you have `y_test` and `y_pred` from your model
# y_true = [0, 1, 1, 0, 1, 0]  # Example ground truth values
# y_pred = [0, 1, 0, 0, 1, 1]  # Example predicted values

# info = extract_classification_report_info(y_true, y_pred)
# print(info)


def get_dataframe(y_test, y_pred, class_names, modelname):
    """
    Function the used the extract_classification_report_info to parse information
    that can be coverted to a dataframe
    Args:
        y_test: test labels from dataset
        y_pred: predicted labels
        class_names: label values
        modelname: classifier model used
    Returns a dataframe with appropriate classification report

    """
    data = extract_classification_report_info(y_test, y_pred)
    labels = data['labels']
    idx = 0
    retdata = []
    for i in labels:
        #print(f"precision {data['precision'][i]}, recall: {data['recall'][i]}, f1: {data['f1-score'][i]}, support: {data['support'][i]}")
        row = dict()
        if idx == 0:
            row['model'] = modelname
        else:
            row['model'] = ""
        row['class'] = class_names[idx]
        if idx == 0:
            row['accuracy'] = data['accuracy']
        else:
            row['accuracy'] = ""
        row['precision'] = data['precision'][i]
        row['recall'] = data['recall'][i]
        row['f1-score'] = data['f1-score'][i]
        row['support'] = data['support'][i]
        idx += 1
        retdata.append(row)
    df = pd.DataFrame(retdata)
    return df

def plotTrainingCurves(history, figuresize = (8,6)):
    """
    Plots the training and loss curves
    Args:
        history: returned from the model training
        figuresize: figsize parameter for matplotlib - a tuple (8,6)
    """

    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = figuresize)
    epochs = range(len(history.history['loss']))

    # Plot accuracy
    ax1.plot(epochs, history.history['accuracy'], label = 'accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], label = 'validation')
    ax1.set(xlabel = 'Epochs', ylabel = 'accuracy')
    ax1.legend()
    ax1.set_title('Training Accuracy')

    # Plot Loss
    ax2.plot(epochs, history.history['loss'], label = 'loss')
    ax2.plot(epochs, history.history['val_loss'], label = 'validation')
    #ax2.ylabel(['loss', 'vla_loss'])
    ax2.set(xlabel='Epochs', ylabel = 'loss')
    ax2.legend()
    ax2.set_title('Training Loss')

    plt.axis(True);


def plotLossCurve(history, loss_name = 'loss', 
                  figuresize = (5,4), 
                  ylim = None):
    """
    Plots the loss curves
    Args:
        history: returned from the model training
        loss_name: name of loss function used
        figuresize: figsize parameter for matplotlib - a tuple (5,4)
        ylim: y-axis limits
    """
    epochs = np.arange(0, len(history.history[loss_name]))
    plt.figure(figsize = figuresize)
    #plt.style.use('ggplot')
    plt.plot(epochs, history.history[loss_name], label = loss_name)
    plt.plot(epochs, history.history['val_' + loss_name], label = 'validation')
    if ylim:
        plt.ylim = ylim

    plt.ylabel(loss_name)
    plt.xlabel('Epochs')
    plt.legend();


def plotLossCurve_PyTorch(history, loss_name = 'loss', 
                  figuresize = (5,4), 
                  ylim = None):
    """
    Plots the loss curves
    Args:
        history: returned from the model training
        loss_name: name of loss function used
        figuresize: figsize parameter for matplotlib - a tuple (5,4)
        ylim: y-axis limits
    """
    epochs = np.arange(0, len(history[loss_name]))
    plt.figure(figsize = figuresize)
    #plt.style.use('ggplot')
    plt.plot(epochs, history[loss_name], label = loss_name)
    plt.plot(epochs, history['val_' + loss_name], label = 'validation')
    if ylim:
        plt.ylim = ylim

    plt.ylabel(loss_name)
    plt.xlabel('Epochs')
    plt.legend();

# if __name__ == '__main__':
#     fname = 'Pneumonia.zip'
#     subfolder = 'Medical'
#     isZipped = True
#     downloadFromCloud(fname, isZipped = isZipped, subfolder = subfolder)

