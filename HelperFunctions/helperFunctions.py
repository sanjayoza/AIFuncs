"""
Helper functions that will be used in various notebooks
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import tensorflow as tf
import subprocess


def walk_through_dir(dir_path):
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


def set_seed(seed: int = 2345):
    """
    Sets a random seed

    Args:
        seed (int, optional): Random seed to set - default 2345

    """

    tf.random.set_seed(seed)
    np.random.seed(seed)


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


def setUpTransferLearning(cloud_file, COLAB = False):
    """
    Set files for transfer learning.
    Copy the zip file from GCS and unzip it
    Args:
        cloud_file: name of the zipfile to download from GCS
        COLAB: flag to indicate if working in COLAB environment
    """

    currLoc = os.getcwd()
    print(f'Current directory location {currLoc}')
    if COLAB is True:
        dest = './'
        upath = dest + '/FoodClasses'
    else:
        dest = '../datasets'
        upath = dest + '/FoodClasses'

    # download the zip file
    getFromGCS(cloud_file, dest)

    fullpath = dest + '/' + cloud_file
    print(f'fullpath: {fullpath}')
    print(f'upath: {upath}')
    os.chdir(dest)

    # unzip the file
    import zipfile
    zref = zipfile.ZipFile(fullpath, 'r')
    zref.extractall()
    zref.close()
    walk_through_dir(upath)
    return upath
