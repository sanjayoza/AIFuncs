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
    Print summary - number of correct and incorrect predictions - pass the confusion matrix
    Args:
        y_test: test labels
        y_pred: predicted labels
    """
    cm = confusion_matrix(y_test, y_pred)
    correct = cm[0, 0] + cm[1, 1]
    error = cm[0, 1] + cm[1,0]
    total = correct + error
    print('Correct predictions: {} of {}'.format(correct, total))
    print('Errored predictions: {} of {}'. format(error, total))


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
    plt.ylabel('False Positive Rate', fontsize = 16)
    plt.xlabel('True Positive Rate', fontsize = 16)
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

# if __name__ == '__main__':
#     fname = 'Pneumonia.zip'
#     subfolder = 'Medical'
#     isZipped = True
#     downloadFromCloud(fname, isZipped = isZipped, subfolder = subfolder)

