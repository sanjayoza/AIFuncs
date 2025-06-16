from pyspark.sql.functions import *
from pyspark.sql.types import *

def getClassMetrics(preds, labelCol = 'label', predCol = 'prediction'):
    '''
    Pass in the predicition dataframe, dataset target column should be label and 
    the prediction column should be called prediction
    Returns: accuracy, precision, recall and f1-measure
    '''
    tp = preds.filter((preds[labelCol] == 1) & (preds[predCol] == 1)).count()
    tn = preds.filter((preds[labelCol] == 0) & (preds[predCol] == 0)).count()
    fp = preds.filter((preds[labelCol] == 0) & (preds[predCol] == 1)).count()
    fn = preds.filter((preds[labelCol] == 1) & (preds[predCol] == 0)).count()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    
    return accuracy, precision, recall, f1
