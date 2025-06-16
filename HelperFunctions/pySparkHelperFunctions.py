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

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

def evaluate_predictions(predictions_df, labelCol, predCol, binary_class = False):
    """
    Evaluates a PySpark DataFrame containing predictions and prints accuracy, false positives, true positives, F1 score, and AUC.
    
    Args:
        predictions_df (DataFrame): PySpark DataFrame with 'label' and 'prediction' columns
        binary_class: flag to indicate if predictions are binary or multi-class classification
    """
    # Accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol=predCol, metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions_df)

    # F1 Score
    f1_evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol=predCol, metricName="f1")
    f1_score = f1_evaluator.evaluate(predictions_df)

    # Precision (True Positives)
    precision_evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol=predCol, metricName="precisionByLabel")
    true_positive = precision_evaluator.evaluate(predictions_df)

    # Recall (False Positives)
    recall_evaluator = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol=predCol, metricName="falsePositiveRateByLabel")
    false_positive = recall_evaluator.evaluate(predictions_df)

    if binary_class == True:
        # AUC (Area Under ROC Curve)
        auc_evaluator = BinaryClassificationEvaluator(labelCol=labelCol, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = auc_evaluator.evaluate(predictions_df)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"True Positive Rate: {true_positive:.4f}")
    print(f"False Positive Rate: {false_positive:.4f}")

    if binary_class == True:
        print(f"AUC (Area Under ROC): {auc:.4f}")

# Example usage:
# evaluate_predictions(predictions, "Class", "prediction", binary_class = True)
