
import datasets

import numpy as np

from sklearn.metrics import (
    classification_report
)


# @datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class EE(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="hello",
            citation="hello",
            inputs_description="hello",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            codebase_urls=["hello"],
            reference_urls=["hello"]
        )

    def _compute(self, predictions, references):
        
        scores = self._scores(predictions, references)
        
        return scores
    
    def _accuracy(self, predictions, references):
        Correct = 0
        for pred, gt in zip(predictions, references):
            if pred == gt:
                Correct += 1 
        return Correct/len(predictions)
    
    def _scores(self, predictions, references):
        
        total_labels = []
        total_prelabels = []
        
        Correct = 0
        
        for pred, gt in zip(predictions, references):
            if gt == 32000:
                total_labels.append(0)
                if pred == 32000:
                    total_prelabels.append(0)
                else:
                    total_prelabels.append(1)
            else:
                total_labels.append(1)
                if pred == gt:
                    total_prelabels.append(1)
                else:
                    total_prelabels.append(0)
            if pred == gt:
                Correct += 1
    
        
        total_prelabels = np.array(total_prelabels)
        total_labels = np.array(total_labels)
                
        classifi_report = classification_report(
            total_labels, total_prelabels, target_names=[0, 1], output_dict=True
        )
        positive_precision = classifi_report[1]["precision"]
        positive_recall = classifi_report[1]["recall"]
        positive_f1_score = classifi_report[1]["f1-score"]
        negative_precision = classifi_report[0]["precision"]
        negative_recall = classifi_report[0]["recall"]
        negative_f1 = classifi_report[0]["f1-score"]

                
                
        return {
            "accuracy": Correct / len(predictions),
            "Positive_Precision": positive_precision,
            "Positive_Recall": positive_recall,
            "Positive_F1": positive_f1_score,
            "Negative_Precision": negative_precision,
            "Negative_Recall": negative_recall,
            "Negative_F1": negative_f1,
            }
                
    
    
    
    